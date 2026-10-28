# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, RotaryPositionEmbedding1D, PositionGetter  # MODIFIED: 1D temporal RoPE
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_FLEX_FN = None   # compiled flex_attention, built on first use (Aggregator._flex_attention)



_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        temporal_every=3,   # NEW: insert one temporal block every `temporal_every` aa-blocks (decision A3)
        # learned column mask on global attention (D1, docs/topics): per global block, one keep
        # weight g = sigmoid(theta) per patch POSITION p', shared by every non-reference frame
        enable_colmask: bool = False,
        colmask_grid=(30, 37),           # patch grid (h, w) theta lives on; SCARED crop = 518x420
        colmask_init: float = 5.0,       # theta init -> g = 0.993
        colmask_rho: float = 0.25,       # target mean keep ratio rho*
        colmask_ramp_frac: float = 0.6,  # rho_t falls 1 -> rho* over this fraction of steps
        colmask_st_frac: float = 0.2,    # last fraction of steps: straight-through hard top-k
        colmask_diag: bool = True,       # a query always sees its own key (AVGGT diagonal)
        colmask_mean: str = "count",     # dropped-key summary token: "count" | "one" | "none"
        colmask_path: Optional[str] = None,  # FIXED hard mask from a .npy bool [depth, h*w]; no theta,
                                             # nothing saved in the ckpt (D1 mask fine-tuning, 3-4)
        # frame-distance tile mask (3-3): mask set by |query frame - key frame| (log buckets) and the
        # key's position modulo an 8x8 tile; one table shared by global blocks [relmask_start, depth)
        enable_relmask: bool = False,
        relmask_path: Optional[str] = None,  # FIXED hard mask .npy bool [n_buckets, th, tw] (3-3 B / eval)
        relmask_buckets=(1, 2, 4, 8, 16),    # bucket upper bounds: 1 | 2 | 3-4 | 5-8 | 9-16 | >=17
        relmask_tile=(8, 8),
        relmask_start: int = 8,
        relmask_grid=(30, 37),
        relmask_init: float = 5.0,           # theta = init + N(0, init_std^2), seeded
        relmask_init_std: float = 0.01,
        relmask_seed: int = 0,
        relmask_rho: float = 0.25,
        relmask_ramp_frac: float = 0.6,
        relmask_st_frac: float = 0.2,
        relmask_budget_S: int = 64,          # budget weights = (query, key>=1) frame pairs per bucket at S
        # 3-7: "per_bucket" = each bucket keeps rho of ITS keys (cell weight = keys per tile slot),
        # tau_b solved in the forward, g = sigmoid((theta - tau_b) / T); "shared" = 3-3 projection.
        relmask_budget: str = "shared",
        relmask_rho_steps: int = 0,          # per_bucket: rho_t 1 -> rho over steps [0, rho_steps]
        relmask_T_start: float = 1.0,
        relmask_T_end: float = 0.05,
        relmask_T_steps=(0, 0),              # T = T_start before T_steps[0], linear to T_end at [1]
        relmask_hard_from: int = 0,          # per_bucket: straight-through hard mask from this step
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        # NEW: temporal RoPE — independent 1D rotary embedding applied only inside temporal attention,
        #      keeping the spatial 2D RoPE untouched so pretrained weights warm-start cleanly (docs §3.1).
        self.temporal_rope = RotaryPositionEmbedding1D(frequency=rope_freq) if rope_freq > 0 else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for i in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for i in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        self.temporal_every = temporal_every  # NEW

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # NEW: temporal attention blocks (Dyn-VGGT contribution ①). Only built when "temporal" is in aa_order,
        #      so default VGGT (aa_order=["frame","global"]) is byte-for-byte unchanged & pretrained-loadable.
        #      Inserted once every `temporal_every` aa-blocks → n_temporal = aa_block_num // temporal_every.
        #      Each block warm-starts as identity via LayerScale gamma=0 (docs §3.2).
        if "temporal" in self.aa_order:
            self.n_temporal = self.aa_block_num // self.temporal_every
            self.temporal_blocks = nn.ModuleList(
                [
                    block_fn(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        init_values=init_values,
                        qk_norm=qk_norm,
                        rope=self.temporal_rope,
                    )
                    for _ in range(self.n_temporal)
                ]
            )
            # γ=0 warm-start: zero both LayerScale gammas so each temporal block is an identity map at init.
            # (Block uses nn.Identity() when init_values is falsy, so we must zero an *existing* LayerScale.)
            for blk in self.temporal_blocks:
                if hasattr(blk.ls1, "gamma"):
                    nn.init.zeros_(blk.ls1.gamma)
                if hasattr(blk.ls2, "gamma"):
                    nn.init.zeros_(blk.ls2.gamma)
        else:
            self.n_temporal = 0
            self.temporal_blocks = None

        # Column mask (D1). Every global block l gets theta_l [h*w]; the key of patch p' in any
        # frame except frame 0 has log g_l(p') = logsigmoid(theta_l(p')) added to its logit.
        # Frame 0 and all special tokens are never masked. Two AVGGT patches (both optional):
        #   diagonal  the query's own key is exempt from its bias
        #   mean      one extra K/V token = the (1-g)-weighted mean of the gated keys, standing in
        #             for the dropped ones. "count" gives it logit weight log(sum(1-g)), i.e. it
        #             approximates the dropped keys' total softmax mass (n * exp(q.k_mean)); "one"
        #             gives it the weight of a single key (capped at log 1).
        # The budget is NOT a loss term: after each optimizer step colmask_post_step_ shifts all
        # theta by one constant so that mean sigmoid(theta) <= rho_t exactly. Only the ORDER of
        # theta is learned, which leaves the per-layer allocation free (shared total budget).
        # Training forward is soft g until the last st_frac of steps, then straight-through on the
        # exact global top-k. Eval forward defaults to that hard top-k at rho* ("hard"); "soft",
        # "off" (plain blocks, bit-identical to the base) and an explicit keep override (random
        # control) are eval switches that do not travel with the checkpoint.
        if isinstance(colmask_mean, bool):          # yaml `colmask_mean: true` = the 3-2 setting
            colmask_mean = "count" if colmask_mean else "none"
        if colmask_mean not in ("count", "one", "none"):
            raise ValueError(f"colmask_mean must be count, one or none, got {colmask_mean}")
        self.enable_colmask = enable_colmask
        self.colmask_eval_mode = "hard"
        self.colmask_keep_override = None    # eval-only bool [depth, h*w]
        self.colmask_theta = None
        self.colmask_fixed = None
        if enable_colmask and colmask_path is not None:
            import numpy as _np
            self.set_colmask_fixed(_np.load(colmask_path), diag=colmask_diag, mean=colmask_mean,
                                   grid=colmask_grid)
        elif enable_colmask:
            gh, gw = int(colmask_grid[0]), int(colmask_grid[1])
            self.colmask_theta = nn.Parameter(torch.full((depth, gh * gw), float(colmask_init)))
            for name, val in (("colmask_grid", torch.tensor([gh, gw])),
                              ("colmask_rho", torch.tensor(float(colmask_rho))),
                              ("colmask_ramp_frac", torch.tensor(float(colmask_ramp_frac))),
                              ("colmask_st_frac", torch.tensor(float(colmask_st_frac))),
                              ("colmask_diag", torch.tensor(int(bool(colmask_diag)))),
                              ("colmask_mean_id", torch.tensor({"none": 0, "one": 1, "count": 2}[colmask_mean])),
                              ("colmask_steps", torch.zeros((), dtype=torch.long)),
                              ("colmask_total_steps", torch.zeros((), dtype=torch.long))):
                self.register_buffer(name, val, persistent=True)

        # Frame-distance tile mask (3-3). Key (frame s', patch (y', x')) seen from a query in frame s
        # gets log g[bucket(|s - s'|), y' mod th, x' mod tw]. Never masked: same frame (|s-s'| = 0,
        # which includes duplicate frames), key frame 0, special keys, the query's own key. One mean
        # token PER QUERY FRAME (its dropped set differs by frame), visible only to that frame.
        # Frame distance uses the REAL frame indices when the trainer passes frame_pos (batch ids),
        # else clip positions (identical on contiguous clips, e.g. every test chunk).
        # Learned (theta, A) or fixed from .npy (B / eval). Budget as in colmask: projection after
        # each step, here on the mean g WEIGHTED by how many keys each cell stands for in a
        # budget_S-frame contiguous chunk (pairs per bucket x cells per tile slot on the grid).
        if enable_relmask and enable_colmask:
            raise ValueError("enable_relmask and enable_colmask are mutually exclusive")
        self.enable_relmask = bool(enable_relmask)
        self.relmask_eval_mode = "hard"
        self.relmask_keep_override = None    # eval-only bool [n_buckets, th, tw]
        self.relmask_theta = None
        if enable_relmask:
            self.relmask_ub = tuple(int(u) for u in relmask_buckets)
            self.relmask_nb = len(self.relmask_ub) + 1
            self.relmask_th, self.relmask_tw = int(relmask_tile[0]), int(relmask_tile[1])
            self.relmask_start = int(relmask_start)
            self.relmask_gh, self.relmask_gw = int(relmask_grid[0]), int(relmask_grid[1])
            self.register_buffer("relmask_cell_w", self._relmask_cell_weights(int(relmask_budget_S)),
                                 persistent=False)
            assert relmask_budget in ("shared", "per_bucket"), relmask_budget
            self.relmask_budget = str(relmask_budget)
            self.relmask_rho_steps = int(relmask_rho_steps)
            self.relmask_T_start, self.relmask_T_end = float(relmask_T_start), float(relmask_T_end)
            self.relmask_T_steps = (int(relmask_T_steps[0]), int(relmask_T_steps[1]))
            self.relmask_hard_from = int(relmask_hard_from)
            if relmask_path is not None:
                import numpy as _np
                keep = torch.as_tensor(_np.load(relmask_path)).bool()
                assert keep.shape == (self.relmask_nb, self.relmask_th, self.relmask_tw), keep.shape
                self.register_buffer("relmask_fixed", keep, persistent=False)
            else:
                self.relmask_fixed = None
                gen = torch.Generator().manual_seed(int(relmask_seed))
                self.relmask_theta = nn.Parameter(
                    float(relmask_init) + float(relmask_init_std)
                    * torch.randn(self.relmask_nb, self.relmask_th, self.relmask_tw, generator=gen))
                for name, val in (("relmask_rho", torch.tensor(float(relmask_rho))),
                                  ("relmask_ramp_frac", torch.tensor(float(relmask_ramp_frac))),
                                  ("relmask_st_frac", torch.tensor(float(relmask_st_frac))),
                                  ("relmask_steps", torch.zeros((), dtype=torch.long)),
                                  ("relmask_total_steps", torch.zeros((), dtype=torch.long))):
                    self.register_buffer(name, val, persistent=True)

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        frame_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            frame_pos: [B, S] optional real frame indices. When given, the temporal blocks' 1D RoPE
                uses each frame's offset from the clip's first frame instead of its position 0..S-1
                (identical on contiguous frames), and relmask measures frame distance with it.

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        # NEW: temporal positions for the 1D time RoPE, shape (B*P, S) integer frame indices (decision B4).
        #      camera token (idx 0) and patch tokens (idx >= patch_start_idx) get the real frame index t;
        #      register tokens (idx 1..patch_start_idx-1) get 0 → identity rotation (no temporal RoPE).
        temporal_pos = None
        if self.n_temporal and self.temporal_rope is not None:
            if frame_pos is None:
                frame_index = torch.arange(S, device=images.device).view(1, 1, S)
            else:
                fp = frame_pos.to(images.device).long()
                frame_index = (fp - fp[:, :1]).view(B, 1, S)
            temporal_pos = frame_index.expand(B, P, S).clone()  # (B, P, S)
            if self.patch_start_idx > 1:
                temporal_pos[:, 1:self.patch_start_idx, :] = 0  # register tokens → no temporal RoPE
            temporal_pos = temporal_pos.reshape(B * P, S)

        frame_idx = 0
        global_idx = 0
        temporal_idx = 0
        output_list = []

        rel_idx = None
        if self.enable_relmask and (self.training or self.relmask_eval_mode != "off") and S > 1:
            if frame_pos is None:
                rp = torch.arange(S, device=images.device).view(1, S).expand(B, S)
            else:
                rp = frame_pos.to(images.device).long()
            rel_idx = self._relmask_index(rp, P, images.device)

        for block_iter in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos, rel_idx=rel_idx,
                    )
                elif attn_type == "temporal":
                    # NEW: run a temporal block once every `temporal_every` aa-blocks.
                    #      It only updates the streaming `tokens`; it does NOT emit an intermediate
                    #      into output_list, so the head input stays [B,S,P,2C] (decision A2).
                    if self.n_temporal and (block_iter % self.temporal_every == self.temporal_every - 1):
                        tokens, temporal_idx = self._process_temporal_attention(
                            tokens, B, S, P, C, temporal_idx, pos=temporal_pos,
                        )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape[:2] != (B * S, P):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, rel_idx=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).

        Extension: relmask (rel_idx given, blocks >= relmask_start) or colmask adds a learned /
        fixed key bias inside flex attention; otherwise the block is the original one.
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape[:2] != (B, S * P):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if rel_idx is not None and global_idx >= self.relmask_start:
                lw_tab, dw_tab = self._relmask_tables(tokens.device)
                blk = self.global_blocks[global_idx]
                _B, _S, _ri = B, S, rel_idx

                def _rm_fn(t, p, lw, dw):  # noqa: E306
                    return self._relmask_global_block_forward(blk, t, p, lw, dw, _ri, _B, _S)

                if self.training:
                    tokens = checkpoint(_rm_fn, tokens, pos, lw_tab, dw_tab, use_reentrant=self.use_reentrant)
                else:
                    tokens = _rm_fn(tokens, pos, lw_tab, dw_tab)
            elif self.enable_colmask and (self.training or self.colmask_eval_mode != "off"):
                key_logw, drop_w = self._colmask_key_bias(global_idx, S, P, tokens.device)
                blk = self.global_blocks[global_idx]
                _B, _S = B, S

                def _cm_fn(t, p, lw, dw):  # noqa: E306
                    return self._colmask_global_block_forward(blk, t, p, lw, dw, _B, _S)

                if self.training:
                    tokens = checkpoint(_cm_fn, tokens, pos, key_logw, drop_w,
                                        use_reentrant=self.use_reentrant)
                else:
                    tokens = _cm_fn(tokens, pos, key_logw, drop_w)
            else:
                if self.training:
                    tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
                else:
                    tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates

    @staticmethod
    def _flex_attention(q, k, v, score_mod, kernel_options=None):
        """flex_attention, compiled once per process (eager flex is several times slower)."""
        global _FLEX_FN
        if _FLEX_FN is None:
            from torch.nn.attention.flex_attention import flex_attention as _fa
            # Frames per batch vary (img_nums [4, 12]), so each distinct shape compiles its own
            # kernel. Dynamo's default cache_size_limit of 8 is hit around step 9, after which it
            # falls back to EAGER flex -- which raises immediately ("get value out of a tracing
            # tensor"). Measured 2026-09-27: 8 good steps then a crash; with the limit raised, 12
            # varied (B, S) combinations all pass. automatic_dynamic_shapes off keeps every shape
            # specialised instead of switching to dynamic (which trips the same error).
            torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
            torch._dynamo.config.automatic_dynamic_shapes = False
            _FLEX_FN = torch.compile(_fa, dynamic=False)
        if kernel_options:
            return _FLEX_FN(q, k, v, score_mod=score_mod, kernel_options=kernel_options)
        return _FLEX_FN(q, k, v, score_mod=score_mod)

    # ---- column mask (D1) -------------------------------------------------------------------
    CM_DROP = -1.0e4     # finite stand-in for log 0: keeps the straight-through sum well-defined

    def _colmask_rho_t(self) -> float:
        """Budget schedule: 1 -> rho* linearly over ramp_frac of the run, then rho*."""
        rho, total = float(self.colmask_rho), int(self.colmask_total_steps)
        ramp = float(self.colmask_ramp_frac) * total
        if total <= 0 or ramp <= 0:
            return rho
        return 1.0 - (1.0 - rho) * min(1.0, float(int(self.colmask_steps)) / ramp)

    def _colmask_st_on(self) -> bool:
        total = int(self.colmask_total_steps)
        return total > 0 and int(self.colmask_steps) >= (1.0 - float(self.colmask_st_frac)) * total

    @torch.no_grad()
    def colmask_hard_keep(self, rho: Optional[float] = None) -> torch.Tensor:
        """Global top-k over ALL layers' theta, k = round(rho * depth * h*w). bool [depth, h*w]."""
        th = self.colmask_theta.detach()
        rho = float(self.colmask_rho) if rho is None else float(rho)
        k = int(round(rho * th.numel()))
        keep = torch.zeros(th.numel(), dtype=torch.bool, device=th.device)
        if k > 0:
            keep[th.flatten().topk(k).indices] = True
        return keep.view_as(th)

    @torch.no_grad()
    def colmask_random_keep(self, seed: int) -> torch.Tensor:
        """Control: the learned mask's per-layer keep COUNTS, at uniformly random positions."""
        counts = self.colmask_hard_keep().sum(1).tolist()
        n = self.colmask_theta.shape[1]
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        keep = torch.zeros(len(counts), n, dtype=torch.bool)
        for l, c in enumerate(counts):
            keep[l, torch.randperm(n, generator=gen)[: int(c)]] = True
        return keep.to(self.colmask_theta.device)

    @torch.no_grad()
    def colmask_post_step_(self):
        """After optimizer.step(): advance the counter, then project theta onto the budget.

        If mean sigmoid(theta) > rho_t, find the one constant c < 0 with mean sigmoid(theta + c)
        == rho_t (bisection; the mean is monotone in c) and add it to every theta. Below the
        budget nothing moves, so the task loss is free to raise keys back up to it.
        """
        if self.colmask_theta is None:
            return
        self.colmask_steps += 1
        th = self.colmask_theta
        rho_t = self._colmask_rho_t()
        if torch.sigmoid(th).mean().item() <= rho_t:
            return
        lo, hi = -60.0, 0.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if torch.sigmoid(th + mid).mean().item() > rho_t:
                hi = mid
            else:
                lo = mid
        th.add_(lo)

    @torch.no_grad()
    def colmask_stats(self) -> dict:
        """Readout: rho_t, soft mean g, and the hard top-k keep fraction per layer."""
        if self.colmask_theta is None:
            return {"fixed": True, "keep_frac": self.colmask_fixed.float().mean().item(),
                    "keep_per_layer": [round(x, 4) for x in self.colmask_fixed.float().mean(1).tolist()]}
        keep = self.colmask_hard_keep()
        return {"step": int(self.colmask_steps), "rho_t": self._colmask_rho_t(),
                "g_mean": torch.sigmoid(self.colmask_theta).mean().item(),
                "st": self._colmask_st_on(),
                "keep_per_layer": [round(x, 4) for x in keep.float().mean(1).tolist()]}

    def _colmask_key_bias(self, global_idx: int, S: int, P: int, device):
        """Per-key additive logit log g [S*P] and dropped weight 1-g [S*P] for one global block.

        Frame 0 and the special tokens get (0, 0). Training: soft, or straight-through on the
        hard top-k in the last st_frac (forward = hard values exactly, backward = d/dtheta of
        the soft ones). Eval: hard top-k (or the override), or soft.
        """
        psi = self.patch_start_idx
        P_patch = P - psi
        gh, gw = (int(x) for x in self.colmask_grid)
        if P_patch != gh * gw:
            raise ValueError(f"colmask theta is a {gh}x{gw} grid ({gh * gw} patches) but this "
                             f"input has {P_patch} patches per frame; train/eval at the same size")
        if self.colmask_theta is None:      # fixed mask (3-4): hard, train and eval alike
            keep = (self.colmask_keep_override if (self.colmask_keep_override is not None and not self.training)
                    else self.colmask_fixed)[global_idx].to(device)
            lw, dw = torch.where(keep, 0.0, self.CM_DROP).float(), (~keep).float()
            zeros_sp = lw.new_zeros(psi)
            frame_lw, frame_dw = torch.cat([zeros_sp, lw]), torch.cat([zeros_sp, dw])
            z = frame_lw.new_zeros(P)
            return (torch.cat([z, frame_lw.repeat(S - 1)]) if S > 1 else z,
                    torch.cat([z, frame_dw.repeat(S - 1)]) if S > 1 else z)
        th = self.colmask_theta[global_idx]
        soft_lw, soft_dw = F.logsigmoid(th), 1.0 - torch.sigmoid(th)
        hard = (not self.training and self.colmask_eval_mode == "hard") or \
               (self.training and self._colmask_st_on())
        if hard:
            keep = self.colmask_keep_override if (self.colmask_keep_override is not None
                                                   and not self.training) else self.colmask_hard_keep()
            keep = keep[global_idx].to(device)
            hard_lw = torch.where(keep, 0.0, self.CM_DROP).to(th.dtype)
            hard_dw = (~keep).to(th.dtype)
            if self.training:
                lw = hard_lw + (soft_lw - soft_lw.detach())
                dw = hard_dw + (soft_dw - soft_dw.detach())
            else:
                lw, dw = hard_lw, hard_dw
        else:
            lw, dw = soft_lw, soft_dw
        zeros_sp = lw.new_zeros(psi)
        frame_lw = torch.cat([zeros_sp, lw])                          # [P]
        frame_dw = torch.cat([zeros_sp, dw])
        if S > 1:
            key_lw = torch.cat([frame_lw.new_zeros(P), frame_lw.repeat(S - 1)])
            key_dw = torch.cat([frame_dw.new_zeros(P), frame_dw.repeat(S - 1)])
        else:
            key_lw, key_dw = frame_lw.new_zeros(P), frame_dw.new_zeros(P)
        return key_lw.to(device), key_dw.to(device)

    def _colmask_global_block_forward(self, block, tokens, pos, key_lw, key_dw, B, S):
        """One global block with the column mask, via flex_attention (nothing N x N in memory).

        score(q, k) = q.k/sqrt(D) + key_lw[k]   except k == q when the diagonal patch is on.
        The mean token is appended as key N with logit weight from colmask_mean.
        """
        N, C_dim = tokens.shape[1], tokens.shape[2]
        H, D = block.attn.num_heads, block.attn.head_dim

        x_norm = block.norm1(tokens)
        q, k, v = block.attn.qkv(x_norm).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = block.attn.q_norm(q), block.attn.k_norm(k)
        if block.attn.rope is not None and pos is not None:
            q, k = block.attn.rope(q, pos), block.attn.rope(k, pos)
        assert (block.attn.attn_drop.p if self.training else 0.0) == 0.0, \
            "flex path does not implement attention dropout"
        q, k = q.to(v.dtype), k.to(v.dtype)       # q_norm/k_norm come out fp32 under autocast

        mean_id = int(self.colmask_mean_id)
        bias = key_lw.float()
        if mean_id != 0:
            w = key_dw.float()
            wsum = w.sum()
            denom = wsum.clamp(min=1e-12)
            k_mean = torch.einsum("bhnd,n->bhd", k.float(), w) / denom          # [B, H, D]
            v_mean = torch.einsum("bhnd,n->bhd", v.float(), w) / denom
            k = torch.cat([k, k_mean.unsqueeze(2).to(k.dtype)], dim=2)
            v = torch.cat([v, v_mean.unsqueeze(2).to(v.dtype)], dim=2)
            # nothing dropped -> weight ~ CM_DROP, so the token vanishes and the block is exact
            logw = wsum.clamp(min=1e-30).log() if mean_id == 2 else wsum.clamp(min=1e-30, max=1.0).log()
            bias = torch.cat([bias, logw.clamp(min=self.CM_DROP).view(1)])
        diag = bool(int(self.colmask_diag))

        if diag:
            def _score_mod(score, b, h, q_idx, kv_idx):
                return torch.where(q_idx == kv_idx, score, score + bias[kv_idx])
        else:
            def _score_mod(score, b, h, q_idx, kv_idx):
                return score + bias[kv_idx]

        attn_out = self._flex_attention(q, k, v, score_mod=_score_mod)        # [B, H, N, D]
        attn_out = block.attn.proj_drop(block.attn.proj(attn_out.permute(0, 2, 1, 3).reshape(B, N, C_dim)))
        tokens = tokens + block.ls1(attn_out)
        return tokens + block.ls2(block.mlp(block.norm2(tokens)))

    def set_colmask_fixed(self, keep, diag=True, mean="count", grid=(30, 37)):
        """Attach a FIXED hard column mask (bool [depth, h*w]) -- for 3-4 fine-tuning and its eval.
        Plain attributes + a non-persistent buffer, so a checkpoint carries no colmask keys and the
        mask always comes from the .npy (eval passes --colmask)."""
        if isinstance(mean, bool):
            mean = "count" if mean else "none"
        keep = torch.as_tensor(keep).bool()
        gh, gw = int(grid[0]), int(grid[1])
        assert keep.shape == (self.depth, gh * gw), f"colmask file shape {tuple(keep.shape)}"
        self.enable_colmask = True
        self.colmask_theta = None
        self.colmask_grid = (gh, gw)
        self.colmask_diag = int(bool(diag))
        self.colmask_mean_id = {"none": 0, "one": 1, "count": 2}[mean]
        if "colmask_fixed" in self.__dict__:
            del self.__dict__["colmask_fixed"]
        dev = self.camera_token.device if hasattr(self, "camera_token") else torch.device("cpu")
        self.register_buffer("colmask_fixed", keep.to(dev), persistent=False)

    def set_relmask_fixed(self, keep, buckets=(1, 2, 4, 8, 16), tile=(8, 8), start=8, grid=(30, 37),
                          budget_S=64):
        """Attach a FIXED frame-distance tile mask (bool [nb, th, tw]) to a model built without it
        (eval of any ckpt with --relmask). Non-persistent, like set_colmask_fixed."""
        self.enable_relmask = True
        self.relmask_eval_mode = "hard"
        self.relmask_keep_override = None
        self.relmask_theta = None
        self.relmask_ub = tuple(int(u) for u in buckets)
        self.relmask_nb = len(self.relmask_ub) + 1
        self.relmask_th, self.relmask_tw = int(tile[0]), int(tile[1])
        self.relmask_start = int(start)
        self.relmask_gh, self.relmask_gw = int(grid[0]), int(grid[1])
        self.relmask_budget = "shared"
        keep = torch.as_tensor(keep).bool()
        assert keep.shape == (self.relmask_nb, self.relmask_th, self.relmask_tw), keep.shape
        dev = self.camera_token.device if hasattr(self, "camera_token") else torch.device("cpu")
        for name, val in (("relmask_cell_w", self._relmask_cell_weights(int(budget_S))),
                          ("relmask_fixed", keep)):
            if name in self.__dict__:
                del self.__dict__[name]
            self.register_buffer(name, val.to(dev), persistent=False)

    # ---- frame-distance tile mask (3-3) -------------------------------------------------------
    def _relmask_bucket(self, d: torch.Tensor) -> torch.Tensor:
        """|s - s'| -> bucket index (0 for d = 1, ..., n_buckets-1 for d > last upper bound)."""
        b = torch.zeros_like(d)
        for u in self.relmask_ub:
            b = b + (d > u).to(d.dtype)
        return b

    def _relmask_cell_weights(self, S: int) -> torch.Tensor:
        """[nb, th, tw]: keys each cell stands for in an S-frame contiguous chunk = (query frame,
        key frame >= 1, different frame) pairs in its bucket x grid cells landing on its tile slot."""
        t = torch.arange(S)
        d = (t[:, None] - t[None, :]).abs()
        valid = (d > 0) & (t[None, :] >= 1)
        w = torch.bincount(self._relmask_bucket(d)[valid], minlength=self.relmask_nb).float()
        y = torch.arange(self.relmask_gh) % self.relmask_th
        x = torch.arange(self.relmask_gw) % self.relmask_tw
        cnt = torch.zeros(self.relmask_th, self.relmask_tw)
        cnt.index_put_((y[:, None].expand(-1, self.relmask_gw).reshape(-1),
                        x[None, :].expand(self.relmask_gh, -1).reshape(-1)),
                       torch.ones(self.relmask_gh * self.relmask_gw), accumulate=True)
        return w[:, None, None] * cnt[None]

    def _relmask_rho_t(self) -> float:
        rho, total = float(self.relmask_rho), int(self.relmask_total_steps)
        ramp = float(self.relmask_ramp_frac) * total
        if total <= 0 or ramp <= 0:
            return rho
        return 1.0 - (1.0 - rho) * min(1.0, float(int(self.relmask_steps)) / ramp)

    def _relmask_st_on(self) -> bool:
        total = int(self.relmask_total_steps)
        return total > 0 and int(self.relmask_steps) >= (1.0 - float(self.relmask_st_frac)) * total

    def _relmask_wmean(self, g: torch.Tensor) -> torch.Tensor:
        w = self.relmask_cell_w.to(g.device)
        return (w * g).sum() / w.sum()

    # ---- 3-7 per-bucket budget -------------------------------------------------------------
    def _relmask_key_w(self, device=None) -> torch.Tensor:
        """[th*tw]: share of a bucket's keys each tile slot stands for (sums to 1)."""
        c = self.relmask_cell_w[0].flatten()
        return (c / c.sum()).to(device if device is not None else c.device)

    def _relmask_pb_rho_t(self) -> float:
        rho, n, r = float(self.relmask_rho), int(self.relmask_steps), int(self.relmask_rho_steps)
        return rho if r <= 0 else 1.0 - (1.0 - rho) * min(1.0, n / r)

    def _relmask_pb_T(self) -> float:
        n, (a, b) = int(self.relmask_steps), self.relmask_T_steps
        if n <= a or b <= a:
            return self.relmask_T_start if n <= a else self.relmask_T_end
        return self.relmask_T_start + (self.relmask_T_end - self.relmask_T_start) * min(1.0, (n - a) / (b - a))

    def _relmask_pb_hard_on(self) -> bool:
        return self.relmask_hard_from > 0 and int(self.relmask_steps) >= self.relmask_hard_from

    @staticmethod
    def _relmask_pb_hard_keep(theta2d: torch.Tensor, rho: float, W: torch.Tensor) -> torch.Tensor:
        """[nb, n]: per bucket, cells by theta descending until the kept key share is closest to rho."""
        h = torch.zeros_like(theta2d, dtype=torch.bool)
        for b in range(theta2d.shape[0]):
            order = theta2d[b].argsort(descending=True)
            n = int((W[order].cumsum(0) - rho).abs().argmin()) + 1
            h[b, order[:n]] = True
        return h

    def _relmask_pb_gate(self, rho: Optional[float] = None, T: Optional[float] = None):
        """(g [nb, n] with grad, tau [nb, 1]): tau_b by bisection so sum_cells W * g = rho per bucket."""
        th = self.relmask_theta.flatten(1)
        W = self._relmask_key_w(th.device).to(th.dtype)
        rho = self._relmask_pb_rho_t() if rho is None else float(rho)
        T = self._relmask_pb_T() if T is None else float(T)
        with torch.no_grad():
            t = th.detach()
            lo = t.min(1, keepdim=True).values - 20 * T
            hi = t.max(1, keepdim=True).values + 20 * T
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                m = (torch.sigmoid((t - mid) / T) * W).sum(1, keepdim=True)
                lo = torch.where(m > rho, mid, lo)
                hi = torch.where(m > rho, hi, mid)
            tau = 0.5 * (lo + hi)
        z = (th - tau) / T
        return torch.sigmoid(z), tau, z

    @torch.no_grad()
    def relmask_hard_keep(self, rho: Optional[float] = None) -> torch.Tensor:
        """theta descending, cells taken until their weighted share reaches rho. bool [nb, th, tw]."""
        if self.relmask_budget == "per_bucket":
            rho = float(self.relmask_rho) if rho is None else float(rho)
            th = self.relmask_theta.detach()
            return self._relmask_pb_hard_keep(th.flatten(1), rho, self._relmask_key_w(th.device)).view_as(th)
        th = self.relmask_theta.detach()
        rho = float(self.relmask_rho) if rho is None else float(rho)
        w = self.relmask_cell_w.to(th.device).flatten()
        order = th.flatten().argsort(descending=True)
        acc = w[order].cumsum(0) / w.sum()
        n = min(int(torch.searchsorted(acc, torch.tensor([rho], device=acc.device)).item()) + 1, th.numel())
        keep = torch.zeros(th.numel(), dtype=torch.bool, device=th.device)
        keep[order[:n]] = True
        return keep.view_as(th)

    @torch.no_grad()
    def relmask_post_step_(self):
        """Counter + projection of theta onto the WEIGHTED budget (see colmask_post_step_)."""
        self.relmask_steps += 1
        if self.relmask_budget == "per_bucket":
            return      # budget lives in the forward (tau_b), theta is never shifted
        th = self.relmask_theta
        rho_t = self._relmask_rho_t()
        if self._relmask_wmean(torch.sigmoid(th)).item() <= rho_t:
            return
        lo, hi = -60.0, 0.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if self._relmask_wmean(torch.sigmoid(th + mid)).item() > rho_t:
                hi = mid
            else:
                lo = mid
        th.add_(lo)

    @torch.no_grad()
    def relmask_stats(self) -> dict:
        if self.relmask_theta is None:
            keep = self.relmask_fixed
            W = self._relmask_key_w(keep.device)
            return {"fixed": True, "keep_per_bucket": [round(x, 4) for x in keep.float().flatten(1).mean(1).tolist()],
                    "keyfrac_per_bucket": [round(x, 4) for x in (keep.float().flatten(1) * W).sum(1).tolist()]}
        if self.relmask_budget == "per_bucket":
            W = self._relmask_key_w(self.relmask_theta.device)
            g, _, _ = self._relmask_pb_gate()
            keep = self.relmask_hard_keep().flatten(1)
            hk = self._relmask_pb_hard_keep(self.relmask_theta.detach().flatten(1), self._relmask_pb_rho_t(), W)
            return {"step": int(self.relmask_steps), "rho_t": round(self._relmask_pb_rho_t(), 5),
                    "T": round(self._relmask_pb_T(), 5), "hard": self._relmask_pb_hard_on(),
                    "g_keyfrac_per_bucket": [round(x, 5) for x in (g * W).sum(1).tolist()],
                    "hard_t_keyfrac_per_bucket": [round(x, 4) for x in (hk.float() * W).sum(1).tolist()],
                    "theta_std": self.relmask_theta.detach().std().item(),
                    "g_mid_frac": ((g > 0.1) & (g < 0.9)).float().mean().item(),
                    "cells_per_bucket": keep.sum(1).tolist(),
                    "keyfrac_per_bucket": [round(x, 4) for x in (keep.float() * W).sum(1).tolist()],
                    "keep_hash": int((keep.flatten().long() * torch.arange(1, keep.numel() + 1,
                                      device=keep.device)).sum().item())}
        keep = self.relmask_hard_keep()
        g = torch.sigmoid(self.relmask_theta)
        return {"step": int(self.relmask_steps), "rho_t": self._relmask_rho_t(),
                "g_wmean": self._relmask_wmean(g).item(), "st": self._relmask_st_on(),
                "theta_std": self.relmask_theta.detach().std().item(),
                "hard_wmean": self._relmask_wmean(keep.float()).item(),
                "g_mid_frac": ((g > 0.1) & (g < 0.9)).float().mean().item(),
                "keep_per_bucket": [round(x, 4) for x in keep.float().flatten(1).mean(1).tolist()],
                "keep_hash": int((keep.flatten().long() * torch.arange(1, keep.numel() + 1,
                                  device=keep.device)).sum().item())}

    def _relmask_tables(self, device):
        """(log g, drop weight) per (bucket, tile slot), each [nb, th*tw]: soft, straight-through
        or hard exactly as _colmask_key_bias decides for the column mask."""
        if self.relmask_theta is None:
            keep = (self.relmask_keep_override if (self.relmask_keep_override is not None and not self.training)
                    else self.relmask_fixed).to(device)
            return (torch.where(keep, 0.0, self.CM_DROP).flatten(1).float(),
                    (~keep).flatten(1).float())
        th = self.relmask_theta
        if self.relmask_budget == "per_bucket":
            if not self.training and self.relmask_eval_mode == "hard":
                keep = (self.relmask_keep_override if self.relmask_keep_override is not None
                        else self.relmask_hard_keep()).to(device)
                return (torch.where(keep, 0.0, self.CM_DROP).flatten(1).float(), (~keep).flatten(1).float())
            g, _, z = self._relmask_pb_gate()
            # logsigmoid, not log(sigmoid): at T ~ 0.05 sigmoid underflows to exactly 0 and
            # log(0).clamp backprops 0 * inf = NaN, which makes GradScaler skip every step.
            soft_lw, soft_dw = F.logsigmoid(z).clamp(min=-30.0), torch.sigmoid(-z)
            if not (self.training and self._relmask_pb_hard_on()):
                return soft_lw, soft_dw
            h = self._relmask_pb_hard_keep(th.detach().flatten(1), self._relmask_pb_rho_t(),
                                           self._relmask_key_w(th.device))
            hard_lw = torch.where(h, 0.0, self.CM_DROP).to(g.dtype)
            hard_dw = (~h).to(g.dtype)
            return hard_lw + soft_lw - soft_lw.detach(), hard_dw + soft_dw - soft_dw.detach()
        soft_lw, soft_dw = F.logsigmoid(th), 1.0 - torch.sigmoid(th)
        hard = (not self.training and self.relmask_eval_mode == "hard") or \
               (self.training and self._relmask_st_on())
        if not hard:
            return soft_lw.flatten(1), soft_dw.flatten(1)
        keep = (self.relmask_keep_override if (self.relmask_keep_override is not None and not self.training)
                else self.relmask_hard_keep()).to(device)
        hard_lw = torch.where(keep, 0.0, self.CM_DROP).to(th.dtype)
        hard_dw = (~keep).to(th.dtype)
        if self.training:
            return ((hard_lw + soft_lw - soft_lw.detach()).flatten(1),
                    (hard_dw + soft_dw - soft_dw.detach()).flatten(1))
        return hard_lw.flatten(1), hard_dw.flatten(1)

    def _relmask_index(self, rel_pos: torch.Tensor, P: int, device):
        """Integer tables for the score_mod: bucket [B,S,S], always-keep [B,S,S], tile slot [P],
        special [P]. rel_pos [B, S] = frame index of each clip position."""
        d = (rel_pos[:, :, None] - rel_pos[:, None, :]).abs()                     # [B, S, S]
        keepf = (d == 0)
        keepf[:, :, 0] = True                                                     # key frame 0
        bt = self._relmask_bucket(d).clamp(max=self.relmask_nb - 1)
        pk = torch.arange(P, device=device) - self.patch_start_idx
        special = pk < 0
        pk = pk.clamp(min=0)
        y, x = pk // self.relmask_gw, pk % self.relmask_gw
        tile = (y % self.relmask_th) * self.relmask_tw + (x % self.relmask_tw)
        return bt.to(device), keepf.to(device), tile, special

    RM_FLEX_OPTS = {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1}

    def _relmask_score_mod(self, lw, idx, mean_w, N, P, S):
        """score_mod for flex: patch keys get lw[bucket, tile] unless always-kept; kv >= N are the
        per-query-frame mean tokens (weight mean_w, -1e4 for other frames). Separate so the startup
        check can evaluate it on hand-picked (q, k) pairs."""
        bt, keepf, tile, special = idx
        drop = float(self.CM_DROP)

        def _score_mod(score, b, h, q_idx, kv_idx):
            fq = q_idx // P
            kvc = torch.minimum(kv_idx, torch.full_like(kv_idx, N - 1))
            fk = kvc // P
            pk = kvc % P
            keep = (q_idx == kv_idx) | keepf[b, fq, fk] | special[pk]
            bias = torch.where(keep, 0.0, lw[bt[b, fq, fk], tile[pk]])
            m = torch.clamp(kv_idx - N, min=0, max=S - 1)
            mean_s = torch.where(fq == m, score + mean_w[b, m], score + drop)
            return torch.where(kv_idx >= N, mean_s, score + bias)
        return _score_mod

    def _relmask_bias_table(self, lw, idx, mean_w, N, P, S):
        """[B, S, N + S]: bias of key kv for any query in frame s (patch keys, then mean tokens)."""
        bt, keepf, tile, special = idx
        B = bt.shape[0]
        pb = lw[bt][..., tile]                                                    # [B, S, S', P]
        pb = pb.masked_fill(keepf.unsqueeze(-1) | special, 0.0).reshape(B, S, N)
        eye = torch.eye(S, dtype=torch.bool, device=pb.device)
        mb = torch.where(eye, mean_w[:, None, :], torch.full_like(mean_w[:, None, :], float(self.CM_DROP)))
        return torch.cat([pb, mb], dim=2)

    @staticmethod
    def _relmask_table_score_mod(table, P):
        def _score_mod(score, b, h, q_idx, kv_idx):
            return score + table[b, q_idx // P, kv_idx]
        return _score_mod

    def _relmask_global_block_forward(self, block, tokens, pos, lw_tab, dw_tab, idx, B, S):
        """One global block with the frame-distance tile mask (flex_attention). kv = N keys + S mean
        tokens; mean token m is visible only to queries of frame m."""
        bt, keepf, tile, special = idx
        N, C_dim = tokens.shape[1], tokens.shape[2]
        P = N // S
        H, D = block.attn.num_heads, block.attn.head_dim
        x_norm = block.norm1(tokens)
        q, k, v = block.attn.qkv(x_norm).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = block.attn.q_norm(q), block.attn.k_norm(k)
        if block.attn.rope is not None and pos is not None:
            q, k = block.attn.rope(q, pos), block.attn.rope(k, pos)
        assert (block.attn.attn_drop.p if self.training else 0.0) == 0.0
        q, k = q.to(v.dtype), k.to(v.dtype)

        # drop weight of key (s', p) as seen from query frame s: [B, S, S, P]
        dmat = dw_tab.float()[bt][..., tile]                                      # [B, S, S, P]
        dmat = dmat * (~keepf).unsqueeze(-1) * (~special)
        wsum = dmat.sum((2, 3))                                                   # [B, S]
        denom = wsum.clamp(min=1e-12)[:, None, :, None]
        k_mean = torch.einsum("bsqp,bhqpd->bhsd", dmat, k.float().view(B, H, S, P, D)) / denom
        v_mean = torch.einsum("bsqp,bhqpd->bhsd", dmat, v.float().view(B, H, S, P, D)) / denom
        k = torch.cat([k, k_mean.to(k.dtype)], dim=2)
        v = torch.cat([v, v_mean.to(v.dtype)], dim=2)
        mean_w = wsum.clamp(min=1e-30).log().clamp(min=self.CM_DROP)               # [B, S]
        # The bias depends only on (query FRAME, key): the query's own key sits in its own frame,
        # which is always kept, so the diagonal rule adds nothing here. Materialise it as one
        # [B, S, N + S] table and let the kernel do a single lookup per score. (The per-score
        # gather version, _relmask_score_mod, ran 410 s/step at S=32 vs 14 s for colmask; it
        # stays as the reference the startup check compares against.)
        table = self._relmask_bias_table(lw_tab.float(), idx, mean_w, N, P, S)
        _score_mod = self._relmask_table_score_mod(table, P)
        # default flex tiles need >101 KB shared memory for this captured table (RTX 5090)
        attn_out = self._flex_attention(q, k, v, score_mod=_score_mod, kernel_options=self.RM_FLEX_OPTS)
        attn_out = block.attn.proj_drop(block.attn.proj(attn_out.permute(0, 2, 1, 3).reshape(B, N, C_dim)))
        tokens = tokens + block.ls1(attn_out)
        return tokens + block.ls2(block.mlp(block.norm2(tokens)))

    def _process_temporal_attention(self, tokens, B, S, P, C, temporal_idx, pos=None):
        # NEW: temporal attention (Dyn-VGGT contribution ①). Reshape so the *time* axis S is the
        #      sequence dim — each spatial position attends across its own S frames (motion/trajectory).
        #      Updates the streaming tokens only; emits NO intermediate (head input stays 2C, decision A2).
        """
        Process one temporal attention block. Tokens are reshaped to (B*P, S, C) so attention runs
        purely along the time axis, then reshaped back to (B*S, P, C) for the next attention type.
        """
        # (B*S, P, C) or (B, S*P, C) -> (B, S, P, C) -> (B*P, S, C)
        tokens = tokens.view(B, S, P, C).permute(0, 2, 1, 3).reshape(B * P, S, C)

        if self.training:
            tokens = checkpoint(self.temporal_blocks[temporal_idx], tokens, pos, use_reentrant=self.use_reentrant)
        else:
            tokens = self.temporal_blocks[temporal_idx](tokens, pos=pos)
        temporal_idx += 1

        # (B*P, S, C) -> (B, P, S, C) -> (B*S, P, C)
        tokens = tokens.view(B, P, S, C).permute(0, 2, 1, 3).reshape(B * S, P, C)

        return tokens, temporal_idx


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
