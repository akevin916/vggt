# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, RotaryPositionEmbedding1D, PositionGetter  # MODIFIED: import 1D temporal RoPE for Dyn-VGGT
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Motion-Gated Camera Aggregation (docs/method.md §3)
# ---------------------------------------------------------------------------

class GatePredictor(nn.Module):
    """
    Mid-aggregator motion gate predictor (Dyn-VGGT gate method, docs/method.md §3).

    Takes patch tokens from the middle of the aggregator (~1/3 depth) and
    predicts a per-patch dynamic logit g ∈ ℝ.  Downstream uses:
      1. Attention bias: bias = clamp(softplus(0) − softplus(g), max=0) added to
         camera/register query rows in subsequent global blocks → structural
         motion gating. DETACHED BY DEFAULT, so the pose loss does not train the
         gate; Aggregator(gate_pose_grad=True) removes the detach and makes the
         downstream loss the gate's teacher (for domains with no mask labels).
         Note the clamp is flat for g < 0, so even then that path carries zero
         gradient there — see _gated_global_block_forward.
      2. Loss supervision (never detached): L_gate = BCE(σ(g), m*)  where m* is
         the GT dynamic mask averaged to patch resolution.

    Zero-init on the last linear so g≡0 at training step 0, which makes the
    initial attention bias ≡0 and keeps the forward pass byte-for-byte equal
    to pretrained VGGT (warm-start property).
    """

    def __init__(self, embed_dim: int, hidden_ratio: int = 4):
        super().__init__()
        hidden_dim = embed_dim // hidden_ratio
        self.norm = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, 1)
        # Zero-init → g=0 at t=0 → bias=0 → pretrained VGGT behaviour preserved
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: [B, S, P_patch, C]  patch tokens only (no camera/register)
        Returns:
            g: [B, S, P_patch]  per-patch dynamic logit
        """
        x = self.norm(patch_tokens)
        x = self.act(self.linear1(x))
        g = self.linear2(x).squeeze(-1)   # [B, S, P_patch]
        return g

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
        # Temporal design change (2026-09-20). Defaults to the ORIGINAL behaviour, so an existing
        # checkpoint reproduces bit-for-bit; switched on per-arm from the config.
        temporal_share_frame_weights: bool = False,
        #   True builds NO temporal blocks. Each temporal step reuses the weights of the frame
        #   block at the same depth (identical shape; RoPE carries no parameters) along the time
        #   axis, and the only new parameters are two LayerScale vectors per temporal step, zero
        #   initialised, so the warm start is still exact. Trainable parameters drop by ~100.8M
        #   (frame_blocks stay frozen), which is the point: with 22 training sequences the extra
        #   capacity of 8 fresh blocks is a liability, not an asset.
        # motion-gated camera aggregation
        enable_gate: bool = False,   # gate predictor + gated global attention
        gate_block_iter: int = 7,    # fire gate after this 0-indexed aa-block iteration (~1/3 of 24)
        gate_pose_grad: bool = False,  # let the downstream (pose) loss train the gate -- see below
        gate_leaky: float = 0.0,       # >0 replaces the bias clamp with a leaky one -- see below
        gate_bias_zero_ref: bool = False,  # drop the softplus(0) reference -- see below
        gate_bias_scale: Optional[float] = None,  # None = legacy bias; float = parametrised form
        gate_bias_a: float = 1.0,      # logit sharpening inside the parametrised bias
        gate_bias_tau: float = 0.0,    # logit offset inside the parametrised bias
        gate_bias_learn: bool = False,  # make the scale a trainable nn.Parameter -- see below
        # frame-distance band bias: a log-Gaussian band over |i-j| per (global block, head)
        enable_dual_stream: bool = False,
        dual_stream_scope: str = "camera",  # "camera": camera/register queries only; "all": every query
        dual_stream_start: int = 8,         # global blocks [start, depth) get the bias
        dual_stream_warmup_steps: int = 3000,   # train steps over which the schedule ramps 0 -> 1
        dual_stream_s_init: tuple = (0.09, 1.0),  # per-head band centres, log-spaced over this range
        dual_stream_sigma_init: float = 0.5,      # band width in log-delta units
        # illumination token (IlluVGGT minimal version): one extra special token per frame
        enable_illu: bool = False,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
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
                for _ in range(depth)
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
                for _ in range(depth)
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
        self.temporal_share_frame_weights = bool(temporal_share_frame_weights)
        if "temporal" in self.aa_order and self.temporal_share_frame_weights:
            # Weight-shared variant: no blocks of our own, just the LayerScales that decide how
            # much of the shared block's output is written back. Gamma 0 => identity at init.
            self.n_temporal = self.aa_block_num // self.temporal_every
            self.temporal_blocks = None
            self.temporal_ls = nn.ParameterList(
                [nn.Parameter(torch.zeros(embed_dim)) for _ in range(2 * self.n_temporal)]
            )
        elif "temporal" in self.aa_order:
            self.n_temporal = self.aa_block_num // self.temporal_every
            self.temporal_ls = None
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
            self.temporal_ls = None

        # motion gate predictor (§3)
        self.enable_gate = enable_gate
        self.gate_block_iter = gate_block_iter
        # The attention bias is detached by default, so L_camera CANNOT train the gate and the
        # only supervision is L_gate's BCE. On a domain with no dynamic annotation (SCARED)
        # that leaves the gate untrainable, so this flag opens the path: with it, the pose loss
        # itself is the gate's teacher. Off by default -- flipping it changes what every
        # existing gate run means, and it is an open question whether the gate converges to
        # anything meaningful or just degenerates into an attention-temperature knob.
        self.gate_pose_grad = gate_pose_grad
        # The bias clamp is flat for g < 0, so under gate_pose_grad the whole static half of
        # the logit range returns exactly zero gradient and a patch that drifts negative can
        # never come back. gate_leaky gives that half a slope (LeakyReLU's fix for dying
        # ReLU), at the cost of letting confident-static patches be boosted by up to
        # gate_leaky * log2 (0.07 at 0.1) instead of exactly 0. 0.0 keeps the hard clamp and
        # is byte-for-byte the old behaviour.
        self.gate_leaky = float(gate_leaky)
        # gate_bias_zero_ref: measure the bias from softplus(-inf)=0 instead of softplus(0).
        # The softplus(0) reference is what puts the clamp's kink at sigma(g)=0.5, i.e. it makes
        # the gate act ONLY on patches it believes are more likely dynamic than not. On this data
        # an honestly-calibrated gate reaches ~0.36 for the average dynamic patch, so that kink
        # leaves the bias identically 0 and the gate inert. With zero_ref the bias is
        # -softplus(g), which is <= 0 everywhere (so the clamp becomes a no-op) and varies
        # smoothly with the evidence: confident-static -> 0 (identical to no gate), mildly
        # suspicious -> mildly suppressed. It differs from gate_leaky=1.0 only by the constant
        # log(2) -- identical ordering AMONG patches, different only relative to the
        # camera/register keys, whose bias slot stays 0.
        # COST: it breaks the warm start. GatePredictor is zero-init so g==0 at step 0, and
        # softplus(0) is exactly what makes bias==0 there, i.e. byte-for-byte pretrained VGGT.
        # Under zero_ref every patch starts at -log(2) instead. Eval-only use is unaffected --
        # there is no initialisation to preserve when scoring a trained ckpt.
        self.gate_bias_zero_ref = bool(gate_bias_zero_ref)
        # Parametrised bias (2026-09-04 sweep):  bias = -s * softplus(a * (g - tau)).
        # It supersedes the reference-point variants above by giving the two things the legacy
        # form cannot express: a DEPTH (s) that is not locked to the BCE-calibrated logit scale,
        # and an OFFSET (tau) that decides where suppression starts. The expression is <= 0
        # everywhere, so no clamp is needed. s == 0 makes bias identically 0 REGARDLESS of g --
        # a strictly stronger warm start than softplus(0), which only holds while g == 0.
        # None keeps the legacy branch byte-for-byte. All three are read at forward time, so an
        # eval sweep can mutate them on a loaded model without rebuilding it.
        self.gate_bias_scale = None if gate_bias_scale is None else float(gate_bias_scale)
        self.gate_bias_a = float(gate_bias_a)
        self.gate_bias_tau = float(gate_bias_tau)
        # gate_bias_learn: promote the scale to an nn.Parameter so L_camera -- and only L_camera,
        # since s appears nowhere else -- decides HOW HARD to suppress. g stays detached into the
        # bias, so the predictor keeps BCE as its only teacher and s cannot re-label anything: a
        # single scalar is identical for every patch and cannot change their ordering. Init 0
        # makes the forward byte-for-byte equal to the warm start regardless of what g is, and s
        # still receives gradient there (d bias/d s = -softplus(a*(g-tau)) != 0). NOTE a and tau
        # stay fixed floats: their gradients are proportional to s, so they would be frozen at 0
        # anyway until s departs -- sweep them, do not train them.
        self.gate_bias_learn = bool(gate_bias_learn)
        if self.gate_bias_learn:
            self.gate_bias_scale_p = nn.Parameter(
                torch.tensor(float(gate_bias_scale or 0.0), dtype=torch.float32)
            )
        if enable_gate:
            self.gate_predictor = GatePredictor(embed_dim)
        else:
            self.gate_predictor = None

        # Dual-stream: a log-Gaussian BAND over frame distance on global block l >= start,
        #   b_{l,h}(i,j) = -beta_l^eff * (log d_ij - log s_{l,h})^2 / (2 sigma_{l,h}^2),
        #   d_ij = |i - j| / (S - 1)  in (0, 1].
        # Each head keeps a preferred temporal distance s (band centre) and a width sigma; at one
        # sigma from the centre the bias is exactly -beta/2, so beta is the DEPTH of the band
        # (per layer) and sigma its WIDTH (per head) -- they are not redundant.
        # The diagonal (own frame, where log d is -inf) and key frame 0 (the reference camera) are
        # forced to bias 0, never suppressed.
        #
        # beta_l^eff = min(1, step / warmup) * beta_l, with beta_l LEARNED from init 1.0. The
        # non-learnable ramp exists to break a chicken-and-egg: d(bias)/d(log s) and
        # d(bias)/d(log sigma) are both proportional to beta, so starting beta near zero (the
        # obvious way to warm-start) would also freeze the band parameters at their init and the
        # run would report "nothing happened" for reasons that have nothing to do with the method.
        # At step 0 the multiplier is exactly 0 and the whole biased path is skipped, so the
        # forward is bit-identical to pretrained VGGT -- a zero attn_mask would NOT be, it swaps
        # the flash kernel for the mem-efficient one (~1e-3 in bf16).
        # The per-head init fan is deliberately identical across layers: any depth-wise structure
        # that shows up in the trained model is then the model's, not the initialisation's.
        if enable_dual_stream and enable_gate:
            raise ValueError("enable_dual_stream and enable_gate are mutually exclusive")
        if dual_stream_scope not in ("camera", "all"):
            raise ValueError(f"dual_stream_scope must be 'camera' or 'all', got {dual_stream_scope}")
        self.enable_dual_stream = enable_dual_stream
        self.dual_stream_scope = dual_stream_scope
        self.dual_stream_start = int(dual_stream_start)
        # Plain attribute only for the disabled case; when the feature is on this is a buffer
        # (registered below) so the ramp length travels with the checkpoint.
        if not enable_dual_stream:
            self.dual_stream_warmup = int(dual_stream_warmup_steps)
        if enable_dual_stream:
            s_lo, s_hi = float(dual_stream_s_init[0]), float(dual_stream_s_init[1])
            # s_lo defaults to 1/11: with img_nums up to 12 the smallest non-zero d seen in
            # training is 1/(12-1), so a band centred below that would sit where the data has no
            # samples, collect no gradient, and masquerade as "the model chose a small scale".
            fan = torch.exp(torch.linspace(math.log(s_lo), math.log(s_hi), num_heads))  # [H]
            self.dual_stream_beta = nn.Parameter(torch.ones(depth))
            self.dual_stream_log_s = nn.Parameter(fan.log().expand(depth, num_heads).clone())
            self.dual_stream_log_sigma = nn.Parameter(
                torch.full((depth, num_heads), math.log(float(dual_stream_sigma_init)))
            )
            # Everything an eval needs to REPRODUCE this forward travels in the state_dict.
            # Anything that changes the bias arithmetic but is only a constructor default is a
            # silent-mismatch waiting to happen: the loader would reinterpret the same trained
            # log_s under different arithmetic and nothing would look wrong.
            #   steps       counter for the ramp; also keeps a resumed run on its schedule
            #   warmup      ramp length; without it a loader divides by its own default
            #   delta_mode  0 = d_ij is |i-j|/(S-1); a future absolute-gap variant takes 1
            #   keep_mode   0 = own frame and key frame 0 are pinned to bias 0
            # (Clamp bounds and the init fan are NOT stored: they shape training only, and what
            # they produced is already in the parameter values themselves.)
            self.register_buffer("dual_stream_steps", torch.zeros((), dtype=torch.long),
                                 persistent=True)
            self.register_buffer("dual_stream_warmup",
                                 torch.tensor(int(dual_stream_warmup_steps), dtype=torch.long),
                                 persistent=True)
            self.register_buffer("dual_stream_delta_mode", torch.zeros((), dtype=torch.long),
                                 persistent=True)
            self.register_buffer("dual_stream_keep_mode", torch.zeros((), dtype=torch.long),
                                 persistent=True)

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # Illumination token: sits right after the register tokens, so the per-frame layout becomes
        # [camera, register x4, illu, patch x P]. One slot shared by every frame -- unlike the
        # camera token there is no reason to single out the reference frame. With the flag off no
        # parameter is built and patch_start_idx is unchanged, so pretrained loading is untouched.
        # Every consumer reads patch_start_idx dynamically, so the shift propagates by itself.
        self.enable_illu = enable_illu
        self.illu_token = nn.Parameter(torch.randn(1, 1, 1, embed_dim)) if enable_illu else None

        # The patch tokens start after the camera, register (and illumination) tokens
        self.patch_start_idx = 1 + num_register_tokens + int(enable_illu)

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)
        if self.illu_token is not None:
            nn.init.normal_(self.illu_token, std=1e-6)

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
        gate_logits_override: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            gate_logits_override: [B, S, P_patch] optional, eval-only. When given, these
                logits (not the model's own GatePredictor output) are used to build the
                camera/register attention bias in every gated global block — for oracle-mask
                ablations (docs/method.md gate diagnostics). The model's own
                gate_logits are still computed and returned unaffected, for logging.

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
        special = [camera_token, register_token]
        if self.illu_token is not None:
            special.append(self.illu_token.reshape(1, 1, C).expand(B * S, 1, C))
        tokens = torch.cat(special + [patch_tokens], dim=1)

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
            frame_index = torch.arange(S, device=images.device)
            temporal_pos = frame_index.view(1, 1, S).expand(B, P, S).clone()  # (B, P, S)
            if self.patch_start_idx > 1:
                temporal_pos[:, 1:self.patch_start_idx, :] = 0  # register tokens → no temporal RoPE
            temporal_pos = temporal_pos.reshape(B * P, S)

        frame_idx = 0
        global_idx = 0
        temporal_idx = 0
        output_list = []

        # gate logits computed lazily after gate_block_iter; None until then
        gate_logits: Optional[torch.Tensor] = None

        # Band bias inputs. None (= plain global blocks, bit-identical to pretrained VGGT) when the
        # feature is off, when the ramp is still at 0, or when S < 2 (single-view eval: d would
        # divide by S-1 = 0).
        # The counter and the range projection are advanced by the TRAINER after optimizer.step()
        # (dual_stream_post_step_), never here: this forward body runs a second time during the
        # activation-checkpoint recomputation in backward, so an in-place side effect here would
        # silently count every step twice and run the schedule at double speed.
        pair_log_delta, pair_warm = None, 0.0
        if self.enable_dual_stream:
            pair_warm = self._dual_stream_warm()
            if pair_warm > 0.0 and S > 1:
                pair_log_delta = self._dual_stream_log_delta(S, images.device)
            elif self.training:
                # warm == 0 (the first optimizer step) or S < 2: the biased path is skipped, so
                # the band parameters would receive no gradient at all and DDP's reducer aborts
                # the NEXT iteration ("parameters that were not used in producing loss").
                # Adding exact 0.0 keeps the forward bit-identical while giving them a graph edge.
                tokens = tokens + 0.0 * (self.dual_stream_beta.sum()
                                         + self.dual_stream_log_s.sum()
                                         + self.dual_stream_log_sigma.sum())

        for block_iter in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    gate_bias_source = gate_logits_override if gate_logits_override is not None else gate_logits
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos, gate_logits=gate_bias_source,
                        pair_log_delta=pair_log_delta, pair_warm=pair_warm,
                    )
                elif attn_type == "temporal":
                    # NEW: run a temporal block once every `temporal_every` aa-blocks.
                    #      It only updates the streaming `tokens`; it does NOT emit an intermediate
                    #      into output_list, so the head input stays [B,S,P,2C] (decision A2).
                    if self.n_temporal and (block_iter % self.temporal_every == self.temporal_every - 1):
                        tokens, temporal_idx = self._process_temporal_attention(
                            tokens, B, S, P, C, temporal_idx, pos=temporal_pos,
                            shared_block=self.frame_blocks[block_iter] if self.temporal_share_frame_weights else None,
                        )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            # compute gate logits from patch tokens after gate_block_iter is complete.
            # tokens is in [B, S*P, C] after global attention; extract patch slice.
            if self.gate_predictor is not None and block_iter == self.gate_block_iter:
                tokens_4d = tokens.view(B, S, P, C)
                patch_tokens_mid = tokens_4d[:, :, self.patch_start_idx:, :]  # [B, S, P_patch, C]
                gate_logits = self.gate_predictor(patch_tokens_mid)            # [B, S, P_patch]

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx, gate_logits

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
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

    # Band-bias parameter ranges. Projected onto these every training forward rather than clamped
    # inside the expression: torch.clamp hands back a ZERO gradient outside the range, so a sigma
    # pushed past the lower bound would be dead there forever -- and "sigma grows until the band
    # switches itself off" is exactly the escape hatch this design wants to keep available.
    DS_BETA_RANGE = (0.0, 4.0)        # negative beta would invert the band into a repulsion
    DS_LOG_S_RANGE = (math.log(0.02), math.log(1.2))     # d lives in (0, 1]
    DS_LOG_SIGMA_RANGE = (math.log(0.3), math.log(3.0))  # lower bound: a band this narrow is a spike
    DS_BIAS_FLOOR = -20.0             # finite stand-in for -inf; keeps softmax well-defined

    @torch.no_grad()
    def dual_stream_post_step_(self):
        """Project the band parameters back into range and advance the schedule counter.

        Called by the trainer right after optimizer.step(), for two separate reasons:
          - counting there makes one increment == one OPTIMIZER step, independent of
            accum_steps and immune to the activation-checkpoint forward re-run;
          - clamping parameters in-place during forward bumps their autograd version counters
            and is exactly the kind of thing DDP hooks and torch.compile object to. After the
            step is just as effective: nothing reads them until the next forward.
        Note Adam keeps the momentum that pushed a parameter out of range, so one held at a
        bound can sit there a while and then jump once the gradient reverses.
        """
        self.dual_stream_beta.clamp_(*self.DS_BETA_RANGE)
        self.dual_stream_log_s.clamp_(*self.DS_LOG_S_RANGE)
        self.dual_stream_log_sigma.clamp_(*self.DS_LOG_SIGMA_RANGE)
        self.dual_stream_steps += 1

    def _dual_stream_warm(self) -> float:
        """Schedule multiplier min(1, step / warmup); 0 at step 0 => the biased path is skipped."""
        warmup = int(self.dual_stream_warmup)
        if warmup <= 0:
            return 1.0
        return min(1.0, float(int(self.dual_stream_steps)) / float(warmup))

    def _dual_stream_log_delta(self, S: int, device) -> torch.Tensor:
        """log d_ij with d = |i-j|/(S-1) in (0,1]. The diagonal (d=0) is a placeholder: the bias
        there is overwritten with 0 in _dual_stream_pair_bias. [S, S]

        A checkpoint carrying a different delta_mode was trained under different arithmetic, and
        scoring it with this one would silently reinterpret its log_s -- so refuse instead."""
        if int(self.dual_stream_delta_mode) != 0:
            raise ValueError(
                f"checkpoint dual_stream_delta_mode={int(self.dual_stream_delta_mode)} but this "
                "build only implements mode 0 (d = |i-j|/(S-1))"
            )
        t = torch.arange(S, device=device, dtype=torch.float32)
        d = (t[:, None] - t[None, :]).abs() / float(S - 1)
        return d.clamp(min=1e-6).log()

    def _dual_stream_pair_bias(self, log_delta, warm, global_idx):
        """b_{l,h}(i,j) = -warm * beta_l * (log d - log s_{l,h})^2 / (2 sigma_{l,h}^2). [H, S, S]

        Own frame and key frame 0 are pinned to 0 -- which also disposes of log(0) on the diagonal.
        """
        log_s = self.dual_stream_log_s[global_idx].view(-1, 1, 1)       # [H,1,1]
        sigma = self.dual_stream_log_sigma[global_idx].exp().view(-1, 1, 1)
        beta = warm * self.dual_stream_beta[global_idx]
        bias = -beta * (log_delta.unsqueeze(0) - log_s).pow(2) / (2.0 * sigma.pow(2))
        bias = bias.clamp(min=self.DS_BIAS_FLOOR)
        if int(self.dual_stream_keep_mode) != 0:
            raise ValueError(
                f"checkpoint dual_stream_keep_mode={int(self.dual_stream_keep_mode)} but this "
                "build only implements mode 0 (own frame + key frame 0 pinned to 0)"
            )
        S = log_delta.shape[-1]
        keep = torch.zeros(S, S, dtype=torch.bool, device=log_delta.device)
        keep[:, 0] = True
        keep.fill_diagonal_(True)
        return bias.masked_fill(keep, 0.0)

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, gate_logits=None,
                                  pair_log_delta=None, pair_warm=0.0):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).

        Extension: when gate_logits [B, S, P_patch] is provided, the camera and register
        query rows receive an additive attention bias of −softplus(gate_logits) on all patch
        key positions, structurally preventing dynamic patches from polluting the camera token.
        Patch↔patch attention is unmodified (flash-friendly, no bias). See §4.
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if pair_log_delta is not None and global_idx >= self.dual_stream_start:
                blk = self.global_blocks[global_idx]
                _B, _S, _psi, _gi, _w = B, S, self.patch_start_idx, global_idx, pair_warm

                def _dual_fn(t, p, ld):  # noqa: E306
                    return self._dual_global_block_forward(
                        blk, t, p, self._dual_stream_pair_bias(ld, _w, _gi), _B, _S, _psi
                    )

                if self.training:
                    tokens = checkpoint(_dual_fn, tokens, pos, pair_log_delta,
                                        use_reentrant=self.use_reentrant)
                else:
                    tokens = _dual_fn(tokens, pos, pair_log_delta)
            elif gate_logits is not None:
                # gated attention — split into patch-query path (flash, no bias) and
                # camera/register-query path (small, with per-patch-key bias).
                # detach unless gate_pose_grad: see __init__ for why the default is detached
                gate_bias = gate_logits if self.gate_pose_grad else gate_logits.detach()
                if self.training:
                    blk = self.global_blocks[global_idx]
                    _B, _S, _psi = B, S, self.patch_start_idx

                    def _gated_fn(t, p, g):  # noqa: E306
                        return self._gated_global_block_forward(blk, t, p, g, _B, _S, _psi)

                    # gate_bias is passed as a checkpoint ARGUMENT, not captured in the closure:
                    # a closure tensor gets no gradient under reentrant checkpointing, which
                    # would silently make gate_pose_grad a no-op if use_reentrant ever flips.
                    tokens = checkpoint(_gated_fn, tokens, pos, gate_bias,
                                        use_reentrant=self.use_reentrant)
                else:
                    tokens = self._gated_global_block_forward(
                        self.global_blocks[global_idx], tokens, pos, gate_bias, B, S, self.patch_start_idx
                    )
            else:
                if self.training:
                    tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
                else:
                    tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates

    def _gated_global_block_forward(
        self,
        block: nn.Module,
        tokens: torch.Tensor,
        pos: Optional[torch.Tensor],
        gate_logits_det: torch.Tensor,
        B: int,
        S: int,
        patch_start_idx: int,
    ) -> torch.Tensor:
        """
        One global attention block with motion-gated camera aggregation (docs/method.md §3.2).

        Splits the attention into two memory-efficient paths:
          • Path 1 — Patch queries → ALL keys: standard F.sdpa, no bias (flash-friendly).
          • Path 2 — Camera+register queries → ALL keys: small F.sdpa with per-patch-key
            additive bias = −softplus(gate_logits_det).  Only ~patch_start_idx*S queries,
            so the extra cost is negligible even without flash attention.

        The MLP sub-layer is unchanged.

        Args:
            block:            global_blocks[i]
            tokens:           [B, S*P, C]  (global layout)
            pos:              [B, S*P, 2]  or None
            gate_logits_det:  [B, S, P_patch]  detached gate logits
            B, S:             batch / sequence dims
            patch_start_idx:  #special tokens per frame (camera + register tokens)
        """
        N = tokens.shape[1]      # S * P
        P = N // S               # tokens per frame
        P_patch = P - patch_start_idx

        C_dim = tokens.shape[2]
        H = block.attn.num_heads
        D = block.attn.head_dim

        # ---- Attention sub-layer ----
        x_norm = block.norm1(tokens)

        # QKV projection: [B, N, 3*C] → split to [3, B, H, N, D]
        qkv = block.attn.qkv(x_norm).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)    # each [B, H, N, D]

        # QK norm (operates on last dim D, safe after reshape)
        q = block.attn.q_norm(q)
        k = block.attn.k_norm(k)

        # Rotary position embedding
        if block.attn.rope is not None and pos is not None:
            q = block.attn.rope(q, pos)
            k = block.attn.rope(k, pos)

        # IMPORTANT: tokens are FRAME-INTERLEAVED, not [all special | all patch].
        # Each frame s occupies positions s*P + [0..patch_start_idx) special
        # (camera+register), then s*P + [patch_start_idx..P) patch tokens. Reshape
        # the sequence axis to (S, P) so special/patch separate cleanly per frame.
        q_sp = q.view(B, H, S, P, D)
        q_special = q_sp[:, :, :, :patch_start_idx, :].reshape(B, H, S * patch_start_idx, D)
        q_patch   = q_sp[:, :, :, patch_start_idx:, :].reshape(B, H, S * P_patch, D)

        # Attention bias for special queries, in the SAME frame-interleaved KEY order:
        # 0 on special keys, min(0, softplus(0)−softplus(g)) on patch keys. [B, 1, 1, N]
        # broadcasts over heads and (special) query positions.
        #
        # Two properties, both wanted:
        #   • warm-start: at g=0 the bias is EXACTLY 0 (softplus(0)−softplus(0)=0), so a
        #     zero-init gate (g≡0) leaves the forward byte-for-byte equal to pretrained
        #     VGGT. (Plain −softplus(g) gives −ln2 at g=0, halving every patch key.)
        #   • suppress-only: clamping at max=0 keeps the gate a pure down-weighting mask —
        #     confident-static patches (g<0) get bias 0 (full weight), never a positive
        #     boost. Dynamic (g→+∞) → −∞ as before; the static-vs-dynamic ordering is
        #     unchanged.
        # The clamp has a kink at g=0. When the bias is detached (the default) no gradient
        # flows through it at all, so the non-smoothness is inert. Under gate_pose_grad=True
        # it is NOT inert: d(bias)/dg is −sigmoid(g) for g > 0 but EXACTLY 0 for g < 0, so
        # the whole static half of the logit range is a gradient dead zone. Any patch the pose
        # loss pushes below 0 stops receiving a direct gradient from there on, and a global
        # slide into g < 0 is an absorbing "gate is a no-op" state. Measured, not inferred.
        # Note the zero-init below only puts a FRESH model on the kink; a run warm-started
        # from a trained gate (e.g. inst_g.pt, whose linear2 is far from zero) starts with
        # some unknown fraction of patches already inside the dead zone. The forward value
        # stays continuous either way.
        bias_key = gate_logits_det.new_zeros(B, S, P)                # [B, S, P]
        if self.gate_bias_learn or self.gate_bias_scale is not None:
            # -s * softplus(a * (g - tau)): already <= 0 everywhere, so the clamp, gate_leaky and
            # the softplus(0) reference are all bypassed -- they are the legacy branch's tools.
            _s = self.gate_bias_scale_p if self.gate_bias_learn else self.gate_bias_scale
            _x = -_s * F.softplus(
                self.gate_bias_a * (gate_logits_det - self.gate_bias_tau)
            )
        else:
            _ref = 0.0 if self.gate_bias_zero_ref else math.log(2.0)
            _x = _ref - F.softplus(gate_logits_det)
            if self.gate_leaky > 0.0:
                # `<=` not `<`: at x == 0 (i.e. g == 0, where a zero-init gate starts) the strict
                # form would take the leaky branch and cut the gradient there by 1/gate_leaky --
                # weakening the one point on the curve that was already healthy.
                _x = torch.where(_x <= 0, _x, self.gate_leaky * _x)
            else:
                _x = torch.clamp(_x, max=0.0)
        bias_key[:, :, patch_start_idx:] = _x   # [B, S, P_patch] slot
        attn_bias = bias_key.reshape(B, 1, 1, N)                     # [B, 1, 1, N]

        drop_p = block.attn.attn_drop.p if self.training else 0.0

        # Path 1: patch queries attend to ALL keys — no bias, flash-friendly
        attn_patch = F.scaled_dot_product_attention(q_patch, k, v, dropout_p=drop_p)
        # Path 2: special queries attend to ALL keys — with bias (tiny op)
        attn_special = F.scaled_dot_product_attention(
            q_special, k, v,
            attn_mask=attn_bias.to(q_special.dtype),
            dropout_p=drop_p,
        )

        # Recombine into original frame-interleaved order [special|patch per frame]
        attn_special = attn_special.view(B, H, S, patch_start_idx, D)
        attn_patch   = attn_patch.view(B, H, S, P_patch, D)
        attn_out = torch.cat([attn_special, attn_patch], dim=3).reshape(B, H, N, D)

        # Output projection
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, C_dim)
        attn_out = block.attn.proj(attn_out)
        attn_out = block.attn.proj_drop(attn_out)

        # LayerScale + residual
        tokens = tokens + block.ls1(attn_out)

        # ---- MLP sub-layer (unchanged) ----
        tokens = tokens + block.ls2(block.mlp(block.norm2(tokens)))

        return tokens

    def _dual_global_block_forward(self, block, tokens, pos, pair_bias, B, S, patch_start_idx):
        """
        One global block with the frame-distance band bias pair_bias [H, S, S] (head, query frame,
        key frame), broadcast to every token of the key frame. Same qkv/norm/rope/ls path as
        _gated_global_block_forward.
          scope "camera": only camera/register queries are biased (patch queries unchanged, tiny op).
          scope "all":    every query is biased; builds an [B, 1, N, N] mask (no flash; flash is
                          already disabled in training by cuda.disable_flash_sdp).
        """
        N, C_dim = tokens.shape[1], tokens.shape[2]
        P = N // S
        H, D = block.attn.num_heads, block.attn.head_dim

        x_norm = block.norm1(tokens)
        q, k, v = block.attn.qkv(x_norm).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = block.attn.q_norm(q), block.attn.k_norm(k)
        if block.attn.rope is not None and pos is not None:
            q, k = block.attn.rope(q, pos), block.attn.rope(k, pos)
        drop_p = block.attn.attn_drop.p if self.training else 0.0

        # pair_bias is per head and batch-independent, so the mask carries a leading size-1 batch
        # dim and broadcasts over B: the [H, ...] axis is the only one that grew vs the per-layer
        # scalar version.
        # Cast the small [H,S,S] bias BEFORE expanding it, so the big mask is materialised once
        # in q's dtype instead of once in fp32 and once cast. (Measured at 50 frames: peak
        # allocation is unchanged, so the peak is set elsewhere -- this just does less work.)
        pair_bias = pair_bias.to(q.dtype)
        if self.dual_stream_scope == "all":
            mask = pair_bias[None, :, :, None, :, None].expand(1, H, S, P, S, P).reshape(1, H, N, N)
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop_p)
        else:
            P_patch = P - patch_start_idx
            q_sp = q.view(B, H, S, P, D)
            q_special = q_sp[:, :, :, :patch_start_idx, :].reshape(B, H, S * patch_start_idx, D)
            q_patch = q_sp[:, :, :, patch_start_idx:, :].reshape(B, H, S * P_patch, D)
            mask = pair_bias[None, :, :, None, :, None].expand(1, H, S, patch_start_idx, S, P)
            mask = mask.reshape(1, H, S * patch_start_idx, N)
            attn_patch = F.scaled_dot_product_attention(q_patch, k, v, dropout_p=drop_p)
            attn_special = F.scaled_dot_product_attention(q_special, k, v, attn_mask=mask,
                                                          dropout_p=drop_p)
            attn_out = torch.cat([attn_special.view(B, H, S, patch_start_idx, D),
                                  attn_patch.view(B, H, S, P_patch, D)], dim=3).reshape(B, H, N, D)

        attn_out = block.attn.proj_drop(block.attn.proj(attn_out.permute(0, 2, 1, 3).reshape(B, N, C_dim)))
        tokens = tokens + block.ls1(attn_out)
        return tokens + block.ls2(block.mlp(block.norm2(tokens)))

    def _shared_temporal_block_forward(self, tokens, block, pos, ls1, ls2):
        """One temporal step that BORROWS `block`'s weights (temporal_share_frame_weights).

        Identical algebra to Block.forward, except the rotary embedding is the 1D temporal one and
        the two LayerScales are ours (zero-init => exact warm start) instead of the frame block's.
        Written out rather than calling block.forward because the rope lives inside block.attn and
        swapping it there would not survive activation-checkpoint recomputation.
        """
        attn = block.attn
        B_, N, C_ = tokens.shape
        H, D = attn.num_heads, attn.head_dim
        q, k, v = attn.qkv(block.norm1(tokens)).reshape(B_, N, 3, H, D).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = attn.q_norm(q), attn.k_norm(k)
        if self.temporal_rope is not None and pos is not None:
            q, k = self.temporal_rope(q, pos), self.temporal_rope(k, pos)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=attn.attn_drop.p if self.training else 0.0)
        out = attn.proj_drop(attn.proj(out.permute(0, 2, 1, 3).reshape(B_, N, C_)))
        tokens = tokens + ls1 * out
        return tokens + ls2 * block.mlp(block.norm2(tokens))

    def _process_temporal_attention(self, tokens, B, S, P, C, temporal_idx, pos=None, shared_block=None):
        # NEW: temporal attention (Dyn-VGGT contribution ①). Reshape so the *time* axis S is the
        #      sequence dim — each spatial position attends across its own S frames (motion/trajectory).
        #      Updates the streaming tokens only; emits NO intermediate (head input stays 2C, decision A2).
        """
        Process one temporal attention block. Tokens are reshaped to (B*P, S, C) so attention runs
        purely along the time axis, then reshaped back to (B*S, P, C) for the next attention type.
        """
        # (B*S, P, C) or (B, S*P, C) -> (B, S, P, C) -> (B*P, S, C)
        tokens = tokens.view(B, S, P, C).permute(0, 2, 1, 3).reshape(B * P, S, C)

        if shared_block is not None:
            ls1, ls2 = self.temporal_ls[2 * temporal_idx], self.temporal_ls[2 * temporal_idx + 1]
            if self.training:
                tokens = checkpoint(self._shared_temporal_block_forward, tokens, shared_block, pos,
                                    ls1, ls2, use_reentrant=self.use_reentrant)
            else:
                tokens = self._shared_temporal_block_forward(tokens, shared_block, pos, ls1, ls2)
        elif self.training:
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
