# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    # MODIFIED: Dyn-VGGT extensions are all opt-in (default flags off) so plain VGGT() and pretrained
    #           checkpoint loading are byte-for-byte unchanged. enable_temporal adds the temporal
    #           blocks; enable_colmask / enable_relmask add the global-attention key masks.
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True,
                 enable_temporal=False,
                 # learned column mask on global attention (D1; aggregator colmask_*)
                 enable_colmask=False, colmask_grid=(30, 37), colmask_init=5.0, colmask_rho=0.25,
                 colmask_ramp_frac=0.6, colmask_st_frac=0.2, colmask_diag=True, colmask_mean="count",
                 head_grad_checkpoint=False,   # checkpoint the DPT heads per frame chunk (memory only)
                 colmask_path=None,            # fixed D1 mask from .npy (3-4 fine-tuning)
                 # frame-distance tile mask (3-3; aggregator relmask_*)
                 enable_relmask=False, relmask_path=None, relmask_buckets=(1, 2, 4, 8, 16),
                 relmask_tile=(8, 8), relmask_start=8, relmask_grid=(30, 37), relmask_init=5.0,
                 relmask_init_std=0.01, relmask_seed=0, relmask_rho=0.25, relmask_ramp_frac=0.6,
                 relmask_st_frac=0.2, relmask_budget_S=64,
                 relmask_budget="shared", relmask_rho_steps=0, relmask_T_start=1.0, relmask_T_end=0.05,
                 relmask_T_steps=(0, 0), relmask_hard_from=0):
        super().__init__()

        # NEW: enable_temporal injects temporal attention into the aggregator (aa_order gains "temporal").
        aa_order = ["frame", "temporal", "global"] if enable_temporal else ["frame", "global"]
        self.aggregator = Aggregator(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, aa_order=aa_order,
            enable_colmask=enable_colmask or colmask_path is not None, colmask_grid=tuple(colmask_grid),
            colmask_init=float(colmask_init), colmask_rho=float(colmask_rho),
            colmask_ramp_frac=float(colmask_ramp_frac), colmask_st_frac=float(colmask_st_frac),
            colmask_diag=bool(colmask_diag),
            colmask_mean=colmask_mean if isinstance(colmask_mean, bool) else str(colmask_mean),
            colmask_path=colmask_path,
            enable_relmask=enable_relmask or relmask_path is not None, relmask_path=relmask_path,
            relmask_buckets=tuple(relmask_buckets), relmask_tile=tuple(relmask_tile),
            relmask_start=int(relmask_start), relmask_grid=tuple(relmask_grid),
            relmask_init=float(relmask_init), relmask_init_std=float(relmask_init_std),
            relmask_seed=int(relmask_seed), relmask_rho=float(relmask_rho),
            relmask_ramp_frac=float(relmask_ramp_frac), relmask_st_frac=float(relmask_st_frac),
            relmask_budget_S=int(relmask_budget_S),
            relmask_budget=str(relmask_budget), relmask_rho_steps=int(relmask_rho_steps),
            relmask_T_start=float(relmask_T_start), relmask_T_end=float(relmask_T_end),
            relmask_T_steps=tuple(relmask_T_steps), relmask_hard_from=int(relmask_hard_from),
        )

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        for _h in (self.point_head, self.depth_head):
            if _h is not None:
                _h.grad_checkpoint = bool(head_grad_checkpoint)
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

        # The v1/v2 motion_head (per-pixel dynamic probability) and flow_head (scene flow Δ) were
        # REMOVED 2026-08-31 together with the losses that consumed them: both lines were
        # diagnosed and abandoned (docs/checkpoints.md §2), and every surviving config had
        # enable_motion/enable_flow False. The two archived v1 checkpoints still carry those
        # weights; eval loads with strict=False, so they come through as ignored unexpected keys
        # and everything still used (trunk, camera/depth/point) loads unchanged.

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None,
                frame_pos: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None
            frame_pos (torch.Tensor, optional): [B, S] real frame indices of the clip; only the
                temporal blocks and relmask read it (see Aggregator.forward). Default: None (positions 0..S-1).

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images, frame_pos=frame_pos)

        predictions = {}

        with torch.amp.autocast('cuda', enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

