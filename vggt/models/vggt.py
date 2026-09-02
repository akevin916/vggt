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
    #           checkpoint loading are byte-for-byte unchanged. Set enable_temporal=True for the
    #           temporal aggregator, enable_gate=True for motion-gated camera aggregation.
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True,
                 enable_temporal=False,
                 # motion-gated camera aggregation
                 enable_gate=False, gate_block_iter=7, gate_pose_grad=False, gate_leaky=0.0,
                 gate_bias_zero_ref=False):
        super().__init__()

        # NEW: enable_temporal injects temporal attention into the aggregator (aa_order gains "temporal").
        aa_order = ["frame", "temporal", "global"] if enable_temporal else ["frame", "global"]
        self.aggregator = Aggregator(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, aa_order=aa_order,
            enable_gate=enable_gate, gate_block_iter=gate_block_iter,
            gate_pose_grad=gate_pose_grad, gate_leaky=gate_leaky,
            gate_bias_zero_ref=gate_bias_zero_ref,
        )

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

        # The v1/v2 motion_head (per-pixel dynamic probability) and flow_head (scene flow Δ) were
        # REMOVED 2026-08-31 together with the losses that consumed them: both lines were
        # diagnosed and abandoned (docs/checkpoints.md §2), and every surviving config had
        # enable_motion/enable_flow False. The two archived v1 checkpoints still carry those
        # weights; eval loads with strict=False, so they come through as ignored unexpected keys
        # and everything still used (trunk, camera/depth/point) loads unchanged.

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None,
                gate_logits_override: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None
            gate_logits_override (torch.Tensor, optional): [B, S, P_patch] eval-only override for
                the gate bias (see Aggregator.forward docstring); bypasses the model's own
                GatePredictor output when building the camera/register attention bias. Used for
                oracle-mask gate ablations. Default: None (use the model's own predicted gate).

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

        aggregated_tokens_list, patch_start_idx, gate_logits = self.aggregator(
            images, gate_logits_override=gate_logits_override
        )

        predictions = {}

        # expose gate logits for L_gate supervision (docs/method.md §3.4)
        if gate_logits is not None:
            predictions["gate_logits"] = gate_logits    # [B, S, P_patch], with gradients

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

