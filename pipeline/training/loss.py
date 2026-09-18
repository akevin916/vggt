# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from dataclasses import dataclass
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from pipeline.training.train_utils.general import check_and_fix_inf_nan
from pipeline.training.ego_flow import ego_flow_from_disp, pixel_grid, relative_w2c
from pipeline.training.flow_mask import pair_flows
from math import ceil, floor


@dataclass(eq=False)
class MultitaskLoss(torch.nn.Module):
    """
    Multi-task loss module that combines different loss types for VGGT.
    
    Supports:
    - Camera loss
    - Depth loss 
    - Point loss
    - Tracking loss (not cleaned yet, dirty code is at the bottom of this file)
    """
    # Every extension config below is optional — if a dict is None the corresponding branch is
    # skipped, so with all of them off this is byte-for-byte the original VGGT loss.
    # `**kwargs` absorbs retired keys (the v1/v2 motion/flow/reproj/tsmooth blocks, removed
    # 2026-08-31), so an old config that still writes `motion: null` loads without error.
    def __init__(self, camera=None, depth=None, point=None, track=None,
                 gate=None,    # gate: motion-gate BCE (docs/method.md §3.4)
                 static_photo=None,   # Extension: static-region photometric consistency ("route B")
                 self_photo=None,     # self-supervised photometric reprojection (predicted depth + pose)
                 smooth=None,         # edge-aware disparity smoothness (pairs with self_photo)
                 flow_geom=None,      # cross-frame 3D consistency via RAFT correspondence (ColonAdapter)
                 camera_smooth=None,  # Extension: camera-trajectory smoothness regularizer
                 ego_flow=None,       # Extension: pixel-space ego-flow reprojection consistency
                 influence=None,      # illumination robustness: cross-view appearance influence loss
                 illu=None,           # illumination-token supervision (IlluVGGT minimal version)
                 **kwargs):
        super().__init__()
        # Loss configuration dictionaries for each task
        self.camera = camera
        self.depth = depth
        self.point = point
        self.track = track
        self.gate = gate          # gate: gate-predictor BCE against GT dynamic mask
        self.static_photo = static_photo   # Extension: static-region photometric consistency
        self.self_photo = self_photo       # self-supervised photometric reprojection (no GT)
        self.smooth = smooth               # edge-aware disparity smoothness
        self.flow_geom = flow_geom         # cross-frame 3D consistency via RAFT correspondence
        self.camera_smooth = camera_smooth  # Extension: camera-trajectory smoothness regularizer
        self.ego_flow = ego_flow  # Extension: ego-flow reprojection consistency (MonST3R geometry, GT target)
        self.influence = influence  # L_inf -- bound one view's appearance influence on the others
        self.illu = illu            # L_illu -- illumination head vs GT-depth shading

    def forward(self, predictions, batch) -> torch.Tensor:
        """
        Compute the total multi-task loss.
        
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks
            
        Returns:
            Dict containing individual losses and total objective
        """
        total_loss = 0
        loss_dict = {}
        
        # Camera pose loss - if pose encodings are predicted AND a camera loss config is provided
        # (MODIFIED: guard on self.camera so stages that predict pose but don't supervise it — e.g. S0 — work)
        if self.camera is not None and "pose_enc_list" in predictions:
            camera_loss_dict = compute_camera_loss(predictions, batch, **self.camera)
            camera_loss = camera_loss_dict["loss_camera"] * self.camera["weight"]   
            total_loss = total_loss + camera_loss
            loss_dict.update(camera_loss_dict)
        
        # Depth estimation loss - if depth maps are predicted AND a depth loss config is provided
        if self.depth is not None and "depth" in predictions:
            depth_loss_dict = compute_depth_loss(predictions, batch, **self.depth)
            depth_loss = depth_loss_dict["loss_conf_depth"] + depth_loss_dict["loss_reg_depth"] + depth_loss_dict["loss_grad_depth"]
            depth_loss = depth_loss * self.depth["weight"]
            total_loss = total_loss + depth_loss
            loss_dict.update(depth_loss_dict)

        # 3D point reconstruction loss - if world points are predicted AND a point loss config is provided
        if self.point is not None and "world_points" in predictions:
            point_loss_dict = compute_point_loss(predictions, batch, **self.point)
            point_loss = point_loss_dict["loss_conf_point"] + point_loss_dict["loss_reg_point"] + point_loss_dict["loss_grad_point"]
            point_loss = point_loss * self.point["weight"]
            total_loss = total_loss + point_loss
            loss_dict.update(point_loss_dict)

        # gate: gate-predictor BCE (L_gate = BCE(σ(g), m*_patch)).  m*_patch is the GT
        # dynamic mask (m*_inst, docs §3.4) averaged to patch resolution.
        # gate_logits are NOT detached here so that gradients flow into the gate
        # predictor weights.
        if self.gate is not None and "gate_logits" in predictions and "motion_mask" in batch:
            gate_loss_dict = compute_gate_loss(predictions, batch, **self.gate)
            total_loss = total_loss + gate_loss_dict["loss_gate"] * self.gate.get("weight", 1.0)
            loss_dict.update(gate_loss_dict)

        # Extension: static-region photometric consistency ("route B" — predicted pose warps a
        # GT-static, GT-depth point into frame t+1 and must land on matching real pixel content).
        # Independent signal from L_cam (target is real image content, not GT-pose-derived).
        if self.static_photo is not None and "pose_enc_list" in predictions and "motion_mask" in batch:
            photo_loss_dict = compute_static_photo_loss(predictions, batch, **self.static_photo)
            total_loss = total_loss + photo_loss_dict["loss_static_photo"] * self.static_photo.get("weight", 1.0)
            loss_dict.update(photo_loss_dict)

        # Self-supervised photometric reprojection: predicted depth + predicted pose warp a
        # neighbouring frame onto this one and must reproduce real pixel content. Needs the depth
        # head live (it is half the geometry) and GT intrinsics; needs NO pose/depth annotation.
        # Pair it with `loss.smooth` -- without the smoothness prior the depth head is free to
        # produce texture-copied garbage in low-gradient regions.
        if self.self_photo is not None and "pose_enc_list" in predictions and "depth" in predictions:
            sp_dict = compute_self_photo_loss(predictions, batch, **self.self_photo)
            total_loss = total_loss + sp_dict["loss_self_photo"] * self.self_photo.get("weight", 1.0)
            loss_dict.update(sp_dict)

        # Edge-aware smoothness on mean-normalised disparity. Only meaningful when the depth head
        # is trainable; harmless (constant) otherwise.
        if self.smooth is not None and "depth" in predictions:
            sm_dict = compute_smooth_loss(predictions, batch, **self.smooth)
            total_loss = total_loss + sm_dict["loss_smooth"] * self.smooth.get("weight", 1.0)
            loss_dict.update(sm_dict)

        # Cross-frame 3D consistency through RAFT correspondence (ColonAdapter geometry_loss).
        # Shares its RAFT evaluations with self_photo's flow mask via batch["_flow_cache"].
        if self.flow_geom is not None and "pose_enc_list" in predictions and "depth" in predictions:
            fg_dict = compute_flow_geom_loss(predictions, batch, **self.flow_geom)
            total_loss = total_loss + fg_dict["loss_flow_geom"] * self.flow_geom.get("weight", 1.0)
            loss_dict.update(fg_dict)

        # Extension: camera-trajectory smoothness regularizer — penalises 2nd-order (acceleration)
        # jumps in the PREDICTED T/quaternion sequence, Δt-normalised by real frame-index gaps
        # (batch["ids"]). Addresses observed trajectory jumps on hard/dynamic eval sequences;
        # purely self-referential (no GT needed) so it's a regularizer, not supervision.
        if self.camera_smooth is not None and "pose_enc_list" in predictions and "ids" in batch:
            smooth_loss_dict = compute_camera_smooth_loss(predictions, batch, **self.camera_smooth)
            total_loss = total_loss + smooth_loss_dict["loss_camera_smooth"] * self.camera_smooth.get("weight", 1.0)
            loss_dict.update(smooth_loss_dict)

        # Extension: ego-flow reprojection consistency — MonST3R's disparity-form ego-flow
        # geometry, scored against the GT-derived ego-flow instead of RAFT. Couples camera head
        # and depth head in pixel space. Needs the depth head (predicted depth is half the
        # geometry) and a depth constraint to anchor it — see compute_ego_flow_loss.
        # "depth" in predictions is only required when the loss actually reads the depth head —
        # use_gt_depth mode does not, and gating on it there would silently disable the term.
        if self.ego_flow is not None and "pose_enc_list" in predictions and "ids" in batch \
                and (self.ego_flow.get("use_gt_depth", False) or "depth" in predictions):
            ego_flow_dict = compute_ego_flow_loss(predictions, batch, **self.ego_flow)
            total_loss = total_loss + ego_flow_dict["loss_ego_flow"] * self.ego_flow.get("weight", 1.0)
            loss_dict.update(ego_flow_dict)

        # Illumination robustness: cross-view appearance influence loss. Active only when the
        # trainer ran the extra clean forward and stashed it in batch["_teacher"] -- i.e. only in
        # the train phase and only past the warmup, so validation and the warmup period keep the
        # plain single-forward loss. See loss.compute_influence_loss for why this is not a
        # restatement of L_sup, and for the constant-output degenerate solution it must be paired
        # with a live `loss.point` to block.
        if self.influence is not None and batch.get("_teacher", None) is not None:
            inf_dict = compute_influence_loss(predictions, batch, **self.influence)
            inf_loss = inf_dict["loss_inf_point"] + inf_dict["loss_inf_pose"]
            total_loss = total_loss + inf_loss * self.influence.get("weight", 1.0)
            loss_dict.update(inf_dict)
            loss_dict["loss_influence"] = inf_loss

        # Illumination token supervision: the head's coarse log map against a shading target
        # derived from GT depth. weight 0 is a real arm (token capacity only), and the term is
        # still computed there so the head stays in the graph -- DDP runs with
        # find_unused_parameters=False and would otherwise fail on the unused head.
        if self.illu is not None and "illu_log16" in predictions:
            illu_dict = compute_illu_loss(predictions, batch, **self.illu)
            total_loss = total_loss + illu_dict["loss_illu"] * self.illu.get("weight", 1.0)
            loss_dict.update(illu_dict)

        # Tracking loss - not cleaned yet, dirty code is at the bottom of this file
        if "track" in predictions:
            raise NotImplementedError("Track loss is not cleaned up yet")

        loss_dict["objective"] = total_loss

        return loss_dict


def compute_camera_loss(
    pred_dict,              # predictions dict, contains pose encodings
    batch_data,             # ground truth and mask batch dict
    loss_type="l1",         # "l1" or "l2" loss
    gamma=0.6,              # temporal decay weight for multi-stage training
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=1.0,       # weight for translation loss
    weight_rot=1.0,         # weight for rotation loss
    weight_focal=0.5,       # weight for focal length loss
    **kwargs
):
    # List of predicted pose encodings per stage
    pred_pose_encodings = pred_dict['pose_enc_list']
    # Binary mask for valid points per frame (B, N, H, W)
    point_masks = batch_data['point_masks']
    # Only consider frames with enough valid points (>100)
    valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100
    # Number of prediction stages
    n_stages = len(pred_pose_encodings)

    # Get ground truth camera extrinsics and intrinsics
    gt_extrinsics = batch_data['extrinsics']
    gt_intrinsics = batch_data['intrinsics']
    image_hw = batch_data['images'].shape[-2:]

    # Encode ground truth pose to match predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )

    # Initialize loss accumulators for translation, rotation, focal length
    total_loss_T = total_loss_R = total_loss_FL = 0

    # Compute loss for each prediction stage with temporal weighting
    for stage_idx in range(n_stages):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_stage = pred_pose_encodings[stage_idx]

        if valid_frame_mask.sum() == 0:
            # If no valid frames, set losses to zero to avoid gradient issues
            loss_T_stage = (pred_pose_stage * 0).mean()
            loss_R_stage = (pred_pose_stage * 0).mean()
            loss_FL_stage = (pred_pose_stage * 0).mean()
        else:
            # Only consider valid frames for loss computation
            loss_T_stage, loss_R_stage, loss_FL_stage = camera_loss_single(
                pred_pose_stage[valid_frame_mask].clone(),
                gt_pose_encoding[valid_frame_mask].clone(),
                loss_type=loss_type
            )
        # Accumulate weighted losses across stages
        total_loss_T += loss_T_stage * stage_weight
        total_loss_R += loss_R_stage * stage_weight
        total_loss_FL += loss_FL_stage * stage_weight

    # Average over all stages
    avg_loss_T = total_loss_T / n_stages
    avg_loss_R = total_loss_R / n_stages
    avg_loss_FL = total_loss_FL / n_stages

    # Compute total weighted camera loss
    total_camera_loss = (
        avg_loss_T * weight_trans +
        avg_loss_R * weight_rot +
        avg_loss_FL * weight_focal
    )

    # Return loss dictionary with individual components
    return {
        "loss_camera": total_camera_loss,
        "loss_T": avg_loss_T,
        "loss_R": avg_loss_R,
        "loss_FL": avg_loss_FL
    }

def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1"):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.
    
    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)
        loss_FL: focal length/intrinsics loss (mean)
    
    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    loss_FL = loss_FL.mean()

    return loss_T, loss_R, loss_FL


def compute_point_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1,
                       **kwargs):
    """
    Compute point loss.

    Args:
        predictions: Dict containing 'world_points' and 'world_points_conf'
        batch: Dict containing ground truth 'world_points' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_points = predictions['world_points']
    pred_points_conf = predictions['world_points_conf']
    gt_points = batch['world_points']
    gt_points_mask = batch['point_masks']

    gt_points = check_and_fix_inf_nan(gt_points, "gt_points")
    
    if gt_points_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_points).mean()
        loss_dict = {f"loss_conf_point": dummy_loss,
                    f"loss_reg_point": dummy_loss,
                    f"loss_grad_point": dummy_loss,}
        return loss_dict
    
    # Compute confidence-weighted regression loss with optional gradient loss
    loss_conf, loss_grad, loss_reg = regression_loss(pred_points, gt_points, gt_points_mask, conf=pred_points_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)
    
    loss_dict = {
        f"loss_conf_point": loss_conf,
        f"loss_reg_point": loss_reg,
        f"loss_grad_point": loss_grad,
    }
    
    return loss_dict


def compute_depth_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, **kwargs):
    """
    Compute depth loss.
    
    Args:
        predictions: Dict containing 'depth' and 'depth_conf'
        batch: Dict containing ground truth 'depths' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_depth = predictions['depth']
    pred_depth_conf = predictions['depth_conf']

    gt_depth = batch['depths']
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
    gt_depth = gt_depth[..., None]              # (B, H, W, 1)
    gt_depth_mask = batch['point_masks'].clone()   # 3D points derived from depth map, so we use the same mask

    if gt_depth_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_depth).mean()
        loss_dict = {f"loss_conf_depth": dummy_loss,
                    f"loss_reg_depth": dummy_loss,
                    f"loss_grad_depth": dummy_loss,}
        return loss_dict

    # NOTE: we put conf inside regression_loss so that we can also apply conf loss to the gradient loss in a multi-scale manner
    # this is hacky, but very easier to implement
    loss_conf, loss_grad, loss_reg = regression_loss(pred_depth, gt_depth, gt_depth_mask, conf=pred_depth_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)

    loss_dict = {
        f"loss_conf_depth": loss_conf,
        f"loss_reg_depth": loss_reg,    
        f"loss_grad_depth": loss_grad,
    }

    return loss_dict


def compute_static_photo_loss(predictions, batch, huber_delta=0.1, dyn_thresh=0.5, min_valid=100, **kwargs):
    """
    Extension: static-region cross-frame photometric consistency ("route B" from the
    method.md §5 discussion — not yet folded into the doc). A static GT point,
    back-projected with GT depth and warped into frame t+1 with the model's PREDICTED pose,
    must land on a pixel whose appearance matches the source pixel. Restricting to the GT
    static mask (m*_inst) keeps this well-posed (dynamic points would violate the rigid-warp
    assumption even under a perfect pose) and, unlike a GT-pose-derived reprojection target,
    the target here is real image content — an independent signal from L_cam, not a
    reparameterization of it.

    Gradient path: only the PREDICTED extrinsics (from pose_enc_list[-1]) receive gradient,
    via the sampling grid inside F.grid_sample. GT depth/intrinsics/images are constants.
    Depth head is untouched (uses GT depth for back-projection), matching this work's "don't touch
    the geometry heads" scope.

    Args:
        predictions: dict with 'pose_enc_list' (list of (B,S,9) pose encodings per refine stage).
        batch: dict with 'images' (B,S,3,H,W) in [0,1], 'depths' (B,S,H,W) GT depth,
            'intrinsics' (B,S,3,3) GT, 'point_masks' (B,S,H,W) bool depth validity,
            'motion_mask' (B,S,H,W) GT dynamic mask m*_inst (1=dynamic).
        huber_delta: Huber delta in normalized [0,1] pixel-intensity units.
        dyn_thresh: threshold on (bilinearly warped) target-frame motion mask to reject
            source points that land on a dynamic region in frame t+1 (occlusion guard).
        min_valid: skip a frame pair if fewer than this many pixels pass all validity checks.
    """
    pose_enc = predictions["pose_enc_list"][-1]                    # (B, S, 9), final refine stage
    images = batch["images"]                                       # (B, S, 3, H, W)
    depths = check_and_fix_inf_nan(batch["depths"], "static_photo_depth")   # (B, S, H, W)
    intr = batch["intrinsics"]                                     # (B, S, 3, 3), GT
    point_masks = batch["point_masks"]                             # (B, S, H, W), bool
    motion_mask = batch["motion_mask"]                             # (B, S, H, W), 1=dynamic (m*_inst)

    B, S, _, H, W = images.shape
    if S < 2:
        return {"loss_static_photo": (0.0 * pose_enc).mean()}

    # Predicted world-to-cam extrinsics; GT intrinsics used throughout so the loss isolates
    # rotation/translation error (FoV is already directly supervised by L_cam).
    extrinsics_pred, _ = pose_encoding_to_extri_intri(pose_enc, (H, W), build_intrinsics=False)
    R = extrinsics_pred[..., :3, :3]                                # (B, S, 3, 3)
    T = extrinsics_pred[..., :3, 3]                                 # (B, S, 3)

    yy, xx = torch.meshgrid(
        torch.arange(H, device=images.device, dtype=images.dtype),
        torch.arange(W, device=images.device, dtype=images.dtype),
        indexing="ij",
    )  # (H, W) each

    total_loss = images.new_tensor(0.0)
    n_pairs = 0
    for t in range(S - 1):
        valid_src = point_masks[:, t] & (motion_mask[:, t] < dyn_thresh)  # (B, H, W) static + valid depth
        if valid_src.sum() < min_valid:
            continue

        fx0 = intr[:, t, 0, 0][:, None, None]; cx0 = intr[:, t, 0, 2][:, None, None]
        fy0 = intr[:, t, 1, 1][:, None, None]; cy0 = intr[:, t, 1, 2][:, None, None]
        Dt = depths[:, t]                                           # (B, H, W)
        Xc = (xx[None] - cx0) / fx0 * Dt
        Yc = (yy[None] - cy0) / fy0 * Dt
        pts_cam_t = torch.stack([Xc, Yc, Dt], dim=-1).reshape(B, H * W, 3)   # (B, HW, 3)

        # cam_t -> world -> cam_{t+1}, using PREDICTED relative pose (row-vector convention:
        # v_row @ R == (R^T v_col)^T, so R^T is applied by right-multiplying by R unchanged).
        Rt, Tt = R[:, t], T[:, t]                                    # (B,3,3), (B,3)
        Rt1, Tt1 = R[:, t + 1], T[:, t + 1]
        pts_world = torch.matmul(pts_cam_t - Tt[:, None, :], Rt)                       # R_t^T @ (X - T_t)
        pts_cam_t1 = torch.matmul(pts_world, Rt1.transpose(-1, -2)) + Tt1[:, None, :]  # R_{t+1} @ X + T_{t+1}
        pts_cam_t1 = pts_cam_t1.reshape(B, H, W, 3)

        fx1 = intr[:, t + 1, 0, 0][:, None, None]; cx1 = intr[:, t + 1, 0, 2][:, None, None]
        fy1 = intr[:, t + 1, 1, 1][:, None, None]; cy1 = intr[:, t + 1, 1, 2][:, None, None]
        Zt1 = pts_cam_t1[..., 2].clamp(min=1e-3)
        u1 = fx1 * pts_cam_t1[..., 0] / Zt1 + cx1
        v1 = fy1 * pts_cam_t1[..., 1] / Zt1 + cy1

        gx = 2.0 * u1 / (W - 1) - 1.0
        gy = 2.0 * v1 / (H - 1) - 1.0
        in_bounds = (gx >= -1) & (gx <= 1) & (gy >= -1) & (gy <= 1) & (pts_cam_t1[..., 2] > 1e-3)
        grid = torch.stack([gx, gy], dim=-1)                         # (B, H, W, 2)

        sampled_rgb = F.grid_sample(images[:, t + 1], grid, mode="bilinear",
                                     align_corners=True, padding_mode="zeros")
        sampled_rgb = sampled_rgb.permute(0, 2, 3, 1)                # (B, H, W, 3)

        # Occlusion guard: reject targets that warp onto a dynamic region in frame t+1.
        sampled_dyn = F.grid_sample(motion_mask[:, t + 1][:, None], grid, mode="bilinear",
                                     align_corners=True, padding_mode="zeros")[:, 0]
        target_static = sampled_dyn < dyn_thresh

        valid = valid_src & in_bounds & target_static
        if valid.sum() < min_valid:
            continue

        img_t = images[:, t].permute(0, 2, 3, 1)                     # (B, H, W, 3)
        err = (sampled_rgb - img_t)[valid]                           # (Nvalid, 3)
        err = check_and_fix_inf_nan(err, "static_photo_err")
        total_loss = total_loss + F.huber_loss(err, torch.zeros_like(err), delta=huber_delta)
        n_pairs += 1

    if n_pairs == 0:
        return {"loss_static_photo": (0.0 * pose_enc).mean()}
    return {"loss_static_photo": total_loss / n_pairs}


def _ssim(x, y):
    """Monodepth2's 3x3 SSIM (avg_pool, reflection pad), returned as the dissimilarity
    (1 - SSIM) / 2 clamped to [0, 1]. Shapes (N, C, H, W) -> (N, C, H, W)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    xp = F.pad(x, (1, 1, 1, 1), mode="reflect")
    yp = F.pad(y, (1, 1, 1, 1), mode="reflect")
    mu_x = F.avg_pool2d(xp, 3, 1)
    mu_y = F.avg_pool2d(yp, 3, 1)
    sigma_x = F.avg_pool2d(xp * xp, 3, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(yp * yp, 3, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(xp * yp, 3, 1) - mu_x * mu_y
    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return ((1 - ssim_n / ssim_d) / 2).clamp(0, 1)


def _photometric_error(pred, target, alpha_ssim=0.85):
    """pe(a, b) = alpha * SSIM_dissim + (1 - alpha) * L1, averaged over channels.
    Shapes (N, 3, H, W) -> (N, H, W)."""
    l1 = (pred - target).abs().mean(1)
    if alpha_ssim <= 0:
        return l1
    return alpha_ssim * _ssim(pred, target).mean(1) + (1 - alpha_ssim) * l1


def _affine_match(src, ref, mask):
    """Closed-form per-image affine brightness match src -> ref over `mask` (N,H,W bool).
    a, b are DETACHED: they correct the appearance gap without giving the network a way to
    explain a geometric residual away. Shapes (N,3,H,W) -> (N,3,H,W)."""
    m = mask[:, None].to(src.dtype)
    cnt = m.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
    mu_s = (src * m).sum(dim=(2, 3), keepdim=True) / cnt
    mu_r = (ref * m).sum(dim=(2, 3), keepdim=True) / cnt
    sd_s = ((((src - mu_s) ** 2) * m).sum(dim=(2, 3), keepdim=True) / cnt).clamp(min=0).sqrt()
    sd_r = ((((ref - mu_r) ** 2) * m).sum(dim=(2, 3), keepdim=True) / cnt).clamp(min=0).sqrt()
    a = ((sd_r + 1e-4) / (sd_s + 1e-4)).clamp(0.5, 2.0).detach()
    b = (mu_r - a * mu_s).detach()
    return (a * src + b).clamp(0.0, 1.0)


def compute_self_photo_loss(predictions, batch, src_offsets=(-1, 1), alpha_ssim=0.85,
                            auto_mask=True, brightness_affine=True, min_valid=100,
                            mask_mode="auto", conf_alpha=None, conf_mix=0.9,
                            symmetric_brightness=False, flow_proc=256, occ_thresh=0.95,
                            **kwargs):
    """Self-supervised photometric reprojection: the model's OWN depth warped by the model's
    OWN pose must reproduce the neighbouring frames' real pixel content.

    This is the label-free sibling of compute_static_photo_loss. The difference is not
    cosmetic: that one back-projects with GT depth inside the GT static mask, so it needs
    annotation and only trains the camera head; this one back-projects with the PREDICTED
    depth, so gradient reaches the depth head too and nothing but the images and the (known)
    intrinsics is required. GT intrinsics are still used -- calibration is always available in
    practice and using them leaks neither pose nor depth.

    Terms, following Monodepth2:
      * per-pixel MINIMUM over source frames rather than an average. Averaging makes every
        pixel that is occluded in one source frame carry an irreducible error that pushes
        depth the wrong way; the min lets each pixel pick the frame that can actually see it.
      * auto-masking: a pixel is dropped when the warp does not beat the identity
        (pe(I_i, I_hat) >= pe(I_i, I_j)), which removes pixels the camera motion cannot
        explain -- a still camera, or content moving with it.
      * per-pair affine brightness alignment (endoscopy-specific, on by default): the light
        source travels with the scope so brightness constancy is violated frame to frame.
        a, b are closed-form from the valid pixels' mean/std and DETACHED, so they correct the
        appearance gap without giving the network a way to explain the residual away.

    ColonAdapter-derived options (all default OFF, so scared_selfsup / _sm05 reproduce exactly;
    see reference/ColonAdapter/trainer.py):
      * mask_mode="flow": replace the auto-mask with RAFT backward-flow coverage
        (pipeline/training/flow_mask.py). A target pixel no source pixel maps onto is treated as
        unseen in that source and gets +inf cost for it, so the per-pixel min picks a source that
        can see it. The auto-mask test is skipped entirely in this mode.
      * conf_alpha: confidence-weighted error using the depth head's `depth_conf`,
            L = conf_mix * mean(conf * e - conf_alpha * log conf) + (1 - conf_mix) * mean(e).
        ColonAdapter uses alpha 0.2 with a 0.9/0.1 mix. The unweighted (1 - conf_mix) floor is
        kept on purpose: docs/status.md §B.3 traces a failed depth arm to exactly this
        -alpha*log(c) term, so alpha should stay low (0.05) and the floor should stay.
      * symmetric_brightness: also affine-match the UN-warped source before the auto-mask
        comparison. Without it the warped branch alone gets a brightness correction and the
        auto-mask is biased toward keeping pixels. Only affects mask_mode="auto".

    GAUGE. Reprojection is invariant to (D, t) -> (sD, st), so global scale carries no
    gradient; depth and translation are divided by the window's mean depth here purely to keep
    the numerics in a fixed range. The unnormalised mean disparity is reported as `disp_mean`
    precisely because the normalisation would otherwise hide a collapse (depth -> infinity
    degenerates the warp into a pure rotation, which small-parallax endoscopy is prone to).

    Returns loss_self_photo plus three diagnostics that are the actual early-warning signals:
    disp_mean (collapse), photo_valid_frac (how much of the image the loss still sees), and
    traj_len_ratio (predicted trajectory length / GT, when GT extrinsics are in the batch --
    this model is known to under-predict SCARED travel by ~47x, and photometric is the only
    term that can see that error).
    """
    pose_enc = predictions["pose_enc_list"][-1]                # (B, S, 9)
    images = batch["images"]                                   # (B, S, 3, H, W), [0, 1]
    intr = batch["intrinsics"]                                 # (B, S, 3, 3), GT
    B, S, _, H, W = images.shape

    offsets = [int(o) for o in src_offsets if int(o) != 0]
    if S < 2 or not offsets:
        return {"loss_self_photo": (0.0 * pose_enc).mean()}

    depth = check_and_fix_inf_nan(predictions["depth"][..., 0], "self_photo_depth")  # (B,S,H,W)
    depth = depth.clamp(min=1e-6)
    scale = depth.mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)     # (B,1,1,1)
    depth_n = depth / scale

    extrinsics, _ = pose_encoding_to_extri_intri(pose_enc, (H, W), build_intrinsics=False)
    R = extrinsics[..., :3, :3]                                 # (B, S, 3, 3)
    T = extrinsics[..., :3, 3] / scale[..., 0]                  # (B, S, 3), same gauge as depth_n

    yy, xx = torch.meshgrid(
        torch.arange(H, device=images.device, dtype=images.dtype),
        torch.arange(W, device=images.device, dtype=images.dtype),
        indexing="ij",
    )

    if mask_mode not in ("auto", "flow"):
        raise ValueError(f"mask_mode must be 'auto' or 'flow', got {mask_mode!r}")
    flows = pair_flows(batch, offsets, proc=flow_proc, occ_thresh=occ_thresh) if mask_mode == "flow" else None

    inf = torch.finfo(images.dtype).max
    cost = images.new_full((len(offsets), B, S, H, W), inf)     # warped error
    ident = images.new_full((len(offsets), B, S, H, W), inf)    # unwarped error (auto-mask)
    ok = torch.zeros((len(offsets), B, S, H, W), dtype=torch.bool, device=images.device)

    for k, off in enumerate(offsets):
        tgt = torch.arange(max(0, -off), min(S, S - off), device=images.device)
        if tgt.numel() == 0:
            continue
        src = tgt + off
        n = tgt.numel()

        def flat(x, idx):                      # (B, S, ...) -> (B*n, ...)
            return x[:, idx].reshape(B * n, *x.shape[2:])

        img_t = flat(images, tgt)              # (N, 3, H, W)
        img_s = flat(images, src)
        D_t = flat(depth_n, tgt)               # (N, H, W)
        K_t, K_s = flat(intr, tgt), flat(intr, src)
        R_t, R_s = flat(R, tgt), flat(R, src)
        T_t, T_s = flat(T, tgt), flat(T, src)
        N = B * n

        fx = K_t[:, 0, 0][:, None, None]; cx = K_t[:, 0, 2][:, None, None]
        fy = K_t[:, 1, 1][:, None, None]; cy = K_t[:, 1, 2][:, None, None]
        pts = torch.stack([(xx[None] - cx) / fx * D_t,
                           (yy[None] - cy) / fy * D_t,
                           D_t], dim=-1).reshape(N, H * W, 3)

        # target camera -> source camera (column convention: X_cam = R X_world + T)
        R_rel = torch.bmm(R_s, R_t.transpose(1, 2))
        t_rel = T_s - torch.bmm(R_rel, T_t[..., None])[..., 0]
        pts_s = torch.bmm(pts, R_rel.transpose(1, 2)) + t_rel[:, None, :]
        pts_s = pts_s.reshape(N, H, W, 3)

        Z = pts_s[..., 2]
        Zc = Z.clamp(min=1e-4)
        u = K_s[:, 0, 0][:, None, None] * pts_s[..., 0] / Zc + K_s[:, 0, 2][:, None, None]
        v = K_s[:, 1, 1][:, None, None] * pts_s[..., 1] / Zc + K_s[:, 1, 2][:, None, None]
        gx = 2.0 * u / (W - 1) - 1.0
        gy = 2.0 * v / (H - 1) - 1.0
        valid = (gx.abs() <= 1) & (gy.abs() <= 1) & (Z > 1e-4)

        warped = F.grid_sample(img_s, torch.stack([gx, gy], dim=-1), mode="bilinear",
                               align_corners=True, padding_mode="border")

        if brightness_affine:
            warped = _affine_match(warped, img_t, valid)

        pe = _photometric_error(warped, img_t, alpha_ssim).reshape(B, n, H, W)
        val = valid.reshape(B, n, H, W)
        if flows is not None:
            val = val & flows[off][2]                     # geometric visibility from flow coverage
            pe_id = pe                                    # unused in flow mode
        else:
            ident_src = img_s
            if brightness_affine and symmetric_brightness:
                ident_src = _affine_match(img_s, img_t, torch.ones_like(valid))
            pe_id = _photometric_error(ident_src, img_t, alpha_ssim).reshape(B, n, H, W)

        cost[k, :, tgt] = torch.where(val, pe, torch.full_like(pe, inf))
        ident[k, :, tgt] = pe_id
        ok[k, :, tgt] = val

    best, _ = cost.min(dim=0)                                   # (B, S, H, W)
    keep = ok.any(dim=0)
    if auto_mask and mask_mode == "auto":
        keep = keep & (best < ident.min(dim=0).values)
    keep = keep.detach()

    n_keep = keep.sum()
    disp_mean = (1.0 / depth.detach()).mean()
    valid_frac = n_keep.to(images.dtype) / float(B * S * H * W)
    diag = {"disp_mean": disp_mean, "photo_valid_frac": valid_frac.detach()}

    if "extrinsics" in batch:
        with torch.no_grad():
            def path_len(Rm, Tm):
                c = -torch.matmul(Rm.transpose(-1, -2), Tm[..., None])[..., 0]   # (B,S,3)
                return (c[:, 1:] - c[:, :-1]).norm(dim=-1).sum(dim=-1)           # (B,)
            gt = batch["extrinsics"]
            l_pred = path_len(R, extrinsics[..., :3, 3])
            l_gt = path_len(gt[..., :3, :3], gt[..., :3, 3])
            diag["traj_len_ratio"] = (l_pred / l_gt.clamp(min=1e-8)).mean()

    if n_keep < min_valid:
        diag["loss_self_photo"] = (0.0 * pose_enc).mean()
        return diag

    err = check_and_fix_inf_nan(best[keep], "self_photo_err")
    if conf_alpha is not None and "depth_conf" in predictions:
        conf = predictions["depth_conf"][keep].clamp(min=1e-6)     # expp1 head: >= 1
        weighted = (conf * err - float(conf_alpha) * torch.log(conf)).mean()
        diag["loss_self_photo"] = float(conf_mix) * weighted + (1.0 - float(conf_mix)) * err.mean()
        diag["photo_conf_mean"] = conf.detach().mean()
    else:
        diag["loss_self_photo"] = err.mean()
    return diag


def compute_flow_geom_loss(predictions, batch, src_offsets=(-1, 1), flow_proc=256,
                           occ_thresh=0.95, min_valid=100, **kwargs):
    """Cross-frame 3D consistency through optical-flow correspondence (ColonAdapter's
    `geometry_loss`, reference/ColonAdapter/trainer.py:579-602, re-expressed for VGGT).

    For a target pixel p in frame t and its RAFT correspondence p' = p + flow(p) in frame s, the
    world point the model places at p (from depth_t + pose_t) and the world point it places at p'
    (from depth_s + pose_s) are the same physical surface point, so they must coincide:

        L = mean_{p visible in s}  || X_t(p) - X_s(p + flow_{t->s}(p)) ||_1

    WHY THIS TERM. scared_selfsup's photometric loss only ever compares colours between adjacent
    frames, so nothing forces frame 5 and frame 50 to put a surface at the same depth -- and the
    measured failure matched that: per-frame RPE improved while full-sequence ATE and depth got
    worse. This is the one term that ties different frames' GEOMETRY to each other directly.

    WHAT IT DOES NOT DO. It penalises disagreement, not error: if every frame is wrong in the same
    way the loss is zero. ColonAdapter pairs it with a point-map/pose term (weight 1.0) that this
    port deliberately leaves out (no second model, no point head -- see
    configs/scared_selfsup_ca3.yaml), so here nothing anchors the geometry to an external value.

    DIFFERENCES FROM COLONADAPTER, both forced by VGGT's outputs:
      * ColonAdapter compares its point head against DUSt3R's `pts3d_in_other_view`. VGGT has no
        such dual output, so both sides here are built from depth + camera instead; the point
        head is not involved (and stays disabled).
      * Flow is RAFT (frozen, no_grad), not a jointly-trained flow network.

    Gauge: points are divided by the window's mean depth, the same normalisation as
    compute_self_photo_loss, so the loss is invariant to global scale and cannot be lowered by
    shrinking the scene. Gradient reaches depth and pose of BOTH frames; flow and the visibility
    mask are constants.
    """
    pose_enc = predictions["pose_enc_list"][-1]
    images = batch["images"]
    intr = batch["intrinsics"]
    B, S, _, H, W = images.shape
    offsets = [int(o) for o in src_offsets if int(o) != 0]
    zero = {"loss_flow_geom": (0.0 * pose_enc).mean()}
    if S < 2 or not offsets:
        return zero

    depth = check_and_fix_inf_nan(predictions["depth"][..., 0], "flow_geom_depth").clamp(min=1e-6)
    scale = depth.mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)           # (B,1,1,1)
    extrinsics, _ = pose_encoding_to_extri_intri(pose_enc, (H, W), build_intrinsics=False)
    R = extrinsics[..., :3, :3]                                                 # (B,S,3,3)
    T = extrinsics[..., :3, 3]                                                  # (B,S,3)

    yy, xx = torch.meshgrid(torch.arange(H, device=images.device, dtype=images.dtype),
                            torch.arange(W, device=images.device, dtype=images.dtype), indexing="ij")
    fx = intr[..., 0, 0][..., None, None]; cx = intr[..., 0, 2][..., None, None]
    fy = intr[..., 1, 1][..., None, None]; cy = intr[..., 1, 2][..., None, None]
    Xc = torch.stack([(xx - cx) / fx * depth, (yy - cy) / fy * depth, depth], dim=-1)  # (B,S,H,W,3)
    # world = R^T (X_cam - T); row-vector form (X - T) @ R
    Xw = torch.matmul((Xc - T[:, :, None, None, :]).reshape(B, S, H * W, 3), R)
    Xw = Xw.reshape(B, S, H, W, 3) / scale[..., None]

    flows = pair_flows(batch, offsets, proc=flow_proc, occ_thresh=occ_thresh)
    total = images.new_tensor(0.0)
    n_terms, n_used, n_all = 0, 0, 0
    for off, (tgt, flow, visible) in flows.items():
        n = tgt.numel()
        src = tgt + off
        X_t = Xw[:, tgt]                                                        # (B,n,H,W,3)
        X_s = Xw[:, src].permute(0, 1, 4, 2, 3).reshape(B * n, 3, H, W)
        u = xx + flow[:, :, 0]
        v = yy + flow[:, :, 1]
        grid = torch.stack([2.0 * u / (W - 1) - 1.0, 2.0 * v / (H - 1) - 1.0], dim=-1)  # (B,n,H,W,2)
        inb = (grid.abs() <= 1).all(dim=-1)
        X_s = F.grid_sample(X_s, grid.reshape(B * n, H, W, 2), mode="bilinear",
                            align_corners=True, padding_mode="border")
        X_s = X_s.reshape(B, n, 3, H, W).permute(0, 1, 3, 4, 2)
        m = (visible & inb).detach()
        n_all += m.numel()
        if m.sum() < min_valid:
            continue
        d = check_and_fix_inf_nan((X_t - X_s).abs().sum(dim=-1)[m], "flow_geom_err")
        total = total + d.mean()
        n_terms += 1
        n_used += int(m.sum())

    if n_terms == 0:
        return zero
    return {"loss_flow_geom": total / n_terms,
            "flow_geom_valid_frac": images.new_tensor(n_used / max(n_all, 1))}


def compute_smooth_loss(predictions, batch, **kwargs):
    """Edge-aware first-order smoothness on MEAN-NORMALISED disparity (Monodepth2 eq. 3).

    The mean normalisation is load-bearing, not cosmetic: on raw disparity this term is
    minimised by shrinking disparity globally, i.e. depth -> infinity, which degenerates the
    photometric warp into a pure rotation. Small-parallax endoscopy sits close to that
    degenerate solution already.
    """
    depth = predictions["depth"][..., 0].clamp(min=1e-6)        # (B, S, H, W)
    B, S, H, W = depth.shape
    disp = (1.0 / depth).reshape(B * S, 1, H, W)
    disp = disp / (disp.mean(dim=(2, 3), keepdim=True) + 1e-8)
    img = batch["images"].reshape(B * S, -1, H, W)

    dx = (disp[..., :, :-1] - disp[..., :, 1:]).abs()
    dy = (disp[..., :-1, :] - disp[..., 1:, :]).abs()
    ix = (img[..., :, :-1] - img[..., :, 1:]).abs().mean(1, keepdim=True)
    iy = (img[..., :-1, :] - img[..., 1:, :]).abs().mean(1, keepdim=True)
    loss = (dx * torch.exp(-ix)).mean() + (dy * torch.exp(-iy)).mean()
    return {"loss_smooth": check_and_fix_inf_nan(loss, "smooth")}


def compute_ego_flow_loss(predictions, batch, beta=1.0, per_pixel_thre=50.0, dyn_thresh=0.5,
                          max_dt=5, bidirectional=True, use_dynamic_mask=True, min_valid=100,
                          max_depth_ratio=None, target_from_pose_encoding=True,
                          use_gt_depth=False, **kwargs):
    """Pixel-space ego-flow (reprojection) consistency between predicted and GT geometry.

    The geometry is MonST3R's flow term ported verbatim (disparity-form ego-flow, smooth-L1,
    per-pixel outlier rejection; see `ego_flow.py`), but the TARGET is the GT-derived
    ego-flow rather than RAFT optical flow. That difference is not cosmetic and must not be
    papered over in the write-up: MonST3R's RAFT target is valuable precisely because it is
    an observation that does NOT know the GT, which is what makes it an anchor during
    test-time optimization. With a GT-derived target this is a REPROJECTION LOSS — a
    pixel-space, depth-weighted restatement of the pose error — not a port of MonST3R's
    flow loss (docs/topics/monst3r_design.md).

    What it still buys over L_cam: the error is measured where it is observable (pixels,
    weighted by disparity) and it couples camera and depth heads through a single
    constraint, the way MonST3R's does.

    Gradient path: predicted extrinsics AND predicted depth AND predicted intrinsics (the
    full-prediction variant). depth_head must be unfrozen for the depth path to matter, and
    a depth constraint (loss.depth) must be on, because ego-flow's translation term is
    `disparity * t`: a depth error and a translation error can cancel inside this residual,
    so the term alone does not pin either. That is a property of the equation, not an
    empirical finding — do NOT cite the RAFT-target headroom probe for it (that probe
    measured a different residual; see diag/ego_flow_residual.py (removed 2026-08-19)).

    SCALE: no alignment needed, and this is a property of the formulation, not luck. The
    translation term of ego-flow is `disparity * t`, invariant to a global scene rescale.
    The predicted depth and translation share VGGT's normalised scale; the GT depth and
    extrinsics are rescaled in lock-step by trainer._process_batch (trainer.py:979-993).
    Each flow is therefore internally consistent and both come out in pixels. (The mixed
    variants in diag/flow_loss_probe.py (removed 2026-08-19) DO need a median-ratio factor — without it they
    measure unit mismatch, a spurious 5x degradation.)

    Not ported: MonST3R's whole-term fuse (`if flow_loss > thre: flow_loss = 0`). It exists
    to survive RAFT's catastrophic outliers early in test-time optimization — a target
    computed from GT geometry has no such failure mode, and a fuse would instead silently
    disable the loss on exactly the hard clips where the pose error is largest. Per-pixel
    rejection is kept (depth discontinuities still produce a heavy tail); its keep rate is
    reported as a diagnostic so the threshold can be set from measurement rather than
    inherited from MonST3R's RAFT-era constant.

    Args:
        predictions: needs 'pose_enc_list' (B,S,9 per stage) and 'depth' (B,S,H,W,1).
        batch: 'extrinsics' (B,S,3,4) GT w2c, 'intrinsics' (B,S,3,3) GT, 'depths' (B,S,H,W)
            GT, 'point_masks' (B,S,H,W) bool, 'ids' (B,S) chronological frame indices,
            'motion_mask' (B,S,H,W) optional (1=dynamic).
        beta: smooth-L1 transition point, in pixels (MonST3R uses 1.0).
        per_pixel_thre: drop pixels whose raw smooth-L1 exceeds this (MonST3R: 50).
        dyn_thresh: motion_mask threshold above which a pixel counts as dynamic.
        max_dt: skip frame pairs further apart than this many real frames (duplicate frames,
            dt == 0, are always skipped — they carry zero flow by construction).
        bidirectional: score t->t+1 and t+1->t, as MonST3R sums both directions.
        use_dynamic_mask: exclude dynamic pixels. The theory that a GT-derived target makes
            this inert is FALSE by measurement (diag/ego_flow_residual.py (removed 2026-08-19): dropping the mask
            raises the residual 65% on Sintel, 24% on Spring). The extra error enters from
            the PREDICTION side — predicted depth is much worse on moving objects — not from
            the target. Kept switchable to quantify that contribution.
        min_valid: skip a frame pair with fewer than this many valid source pixels.
        max_depth_ratio: if set, drop source pixels whose GT depth exceeds this multiple of
            the frame's median valid GT depth (e.g. 5.0 = drop anything past 5x median).
            Far pixels are where this loss is least trustworthy: ego-flow's translation term
            is `disparity * t`, so a far point contributes almost no translation signal while
            its depth is the least reliable thing the model predicts — a bad ratio in a term
            that has no other way to tell depth error from pose error. Measured effect: on
            Waymo (a driving set that is mostly far road/sky) this term is 50% of the
            objective at weight 1.0 versus 0.4% on PointOdyssey, so without a cap most of the
            gradient goes into distant depth. The cut uses GT depth, never predicted depth:
            a mask keyed on the prediction would reward pushing depth outward to escape the
            loss. None keeps every pixel (original behaviour).
        target_from_pose_encoding: build the GT ego-flow from GT extrinsics/intrinsics that
            have been round-tripped through the pose encoding, instead of the raw ones.
            This removes an otherwise irreducible floor. Measured on PointOdyssey
            (diag/ego_flow_selftest.py --dataset po, (removed 2026-08-19)): the GT rotation matrices are off SO(3)
            by 5e-4 — float32 drift through the dataset's crop/resize/rotate path — and the
            quaternion in the pose encoding can only represent a proper rotation, so decoding
            silently re-orthonormalises. The prediction therefore CANNOT reproduce the raw GT
            flow: identity residual 0.061 px, against a signal of 0.13 px on the same set.
            (The dropped principal point costs nothing by comparison: shifting it translates
            the whole flow field rather than changing its values.) compute_camera_loss has no
            such floor because it encodes the GT and compares in encoding space; this makes
            the pixel-space term consistent with that. Off = compare against the raw GT, i.e.
            penalise the model for not reproducing a camera it cannot represent.
        use_gt_depth: build the PREDICTED ego-flow from GT depth instead of predicted depth.
            Both flows then share one disparity field, the depth term cancels in the difference,
            and the residual reduces to pure relative-pose error weighted by observability:

                flow_pred - flow_gt  ~  [rotation term] + disp_gt * K (t_rel_pred - t_rel_gt)

            Two things this buys. (1) No gradient reaches depth, so the term cannot buy a lower
            residual by distorting depth to compensate a pose error — and that shortcut is not
            hypothetical: ego-flow's translation term is `disparity * t`, so a global depth-scale
            error is EXACTLY cancelled by a translation-scale error, and translation scale is
            what ATE measures. (2) It needs no depth head, so depth_head stays frozen and L_depth
            stays off, leaving this loss as the only difference from the baseline run.
            Freezing depth_head alone would NOT close that path: train_utils/freeze.py sets
            requires_grad=False, which stops the head's parameters from updating but not gradient
            from flowing THROUGH it into the unfrozen aggregator trunk.
            Scale is consistent because trainer._process_batch rescales GT extrinsics and depths
            in lock-step to unit average distance and the model predicts in that same convention
            (compute_camera_loss compares them directly). Residual scale error in the prediction
            is genuine pose error and SHOULD show up here.
    """
    pose_enc = predictions["pose_enc_list"][-1]                 # (B,S,9), final refine stage
    images = batch["images"]
    B, S, _, H, W = images.shape
    zero = (0.0 * pose_enc).mean()
    if S < 2:
        return {"loss_ego_flow": zero, "loss_ego_flow_kept": zero.detach(),
                "loss_ego_flow_pairs": zero.detach()}

    # fp32 throughout: the perspective divide in ego_flow_from_disp loses pixel-level
    # precision under bf16, and this loss is measured in pixels.
    with torch.amp.autocast("cuda", enabled=False):
        pr_extri, pr_intri = pose_encoding_to_extri_intri(pose_enc.float(), (H, W), build_intrinsics=True)
        # With use_gt_depth the depth head is not consulted at all, so this must not require it:
        # the whole point of that mode is to run with depth_head frozen (or absent).
        pr_depth = None if use_gt_depth else predictions["depth"][..., 0].float()   # (B,S,H,W)
        gt_extri = batch["extrinsics"].float()
        gt_intri = batch["intrinsics"].float()
        if target_from_pose_encoding:
            # Project the GT onto what a pose encoding can express (proper rotation, centred
            # principal point), so the target is reachable. See the docstring.
            gt_extri, gt_intri = pose_encoding_to_extri_intri(
                extri_intri_to_pose_encoding(gt_extri, gt_intri, (H, W)), (H, W),
                build_intrinsics=True,
            )
        gt_depth = check_and_fix_inf_nan(batch["depths"].float(), "ego_flow_gt_depth")
        point_masks = batch["point_masks"]
        motion_mask = batch.get("motion_mask", None)
        ids = batch["ids"]

        coord = pixel_grid(H, W, images.device, torch.float32)
        eps = 1e-6

        # Per-frame far-pixel cap, from GT depth only (see max_depth_ratio in the docstring).
        # Computed once for all frames rather than per pair, and on a strided subsample: the
        # median of ~16k pixels is more than accurate enough to set a cut, and quantile() over
        # the full 518x518 grid would sort a quarter of a million values per frame per step.
        depth_cap = None
        if max_depth_ratio is not None:
            with torch.no_grad():
                sub = max(1, (H * W) // 16384)
                d = gt_depth.reshape(B * S, -1)[:, ::sub]
                mk = point_masks.reshape(B * S, -1)[:, ::sub]
                med = torch.nanquantile(d.masked_fill(~mk, float("nan")), 0.5, dim=-1)
                # Frames with no valid depth yield NaN -> no cap rather than an empty mask.
                depth_cap = torch.where(torch.isnan(med), torch.full_like(med, float("inf")),
                                        med * max_depth_ratio).reshape(B, S)

        total_loss = pose_enc.new_zeros(())
        n_pairs = 0
        kept_num = 0.0
        kept_den = 0.0

        directions = [(0, 1)] + ([(1, 0)] if bidirectional else [])
        for t in range(S - 1):
            dt = (ids[:, t + 1] - ids[:, t]).float()             # (B,)
            pair_ok = (dt > 0) & (dt <= max_dt)                  # duplicate frames carry no flow
            if not bool(pair_ok.any()):
                continue

            for off_src, off_tgt in directions:
                s_idx, t_idx = t + off_src, t + off_tgt

                gt_disp = 1.0 / gt_depth[:, s_idx].clamp(min=eps)
                # Same disparity on both sides -> the depth term cancels in the difference.
                pr_disp = gt_disp if use_gt_depth else 1.0 / pr_depth[:, s_idx].clamp(min=eps)

                R_pr, t_pr = relative_w2c(pr_extri[:, s_idx], pr_extri[:, t_idx])
                R_gt, t_gt = relative_w2c(gt_extri[:, s_idx], gt_extri[:, t_idx])

                flow_pr, z_pr = ego_flow_from_disp(
                    R_pr, t_pr, pr_disp[:, None], pr_intri[:, t_idx],
                    torch.linalg.inv(pr_intri[:, s_idx]), coord,
                )
                flow_gt, z_gt = ego_flow_from_disp(
                    R_gt, t_gt, gt_disp[:, None], gt_intri[:, t_idx],
                    torch.linalg.inv(gt_intri[:, s_idx]), coord,
                )

                # Source-frame validity. z <= 0 means the point warps behind the target
                # camera, where the perspective divide flips sign and the gradient points
                # the wrong way.
                valid = point_masks[:, s_idx] & (z_pr > eps) & (z_gt > eps)
                if use_dynamic_mask and motion_mask is not None:
                    valid = valid & (motion_mask[:, s_idx] < dyn_thresh)
                if depth_cap is not None:
                    valid = valid & (gt_depth[:, s_idx] <= depth_cap[:, s_idx, None, None])
                valid = valid & pair_ok[:, None, None]
                m = valid.float()[:, None]                        # (B,1,H,W)

                raw = F.smooth_l1_loss(flow_pr * m, flow_gt * m, beta=beta, reduction="none")
                keep = m.expand_as(raw) if per_pixel_thre <= 0 else (raw < per_pixel_thre).float() * m

                n_valid_px = m.sum(dim=(1, 2, 3))                 # (B,)
                denom = keep.sum(dim=(1, 2, 3))                   # (B,), counts both channels
                usable = (n_valid_px >= min_valid) & (denom > 0)
                if not bool(usable.any()):
                    continue

                per_sample = (raw * keep).sum(dim=(1, 2, 3)) / denom.clamp(min=1.0)
                total_loss = total_loss + per_sample[usable].sum()
                n_pairs += int(usable.sum())

                kept_num += float(keep.sum().detach())
                kept_den += float(m.sum().detach()) * raw.shape[1]

    if n_pairs == 0:
        return {"loss_ego_flow": zero, "loss_ego_flow_kept": zero.detach(),
                "loss_ego_flow_pairs": zero.detach()}

    loss = check_and_fix_inf_nan(total_loss / n_pairs, "loss_ego_flow")
    kept = torch.as_tensor(kept_num / max(kept_den, 1.0), device=loss.device, dtype=loss.dtype)
    pairs = torch.as_tensor(float(n_pairs), device=loss.device, dtype=loss.dtype)
    return {"loss_ego_flow": loss, "loss_ego_flow_kept": kept, "loss_ego_flow_pairs": pairs}


def compute_camera_smooth_loss(predictions, batch, weight_trans=1.0, weight_rot=1.0, gamma=0.6,
                               orders=(2,), order_weights=None, **kwargs):
    """
    Extension: camera-trajectory smoothness regularizer (conversation notes, not yet folded
    into method.md). Penalises k-th order discontinuities in the PREDICTED pose
    sequence — observed as visible trajectory "jumps" on hard/dynamic Sintel sequences, present
    in native VGGT-1B too (not a regression introduced by this work). Purely self-referential (no GT pose
    used): a regularizer on the network's own output, analogous to MonST3R's trajectory-
    smoothness prior but applied as a training loss rather than a test-time optimization term.

    ORDER (`orders` / `order_weights`). Default (2,) reproduces the original 2nd-order-only
    behaviour exactly — that is the term that produced the current best checkpoint (run 2,
    0.1714 -> 0.1343). Order 1 penalises the motion itself ("the camera barely moves", which is
    MonST3R's relative_pose_loss); order 2 penalises acceleration, so constant velocity is free.

    Why order 1 is worth having despite the obvious objection that it fights legitimate constant
    motion (Waymo drives forward, TartanAir flies): the TTO line measured that a 1st-order prior
    is the ONLY thing that substantially rescues badly-initialised sequences —
        seq        base      2nd order     1st order
        cave_2     0.8298    0.7483 (-10%) 0.5527 (-33%)   [with flow: 0.4600, -45%]
        temple_3   0.4505    0.3849 (-15%) 0.1850 (-59%)   [with flow: 0.1770, -61%]
    and that 2nd order SATURATES: its TTO weight sweep is flat (w10 0.1154 / w30 0.1128 /
    w100 0.1134 / w300 0.1147), so no amount of 2nd-order weight buys what order 1 buys.
    Order 1 is aggressive and needs a brake: in TTO that brake is the flow term (without it,
    1st order blows up well-initialised sequences by up to 8.7x). In training L_cam plays that
    role and is exact, which is why the training-side behaviour should resemble TTO's
    "1st order + flow" column rather than its "1st order alone" column.
    Use it ADDITIVELY (orders=(1, 2)) rather than as a replacement, and read per-sequence
    results: order 1's value is concentrated in the few worst sequences and a mean can hide it
    entirely.

    order_weights defaults to 1.0 for every entry of `orders`. Note orders are NOT on a common
    scale (order 1 is a velocity, order 2 an acceleration), so their weights need separate
    calibration — measure both from a smoke run rather than assuming parity.

    Δt-normalization: training clips (PointOdyssey/TartanAir, get_nearby=True) sample frames
    from a local window with irregular spacing and possible duplicates (replace=True), not a
    fixed stride. batch["ids"] holds the real chronological frame indices, so velocity is
    computed as ΔT/Δt rather than raw ΔT — this makes the term valid despite irregular/duplicate
    gaps. Pairs with Δt=0 (duplicate sampled frame) are excluded.

    Rotation: quaternions are sign-corrected (consecutive-frame hemisphere alignment) before
    differencing, since q and -q represent the same rotation and a sign flip would otherwise
    look like a large spurious rotation jump. This is a simplified (component-wise) smoothness,
    not a strict geodesic one.

    FoV is intentionally not smoothed (unrelated to the observed jump symptom).

    Args:
        predictions: dict with 'pose_enc_list' (list of (B,S,9) pose encodings per refine stage).
        batch: dict with 'ids' (B,S) real chronological frame indices, 'point_masks' (B,S,H,W).
    """
    pred_pose_encodings = predictions["pose_enc_list"]
    n_stages = len(pred_pose_encodings)

    point_masks = batch["point_masks"]
    valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100  # same convention as compute_camera_loss

    ids = batch["ids"]
    B, S = ids.shape

    orders = [int(o) for o in orders]
    if any(o < 1 for o in orders):
        raise ValueError(f"orders must all be >= 1, got {orders}")
    w_order = [1.0] * len(orders) if order_weights is None else [float(w) for w in order_weights]
    if len(w_order) != len(orders):
        raise ValueError(f"order_weights {w_order} must match orders {orders}")
    max_order = max(orders)

    # An order-k difference needs k+1 frames (and k+1 chronologically increasing ids).
    if S < max_order + 1 or valid_frame_mask.sum() == 0:
        zero = (pred_pose_encodings[-1] * 0).mean()
        out = {"loss_camera_smooth": zero, "loss_smooth_T": zero, "loss_smooth_R": zero}
        out.update({f"loss_smooth_{c}{o}": zero for o in orders for c in ("T", "R")})
        return out

    ids = ids[valid_frame_mask].float()                    # (B', S)
    dt = ids[:, 1:] - ids[:, :-1]                          # (B', S-1)
    pair_valid = dt > 0
    safe_dt = dt.clamp(min=1.0).unsqueeze(-1)              # (B', S-1, 1)

    # valid_of_order[k] marks the order-k differences whose k underlying frame pairs are all
    # real (no duplicate-frame gaps). Order 1 is the pair mask itself; each further difference
    # consumes one more element and must AND the two it was built from.
    valid_of_order = {1: pair_valid}
    for k in range(2, max_order + 1):
        prev = valid_of_order[k - 1]
        valid_of_order[k] = prev[:, 1:] & prev[:, :-1]

    def _diff(x, order):
        """x is the order-1 quantity (velocity); difference it order-1 more times."""
        for _ in range(order - 1):
            x = x[:, 1:] - x[:, :-1]
        return x

    per_order = {o: [0.0, 0.0] for o in orders}            # order -> [T, R] accumulated over stages
    total_smooth_T = total_smooth_R = 0
    for stage_idx in range(n_stages):
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pose = pred_pose_encodings[stage_idx][valid_frame_mask]   # (B', S, 9)
        T = pose[..., :3]
        quat = pose[..., 3:7]

        # Sign-fix quaternion hemisphere so consecutive frames don't spuriously flip.
        quat_fixed = [quat[:, 0]]
        for t in range(1, S):
            dot = (quat_fixed[-1] * quat[:, t]).sum(-1, keepdim=True)
            sign = torch.where(dot < 0, -torch.ones_like(dot), torch.ones_like(dot))
            quat_fixed.append(quat[:, t] * sign)
        quat_fixed = torch.stack(quat_fixed, dim=1)   # (B', S, 4)

        v_T = (T[:, 1:] - T[:, :-1]) / safe_dt             # (B', S-1, 3) velocity
        v_R = (quat_fixed[:, 1:] - quat_fixed[:, :-1]) / safe_dt

        loss_T_stage = loss_R_stage = 0
        for o, w in zip(orders, w_order):
            valid = valid_of_order[o]
            if valid.sum() == 0:
                d_T = (T * 0).mean()
                d_R = (quat_fixed * 0).mean()
            else:
                dT, dR = _diff(v_T, o), _diff(v_R, o)
                mask_T = valid.unsqueeze(-1).expand_as(dT).float()
                mask_R = valid.unsqueeze(-1).expand_as(dR).float()
                d_T = (dT.abs() * mask_T).sum() / mask_T.sum().clamp(min=1)
                d_R = (dR.abs() * mask_R).sum() / mask_R.sum().clamp(min=1)
            per_order[o][0] = per_order[o][0] + d_T * stage_weight
            per_order[o][1] = per_order[o][1] + d_R * stage_weight
            loss_T_stage = loss_T_stage + d_T * w
            loss_R_stage = loss_R_stage + d_R * w

        total_smooth_T += loss_T_stage * stage_weight
        total_smooth_R += loss_R_stage * stage_weight

    avg_T = total_smooth_T / n_stages
    avg_R = total_smooth_R / n_stages
    loss_camera_smooth = avg_T * weight_trans + avg_R * weight_rot
    loss_camera_smooth = check_and_fix_inf_nan(loss_camera_smooth, "loss_camera_smooth")

    out = {
        "loss_camera_smooth": loss_camera_smooth,
        "loss_smooth_T": avg_T,
        "loss_smooth_R": avg_R,
    }
    # Per-order values are UNWEIGHTED. They exist to calibrate order_weights: a velocity and an
    # acceleration are not on the same scale, so the two cannot be given the same weight blind.
    for o in orders:
        out[f"loss_smooth_T{o}"] = per_order[o][0] / n_stages
        out[f"loss_smooth_R{o}"] = per_order[o][1] / n_stages
    return out


def compute_gate_loss(predictions, batch, patch_size=14, alpha_m=10.0, beta_m=0.1,
                      hard_lo=None, hard_hi=None, **kwargs):
    """
    gate-predictor loss  L_gate = BCE( σ(g), m*_patch )  (docs/method.md §3.4).

    gate_logits g [B, S, P_patch] are supervised against m*_patch: the GT dynamic
    mask, averaged/pooled to patch resolution.

    m*_patch is derived from (in order of priority):
      1. batch["motion_mask"]  [B, S, H, W]  — binary GT dynamic mask m*_inst (§3.4,
         instance × 3D scene-flow for PointOdyssey). Average-pooled to patch grid to
         obtain a soft per-patch probability in [0, 1] (= m*_patch).
      2. (future) m*_geo: geometric residual ‖f^gt − f^cam‖ thresholded by α_m / β_m — §5.1.

    The loss is a standard binary cross-entropy on the logits (autocast-safe).

    C-1 (hard label + ignore-band). When `hard_lo`/`hard_hi` are set, the soft pooled
    label is discretised: patches with m*_patch >= hard_hi -> 1, <= hard_lo -> 0, and the
    boundary band (hard_lo, hard_hi) is IGNORED (excluded from the mean). This removes the
    irreducible BCE floor that soft boundary targets impose, which otherwise trains the gate
    to hedge (σ(g) capped ~0.3-0.5). Defaults (None) keep the original soft-label behaviour
    byte-for-byte. Typical: hard_lo=0.3, hard_hi=0.7.
    """
    gate_logits = predictions["gate_logits"]       # [B, S, P_patch], NOT detached → gradient flows
    motion_mask = batch["motion_mask"]             # [B, S, H, W], float or bool — m*_inst (§3.4)

    B, S, P_patch = gate_logits.shape
    _, _, H, W = motion_mask.shape

    # Derive patch grid dimensions
    P_h = H // patch_size
    P_w = W // patch_size

    # Average-pool GT mask to patch resolution → soft probability ∈ [0, 1] (= m*_patch)
    m_star_patch = F.adaptive_avg_pool2d(
        motion_mask.reshape(B * S, 1, H, W).float(),
        (P_h, P_w),
    ).reshape(B, S, P_h * P_w)  # [B, S, P_patch]

    if hard_lo is not None and hard_hi is not None:
        # C-1: hard label + boundary ignore-band. Keep only confident patches.
        keep = (m_star_patch <= hard_lo) | (m_star_patch >= hard_hi)  # [B, S, P_patch] bool
        target = (m_star_patch >= hard_hi).to(gate_logits.dtype)
        per_patch = F.binary_cross_entropy_with_logits(
            gate_logits, target, reduction="none",
        )
        keep = keep.to(per_patch.dtype)
        denom = keep.sum().clamp_min(1.0)
        loss = (per_patch * keep).sum() / denom
    else:
        # BCE on logits (more numerically stable than BCE on probabilities)
        loss = F.binary_cross_entropy_with_logits(
            gate_logits, m_star_patch.to(gate_logits.dtype),
        )
    loss = check_and_fix_inf_nan(loss, "loss_gate")
    return {"loss_gate": loss}


def _gaussian_blur_nc(x, sigma):
    """Separable Gaussian blur of [N,1,H,W], replicate-padded."""
    rad = max(1, int(ceil(3 * sigma)))
    t = torch.arange(-rad, rad + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (rad, rad, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, rad, rad), mode="replicate"), k.view(1, 1, -1, 1))


def illu_shading_target(depths, intrinsics, valid, target="phys", smooth_sigma=1.0):
    """Log near-field shading from depth, per pixel. The light is assumed at the camera centre.

      phys:  log( |n . v| / r^2 )   n = surface normal, v = unit direction to the camera
      invr2: log( 1 / r^2 )         the distance term alone -- the control that separates
                                    "illumination supervision" from "depth supervision in disguise"

    depths [N,H,W], intrinsics [N,3,3], valid [N,H,W] bool -> (log_s [N,H,W], valid [N,H,W]).
    Depth is smoothed first by NORMALIZED convolution (invalid pixels neither contribute nor get
    averaged in as zeros); raw depth quantisation otherwise dominates the finite-difference normals.
    Absolute depth scale only shifts log_s by a constant, which the loss removes per frame.
    """
    N, H, W = depths.shape
    m = valid.float().unsqueeze(1)
    d = depths.unsqueeze(1) * m
    if smooth_sigma > 0:
        den = _gaussian_blur_nc(m, smooth_sigma)
        d = _gaussian_blur_nc(d, smooth_sigma) / den.clamp_min(1e-3)
        m = m * (den > 1e-3).float()
    d = d[:, 0]
    valid = (m[:, 0] > 0) & (d > 0)

    fx, fy = intrinsics[:, 0, 0].view(N, 1, 1), intrinsics[:, 1, 1].view(N, 1, 1)
    cx, cy = intrinsics[:, 0, 2].view(N, 1, 1), intrinsics[:, 1, 2].view(N, 1, 1)
    v, u = torch.meshgrid(torch.arange(H, device=d.device, dtype=d.dtype),
                          torch.arange(W, device=d.device, dtype=d.dtype), indexing="ij")
    P = torch.stack([(u - cx) / fx * d, (v - cy) / fy * d, d], dim=-1)  # [N,H,W,3]
    r = P.norm(dim=-1).clamp_min(1e-6)
    log_s = -2.0 * torch.log(r)

    if target == "phys":
        dx = torch.zeros_like(P)
        dy = torch.zeros_like(P)
        dx[:, :, 1:-1] = P[:, :, 2:] - P[:, :, :-2]
        dy[:, 1:-1] = P[:, 2:] - P[:, :-2]
        n = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
        cos = (n * (-P / r.unsqueeze(-1))).sum(-1).abs()  # visible surface faces the camera
        log_s = log_s + torch.log(cos.clamp_min(1e-3))
        # a central difference needs both neighbours valid; also drop the image border
        nb = -F.max_pool2d(-valid.float().unsqueeze(1), 3, stride=1, padding=1)[:, 0]
        valid = valid & (nb > 0)
        valid[:, [0, -1], :] = False
        valid[:, :, [0, -1]] = False
    elif target != "invr2":
        raise ValueError(f"illu target must be 'phys' or 'invr2', got {target}")
    return log_s, valid


def compute_illu_loss(predictions, batch, target="phys", smooth_sigma=1.0, min_valid_frac=0.5, **kwargs):
    """Scale-invariant L1 between the illumination head's log map and GT-depth shading.

    The target is pooled to the head's own grid in log space (valid pixels only), and a cell
    counts only if at least `min_valid_frac` of it is valid. Per frame the residual is shifted by
    its (detached) median before the L1, so neither the trainer's per-sample depth normalisation
    nor the unknown light intensity matters. Computed at the grid, not upsampled: one token
    cannot express texture, and a full-resolution comparison would bake that in as an
    irreducible floor (the same trap as the gate's soft-label BCE).

    Also returns `loss_illu_corr`, the per-frame Pearson correlation over kept cells averaged
    over frames -- logged only, never added to the objective. It answers "did the head learn the
    target at all", which the shifted L1 alone does not.
    """
    pred = predictions["illu_log16"].float()  # [B,S,g,g]
    B, S, g, _ = pred.shape
    depths = batch["depths"].float()
    H, W = depths.shape[-2:]

    with torch.no_grad():
        log_s, valid = illu_shading_target(
            depths.reshape(B * S, H, W), batch["intrinsics"].float().reshape(B * S, 3, 3),
            batch["point_masks"].reshape(B * S, H, W).bool(), target=target, smooth_sigma=smooth_sigma,
        )
        vf = valid.float().unsqueeze(1)
        frac = F.adaptive_avg_pool2d(vf, g)
        tgt = F.adaptive_avg_pool2d(torch.where(valid, log_s, 0.0).unsqueeze(1), g) / frac.clamp_min(1e-6)
        tgt = tgt.view(B, S, g * g)
        keep = (frac.view(B, S, g * g) >= min_valid_frac)

    e = pred.view(B, S, g * g) - tgt
    losses, corrs = [], []
    for b in range(B):
        for s in range(S):
            k = keep[b, s]
            if k.sum() < 4:
                continue
            eb = e[b, s][k]
            losses.append((eb - eb.median().detach()).abs().mean())
            with torch.no_grad():
                p = pred[b, s].reshape(-1)[k]
                t = tgt[b, s][k]
                p, t = p - p.mean(), t - t.mean()
                corrs.append((p * t).sum() / (p.norm() * t.norm()).clamp_min(1e-8))

    if not losses:  # no frame had enough valid depth: keep the head in the graph, contribute 0
        return {"loss_illu": pred.sum() * 0.0, "loss_illu_corr": torch.zeros((), device=pred.device)}
    loss = check_and_fix_inf_nan(torch.stack(losses).mean(), "loss_illu")
    return {"loss_illu": loss, "loss_illu_corr": torch.stack(corrs).mean()}


def oracle_gate_logits_from_mask(motion_mask: torch.Tensor, patch_size: int = 14, k: float = 30.0) -> torch.Tensor:
    """Build a gate_logits_override straight from the GT dynamic mask (m*_inst), for the
    oracle-gate training ablation (the oracle_camera_only ablation; that config was never kept on disk): tests whether a
    camera token that structurally only aggregates static patches yields better pose, isolated
    from whether the learned gate predictor is accurate (docs/method.md gate
    diagnostics). Same construction as diag/gate_eval.py's oracle mode:
    static patch -> -k, dynamic patch -> +k, saturating softplus so bias is ~0 / ~-k.

    Args:
        motion_mask: [B, S, H, W] GT dynamic mask (m*_inst)
        patch_size: ViT patch size
        k: logit magnitude
    Returns:
        [B, S, P_patch] gate logits, ready for Aggregator.forward's gate_logits_override
    """
    B, S, H, W = motion_mask.shape
    P_h, P_w = H // patch_size, W // patch_size
    m_patch = F.adaptive_avg_pool2d(
        motion_mask.reshape(B * S, 1, H, W).float(), (P_h, P_w)
    ).reshape(B, S, P_h * P_w)
    return (m_patch * 2.0 - 1.0) * k


def compute_influence_loss(predictions, batch, use_point=True, use_pose=True,
                           w_point=1.0, w_pose=1.0, valid_range=0.98, **kwargs):
    """L_inf -- bound how much ONE view's APPEARANCE is allowed to move the OTHER views' geometry.

    The batch is forwarded twice: once clean under no_grad (the teacher, batch["_teacher"]) and
    once with a photometric corruption applied to a few frames (the student, `predictions`).
    This term scores, on the frames that were NOT corrupted, how far the student drifted from
    what the same model predicted before its neighbour went bad:

        L_inf = mean_{s not corrupted} || Y_s(I with frame k corrupted) - sg[ Y_s(I) ] ||_1

    WHY THIS IS NOT A RESTATEMENT OF L_sup. L_sup can only say "each frame's output should be
    near GT"; it has no way to say "frame s's output must not DEPEND on frame k's appearance".
    That is a statement about the Jacobian, and the difference above is precisely a finite
    difference of dY_s / d(appearance of I_k). Because the corruption changes pixel values and
    nothing else (data/photometric_corruption.py enforces that), the probe direction spans only
    the nuisance subspace: it cannot ask the model to discard frame k's GEOMETRIC contribution,
    only its appearance-induced one. Multi-view information is therefore left intact.

    Corrupted frames are excluded from the sum deliberately. A frame that was just turned green
    SHOULD predict worse -- information really was removed from it. What must not happen is that
    it drags its neighbours down with it, which is the failure this term exists to stop.

    THE DEGENERATE SOLUTION, and what blocks it. Alone, this loss is minimised by a model that
    ignores its input entirely and emits a constant. Two things stand in the way: the point head
    must also be under L_sup (`loss.point` non-null -- enabling the head without it hands L_inf
    exactly that shortcut), and only 1-2 of S frames are ever corrupted, so collapsing to
    single-view reconstruction costs accuracy on every sequence and never pays.

    The point term is normalised by the teacher's own point scale so that `weight` means the same
    thing regardless of how the scene normalisation happened to scale this batch.

    Args:
        predictions: student (corrupted-input) forward.
        batch: must carry "_teacher" (detached clean predictions) and "corrupt_frame_mask" [B,S].
        use_point / use_pose: which channels participate.
        valid_range: outlier quantile for the point channel, matching loss.point's own
            valid_range. Set to 0 to disable (this is what broke the first run).
        w_point / w_pose: relative weight between the two channels (the overall scale is the
            loss block's `weight`).
    """
    teacher = batch.get("_teacher", None)
    cmask = batch.get("corrupt_frame_mask", None)

    def _zero():
        ref = predictions.get("world_points", None)
        if ref is None:
            ref = predictions["pose_enc_list"][-1]
        z = (0.0 * ref).mean()
        return {"loss_inf_point": z, "loss_inf_pose": z}

    if teacher is None or cmask is None:
        return _zero()

    keep = ~cmask.to(predictions["pose_enc_list"][-1].device)   # [B, S], True = uncorrupted
    if keep.sum() == 0:
        return _zero()

    loss_point = None
    loss_pose = None

    if use_point and "world_points" in predictions and "world_points" in teacher:
        ps = check_and_fix_inf_nan(predictions["world_points"], "inf_student_points")
        pt = teacher["world_points"]
        # Per-PIXEL distance, then the SAME outlier treatment compute_point_loss gives its own
        # residuals (filter_by_quantile: clamp, then drop everything above `valid_range`).
        #
        # This is not defensive coding, it is the fix for a measured failure. The first
        # scared_point_inf run took a plain mean here and it destroyed the run:
        # diag/influence_scale_probe.py showed that by epoch 10 the point map's median was
        # 0.2684 and its p99 was 1.33 -- both unchanged from the warm start -- while its MAX had
        # reached 8.9e4. A few dozen pixels out of ~2M carried essentially the whole mean, so
        # loss_inf_point read 0.63 and, at weight 30, became the entire objective; every gradient
        # went into those pixels and the aggregator (and with it the pose) was wrecked.
        #
        # Those pixels diverged in the first place BECAUSE the two losses disagreed about
        # outliers: compute_point_loss's valid_range=0.98 filter drops the top 2% of residuals,
        # so the runaway pixels were never supervised and grew unwatched behind it, while L_inf
        # took their raw value at face value. Matching the filter here makes the two consistent.
        # On the same measurement the well-behaved 99% went the RIGHT way -- median drift fell
        # 0.00082 -> 0.00024 -- which is the signal this term is supposed to be reading.
        d = (ps - pt).abs().mean(dim=-1)                          # [B, S, H, W]
        keep_px = keep.view(*keep.shape, 1, 1).expand_as(d)
        vals = d[keep_px]
        if valid_range is not None and valid_range > 0:
            vals = filter_by_quantile(vals, valid_range)
        # MEDIAN, not mean, for the divisor: the same probe measured the teacher's mean |X| at
        # 12.62 against a median of 0.2684, i.e. the mean was itself outlier-dominated and would
        # have silently rescaled the loss by ~30x. The teacher runs under no_grad, so this
        # divisor carries no gradient and cannot be gamed by inflating the point map.
        scale = pt.abs().median().clamp(min=1e-3)
        loss_point = vals.mean() / scale

    if use_pose and "pose_enc_list" in predictions and "pose_enc_list" in teacher:
        qs = predictions["pose_enc_list"][-1]
        qt = teacher["pose_enc_list"][-1]
        dq = (qs - qt).abs().mean(dim=-1)                   # [B, S]
        loss_pose = (dq * keep).sum() / keep.sum()

    zero = _zero()["loss_inf_point"]
    return {
        "loss_inf_point": (loss_point * w_point) if loss_point is not None else zero,
        "loss_inf_pose": (loss_pose * w_pose) if loss_pose is not None else zero,
    }


def regression_loss(pred, gt, mask, conf=None, gradient_loss_fn=None, gamma=1.0, alpha=0.2, valid_range=-1):
    """
    Core regression loss function with confidence weighting and optional gradient loss.
    
    Computes:
    1. gamma * ||pred - gt||^2 * conf - alpha * log(conf)
    2. Optional gradient loss
    
    Args:
        pred: (B, S, H, W, C) predicted values
        gt: (B, S, H, W, C) ground truth values
        mask: (B, S, H, W) valid pixel mask
        conf: (B, S, H, W) confidence weights (optional)
        gradient_loss_fn: Type of gradient loss ("normal", "grad", etc.)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        valid_range: Quantile range for outlier filtering
    
    Returns:
        loss_conf: Confidence-weighted loss
        loss_grad: Gradient loss (0 if not specified)
        loss_reg: Regular L2 loss
    """
    bb, ss, hh, ww, nc = pred.shape

    # Compute L2 distance between predicted and ground truth points
    loss_reg = torch.norm(gt[mask] - pred[mask], dim=-1)
    loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

    # Confidence-weighted loss: gamma * loss * conf - alpha * log(conf)
    # This encourages the model to be confident on easy examples and less confident on hard ones
    loss_conf = gamma * loss_reg * conf[mask] - alpha * torch.log(conf[mask])
    loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf")
        
    # Initialize gradient loss
    loss_grad = 0

    # Prepare confidence for gradient loss if needed
    if "conf" in gradient_loss_fn:
        to_feed_conf = conf.reshape(bb*ss, hh, ww)
    else:
        to_feed_conf = None

    # Compute gradient loss if specified for spatial smoothness
    if "normal" in gradient_loss_fn:
        # Surface normal-based gradient loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=normal_loss,
            scales=3,
            conf=to_feed_conf,
        )
    elif "grad" in gradient_loss_fn:
        # Standard gradient-based loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=gradient_loss,
            conf=to_feed_conf,
        )

    # Process confidence-weighted loss
    if loss_conf.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_conf = filter_by_quantile(loss_conf, valid_range)

        loss_conf = check_and_fix_inf_nan(loss_conf, f"loss_conf_depth")
        loss_conf = loss_conf.mean()
    else:
        loss_conf = (0.0 * pred).mean()

    # Process regular regression loss
    if loss_reg.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_reg = filter_by_quantile(loss_reg, valid_range)

        loss_reg = check_and_fix_inf_nan(loss_reg, f"loss_reg_depth")
        loss_reg = loss_reg.mean()
    else:
        loss_reg = (0.0 * pred).mean()

    return loss_conf, loss_grad, loss_reg


def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn = None, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values  
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None, gamma=1.0, alpha=0.2):
    """
    Surface normal-based loss for geometric consistency.
    
    Computes surface normals from 3D point maps using cross products of neighboring points,
    then measures the angle between predicted and ground truth normals.
    
    Args:
        prediction: (B, H, W, 3) predicted 3D coordinates/points
        target: (B, H, W, 3) ground-truth 3D coordinates/points
        mask: (B, H, W) valid pixel mask
        cos_eps: Epsilon for numerical stability in cosine computation
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Convert point maps to surface normals using cross products
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids   = point_map_to_normal(target,     mask, eps=cos_eps)

    # Only consider regions where both predicted and GT normals are valid
    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    # Extract valid normals
    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    dot = torch.sum(pred_normals * gt_normals, dim=-1)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot

    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            # Apply confidence weighting
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            loss = gamma * loss * conf - alpha * torch.log(conf)
            return loss.mean()
        else:
            return loss.mean()


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    # Sum gradients and normalize by number of valid pixels
    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        grad_loss = torch.sum(grad_loss) / divisor

    return grad_loss


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    Convert 3D point map to surface normal vectors using cross products.
    
    Computes normals by taking cross products of neighboring point differences.
    Uses 4 different cross-product directions for robustness.
    
    Args:
        point_map: (B, H, W, 3) 3D points laid out in a 2D grid
        mask: (B, H, W) valid pixels (bool)
        eps: Epsilon for numerical stability in normalization
    
    Returns:
        normals: (4, B, H, W, 3) normal vectors for each of the 4 cross-product directions
        valids: (4, B, H, W) corresponding valid masks
    """
    with torch.amp.autocast('cuda', enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1,1,1,1), mode='constant', value=0).permute(0, 2, 3, 1)

        # Get neighboring points for each pixel
        center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
        up     = pts[:, :-2,  1:-1, :]
        left   = pts[:, 1:-1, :-2 , :]
        down   = pts[:, 2:,   1:-1, :]
        right  = pts[:, 1:-1, 2:,   :]

        # Compute direction vectors from center to neighbors
        up_dir    = up    - center
        left_dir  = left  - center
        down_dir  = down  - center
        right_dir = right - center

        # Compute four cross products for different normal directions
        n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir,up_dir,    dim=-1)  # right x up

        # Validity masks - require both direction pixels to be valid
        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        # Stack normals and validity masks
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids  = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize normal vectors
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filter loss tensor by keeping only values below a certain quantile threshold.
    
    This helps remove outliers that could destabilize training.
    
    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss
    
    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= min_elements:
        # Too few elements, just return as-is
        return loss_tensor

    # Randomly sample if tensor is too large to avoid memory issues
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values to prevent extreme outliers
    loss_tensor = loss_tensor.clamp(max=hard_max)

    # Compute quantile threshold
    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def torch_quantile(
    input,
    q,
    dim = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Validate out parameter
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out


########################################################################################
########################################################################################

# Dirty code for tracking loss:

########################################################################################
########################################################################################

'''
def _compute_losses(self, coord_preds, vis_scores, conf_scores, batch):
    """Compute tracking losses using sequence_loss"""
    gt_tracks = batch["tracks"]  # B, S, N, 2
    gt_track_vis_mask = batch["track_vis_mask"]  # B, S, N

    # if self.training and hasattr(self, "train_query_points"):
    train_query_points = coord_preds[-1].shape[2]
    gt_tracks = gt_tracks[:, :, :train_query_points]
    gt_tracks = check_and_fix_inf_nan(gt_tracks, "gt_tracks", hard_max=None)

    gt_track_vis_mask = gt_track_vis_mask[:, :, :train_query_points]

    # Create validity mask that filters out tracks not visible in first frame
    valids = torch.ones_like(gt_track_vis_mask)
    mask = gt_track_vis_mask[:, 0, :] == True
    valids = valids * mask.unsqueeze(1)



    if not valids.any():
        print("No valid tracks found in first frame")
        print("seq_name: ", batch["seq_name"])
        print("ids: ", batch["ids"])
        print("time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        dummy_coord = coord_preds[0].mean() * 0          # keeps graph & grads
        dummy_vis = vis_scores.mean() * 0
        if conf_scores is not None:
            dummy_conf = conf_scores.mean() * 0
        else:
            dummy_conf = 0
        return dummy_coord, dummy_vis, dummy_conf                # three scalar zeros


    # Compute tracking loss using sequence_loss
    track_loss = sequence_loss(
        flow_preds=coord_preds,
        flow_gt=gt_tracks,
        vis=gt_track_vis_mask,
        valids=valids,
        **self.loss_kwargs
    )

    vis_loss = F.binary_cross_entropy_with_logits(vis_scores[valids], gt_track_vis_mask[valids].float())

    vis_loss = check_and_fix_inf_nan(vis_loss, "vis_loss", hard_max=None)


    # within 3 pixels
    if conf_scores is not None:
        gt_conf_mask = (gt_tracks - coord_preds[-1]).norm(dim=-1) < 3
        conf_loss = F.binary_cross_entropy_with_logits(conf_scores[valids], gt_conf_mask[valids].float())
        conf_loss = check_and_fix_inf_nan(conf_loss, "conf_loss", hard_max=None)
    else:
        conf_loss = 0

    return track_loss, vis_loss, conf_loss



def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    for a, b in zip(x.size(), mask.size()):
        assert a == b
    prod = x * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / denom.clamp(min=1)
    mean = torch.where(denom > 0,
                       mean,
                       torch.zeros_like(mean))
    return mean


def sequence_loss(flow_preds, flow_gt, vis, valids, gamma=0.8, vis_aware=False, huber=False, delta=10, vis_aware_w=0.1, **kwargs):
    """Loss function defined over sequence of flow predictions"""
    B, S, N, D = flow_gt.shape
    assert D == 2
    B, S1, N = vis.shape
    B, S2, N = valids.shape
    assert S == S1
    assert S == S2
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        flow_pred = flow_preds[i]

        i_loss = (flow_pred - flow_gt).abs()  # B, S, N, 2
        i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_{i}", hard_max=None)

        i_loss = torch.mean(i_loss, dim=3) # B, S, N

        # Combine valids and vis for per-frame valid masking.
        combined_mask = torch.logical_and(valids, vis)

        num_valid_points = combined_mask.sum()

        if vis_aware:
            combined_mask = combined_mask.float() * (1.0 + vis_aware_w)  # Add, don't add to the mask itself.
            flow_loss += i_weight * reduce_masked_mean(i_loss, combined_mask)
        else:
            if num_valid_points > 2:
                i_loss = i_loss[combined_mask]
                flow_loss += i_weight * i_loss.mean()
            else:
                i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_safe_check_{i}", hard_max=None)
                flow_loss += 0 * i_loss.mean()

    # Avoid division by zero if n_predictions is 0 (though it shouldn't be).
    if n_predictions > 0:
        flow_loss = flow_loss / n_predictions

    return flow_loss
'''


