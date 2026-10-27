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
                 self_photo=None,     # self-supervised photometric reprojection (predicted depth + pose)
                 smooth=None,         # edge-aware disparity smoothness (pairs with self_photo)
                 flow_geom=None,      # cross-frame 3D consistency via RAFT correspondence (ColonAdapter)
                 camera_smooth=None,  # Extension: camera-trajectory smoothness regularizer
                 **kwargs):
        super().__init__()
        # Loss configuration dictionaries for each task
        self.camera = camera
        self.depth = depth
        self.point = point
        self.track = track
        self.self_photo = self_photo       # self-supervised photometric reprojection (no GT)
        self.smooth = smooth               # edge-aware disparity smoothness
        self.flow_geom = flow_geom         # cross-frame 3D consistency via RAFT correspondence
        self.camera_smooth = camera_smooth  # Extension: camera-trajectory smoothness regularizer

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

    This is the label-free sibling of compute_static_photo_loss (removed; tag archive-pre-slim). The difference is not
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


