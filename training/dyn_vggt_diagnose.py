#!/usr/bin/env python3
"""
Dyn-VGGT diagnostic evaluation (items ①②⑤⑥).

Runs a fixed set of Sintel clips through one or more checkpoints and produces
per-pixel heatmaps, histograms, calibration curves, scatter plots, and temporal
consistency analysis.  All outputs land in  logs/diag/<stage>/ .

Usage (from training/):
  # Sintel
  python dyn_vggt_diagnose.py \
      --ckpts checkpoints/VGGT-1B.pt:vggt_base checkpoints/dyn_vggt_s1a.pt:s1a \
      --sintel_root /media/cvml-75/ssd2t1/data/sintel/training \
      --out_dir logs/diag

  # PointOdyssey (has native GT motion masks)
  python dyn_vggt_diagnose.py --dataset po \
      --ckpts checkpoints/dyn_vggt_s1a.pt:s1a \
      --n_clips 10 --img_per_seq 6 \
      --out_dir logs/diag_po
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.sintel_io import (
    compute_preprocess_meta,
    load_sintel_gt_depths,
    load_sintel_gt_poses,
    load_sintel_rgb_paths,
    read_sintel_depth,
    resize_pred_to_gt,
    sintel_cam_read,
    sintel_seq_paths,
)
from eval.vggt_infer import load_dyn_vggt

from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

DIAG_SEQUENCES = ["sleeping_2", "temple_2", "market_5", "cave_2", "temple_3"]

TAG_FLOAT = 202021.25


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_flo(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        assert magic == TAG_FLOAT, f"Bad .flo magic: {magic}"
        w = int(np.fromfile(f, np.int32, count=1)[0])
        h = int(np.fromfile(f, np.int32, count=1)[0])
        return np.fromfile(f, np.float32, count=h * w * 2).reshape((h, w, 2))


def load_sintel_gt_flows(sintel_root: str, seq: str, rgb_paths: List[str]) -> List[Optional[np.ndarray]]:
    flow_dir = os.path.join(sintel_root, "flow", seq)
    flows: List[Optional[np.ndarray]] = []
    for p in rgb_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        flo_path = os.path.join(flow_dir, f"{stem}.flo")
        if os.path.isfile(flo_path):
            flows.append(read_flo(flo_path))
        else:
            flows.append(None)
    return flows


def compute_ego_flow(
    depth: np.ndarray,
    K: np.ndarray,
    ext_cur: np.ndarray,
    ext_next: np.ndarray,
) -> np.ndarray:
    """Camera-induced optical flow from depth + relative pose (GT)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    ones = np.ones_like(u)
    K_inv = np.linalg.inv(K.astype(np.float64))

    pts_cam = np.stack([u, v, ones], axis=-1)  # (H,W,3)
    pts_cam = (K_inv @ pts_cam[..., None])[..., 0]  # unproject
    pts_cam = pts_cam * depth[..., None].astype(np.float64)

    w2c_cur = np.vstack([ext_cur, [0, 0, 0, 1]]).astype(np.float64)
    w2c_next = np.vstack([ext_next, [0, 0, 0, 1]]).astype(np.float64)
    rel = w2c_next @ np.linalg.inv(w2c_cur)

    R_rel = rel[:3, :3]
    t_rel = rel[:3, 3]
    pts_next = (R_rel @ pts_cam[..., None])[..., 0] + t_rel[None, None, :]

    proj = (K.astype(np.float64) @ pts_next[..., None])[..., 0]
    z = np.clip(proj[..., 2], 1e-8, None)
    u_next = proj[..., 0] / z
    v_next = proj[..., 1] / z

    flow_ego = np.stack([u_next - u, v_next - v], axis=-1).astype(np.float32)
    return flow_ego


def derive_motion_mask(
    gt_flow: np.ndarray,
    ego_flow: np.ndarray,
    threshold: float = 2.0,
) -> np.ndarray:
    residual = np.linalg.norm(gt_flow - ego_flow, axis=-1)
    return (residual > threshold).astype(np.float32)


# ---------------------------------------------------------------------------
# Extended inference: return full pred dict (not just depth/pose)
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_full(
    model,
    image_paths: List[str],
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    images = load_and_preprocess_images(image_paths, mode="crop").to(device)
    if images.dim() == 4:
        images = images.unsqueeze(0)
    with torch.amp.autocast("cuda", dtype=dtype, enabled=(device == "cuda")):
        pred = model(images=images)

    h, w = images.shape[-2], images.shape[-1]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], image_size_hw=(h, w))

    out: Dict[str, Any] = {
        "extrinsic": extrinsic.squeeze(0).float().cpu().numpy(),
        "intrinsic": intrinsic.squeeze(0).float().cpu().numpy(),
        "pose_enc": pred["pose_enc"].squeeze(0).float().cpu().numpy(),
        "input_hw": np.array([h, w], dtype=np.int32),
    }

    depth = pred["depth"]
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    out["depth"] = depth.squeeze(0).float().cpu().numpy()

    if "world_points" in pred:
        out["world_points"] = pred["world_points"].squeeze(0).float().cpu().numpy()
    if "world_points_dyn" in pred:
        out["world_points_dyn"] = pred["world_points_dyn"].squeeze(0).float().cpu().numpy()
    if "motion_prob" in pred:
        mp = pred["motion_prob"]
        if mp.ndim == 5 and mp.shape[-1] == 1:
            mp = mp[..., 0]
        out["motion_prob"] = mp.squeeze(0).float().cpu().numpy()
    if "scene_flow" in pred:
        out["scene_flow"] = pred["scene_flow"].squeeze(0).float().cpu().numpy()

    return out


@torch.no_grad()
def infer_from_tensor(
    model,
    images_tensor: torch.Tensor,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Run model on a pre-processed image tensor (B,S,C,H,W) or (S,C,H,W)."""
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    if images_tensor.dim() == 4:
        images_tensor = images_tensor.unsqueeze(0)
    images_tensor = images_tensor.to(device)
    with torch.amp.autocast("cuda", dtype=dtype, enabled=(device == "cuda")):
        pred = model(images=images_tensor)

    h, w = images_tensor.shape[-2], images_tensor.shape[-1]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], image_size_hw=(h, w))

    out: Dict[str, Any] = {
        "extrinsic": extrinsic.squeeze(0).float().cpu().numpy(),
        "intrinsic": intrinsic.squeeze(0).float().cpu().numpy(),
        "pose_enc": pred["pose_enc"].squeeze(0).float().cpu().numpy(),
        "input_hw": np.array([h, w], dtype=np.int32),
    }

    depth = pred["depth"]
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    out["depth"] = depth.squeeze(0).float().cpu().numpy()

    if "world_points" in pred:
        out["world_points"] = pred["world_points"].squeeze(0).float().cpu().numpy()
    if "world_points_dyn" in pred:
        out["world_points_dyn"] = pred["world_points_dyn"].squeeze(0).float().cpu().numpy()
    if "motion_prob" in pred:
        mp = pred["motion_prob"]
        if mp.ndim == 5 and mp.shape[-1] == 1:
            mp = mp[..., 0]
        out["motion_prob"] = mp.squeeze(0).float().cpu().numpy()
    if "scene_flow" in pred:
        out["scene_flow"] = pred["scene_flow"].squeeze(0).float().cpu().numpy()

    return out


# ---------------------------------------------------------------------------
# PointOdyssey data helpers
# ---------------------------------------------------------------------------

def load_po_dataset(po_root: str, img_size: int = 518, img_per_seq: int = 6, min_num_images: int = 6):
    from types import SimpleNamespace
    from data.datasets.pointodyssey import PointOdysseyDataset

    common = SimpleNamespace(
        img_size=img_size, patch_size=14,
        augs=SimpleNamespace(scales=None), rescale=True, rescale_aug=False,
        landscape_check=False, debug=False, training=False, get_nearby=True,
        load_depth=True, inside_random=False, allow_duplicate_img=False,
    )
    ds = PointOdysseyDataset(common_conf=common, split="test", PO_DIR=po_root, min_num_images=min_num_images)
    return ds


def collate_batch(batch: dict) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collate a dataset batch dict into tensors for inference + GT arrays.
    Works for PO (has motion_mask), TA/WO (no motion_mask → all-zeros)."""
    img = torch.from_numpy(np.stack(batch["images"]).astype(np.float32)).permute(0, 3, 1, 2).div(255)
    depths = np.stack(batch["depths"]).astype(np.float32)
    extrinsics = np.stack(batch["extrinsics"]).astype(np.float32)
    intrinsics = np.stack(batch["intrinsics"]).astype(np.float32)
    rgb_images = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    if "motion_mask" in batch:
        motion_masks = np.stack(batch["motion_mask"]).astype(np.float32)
    else:
        motion_masks = np.zeros(depths.shape, dtype=np.float32)
    return img, depths, motion_masks, extrinsics, intrinsics, rgb_images


def load_ta_dataset(ta_root: str, img_size: int = 518, min_num_images: int = 6):
    from types import SimpleNamespace
    from data.datasets.tartanair import TartanAirDataset

    common = SimpleNamespace(
        img_size=img_size, patch_size=14,
        augs=SimpleNamespace(scales=None), rescale=True, rescale_aug=False,
        landscape_check=False, debug=False, training=False, get_nearby=True,
        load_depth=True, inside_random=False, allow_duplicate_img=False,
    )
    return TartanAirDataset(common_conf=common, TARTANAIR_DIR=ta_root, min_num_images=min_num_images)


def load_wo_dataset(wo_root: str, img_size: int = 518, min_num_images: int = 6):
    from types import SimpleNamespace
    from data.datasets.waymo import WaymoDataset

    common = SimpleNamespace(
        img_size=img_size, patch_size=14,
        augs=SimpleNamespace(scales=None), rescale=True, rescale_aug=False,
        landscape_check=False, debug=False, training=False, get_nearby=True,
        load_depth=True, inside_random=False, allow_duplicate_img=False,
    )
    return WaymoDataset(common_conf=common, WAYMO_DIR=wo_root, min_num_images=min_num_images)


# ---------------------------------------------------------------------------
# ① Motion mask calibration
# ---------------------------------------------------------------------------

def diag_motion_calibration(
    pred: Dict[str, np.ndarray],
    gt_motion_masks: List[np.ndarray],
    rgb_paths: Optional[List[str]],
    out_dir: str,
    seq: str,
    rgb_images: Optional[np.ndarray] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    if "motion_prob" not in pred:
        return {}

    m_pred = pred["motion_prob"]  # (S, H_model, W_model)
    S = m_pred.shape[0]
    n_gt = len(gt_motion_masks)
    n_frames = min(S, n_gt)

    all_probs, all_labels = [], []

    for i in range(n_frames):
        gt_mask_orig = gt_motion_masks[i]
        if gt_mask_orig is None:
            continue
        m_frame = m_pred[i]
        gt_resized = cv2.resize(gt_mask_orig, (m_frame.shape[1], m_frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        all_probs.append(m_frame.flatten())
        all_labels.append(gt_resized.flatten())

        if i < 3:
            if rgb_images is not None:
                rgb_small = cv2.resize(rgb_images[i], (m_frame.shape[1], m_frame.shape[0]))
                rgb_small = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR)
            elif rgb_paths is not None:
                rgb = cv2.imread(rgb_paths[i])
                rgb_small = cv2.resize(rgb, (m_frame.shape[1], m_frame.shape[0]))
            else:
                continue
            m_vis = (m_frame * 255).astype(np.uint8)
            m_color = cv2.applyColorMap(m_vis, cv2.COLORMAP_JET)
            gt_vis = (gt_resized * 255).astype(np.uint8)
            gt_color = cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2BGR)
            row = np.concatenate([rgb_small, m_color, gt_color], axis=1)
            cv2.imwrite(os.path.join(out_dir, f"{seq}_motion_triplet_f{i}.png"), row)

    if not all_probs:
        return {}

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    labels_bin = (labels > 0.5).astype(np.int32)

    # Reliability diagram (10 bins)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_acc = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (probs >= bin_edges[b]) & (probs < bin_edges[b + 1])
        if b == n_bins - 1:
            mask |= probs == bin_edges[b + 1]
        bin_count[b] = mask.sum()
        if bin_count[b] > 0:
            bin_acc[b] = labels_bin[mask].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.bar(bin_centers, bin_acc, width=0.08, alpha=0.7, label="Observed")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Actual dynamic fraction")
    ax.set_title(f"Reliability Diagram — {seq}")
    ax.legend()

    # Precision-Recall
    from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score

    subsample = min(500000, len(probs))
    idx = np.random.choice(len(probs), subsample, replace=False)
    p_sub, l_sub = probs[idx], labels_bin[idx]

    if l_sub.sum() > 0 and (1 - l_sub).sum() > 0:
        precision, recall, thresholds = precision_recall_curve(l_sub, p_sub)
        ap = average_precision_score(l_sub, p_sub)
        auc_roc = roc_auc_score(l_sub, p_sub)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
        best_idx = np.argmax(f1_scores)
        best_thr = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        ax = axes[1]
        ax.plot(recall, precision, "b-")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve — AP={ap:.3f}, ROC-AUC={auc_roc:.3f}, best_thr={best_thr:.2f}")
    else:
        ap = auc_roc = best_thr = 0.0
        axes[1].set_title("PR Curve — insufficient pos/neg samples")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{seq}_motion_calibration.png"), dpi=150)
    plt.close(fig)

    return {
        "seq": seq,
        "ap": float(ap),
        "roc_auc": float(auc_roc),
        "best_threshold": float(best_thr),
        "dynamic_fraction": float(labels_bin.mean()),
        "mean_prob_on_dynamic": float(probs[labels_bin == 1].mean()) if labels_bin.sum() > 0 else 0.0,
        "mean_prob_on_static": float(probs[labels_bin == 0].mean()) if (1 - labels_bin).sum() > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# ② Dual-field decoupling
# ---------------------------------------------------------------------------

def diag_dual_field(
    pred: Dict[str, np.ndarray],
    gt_motion_masks: List[np.ndarray],
    rgb_paths: Optional[List[str]],
    out_dir: str,
    seq: str,
    rgb_images: Optional[np.ndarray] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    if "scene_flow" not in pred or "world_points" not in pred:
        return {}

    scene_flow = pred["scene_flow"]  # (S,H,W,3)
    world_pts = pred["world_points"]  # (S,H,W,3)
    S = scene_flow.shape[0]
    n_gt = len(gt_motion_masks)
    n_frames = min(S, n_gt)

    delta_norms_static, delta_norms_dynamic = [], []

    for i in range(n_frames):
        gt_mask = gt_motion_masks[i]
        if gt_mask is None:
            continue

        sf = scene_flow[i]  # (H,W,3)
        delta_norm = np.linalg.norm(sf, axis=-1)  # (H,W)

        gt_resized = cv2.resize(gt_mask, (delta_norm.shape[1], delta_norm.shape[0]), interpolation=cv2.INTER_NEAREST)
        dyn_mask = gt_resized > 0.5
        stat_mask = ~dyn_mask

        delta_norms_static.append(delta_norm[stat_mask])
        delta_norms_dynamic.append(delta_norm[dyn_mask])

        if i < 3:
            fig = plt.figure(figsize=(16, 4))
            gs = GridSpec(1, 4, figure=fig)

            # Panel 1: RGB
            ax1 = fig.add_subplot(gs[0, 0])
            if rgb_images is not None:
                rgb_show = cv2.resize(rgb_images[i], (delta_norm.shape[1], delta_norm.shape[0]))
            elif rgb_paths is not None:
                rgb_show = cv2.imread(rgb_paths[i])
                rgb_show = cv2.resize(rgb_show, (delta_norm.shape[1], delta_norm.shape[0]))
                rgb_show = cv2.cvtColor(rgb_show, cv2.COLOR_BGR2RGB)
            else:
                rgb_show = np.zeros((delta_norm.shape[0], delta_norm.shape[1], 3), dtype=np.uint8)
            ax1.imshow(rgb_show)
            ax1.set_title("RGB")
            ax1.axis("off")

            # Panel 2: ||Δ|| heatmap
            ax2 = fig.add_subplot(gs[0, 1])
            vmax = np.percentile(delta_norm, 99)
            ax2.imshow(delta_norm, cmap="hot", vmin=0, vmax=max(vmax, 0.01))
            ax2.set_title("‖Δ‖ (scene flow magnitude)")
            ax2.axis("off")

            # Panel 3: GT motion mask
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.imshow(gt_resized, cmap="gray", vmin=0, vmax=1)
            ax3.set_title("GT motion mask")
            ax3.axis("off")

            # Panel 4: histogram
            ax4 = fig.add_subplot(gs[0, 3])
            clip_val = np.percentile(delta_norm, 99.5)
            if stat_mask.sum() > 0:
                ax4.hist(delta_norm[stat_mask].clip(0, clip_val), bins=50, alpha=0.6, label="Static", density=True, color="blue")
            if dyn_mask.sum() > 0:
                ax4.hist(delta_norm[dyn_mask].clip(0, clip_val), bins=50, alpha=0.6, label="Dynamic", density=True, color="red")
            ax4.set_xlabel("‖Δ‖")
            ax4.set_ylabel("Density")
            ax4.set_title("‖Δ‖ by motion class")
            ax4.legend()

            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{seq}_dual_field_f{i}.png"), dpi=150)
            plt.close(fig)

    all_static = np.concatenate(delta_norms_static) if delta_norms_static else np.array([])
    all_dynamic = np.concatenate(delta_norms_dynamic) if delta_norms_dynamic else np.array([])

    # Aggregated histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    clip_val = np.percentile(np.concatenate([all_static, all_dynamic]), 99.5) if len(all_static) + len(all_dynamic) > 0 else 1
    if len(all_static) > 0:
        ax.hist(all_static.clip(0, clip_val), bins=80, alpha=0.6, label=f"Static (med={np.median(all_static):.4f})", density=True, color="blue")
    if len(all_dynamic) > 0:
        ax.hist(all_dynamic.clip(0, clip_val), bins=80, alpha=0.6, label=f"Dynamic (med={np.median(all_dynamic):.4f})", density=True, color="red")
    ax.set_xlabel("‖Δ‖ (scene flow magnitude)")
    ax.set_title(f"Dual-field decoupling — {seq}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{seq}_dual_field_hist.png"), dpi=150)
    plt.close(fig)

    # Cross-frame X^can stability for dynamic pixels
    xcan_dyn_per_frame = []
    for i in range(n_frames):
        gt_mask = gt_motion_masks[i]
        if gt_mask is None:
            continue
        gt_resized = cv2.resize(gt_mask, (world_pts.shape[2], world_pts.shape[1]), interpolation=cv2.INTER_NEAREST)
        dyn_mask = gt_resized > 0.5
        if dyn_mask.sum() > 0:
            xcan_dyn_per_frame.append(world_pts[i][dyn_mask].mean(axis=0))
    xcan_dyn_per_frame = np.array(xcan_dyn_per_frame) if xcan_dyn_per_frame else np.zeros((0, 3))

    assembled_dyn_per_frame = []
    if "world_points_dyn" in pred:
        wpts_dyn = pred["world_points_dyn"]
        for i in range(n_frames):
            gt_mask = gt_motion_masks[i]
            if gt_mask is None:
                continue
            gt_resized = cv2.resize(gt_mask, (wpts_dyn.shape[2], wpts_dyn.shape[1]), interpolation=cv2.INTER_NEAREST)
            dyn_mask = gt_resized > 0.5
            if dyn_mask.sum() > 0:
                assembled_dyn_per_frame.append(wpts_dyn[i][dyn_mask].mean(axis=0))
    assembled_dyn_per_frame = np.array(assembled_dyn_per_frame) if assembled_dyn_per_frame else np.zeros((0, 3))

    # Plot X^can centroid vs assembled centroid trajectory
    if xcan_dyn_per_frame.shape[0] >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        labels_3d = ["X", "Y", "Z"]
        for d in range(3):
            axes[d].plot(xcan_dyn_per_frame[:, d], "b-o", markersize=3, label="X^can centroid")
            if assembled_dyn_per_frame.shape[0] >= 2:
                axes[d].plot(assembled_dyn_per_frame[:, d], "r-o", markersize=3, label="Assembled centroid")
            axes[d].set_xlabel("Frame")
            axes[d].set_ylabel(labels_3d[d])
            axes[d].set_title(f"Dynamic obj {labels_3d[d]}-trajectory")
            axes[d].legend(fontsize=7)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{seq}_dual_field_trajectories.png"), dpi=150)
        plt.close(fig)

    metrics = {
        "seq": seq,
        "static_delta_median": float(np.median(all_static)) if len(all_static) > 0 else None,
        "static_delta_mean": float(all_static.mean()) if len(all_static) > 0 else None,
        "dynamic_delta_median": float(np.median(all_dynamic)) if len(all_dynamic) > 0 else None,
        "dynamic_delta_mean": float(all_dynamic.mean()) if len(all_dynamic) > 0 else None,
        "separation_ratio": (
            float(np.median(all_dynamic) / (np.median(all_static) + 1e-8))
            if len(all_static) > 0 and len(all_dynamic) > 0
            else None
        ),
    }
    # X^can stability: std of centroid across frames
    if xcan_dyn_per_frame.shape[0] >= 2:
        metrics["xcan_centroid_std"] = float(xcan_dyn_per_frame.std(axis=0).mean())
    if assembled_dyn_per_frame.shape[0] >= 2:
        metrics["assembled_centroid_std"] = float(assembled_dyn_per_frame.std(axis=0).mean())

    return metrics


# ---------------------------------------------------------------------------
# ②-oracle: Dual-field with GT mask replacing predicted m
# ---------------------------------------------------------------------------

def diag_dual_field_oracle(
    pred: Dict[str, np.ndarray],
    gt_motion_masks: List[np.ndarray],
    rgb_paths: Optional[List[str]],
    out_dir: str,
    seq: str,
    rgb_images: Optional[np.ndarray] = None,
):
    """Re-run ② analysis but assemble points as X^can + gt_mask·Δ instead of m·Δ.

    If ‖Δ‖ histogram shows bimodal separation under GT mask → architecture is fine,
    motion head is the bottleneck.  If still unimodal → identifiability problem in
    the dual-field decomposition itself.
    """
    os.makedirs(out_dir, exist_ok=True)
    if "scene_flow" not in pred or "world_points" not in pred:
        return {}

    scene_flow = pred["scene_flow"]  # (S,H,W,3)
    world_pts = pred["world_points"]  # (S,H,W,3)
    S = scene_flow.shape[0]
    n_gt = len(gt_motion_masks)
    n_frames = min(S, n_gt)

    delta_norms_static, delta_norms_dynamic = [], []

    for i in range(n_frames):
        gt_mask = gt_motion_masks[i]
        if gt_mask is None:
            continue

        sf = scene_flow[i]  # (H,W,3)
        delta_norm = np.linalg.norm(sf, axis=-1)  # (H,W)

        gt_resized = cv2.resize(
            gt_mask, (delta_norm.shape[1], delta_norm.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        dyn_mask = gt_resized > 0.5
        stat_mask = ~dyn_mask

        delta_norms_static.append(delta_norm[stat_mask])
        delta_norms_dynamic.append(delta_norm[dyn_mask])

        if i < 3:
            # Oracle-assembled points: X^can + gt_mask·Δ
            oracle_assembled = world_pts[i] + gt_resized[..., None] * sf  # (H,W,3)
            pred_assembled = pred.get("world_points_dyn", world_pts + pred.get("motion_prob", np.zeros_like(delta_norm))[..., None] * sf)
            if pred_assembled.ndim == 4:
                pred_assembled_i = pred_assembled[i]
            else:
                pred_assembled_i = pred_assembled

            fig = plt.figure(figsize=(20, 5))
            gs = GridSpec(1, 5, figure=fig)

            # Panel 1: RGB
            ax = fig.add_subplot(gs[0, 0])
            if rgb_images is not None:
                rgb_show = cv2.resize(rgb_images[i], (delta_norm.shape[1], delta_norm.shape[0]))
            elif rgb_paths is not None:
                rgb_show = cv2.imread(rgb_paths[i])
                rgb_show = cv2.resize(rgb_show, (delta_norm.shape[1], delta_norm.shape[0]))
                rgb_show = cv2.cvtColor(rgb_show, cv2.COLOR_BGR2RGB)
            else:
                rgb_show = np.zeros((delta_norm.shape[0], delta_norm.shape[1], 3), dtype=np.uint8)
            ax.imshow(rgb_show)
            ax.set_title("RGB")
            ax.axis("off")

            # Panel 2: ‖Δ‖ heatmap
            ax = fig.add_subplot(gs[0, 1])
            vmax = np.percentile(delta_norm, 99)
            ax.imshow(delta_norm, cmap="hot", vmin=0, vmax=max(vmax, 0.01))
            ax.set_title("‖Δ‖")
            ax.axis("off")

            # Panel 3: GT mask
            ax = fig.add_subplot(gs[0, 2])
            ax.imshow(gt_resized, cmap="gray", vmin=0, vmax=1)
            ax.set_title("GT motion mask (oracle)")
            ax.axis("off")

            # Panel 4: ‖Δ‖ histogram split by GT mask
            ax = fig.add_subplot(gs[0, 3])
            clip_val = np.percentile(delta_norm, 99.5)
            if stat_mask.sum() > 0:
                ax.hist(delta_norm[stat_mask].clip(0, clip_val), bins=50, alpha=0.6,
                        label="Static", density=True, color="blue")
            if dyn_mask.sum() > 0:
                ax.hist(delta_norm[dyn_mask].clip(0, clip_val), bins=50, alpha=0.6,
                        label="Dynamic", density=True, color="red")
            ax.set_xlabel("‖Δ‖")
            ax.set_ylabel("Density")
            ax.set_title("‖Δ‖ by GT class")
            ax.legend()

            # Panel 5: oracle vs pred assembled error (if GT depth available)
            ax = fig.add_subplot(gs[0, 4])
            oracle_norm = np.linalg.norm(oracle_assembled, axis=-1)
            pred_norm = np.linalg.norm(pred_assembled_i, axis=-1)
            diff = np.abs(oracle_norm - pred_norm)
            ax.imshow(diff, cmap="magma", vmin=0, vmax=np.percentile(diff, 99))
            ax.set_title("|oracle − pred| assembled")
            ax.axis("off")

            plt.suptitle(f"Oracle mask ② — {seq} frame {i}", fontsize=12)
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{seq}_oracle_dual_field_f{i}.png"), dpi=150)
            plt.close(fig)

    all_static = np.concatenate(delta_norms_static) if delta_norms_static else np.array([])
    all_dynamic = np.concatenate(delta_norms_dynamic) if delta_norms_dynamic else np.array([])

    # Aggregated oracle histogram
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: oracle ‖Δ‖ histogram
    ax = axes[0]
    clip_val = (
        np.percentile(np.concatenate([all_static, all_dynamic]), 99.5)
        if len(all_static) + len(all_dynamic) > 0
        else 1
    )
    if len(all_static) > 0:
        ax.hist(all_static.clip(0, clip_val), bins=80, alpha=0.6,
                label=f"Static (med={np.median(all_static):.4f})", density=True, color="blue")
    if len(all_dynamic) > 0:
        ax.hist(all_dynamic.clip(0, clip_val), bins=80, alpha=0.6,
                label=f"Dynamic (med={np.median(all_dynamic):.4f})", density=True, color="red")
    ax.set_xlabel("‖Δ‖ (scene flow magnitude)")
    ax.set_title(f"Oracle ② — {seq}")
    ax.legend()

    # Right: diagnosis verdict
    ax = axes[1]
    ax.axis("off")
    if len(all_static) > 0 and len(all_dynamic) > 0:
        sep_ratio = float(np.median(all_dynamic) / (np.median(all_static) + 1e-8))
        overlap = _histogram_overlap(all_static, all_dynamic, clip_val)
        verdict_lines = [
            f"Static  Δ median: {np.median(all_static):.5f}",
            f"Dynamic Δ median: {np.median(all_dynamic):.5f}",
            f"Separation ratio: {sep_ratio:.2f}x",
            f"Histogram overlap: {overlap:.1%}",
            "",
            "DIAGNOSIS:",
        ]
        if sep_ratio > 2.0 and overlap < 0.6:
            verdict_lines.append("✓ Bimodal separation under GT mask")
            verdict_lines.append("→ Architecture OK, problem is motion head")
        elif sep_ratio > 1.3:
            verdict_lines.append("~ Weak separation under GT mask")
            verdict_lines.append("→ Partial identifiability; motion head + mild Δ prior may help")
        else:
            verdict_lines.append("✗ No separation even with GT mask")
            verdict_lines.append("→ Architecture-level identifiability issue")
            verdict_lines.append("  (needs Δ shrinkage prior / X^can anchor)")
        ax.text(0.05, 0.95, "\n".join(verdict_lines), transform=ax.transAxes,
                fontsize=11, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    else:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{seq}_oracle_dual_field_hist.png"), dpi=150)
    plt.close(fig)

    metrics = {
        "seq": seq,
        "mode": "oracle",
        "static_delta_median": float(np.median(all_static)) if len(all_static) > 0 else None,
        "dynamic_delta_median": float(np.median(all_dynamic)) if len(all_dynamic) > 0 else None,
        "separation_ratio": (
            float(np.median(all_dynamic) / (np.median(all_static) + 1e-8))
            if len(all_static) > 0 and len(all_dynamic) > 0
            else None
        ),
    }
    if len(all_static) > 0 and len(all_dynamic) > 0:
        metrics["histogram_overlap"] = float(_histogram_overlap(all_static, all_dynamic, clip_val))

    return metrics


def _histogram_overlap(a: np.ndarray, b: np.ndarray, clip_val: float, n_bins: int = 80) -> float:
    bins = np.linspace(0, clip_val, n_bins + 1)
    ha, _ = np.histogram(a.clip(0, clip_val), bins=bins, density=True)
    hb, _ = np.histogram(b.clip(0, clip_val), bins=bins, density=True)
    bin_width = bins[1] - bins[0]
    return float(np.minimum(ha, hb).sum() * bin_width)


# ---------------------------------------------------------------------------
# ⑤ Pose vs dynamic fraction correlation
# ---------------------------------------------------------------------------

def diag_pose_dynamic_corr(
    pred: Dict[str, np.ndarray],
    gt_motion_masks: List[np.ndarray],
    gt_tum: np.ndarray,
    gt_timestamps: np.ndarray,
    out_dir: str,
    seq: str,
    stage: str,
):
    os.makedirs(out_dir, exist_ok=True)

    from eval.pose_metrics import extrinsics_w2c_to_tum
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation

    pred_tum, pred_ts = extrinsics_w2c_to_tum(pred["extrinsic"])

    S = pred["extrinsic"].shape[0]
    n_gt = len(gt_motion_masks)
    n_frames = min(S, n_gt)

    dynamic_fractions = []
    for i in range(n_frames):
        gt_mask = gt_motion_masks[i]
        if gt_mask is not None:
            dynamic_fractions.append(float(gt_mask.mean()))
        else:
            dynamic_fractions.append(0.0)

    # Per-frame pose error: compute RPE for each consecutive pair
    pred_traj = PoseTrajectory3D(
        positions_xyz=pred_tum[:n_frames, :3],
        orientations_quat_wxyz=pred_tum[:n_frames, 3:],
        timestamps=gt_timestamps[:n_frames].flatten(),
    )
    gt_traj = PoseTrajectory3D(
        positions_xyz=gt_tum[:n_frames, :3],
        orientations_quat_wxyz=gt_tum[:n_frames, 3:],
        timestamps=gt_timestamps[:n_frames].flatten(),
    )

    gt_traj_s, pred_traj_s = sync.associate_trajectories(gt_traj, pred_traj)

    ape_result = main_ape.ape(
        gt_traj_s, pred_traj_s,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True, correct_scale=True,
    )

    per_frame_errors = ape_result.np_arrays["error_array"]
    n_err = min(len(per_frame_errors), len(dynamic_fractions))
    dyn_frac = np.array(dynamic_fractions[:n_err])
    pose_err = per_frame_errors[:n_err]

    return {
        "seq": seq,
        "stage": stage,
        "dynamic_fractions": dyn_frac.tolist(),
        "pose_errors": pose_err.tolist(),
    }


def plot_pose_dynamic_scatter(
    all_data: Dict[str, List[Dict]],
    out_dir: str,
):
    """Scatter plot across all stages and sequences."""
    os.makedirs(out_dir, exist_ok=True)

    colors = {
        "vggt_base": "gray",
        "s0": "blue",
        "s1a": "green",
        "s1b": "orange",
        "s2": "red",
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    slopes = {}

    for stage, records in all_data.items():
        all_dyn = np.concatenate([np.array(r["dynamic_fractions"]) for r in records])
        all_err = np.concatenate([np.array(r["pose_errors"]) for r in records])
        c = colors.get(stage, "purple")
        ax.scatter(all_dyn, all_err, alpha=0.3, s=8, color=c, label=stage)

        if len(all_dyn) > 2 and all_dyn.std() > 1e-6:
            try:
                coeffs = np.polyfit(all_dyn, all_err, 1)
                slopes[stage] = float(coeffs[0])
                x_fit = np.linspace(all_dyn.min(), all_dyn.max(), 50)
                ax.plot(x_fit, np.polyval(coeffs, x_fit), color=c, linewidth=2, alpha=0.8)
            except np.linalg.LinAlgError:
                pass

    ax.set_xlabel("Dynamic pixel fraction (per frame)")
    ax.set_ylabel("Pose APE (translation)")
    ax.set_title("⑤ Pose error vs dynamic fraction — does decoupling help?")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "pose_vs_dynamic_scatter.png"), dpi=150)
    plt.close(fig)

    return slopes


# ---------------------------------------------------------------------------
# ⑥ Temporal consistency
# ---------------------------------------------------------------------------

def diag_temporal_consistency(
    pred: Dict[str, np.ndarray],
    gt_depths: List[np.ndarray],
    gt_flows: List[Optional[np.ndarray]],
    rgb_paths: Optional[List[str]],
    out_dir: str,
    seq: str,
    stage: str,
    max_depth: float = 80.0,
):
    os.makedirs(out_dir, exist_ok=True)

    depth_pred = pred["depth"]  # (S, H_model, W_model)
    S = depth_pred.shape[0]

    # Resize pred depths to GT resolution (skip if no rgb_paths, i.e. PO mode)
    pred_on_gt = []
    for i in range(S):
        fd = depth_pred[i]
        if fd.ndim == 3:
            fd = fd[..., 0]
        if rgb_paths is not None:
            meta = compute_preprocess_meta(rgb_paths[i])
            pred_on_gt.append(resize_pred_to_gt(fd, meta))
        else:
            pred_on_gt.append(fd)

    # OPW: temporal depth consistency via GT optical flow warp
    opw_errors = []
    for i in range(min(S - 1, len(gt_flows))):
        flow = gt_flows[i]
        if flow is None:
            continue
        d_cur = pred_on_gt[i]
        d_next = pred_on_gt[i + 1]
        h, w = d_cur.shape

        if flow.shape[:2] != (h, w):
            flow = cv2.resize(flow, (w, h))

        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        u_warp = u + flow[..., 0]
        v_warp = v + flow[..., 1]

        u_warp_int = np.clip(np.round(u_warp).astype(int), 0, w - 1)
        v_warp_int = np.clip(np.round(v_warp).astype(int), 0, h - 1)

        d_warped = d_next[v_warp_int, u_warp_int]

        valid = (d_cur > 0) & (d_warped > 0) & (d_cur < max_depth) & (d_warped < max_depth)
        if valid.sum() > 0:
            scale_cur = np.median(d_cur[valid])
            scale_warped = np.median(d_warped[valid])
            if scale_cur > 1e-6 and scale_warped > 1e-6:
                ratio = scale_warped / scale_cur
                d_warped_aligned = d_warped / ratio

                abs_diff = np.abs(d_cur - d_warped_aligned)
                opw = float(abs_diff[valid].mean() / d_cur[valid].mean())
                opw_errors.append(opw)

    # Temporal depth curve for a sample pixel
    sample_pixels = []
    h_model, w_model = depth_pred.shape[1], depth_pred.shape[2]
    pixel_coords = [
        (h_model // 2, w_model // 2),
        (h_model // 3, w_model // 3),
        (2 * h_model // 3, 2 * w_model // 3),
    ]
    for py, px in pixel_coords:
        curve = depth_pred[:, py, px]
        sample_pixels.append(curve)

    # Plot temporal depth curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for k, (py, px) in enumerate(pixel_coords):
        ax.plot(sample_pixels[k], "-o", markersize=2, label=f"px({py},{px})")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Depth")
    ax.set_title(f"Temporal depth curves — {seq} [{stage}]")
    ax.legend(fontsize=7)

    ax = axes[1]
    if opw_errors:
        ax.plot(opw_errors, "b-o", markersize=3)
        ax.set_xlabel("Frame pair index")
        ax.set_ylabel("OPW (normalized)")
        ax.set_title(f"OPW per frame pair — mean={np.mean(opw_errors):.4f}")
    else:
        ax.set_title("OPW — no valid flow pairs")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{seq}_temporal_{stage}.png"), dpi=150)
    plt.close(fig)

    return {
        "seq": seq,
        "stage": stage,
        "opw_mean": float(np.mean(opw_errors)) if opw_errors else None,
        "opw_std": float(np.std(opw_errors)) if opw_errors else None,
        "n_flow_pairs": len(opw_errors),
        "depth_curve_std": [float(c.std()) for c in sample_pixels],
    }


def plot_temporal_cross_stage(
    all_data: Dict[str, List[Dict]],
    out_dir: str,
):
    """Bar chart: OPW across stages."""
    os.makedirs(out_dir, exist_ok=True)

    stages = list(all_data.keys())
    seqs = set()
    for recs in all_data.values():
        for r in recs:
            seqs.add(r["seq"])
    seqs = sorted(seqs)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(seqs))
    width = 0.8 / max(len(stages), 1)

    for si, stage in enumerate(stages):
        opws = []
        for seq in seqs:
            match = [r for r in all_data[stage] if r["seq"] == seq]
            if match and match[0]["opw_mean"] is not None:
                opws.append(match[0]["opw_mean"])
            else:
                opws.append(0)
        ax.bar(x + si * width, opws, width, label=stage, alpha=0.8)

    ax.set_xticks(x + width * len(stages) / 2)
    ax.set_xticklabels(seqs, rotation=30, ha="right")
    ax.set_ylabel("OPW (lower = more temporally consistent)")
    ax.set_title("⑥ Temporal consistency (OPW) across stages")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "temporal_opw_comparison.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Dyn-VGGT diagnostic evaluation (①②⑤⑥)")
    ap.add_argument(
        "--ckpts", nargs="+", required=True,
        help="Checkpoint specs as path:variant (e.g. checkpoints/VGGT-1B.pt:vggt_base)",
    )
    ap.add_argument("--dataset", choices=["sintel", "po", "ta", "wo"], default="sintel",
                    help="Dataset: sintel (derived masks), po (native GT masks), ta (static), wo (static)")
    ap.add_argument("--sintel_root", default="/media/cvml-75/ssd2t1/data/sintel/training")
    ap.add_argument("--po_root", default="/media/cvml-75/ssd2t1/data/point_odyssey")
    ap.add_argument("--ta_root", default="/media/cvml-75/ssd2t1/data/tartanair")
    ap.add_argument("--wo_root", default="/media/cvml-75/ssd2t1/data/waymo_processed")
    ap.add_argument("--out_dir", default="logs/diag")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq_list", nargs="*", default=None)
    ap.add_argument("--n_clips", type=int, default=10, help="Number of PO clips to evaluate")
    ap.add_argument("--img_per_seq", type=int, default=6, help="Frames per PO clip")
    ap.add_argument("--motion_thr", type=float, default=2.0, help="Flow residual threshold for motion mask derivation")
    ap.add_argument("--max_depth", type=float, default=80.0)
    ap.add_argument("--oracle_mask", action="store_true",
                    help="Oracle mode: replace predicted m with GT motion mask for ② analysis. "
                         "If Δ separates under GT mask → problem is motion head; if not → architecture.")
    return ap.parse_args()


def main():
    args = parse_args()

    # Parse checkpoint specs
    ckpt_specs = []
    for spec in args.ckpts:
        if ":" in spec:
            path, variant = spec.rsplit(":", 1)
        else:
            path = spec
            variant = os.path.splitext(os.path.basename(path))[0]
        ckpt_specs.append((path, variant))

    if args.dataset == "sintel":
        main_sintel(args, ckpt_specs)
    else:
        main_generic(args, ckpt_specs)


def main_sintel(args, ckpt_specs):
    sequences = args.seq_list or DIAG_SEQUENCES

    # Pre-load GT data for all sequences
    print("Loading Sintel GT data...")
    gt_data: Dict[str, Dict[str, Any]] = {}
    for seq in sequences:
        rgb_paths = load_sintel_rgb_paths(args.sintel_root, seq)
        _, _, cam_dir = sintel_seq_paths(args.sintel_root, seq)
        gt_tum, gt_ts = load_sintel_gt_poses(cam_dir, rgb_paths)
        gt_depths = load_sintel_gt_depths(args.sintel_root, seq, rgb_paths)
        gt_flows = load_sintel_gt_flows(args.sintel_root, seq, rgb_paths)

        intrinsics, extrinsics = [], []
        for p in rgb_paths:
            from eval.sintel_io import matching_cam_path, sintel_cam_read
            cam_path = matching_cam_path(cam_dir, p)
            K, ext = sintel_cam_read(cam_path)
            intrinsics.append(K)
            extrinsics.append(ext)

        motion_masks = []
        for i in range(len(rgb_paths)):
            if i < len(gt_flows) and gt_flows[i] is not None and i + 1 < len(extrinsics):
                ego_flow = compute_ego_flow(gt_depths[i], intrinsics[i], extrinsics[i], extrinsics[i + 1])
                mm = derive_motion_mask(gt_flows[i], ego_flow, threshold=args.motion_thr)
                motion_masks.append(mm)
            else:
                motion_masks.append(None)

        gt_data[seq] = {
            "rgb_paths": rgb_paths,
            "gt_tum": gt_tum,
            "gt_ts": gt_ts,
            "gt_depths": gt_depths,
            "gt_flows": gt_flows,
            "motion_masks": motion_masks,
        }

    pose_dyn_all: Dict[str, List[Dict]] = {}
    temporal_all: Dict[str, List[Dict]] = {}
    summary: Dict[str, Dict[str, Any]] = {}

    for ckpt_path, variant in ckpt_specs:
        print(f"\n{'='*60}")
        print(f"Stage: {variant}  ckpt: {ckpt_path}")
        print(f"{'='*60}")

        is_base = variant == "vggt_base"
        model = load_dyn_vggt(
            ckpt_path, temporal=not is_base, motion=not is_base,
            flow=not is_base, device=args.device,
        )

        stage_dir = os.path.join(args.out_dir, variant)
        os.makedirs(stage_dir, exist_ok=True)
        stage_results: Dict[str, Any] = {"motion": [], "dual_field": [], "oracle_dual_field": [], "pose_dyn": [], "temporal": []}
        pose_dyn_all[variant] = []
        temporal_all[variant] = []

        for seq in tqdm(sequences, desc=f"[{variant}]"):
            gd = gt_data[seq]
            rgb_paths = gd["rgb_paths"]

            print(f"  Inferring {seq} ({len(rgb_paths)} frames)...")
            pred = infer_full(model, rgb_paths, device=args.device)

            if not is_base:
                print(f"    ① Motion calibration...")
                m1 = diag_motion_calibration(pred, gd["motion_masks"], rgb_paths, stage_dir, seq)
                stage_results["motion"].append(m1)

                print(f"    ② Dual-field decoupling...")
                m2 = diag_dual_field(pred, gd["motion_masks"], rgb_paths, stage_dir, seq)
                stage_results["dual_field"].append(m2)

                if args.oracle_mask:
                    print(f"    ②-oracle Dual-field with GT mask...")
                    m2o = diag_dual_field_oracle(pred, gd["motion_masks"], rgb_paths, stage_dir, seq)
                    stage_results["oracle_dual_field"].append(m2o)

            print(f"    ⑤ Pose-dynamic correlation...")
            try:
                m5 = diag_pose_dynamic_corr(
                    pred, gd["motion_masks"], gd["gt_tum"], gd["gt_ts"],
                    stage_dir, seq, variant,
                )
                stage_results["pose_dyn"].append(m5)
                pose_dyn_all[variant].append(m5)
            except Exception as e:
                print(f"      ⑤ skipped: {e}")

            print(f"    ⑥ Temporal consistency...")
            m6 = diag_temporal_consistency(
                pred, gd["gt_depths"], gd["gt_flows"], rgb_paths,
                stage_dir, seq, variant, max_depth=args.max_depth,
            )
            stage_results["temporal"].append(m6)
            temporal_all[variant].append(m6)

        summary[variant] = stage_results
        json_path = os.path.join(stage_dir, "diag_results.json")
        with open(json_path, "w") as f:
            json.dump(stage_results, f, indent=2, default=str)
        print(f"  Saved {json_path}")
        del model
        torch.cuda.empty_cache()

    _finalize(args, summary, pose_dyn_all, temporal_all)


def main_generic(args, ckpt_specs):
    from eval.pose_metrics import extrinsics_w2c_to_tum

    if args.dataset == "po":
        print(f"Loading PointOdyssey test set from {args.po_root}...")
        ds = load_po_dataset(args.po_root, img_per_seq=args.img_per_seq)
    elif args.dataset == "ta":
        print(f"Loading TartanAir from {args.ta_root}...")
        ds = load_ta_dataset(args.ta_root, min_num_images=args.img_per_seq)
    elif args.dataset == "wo":
        print(f"Loading Waymo from {args.wo_root}...")
        ds = load_wo_dataset(args.wo_root, min_num_images=args.img_per_seq)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    print(f"  {ds.sequence_list_len} sequences available")

    n_clips = min(args.n_clips, ds.sequence_list_len)
    torch.manual_seed(0)
    np.random.seed(0)

    # Pre-load clips
    print(f"Sampling {n_clips} clips × {args.img_per_seq} frames...")
    clips: List[Dict[str, Any]] = []
    for ci in range(n_clips):
        seq_idx = ci % ds.sequence_list_len
        seq_name = ds.sequence_list[seq_idx]
        batch = ds.get_data(seq_index=seq_idx, img_per_seq=args.img_per_seq, aspect_ratio=1.0)
        img_tensor, depths, motion_masks, extrinsics, intrinsics, rgb_images = collate_batch(batch)

        gt_tum, gt_ts = extrinsics_w2c_to_tum(extrinsics)
        gt_depths = [depths[i] for i in range(depths.shape[0])]
        gt_masks = [motion_masks[i] for i in range(motion_masks.shape[0])]
        dyn_frac = float(motion_masks.mean())

        clips.append({
            "seq_name": f"{seq_name}_c{ci}",
            "img_tensor": img_tensor,
            "gt_depths": gt_depths,
            "motion_masks": gt_masks,
            "gt_tum": gt_tum,
            "gt_ts": gt_ts,
            "rgb_images": rgb_images,
            "dyn_frac": dyn_frac,
        })
        print(f"  clip {ci}: {seq_name} ({args.img_per_seq}f, dyn={dyn_frac:.1%})")

    pose_dyn_all: Dict[str, List[Dict]] = {}
    temporal_all: Dict[str, List[Dict]] = {}
    summary: Dict[str, Dict[str, Any]] = {}

    for ckpt_path, variant in ckpt_specs:
        print(f"\n{'='*60}")
        print(f"Stage: {variant}  ckpt: {ckpt_path}")
        print(f"{'='*60}")

        is_base = variant == "vggt_base"
        model = load_dyn_vggt(
            ckpt_path, temporal=not is_base, motion=not is_base,
            flow=not is_base, device=args.device,
        )

        stage_dir = os.path.join(args.out_dir, variant)
        os.makedirs(stage_dir, exist_ok=True)
        stage_results: Dict[str, Any] = {"motion": [], "dual_field": [], "oracle_dual_field": [], "pose_dyn": [], "temporal": []}
        pose_dyn_all[variant] = []
        temporal_all[variant] = []

        for clip in tqdm(clips, desc=f"[{variant}]"):
            seq = clip["seq_name"]
            print(f"  Inferring {seq}...")
            pred = infer_from_tensor(model, clip["img_tensor"], device=args.device)

            if not is_base:
                print(f"    ① Motion calibration...")
                m1 = diag_motion_calibration(
                    pred, clip["motion_masks"], None, stage_dir, seq,
                    rgb_images=clip["rgb_images"],
                )
                stage_results["motion"].append(m1)

                print(f"    ② Dual-field decoupling...")
                m2 = diag_dual_field(
                    pred, clip["motion_masks"], None, stage_dir, seq,
                    rgb_images=clip["rgb_images"],
                )
                stage_results["dual_field"].append(m2)

                if args.oracle_mask:
                    print(f"    ②-oracle Dual-field with GT mask...")
                    m2o = diag_dual_field_oracle(
                        pred, clip["motion_masks"], None, stage_dir, seq,
                        rgb_images=clip["rgb_images"],
                    )
                    stage_results["oracle_dual_field"].append(m2o)

            print(f"    ⑤ Pose-dynamic correlation...")
            try:
                m5 = diag_pose_dynamic_corr(
                    pred, clip["motion_masks"], clip["gt_tum"], clip["gt_ts"],
                    stage_dir, seq, variant,
                )
                stage_results["pose_dyn"].append(m5)
                pose_dyn_all[variant].append(m5)
            except Exception as e:
                print(f"      ⑤ skipped: {e}")

            print(f"    ⑥ Temporal consistency (depth curves only)...")
            m6 = diag_temporal_consistency(
                pred, clip["gt_depths"], [], None,
                stage_dir, seq, variant, max_depth=args.max_depth,
            )
            stage_results["temporal"].append(m6)
            temporal_all[variant].append(m6)

        summary[variant] = stage_results
        json_path = os.path.join(stage_dir, "diag_results.json")
        with open(json_path, "w") as f:
            json.dump(stage_results, f, indent=2, default=str)
        print(f"  Saved {json_path}")
        del model
        torch.cuda.empty_cache()

    _finalize(args, summary, pose_dyn_all, temporal_all)


def _finalize(args, summary, pose_dyn_all, temporal_all):
    cross_dir = os.path.join(args.out_dir, "cross_stage")
    os.makedirs(cross_dir, exist_ok=True)

    if pose_dyn_all:
        slopes = plot_pose_dynamic_scatter(pose_dyn_all, cross_dir)
        print(f"\n⑤ Pose-dynamic slopes: {slopes}")

    if temporal_all:
        plot_temporal_cross_stage(temporal_all, cross_dir)
        opw_summary = {}
        for stage, recs in temporal_all.items():
            valid = [r["opw_mean"] for r in recs if r["opw_mean"] is not None]
            opw_summary[stage] = float(np.mean(valid)) if valid else None
        print(f"\n⑥ OPW means: {opw_summary}")

    final_path = os.path.join(args.out_dir, "diag_summary.json")
    with open(final_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nAll diagnostics saved to {args.out_dir}/")
    print_final_summary(summary)


def print_final_summary(summary: Dict[str, Dict[str, Any]]):
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    for stage, data in summary.items():
        print(f"\n--- {stage} ---")

        # ① Motion
        if data.get("motion"):
            valid = [m for m in data["motion"] if m and "ap" in m]
            if valid:
                mean_ap = np.mean([m["ap"] for m in valid])
                mean_auc = np.mean([m["roc_auc"] for m in valid])
                print(f"  ① Motion: AP={mean_ap:.3f}  ROC-AUC={mean_auc:.3f}")

        # ② Dual field
        if data.get("dual_field"):
            valid = [m for m in data["dual_field"] if m and m.get("separation_ratio") is not None]
            if valid:
                mean_sep = np.mean([m["separation_ratio"] for m in valid])
                mean_stat = np.mean([m["static_delta_median"] for m in valid])
                mean_dyn = np.mean([m["dynamic_delta_median"] for m in valid])
                print(f"  ② Dual-field: static_Δ_med={mean_stat:.4f}  dynamic_Δ_med={mean_dyn:.4f}  ratio={mean_sep:.1f}x")

        # ②-oracle
        if data.get("oracle_dual_field"):
            valid = [m for m in data["oracle_dual_field"] if m and m.get("separation_ratio") is not None]
            if valid:
                mean_sep = np.mean([m["separation_ratio"] for m in valid])
                mean_stat = np.mean([m["static_delta_median"] for m in valid])
                mean_dyn = np.mean([m["dynamic_delta_median"] for m in valid])
                overlap = np.mean([m.get("histogram_overlap", 0) for m in valid])
                print(f"  ②-oracle: static_Δ_med={mean_stat:.4f}  dynamic_Δ_med={mean_dyn:.4f}  ratio={mean_sep:.1f}x  overlap={overlap:.1%}")
                if mean_sep > 2.0 and overlap < 0.6:
                    print(f"    → VERDICT: Architecture OK — problem is motion head")
                elif mean_sep > 1.3:
                    print(f"    → VERDICT: Weak separation — motion head + mild Δ prior may help")
                else:
                    print(f"    → VERDICT: Architecture-level identifiability issue (Δ prior / X^can anchor needed)")

        # ⑤ Pose-dynamic
        if data.get("pose_dyn"):
            all_dyn = np.concatenate([np.array(r["dynamic_fractions"]) for r in data["pose_dyn"]])
            all_err = np.concatenate([np.array(r["pose_errors"]) for r in data["pose_dyn"]])
            if len(all_dyn) > 2 and all_dyn.std() > 1e-6:
                try:
                    slope = np.polyfit(all_dyn, all_err, 1)[0]
                    print(f"  ⑤ Pose-dyn slope={slope:.4f} (lower = better decoupling)")
                except np.linalg.LinAlgError:
                    print(f"  ⑤ Pose-dyn slope: fit failed (insufficient variance)")

        # ⑥ Temporal
        if data.get("temporal"):
            valid_opw = [m["opw_mean"] for m in data["temporal"] if m.get("opw_mean") is not None]
            if valid_opw:
                print(f"  ⑥ OPW mean={np.mean(valid_opw):.4f} (lower = more consistent)")


if __name__ == "__main__":
    main()
