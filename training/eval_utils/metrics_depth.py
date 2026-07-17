"""Depth evaluation metrics with median scale alignment."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


def depth_evaluation(
    predicted_depth: np.ndarray,
    ground_truth_depth: np.ndarray,
    max_depth: float = 80.0,
    custom_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Evaluate depth with median scaling (MonST3R default)."""
    pred = torch.from_numpy(predicted_depth.astype(np.float32))
    gt = torch.from_numpy(ground_truth_depth.astype(np.float32))
    if custom_mask is not None:
        custom_mask_t = torch.from_numpy(custom_mask.astype(bool))
    else:
        custom_mask_t = None

    mask = (gt > 0) & (gt < max_depth)
    if custom_mask_t is not None:
        mask = mask & custom_mask_t

    pred_flat = pred[mask]
    gt_flat = gt[mask]
    if pred_flat.numel() == 0:
        return {
            "abs_rel": 0.0,
            "sq_rel": 0.0,
            "rmse": 0.0,
            "log_rmse": 0.0,
            "delta_1": 0.0,
            "delta_2": 0.0,
            "delta_3": 0.0,
            "valid_pixels": 0,
        }

    scale = torch.median(gt_flat) / torch.median(pred_flat)
    pred_aligned = pred_flat * scale

    abs_rel = torch.mean(torch.abs(pred_aligned - gt_flat) / gt_flat).item()
    sq_rel = torch.mean(((pred_aligned - gt_flat) ** 2) / gt_flat).item()
    rmse = torch.sqrt(torch.mean((pred_aligned - gt_flat) ** 2)).item()
    pred_clipped = torch.clamp(pred_aligned, min=1e-5)
    log_rmse = torch.sqrt(torch.mean((torch.log(pred_clipped) - torch.log(gt_flat)) ** 2)).item()
    ratio = torch.maximum(pred_aligned / gt_flat, gt_flat / pred_aligned)
    delta_1 = torch.mean((ratio < 1.25).float()).item()
    delta_2 = torch.mean((ratio < 1.25**2).float()).item()
    delta_3 = torch.mean((ratio < 1.25**3).float()).item()

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "log_rmse": log_rmse,
        "delta_1": delta_1,
        "delta_2": delta_2,
        "delta_3": delta_3,
        "valid_pixels": int(pred_flat.numel()),
    }


def eval_sequence_depth(
    pred_depths: List[np.ndarray],
    gt_depths: List[np.ndarray],
    max_depth: float = 80.0,
) -> Dict[str, float]:
    keys = ["abs_rel", "sq_rel", "rmse", "log_rmse", "delta_1", "delta_2", "delta_3"]
    acc = {k: 0.0 for k in keys}
    n = 0
    total_valid = 0
    for pred, gt in zip(pred_depths, gt_depths):
        m = depth_evaluation(pred, gt, max_depth=max_depth)
        if m["valid_pixels"] == 0:
            continue
        for k in keys:
            acc[k] += m[k]
        total_valid += m["valid_pixels"]
        n += 1
    if n == 0:
        return {k: 0.0 for k in keys} | {"valid_pixels": 0, "num_frames": 0}
    out = {k: acc[k] / n for k in keys}
    out["valid_pixels"] = total_valid
    out["num_frames"] = n
    return out


def average_depth_results(per_seq: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    keys = ["abs_rel", "sq_rel", "rmse", "log_rmse", "delta_1", "delta_2", "delta_3"]
    if not per_seq:
        return {k: 0.0 for k in keys}
    out = {}
    for k in keys:
        out[k] = float(np.mean([v[k] for v in per_seq.values() if v.get("num_frames", 0) > 0]))
    return out
