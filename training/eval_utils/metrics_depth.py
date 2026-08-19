"""Depth evaluation metrics.

Two alignment regimes live here on purpose:

* ``depth_evaluation(..., align_with_lad2=False)`` -- cheap **per-frame median**
  scaling (1 DOF), the default. Currently unused by any in-repo caller (the trainer's
  channel-A validation is loss-only, see trainer.val_epoch); kept as a cheap alternative
  to the lad2 fit below for future callers that need fast, single-frame alignment.
* ``eval_sequence_depth`` -- the **MonST3R Sintel protocol**: stack every frame of a
  sequence, fit ONE global **scale+shift** (``align_with_lad2``, 2 DOF) over the whole
  sequence, pool the metrics over all valid pixels, and average sequences weighted by
  valid-pixel count. ``absolute_value_scaling2`` and the lad2 branch of
  ``depth_evaluation`` are ported verbatim from
  reference/monst3r/dust3r/depth_eval.py so the benchmark numbers use MonST3R's method.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

METRIC_KEYS = ["abs_rel", "sq_rel", "rmse", "log_rmse", "delta_1", "delta_2", "delta_3"]


def _np_median(x: torch.Tensor) -> torch.Tensor:
    """``np.median`` semantics: on an even-sized input average the two central values.
    ``torch.median`` returns the lower one instead, which shifts the median scale factor
    by half a pixel-pair and makes our numbers disagree with the reference implementations
    (AF-SfMLearner scales with ``np.median``). Used by the 1-DOF median branch only."""
    return torch.quantile(x.flatten().float(), 0.5)


def absolute_value_scaling2(
    predicted_depth: torch.Tensor,
    ground_truth_depth: torch.Tensor,
    s_init: float = 1.0,
    t_init: float = 0.0,
    lr: float = 1e-4,
    max_iters: int = 1000,
    tol: float = 1e-6,
):
    """Fit scale ``s`` and shift ``t`` minimising L1(s*pred + t - gt) via Adam.

    Ported from MonST3R (dust3r/depth_eval.py). Used for the lad2 alignment.

    Runs its own local optimization loop, so it must work regardless of an outer
    ``torch.no_grad()`` context (e.g. trainer.run_pose_eval wraps the whole full-sequence
    Sintel eval in no_grad to save memory on the forward passes) -- force-enable autograd
    here so ``loss.backward()`` always has a graph to walk.
    """
    with torch.enable_grad():
        s = torch.tensor([s_init], requires_grad=True, device=predicted_depth.device, dtype=predicted_depth.dtype)
        t = torch.tensor([t_init], requires_grad=True, device=predicted_depth.device, dtype=predicted_depth.dtype)
        optimizer = torch.optim.Adam([s, t], lr=lr)
        prev_loss = None
        for _ in range(max_iters):
            optimizer.zero_grad()
            predicted_aligned = s * predicted_depth + t
            loss = torch.sum(torch.abs(predicted_aligned - ground_truth_depth))
            loss.backward()
            optimizer.step()
            if prev_loss is not None and abs(prev_loss - loss.item()) < tol:
                break
            prev_loss = loss.item()
    return s.detach().item(), t.detach().item()


def depth_evaluation(
    predicted_depth: np.ndarray,
    ground_truth_depth: np.ndarray,
    max_depth: float = 80.0,
    min_depth: float = 0.0,
    custom_mask: Optional[np.ndarray] = None,
    align_with_lad2: bool = False,
    post_clip_min: Optional[float] = None,
    post_clip_max: Optional[float] = None,
    lr: float = 1e-4,
    max_iters: int = 1000,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """Scale (or scale+shift) align ``predicted_depth`` to GT, then score.

    ``align_with_lad2=False`` -> median scale (per-call, 1 DOF); the trainer default.
    ``align_with_lad2=True``  -> optimise scale+shift over all valid pixels (MonST3R).
    Inputs may be single frames (H,W) or whole-sequence stacks (T,H,W); alignment and
    metrics are computed jointly over every valid pixel of the input.
    """
    if device is None:
        device = "cuda" if (align_with_lad2 and torch.cuda.is_available()) else "cpu"

    gt_orig = torch.from_numpy(ground_truth_depth.astype(np.float32)).to(device)
    pred_orig = torch.from_numpy(predicted_depth.astype(np.float32)).to(device)

    mask = (gt_orig > min_depth) & (gt_orig < max_depth)
    if custom_mask is not None:
        mask = mask & torch.from_numpy(custom_mask.astype(bool)).to(device)

    pred = pred_orig[mask]
    gt = gt_orig[mask]
    if pred.numel() == 0:
        return {k: 0.0 for k in METRIC_KEYS} | {"valid_pixels": 0}

    if align_with_lad2:
        s_init = (torch.median(gt) / torch.median(pred)).item()
        s, t = absolute_value_scaling2(pred, gt, s_init=s_init, t_init=0.0, lr=lr, max_iters=max_iters)
        pred = s * pred + t
    else:
        scale = _np_median(gt) / _np_median(pred)
        pred = pred * scale

    if post_clip_min is not None:
        pred = torch.clamp(pred, min=post_clip_min)
    if post_clip_max is not None:
        pred = torch.clamp(pred, max=post_clip_max)

    abs_rel = torch.mean(torch.abs(pred - gt) / gt).item()
    sq_rel = torch.mean(((pred - gt) ** 2) / gt).item()
    rmse = torch.sqrt(torch.mean((pred - gt) ** 2)).item()
    pred_c = torch.clamp(pred, min=1e-5)
    log_rmse = torch.sqrt(torch.mean((torch.log(pred_c) - torch.log(gt)) ** 2)).item()
    ratio = torch.maximum(pred / gt, gt / pred)
    delta_1 = torch.mean((ratio < 1.25).float()).item()
    delta_2 = torch.mean((ratio < 1.25 ** 2).float()).item()
    delta_3 = torch.mean((ratio < 1.25 ** 3).float()).item()

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "log_rmse": log_rmse,
        "delta_1": delta_1,
        "delta_2": delta_2,
        "delta_3": delta_3,
        "valid_pixels": int(mask.sum().item()),
    }


def eval_sequence_depth(
    pred_depths: List[np.ndarray],
    gt_depths: List[np.ndarray],
    max_depth: float = 70.0,
    align_with_lad2: bool = True,
    post_clip_max: Optional[float] = 70.0,
    device: Optional[str] = None,
    per_frame: bool = False,
    min_depth: float = 0.0,
    post_clip_min: Optional[float] = None,
) -> Dict[str, float]:
    """MonST3R Sintel depth protocol: one global scale+shift over the whole sequence,
    metrics pooled over all valid pixels. Defaults (max_depth=70, lad2, clip=70) match
    MonST3R's Sintel command. ``pred_depths[i]`` must already be at ``gt_depths[i]``'s
    resolution (eval_sintel resizes each frame before calling).

    ``per_frame=True`` switches to the AF-SfMLearner SCARED protocol instead: align and
    score every frame on its own, then take the UNWEIGHTED mean over frames. Frames with
    no valid GT pixel are dropped rather than counted as zero."""
    if per_frame:
        per = [
            depth_evaluation(
                p, g, max_depth=max_depth, min_depth=min_depth,
                align_with_lad2=False, post_clip_min=post_clip_min,
                post_clip_max=post_clip_max, device=device,
            )
            for p, g in zip(pred_depths, gt_depths)
        ]
        per = [m for m in per if m["valid_pixels"] > 0]
        if not per:
            return {k: 0.0 for k in METRIC_KEYS} | {"valid_pixels": 0, "num_frames": 0}
        out = {k: float(np.mean([m[k] for m in per])) for k in METRIC_KEYS}
        out["valid_pixels"] = int(sum(m["valid_pixels"] for m in per))
        out["num_frames"] = len(per)
        return out

    pred = np.stack(pred_depths, axis=0)
    gt = np.stack(gt_depths, axis=0)
    out = depth_evaluation(
        pred, gt, max_depth=max_depth, min_depth=min_depth,
        align_with_lad2=align_with_lad2, post_clip_min=post_clip_min,
        post_clip_max=post_clip_max, device=device,
    )
    out["num_frames"] = len(pred_depths)
    return out


def average_depth_results(per_seq: Dict[str, Dict[str, float]],
                          weight_key: str = "valid_pixels") -> Dict[str, float]:
    """Combine per-sequence results into one number.

    ``weight_key="valid_pixels"`` -- MonST3R aggregation, the default: a sequence counts in
    proportion to how much valid GT it has.
    ``weight_key="num_frames"`` -- the AF-SfMLearner/SCARED aggregation. AF never groups by
    sequence at all: it pools all 550 test frames into one list and takes ``errors.mean(0)``.
    Weighting each sequence's per-frame mean by its frame count is algebraically identical to
    that pooled mean, so this reproduces it without restructuring the caller. Using the
    valid-pixel weights instead would silently shift abs_rel away from the published tables,
    because SCARED's valid fraction swings from 25% to 90% across keyframes.
    """
    valid = [v for v in per_seq.values() if v.get(weight_key, 0) > 0]
    if not valid:
        return {k: 0.0 for k in METRIC_KEYS}
    weights = np.array([v[weight_key] for v in valid], dtype=np.float64)
    return {k: float(np.average([v[k] for v in valid], weights=weights)) for k in METRIC_KEYS}
