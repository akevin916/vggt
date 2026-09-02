"""Point-cloud accuracy / completeness between a predicted and a GT depth-derived cloud.

WHY THIS EXISTS SEPARATELY FROM metrics_depth. Per-pixel depth error scores each pixel against
the GT at the same pixel, so it can only ever say "this pixel's depth was off". It cannot say
that a region produced points that correspond to no surface at all, nor that a region produced
no points where a surface exists. Glare does both, and they are different failures:

    accuracy     -- for each PREDICTED point, distance to the nearest GT point.
                    "Is what the model invented actually there?"  Hallucinated geometry
                    (a specular blob read as a bulge) shows up here and nowhere else.
    completeness -- for each GT point, distance to the nearest PREDICTED point.
                    "Did the model reconstruct what is there?"  A washed-out surface the model
                    declined to place shows up here.

Both are reported as mean and median. The median is the honest headline: these clouds have long
tails from a handful of far-flung outliers, and a mean lets ten stray points outvote a hundred
thousand good ones.

SCALE. Everything is in the GT's units (mm for SCARED), which only holds after the predicted
cloud has been Sim3-aligned to GT -- VGGT's output is scale-free. Alignment is the caller's job
(vggt_infer.umeyama_sim3 on camera centres); this module assumes the two clouds already live in
one frame and will happily return meaningless numbers if they do not.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree


def unproject(depth: np.ndarray, K: np.ndarray, c2w: Optional[np.ndarray] = None,
              valid: Optional[np.ndarray] = None, stride: int = 1):
    """Back-project a depth map to 3D. Returns (points [N,3], pixel indices [N,2] as (y,x)).

    Pixel indices come back alongside the points so a caller can carry per-pixel labels (which
    stratum a pixel belongs to) through to the cloud without re-deriving them.
    """
    h, w = depth.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    d = depth[::stride, ::stride]
    m = np.isfinite(d) & (d > 0)
    if valid is not None:
        m &= valid[::stride, ::stride].astype(bool)
    ys, xs, d = ys[m], xs[m], d[m]
    if d.size == 0:
        return np.zeros((0, 3), np.float64), np.zeros((0, 2), np.int64)

    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    cam = np.stack([(xs - cx) / fx * d, (ys - cy) / fy * d, d], axis=1)
    if c2w is not None:
        cam = cam @ c2w[:3, :3].T + c2w[:3, 3]
    return cam.astype(np.float64), np.stack([ys, xs], axis=1)


def voxel_downsample(points: np.ndarray, voxel: float, labels: Optional[np.ndarray] = None):
    """Keep one point per ``voxel``-sized cell.

    Without this the two clouds are dominated by wherever the camera dwelt longest: a surface
    seen in 40 frames contributes 40x the points of one seen once, and both metrics then quietly
    become "how well did we do on the most-visited surface". Downsampling makes them per-surface.
    """
    if len(points) == 0 or voxel <= 0:
        return (points, labels) if labels is not None else points
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx.sort()
    return (points[idx], labels[idx]) if labels is not None else points[idx]


def cloud_accuracy_completeness(pred: np.ndarray, gt: np.ndarray,
                                trunc: Optional[float] = None,
                                pred_labels: Optional[np.ndarray] = None,
                                label_names: Optional[Dict[int, str]] = None) -> Dict[str, float]:
    """Nearest-neighbour distances both ways.

    ``trunc`` clips each distance at a ceiling (mm). Untruncated means are unstable here: one
    predicted point flung far behind the tissue can move the mean by more than a systematic
    half-millimetre bias across the whole surface, which is the opposite of what we want to read.

    ``pred_labels`` (int per predicted point, e.g. which stratum the source pixel was in) adds
    ``acc_mean_<name>`` / ``acc_median_<name>`` / ``n_<name>`` columns, which is what answers
    "are the points grown out of the glare worse than the rest?".
    """
    out: Dict[str, float] = {"n_pred": int(len(pred)), "n_gt": int(len(gt))}
    if len(pred) == 0 or len(gt) == 0:
        return out

    d_pred = cKDTree(gt).query(pred, k=1)[0]      # accuracy
    d_gt = cKDTree(pred).query(gt, k=1)[0]        # completeness
    if trunc is not None:
        d_pred = np.minimum(d_pred, trunc)
        d_gt = np.minimum(d_gt, trunc)

    out.update(
        acc_mean=float(d_pred.mean()), acc_median=float(np.median(d_pred)),
        comp_mean=float(d_gt.mean()), comp_median=float(np.median(d_gt)),
        # Chamfer here is the symmetric mean of the two, i.e. one number when a table has room
        # for only one -- it is not a substitute for reading both.
        chamfer=float(0.5 * (d_pred.mean() + d_gt.mean())),
    )
    # Fraction of predicted points within a tight tolerance: less outlier-sensitive than any mean.
    for tol in (1.0, 2.0, 5.0):
        out[f"acc_frac_{tol:g}mm"] = float((d_pred <= tol).mean())
        out[f"comp_frac_{tol:g}mm"] = float((d_gt <= tol).mean())

    if pred_labels is not None and label_names:
        for val, name in label_names.items():
            sel = pred_labels == val
            if sel.sum() == 0:
                continue
            out[f"acc_mean_{name}"] = float(d_pred[sel].mean())
            out[f"acc_median_{name}"] = float(np.median(d_pred[sel]))
            out[f"n_{name}"] = int(sel.sum())
    return out
