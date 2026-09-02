"""Sintel geometric-residual dynamic mask (m_geo) derivation + on-disk cache.

m_geo = |GT optical flow - ego flow| > threshold, where ego flow is reprojected
from GT depth + GT relative pose. Precompute masks with
``data/preprocess/sintel_geo_dynmask.py`` (writes to ``<sintel_root>/mask/``);
eval/vis then load the cache instead of recomputing every run.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np

TAG_FLOAT = 202021.25

DIAG_SEQUENCES = ["sleeping_2", "temple_2", "market_5", "cave_2", "temple_3"]

# On-disk cache layout: <sintel_root>/mask/<seq>/<frame_stem>.png (uint8 0/255)
# + <sintel_root>/mask/meta.json recording the threshold the cache was built at.
MASK_CACHE_SUBDIR = "mask"
MASK_CACHE_VERSION = 1


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

    pts_cam = np.stack([u, v, ones], axis=-1)
    pts_cam = (K_inv @ pts_cam[..., None])[..., 0]
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

    return np.stack([u_next - u, v_next - v], axis=-1).astype(np.float32)


def derive_motion_mask(
    gt_flow: np.ndarray,
    ego_flow: np.ndarray,
    threshold: float = 2.0,
    rel_threshold: float = 0.0,
) -> np.ndarray:
    """m_geo = |gt_flow - ego_flow| > bar.

    ``bar = max(threshold, rel_threshold * |gt_flow|)``. The relative term only
    bites where the camera flow is large (fast-motion frames), where ego-flow
    reprojection error otherwise flags the whole frame; slow frames stay governed
    by the absolute floor and are essentially unchanged. rel_threshold=0 recovers
    the plain absolute-threshold behaviour.
    """
    residual = np.linalg.norm(gt_flow - ego_flow, axis=-1)
    if rel_threshold > 0.0:
        bar = np.maximum(threshold, rel_threshold * np.linalg.norm(gt_flow, axis=-1))
    else:
        bar = threshold
    return (residual > bar).astype(np.float32)


# --------------------------------------------------------------------------- #
# On-disk cache (see module docstring)
# --------------------------------------------------------------------------- #

def mask_cache_dir(sintel_root: str) -> str:
    return os.path.join(sintel_root, MASK_CACHE_SUBDIR)


def mask_cache_meta_path(sintel_root: str) -> str:
    return os.path.join(mask_cache_dir(sintel_root), "meta.json")


def read_mask_cache_meta(sintel_root: str) -> Optional[dict]:
    p = mask_cache_meta_path(sintel_root)
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _frame_stem(rgb_path: str) -> str:
    return os.path.splitext(os.path.basename(rgb_path))[0]


def cached_mask_path(sintel_root: str, seq: str, rgb_path: str) -> str:
    return os.path.join(mask_cache_dir(sintel_root), seq, f"{_frame_stem(rgb_path)}.png")


def load_masks(
    sintel_root: str,
    seq: str,
    rgb_paths: List[str],
    threshold: float = 2.0,
    rel_threshold: float = 0.0,
    gt_flows=None,
    gt_depths=None,
    intrinsics=None,
    extrinsics=None,
) -> List[Optional[np.ndarray]]:
    """Return per-frame m_geo aligned to ``rgb_paths`` (None where no flow/next pose).

    Prefers the on-disk cache under ``<sintel_root>/mask/`` when it exists and was
    built at the requested ``threshold``; otherwise computes live from the GT
    fields (which must then be supplied). Values are float32 {0,1}.
    """
    import cv2

    meta = read_mask_cache_meta(sintel_root)
    if meta is not None and float(meta.get("threshold", -1)) == float(threshold):
        masks: List[Optional[np.ndarray]] = []
        for p in rgb_paths:
            mp = cached_mask_path(sintel_root, seq, p)
            if os.path.isfile(mp):
                m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                masks.append((m.astype(np.float32) / 255.0) if m is not None else None)
            else:
                masks.append(None)
        return masks

    # Cache miss / threshold mismatch -> compute live.
    if gt_flows is None or gt_depths is None or intrinsics is None or extrinsics is None:
        raise ValueError(
            f"m_geo cache miss for {seq} at threshold={threshold} and GT fields not "
            "supplied for live compute; run data/preprocess/sintel_geo_dynmask.py first."
        )
    masks = []
    for i in range(len(rgb_paths)):
        if gt_flows[i] is None or i + 1 >= len(extrinsics):
            masks.append(None)
            continue
        ego = compute_ego_flow(gt_depths[i], intrinsics[i], extrinsics[i], extrinsics[i + 1])
        masks.append(derive_motion_mask(gt_flows[i], ego, threshold=threshold, rel_threshold=rel_threshold))
    return masks


def sintel_masks_and_fraction(
    sintel_root: str, seq: str, rgb_paths: List[str], motion_thr: float
) -> Tuple[List[Optional[np.ndarray]], float]:
    """``load_masks`` plus the mean dynamic fraction, reading GT only on a cache miss."""
    from pipeline.data.sintel_io import (
        load_sintel_gt_depths,
        matching_cam_path,
        sintel_cam_read,
        sintel_seq_paths,
    )

    meta = read_mask_cache_meta(sintel_root)
    if meta is not None and float(meta.get("threshold", -1)) == float(motion_thr):
        # cache hit: skip the (slow) GT flow/depth/cam reads entirely
        masks = load_masks(sintel_root, seq, rgb_paths, motion_thr)
    else:
        _, _, cam_dir = sintel_seq_paths(sintel_root, seq)
        gt_flows = load_sintel_gt_flows(sintel_root, seq, rgb_paths)
        gt_depths = load_sintel_gt_depths(sintel_root, seq, rgb_paths)
        intrinsics, extrinsics = [], []
        for p in rgb_paths:
            K, ext = sintel_cam_read(matching_cam_path(cam_dir, p))
            intrinsics.append(K)
            extrinsics.append(ext)
        masks = load_masks(
            sintel_root, seq, rgb_paths, motion_thr,
            gt_flows=gt_flows, gt_depths=gt_depths,
            intrinsics=intrinsics, extrinsics=extrinsics,
        )

    fracs = [float(m.mean()) for m in masks if m is not None]
    mean_frac = float(np.mean(fracs)) if fracs else float("nan")
    return masks, mean_frac
