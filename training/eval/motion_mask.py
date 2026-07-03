"""Sintel flow-residual dynamic mask derivation (shared by gate eval and RAFT precompute)."""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

TAG_FLOAT = 202021.25

DIAG_SEQUENCES = ["sleeping_2", "temple_2", "market_5", "cave_2", "temple_3"]


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
) -> np.ndarray:
    residual = np.linalg.norm(gt_flow - ego_flow, axis=-1)
    return (residual > threshold).astype(np.float32)
