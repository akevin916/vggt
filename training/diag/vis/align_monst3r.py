#!/usr/bin/env python3
"""Bring MonST3R's exported point clouds into the shared GT frame, next to base/inst.

MonST3R dumps, per Sintel sequence, a fused ``pointcloud.ply`` plus ``pred_traj.txt`` (TUM
camera-to-world) -- both in MonST3R's own world frame. This aligns each to Sintel GT with the
*same* recipe pointcloud_gt.py uses for the VGGT clouds: a camera-centre Umeyama (Sim3) seed,
then a scaled ICP on the points so only the geometry difference survives. Output lands in the
comparison layout ``outputs/point_cloud/<scene>/monst3r.ply`` (gt.ply is left untouched -- it is
written at full resolution by the base/inst run).
"""

from __future__ import annotations

import os
import sys

_TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO_DIR = os.path.dirname(_TRAINING_DIR)
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), _TRAINING_DIR, _REPO_DIR]

import argparse

import numpy as np
import trimesh

from data.sintel_io import (
    SINTEL_EVAL_SEQUENCES,
    load_sintel_rgb_paths,
    matching_cam_path,
    resolve_sintel_root,
    sintel_cam_read,
    sintel_seq_paths,
)
from scipy.spatial import cKDTree

from eval_utils.paths import OUTPUTS_DIR
from eval_utils.ply_io import write_ply
from pointcloud_gt import apply_sim3, camera_centres, gt_cloud, umeyama_sim3

DEFAULT_MONST3R_DIR = os.path.join(_REPO_DIR, "reference", "monst3r", "results", "sintel_pose")


def parse_args():
    ap = argparse.ArgumentParser(description="Align MonST3R point clouds to Sintel GT")
    ap.add_argument("--seqs", nargs="*", default=None, help="Default: all SINTEL_EVAL_SEQUENCES")
    ap.add_argument("--monst3r_dir", default=DEFAULT_MONST3R_DIR)
    ap.add_argument("--ply_name", default="pointcloud.ply", help="pointcloud.ply (full) or pointcloud_static.ply")
    ap.add_argument("--sintel_root", default=None)
    ap.add_argument("--max_depth", type=float, default=80.0, help="GT cloud depth cap (sky)")
    ap.add_argument("--gt_stride", type=int, default=3, help="GT subsampling for the KD-tree reference")
    ap.add_argument(
        "--gt_dist_thresh",
        type=float,
        default=0.1,
        help="Drop monst3r points whose nearest GT point is farther than this x the GT diagonal "
        "(removes MonST3R's far background junk without rescaling the cloud).",
    )
    return ap.parse_args()


def load_monst3r_ply(path: str):
    """Robustly read a MonST3R ply -> (points[N,3] float64, colors[N,3] in [0,1])."""
    obj = trimesh.load(path, process=False)
    pts = np.asarray(obj.vertices, dtype=np.float64)
    cols = None
    if getattr(obj, "colors", None) is not None and len(obj.colors) == len(pts):
        cols = np.asarray(obj.colors)[:, :3] / 255.0
    elif hasattr(obj, "visual") and getattr(obj.visual, "vertex_colors", None) is not None:
        vc = np.asarray(obj.visual.vertex_colors)
        if len(vc) == len(pts):
            cols = vc[:, :3] / 255.0
    if cols is None:
        cols = np.full((len(pts), 3), 0.6)
    return pts, cols


def gt_camera_centres(sintel_root: str, seq: str, rgb_paths):
    _, _, cam_dir = sintel_seq_paths(sintel_root, seq)
    w2c = np.stack([sintel_cam_read(matching_cam_path(cam_dir, p))[1] for p in rgb_paths])
    return camera_centres(w2c.astype(np.float64))


def process(seq: str, args, sintel_root: str) -> bool:
    seq_dir = os.path.join(args.monst3r_dir, seq)
    ply_path = os.path.join(seq_dir, args.ply_name)
    traj_path = os.path.join(seq_dir, "pred_traj.txt")
    if not (os.path.exists(ply_path) and os.path.exists(traj_path)):
        print(f"[{seq}] missing {args.ply_name} or pred_traj.txt -- skipped")
        return False

    m_pts, m_cols = load_monst3r_ply(ply_path)
    traj = np.loadtxt(traj_path)               # (N, 8) TUM: ts tx ty tz qx qy qz qw
    m_centres = traj[:, 1:4].astype(np.float64)  # camera-to-world translation = camera centre

    rgb_paths = load_sintel_rgb_paths(sintel_root, seq)[: len(m_centres)]
    n = min(len(rgb_paths), len(m_centres))
    gt_centres = gt_camera_centres(sintel_root, seq, rgb_paths[:n])
    m_centres = m_centres[:n]

    print(f"[{seq}] monst3r pts={m_pts.shape[0]}  frames={n}")
    gt_pts, _, _ = gt_cloud(sintel_root, seq, rgb_paths[:n], args.max_depth, args.gt_stride)
    gt_diag = float(np.linalg.norm(gt_pts.max(0) - gt_pts.min(0)))

    # Rotation comes from the cameras (Umeyama), but SCALE and TRANSLATION come from the point
    # clouds' own robust extent, not the cameras: on low-motion sequences the camera baseline is
    # too short to constrain scale, so a camera-only Sim3 wildly mis-sizes the (far-reaching) cloud.
    _, rot, _ = umeyama_sim3(m_centres, gt_centres)
    m_rot = m_pts @ rot.T

    def robust_center_radius(p):
        c = np.median(p, axis=0)
        r = float(np.percentile(np.linalg.norm(p - c, axis=1), 75))
        return c, max(r, 1e-9)

    mc, mr = robust_center_radius(m_rot)
    gc, gr = robust_center_radius(gt_pts)
    scale = gr / mr
    m_pts = scale * m_rot + (gc - scale * mc)
    print(f"  align: rot from cameras, scale={scale:.4f} from cloud radius (gt {gr:.2f} / m {mr:.2f})")

    # Strip MonST3R's far background junk: keep points that actually have GT geometry near them.
    d, _ = cKDTree(gt_pts).query(m_pts, workers=-1)
    keep = d < args.gt_dist_thresh * gt_diag
    print(f"  near-GT clip (<{args.gt_dist_thresh:.3f}x diag): kept {int(keep.sum())}/{len(m_pts)} pts")
    m_pts, m_cols = m_pts[keep], m_cols[keep]

    out_dir = os.path.join(OUTPUTS_DIR, "point_cloud", seq)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "monst3r.ply")
    nn = write_ply(out_path, m_pts, m_cols)
    m_diag = np.linalg.norm(m_pts.max(0) - m_pts.min(0)) if nn else 0.0
    print(f"  saved {out_path}  ({nn} pts)  extent monst3r/gt={m_diag / gt_diag:.2f}")
    return True


def main():
    args = parse_args()
    sintel_root = resolve_sintel_root(args.sintel_root)
    seqs = args.seqs or SINTEL_EVAL_SEQUENCES
    done = sum(process(seq, args, sintel_root) for seq in seqs)
    print(f"aligned {done}/{len(seqs)} sequences")


if __name__ == "__main__":
    main()
