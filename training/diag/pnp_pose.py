#!/usr/bin/env python3
"""Point-head PnP diagnostic: can a robust estimator recover pose the camera head can't?

Reads VGGT's point map (``world_points``, in the frame-0 camera frame) and solves each
frame's pose by PnP-RANSAC against the pixel grid -- the DUSt3R/MonST3R route -- then
scores it next to the camera head's own pose on the same sequences.

Three pose columns, one question each:

  camera_head  the aggregator's 1-token-per-frame pose               (the anchor: ATE 0.1714 on f50)
  pnp_all      PnP over all confident pixels                          (is the point head's geometry
                                                                       even a viable pose source?)
  pnp_static   PnP over confident *static* pixels only (m_geo)        (THE question: how much pose is
                                                                       recoverable by simply dropping
                                                                       the dynamic pixels?)

``pnp_all`` vs ``pnp_static`` is a gate-free, training-free measurement of the gate thesis.
RANSAC already rejects dynamic pixels as outliers, so pnp_static isolates what an
*explicit* static prior buys on top of that. If neither moves toward MonST3R's 0.108,
the Sintel pose gap is not primarily caused by dynamic content -- which is what results_natural.md's
f50 oracle (+0.7%) already hints at.

Two depth columns as a side product:

  depth_head    the depth head's own depth
  point_map_z   the point map transformed into each camera by the predicted pose

Their gap is the model's pose/point self-inconsistency, which per-frame median scaling
(eval_utils/metrics_depth.py) otherwise hides.
"""

from __future__ import annotations

import os
import sys

_TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_TRAINING_DIR, os.path.dirname(_TRAINING_DIR)]

import argparse
import json
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from data.motion_mask import sintel_masks_and_fraction
from data.sintel_io import (
    compute_preprocess_meta,
    list_sintel_sequences,
    load_sintel_gt_depths,
    load_sintel_gt_poses,
    load_sintel_rgb_paths,
    resize_gt_to_pred,
    resize_pred_to_gt,
    resolve_sintel_root,
    sintel_seq_paths,
)
from eval_utils.metrics_depth import average_depth_results, eval_sequence_depth
from eval_utils.metrics_pose import eval_pose_metrics
from eval_utils.paths import PNP_POSE, default_output_dir
from eval_utils.vggt_infer import graft_point_head, infer_sequence, load_vggt_for_eval

POSE_MODES = ["camera_head", "pnp_all", "pnp_static"]
DEPTH_MODES = ["depth_head", "point_map_z"]


def parse_args():
    ap = argparse.ArgumentParser(description="PnP-from-pointmap pose diagnostic on Sintel")
    ap.add_argument("--ckpt", type=str, default="checkpoints/VGGT-1B.pt")
    ap.add_argument(
        "--point_head_from",
        type=str,
        default=None,
        help="Borrow point_head weights from this checkpoint (e.g. checkpoints/VGGT-1B.pt) "
        "for a ckpt that has none -- every gate-lineage run disables the point head. The grafted "
        "head reads a trunk it never trained on; read the result as indicative only.",
    )
    ap.add_argument("--sintel_root", type=str, default=None)
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help=f"Default: outputs/{PNP_POSE}/<exp>/f<max_frames>",
    )
    ap.add_argument("--seq_list", type=str, nargs="*", default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_frames", type=int, default=50, help="Truncate each sequence (0 = all)")
    ap.add_argument("--motion_thr", type=float, default=2.0, help="m_geo flow-residual threshold (px)")
    ap.add_argument("--dyn_thr", type=float, default=0.5, help="Binarize the resized m_geo above this")
    ap.add_argument("--conf_percentile", type=float, default=50.0, help="Keep pixels above this world_points_conf percentile")
    ap.add_argument("--n_points", type=int, default=8000, help="Pixels subsampled per frame for PnP")
    ap.add_argument("--ransac_thr", type=float, default=2.0, help="PnP-RANSAC reprojection threshold (px)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_depth", type=float, default=80.0)
    return ap.parse_args()


def pixel_grid(h: int, w: int) -> np.ndarray:
    """(h, w, 2) of pixel centers in model crop coordinates."""
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    return np.stack([u, v], axis=-1)


def solve_frame_pnp(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    K: np.ndarray,
    ransac_thr: float,
    seed: int,
) -> Optional[np.ndarray]:
    """PnP-RANSAC -> (3,4) world-to-camera extrinsic, or None if it fails."""
    if len(obj_pts) < 6:
        return None
    ok, rvec, tvec, _ = cv2.solvePnPRansac(
        obj_pts.astype(np.float64),
        img_pts.astype(np.float64),
        K.astype(np.float64),
        None,
        reprojectionError=ransac_thr,
        iterationsCount=1000,
        confidence=0.9999,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return np.hstack([R, tvec.reshape(3, 1)])


def pnp_trajectory(
    world_points: np.ndarray,
    conf: np.ndarray,
    intrinsics: np.ndarray,
    static_masks: Optional[List[Optional[np.ndarray]]],
    args,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """PnP each frame independently. Returns (S,3,4) extrinsics, or None if any frame fails.

    ``static_masks[i]`` is a bool array in model crop space (True = keep). None for a
    frame means no mask was available (m_geo needs frame i+1's flow, so the last frame
    never has one) -- that frame falls back to the confidence filter alone.
    """
    S, H, W, _ = world_points.shape
    grid = pixel_grid(H, W).reshape(-1, 2)
    rng = np.random.default_rng(args.seed)

    extrinsics = []
    n_used, n_fallback = [], 0
    for i in range(S):
        pts = world_points[i].reshape(-1, 3)
        keep = np.isfinite(pts).all(axis=1)

        c = conf[i].reshape(-1)
        if np.any(keep):
            keep &= c >= np.percentile(c[keep], args.conf_percentile)

        if static_masks is not None:
            m = static_masks[i]
            if m is None:
                n_fallback += 1
            else:
                keep &= m.reshape(-1)

        idx = np.flatnonzero(keep)
        if len(idx) > args.n_points:
            idx = rng.choice(idx, size=args.n_points, replace=False)

        ext = solve_frame_pnp(pts[idx], grid[idx], intrinsics[i], args.ransac_thr, args.seed)
        if ext is None:
            return None, {"failed_frame": i, "n_candidates": int(len(idx))}
        extrinsics.append(ext)
        n_used.append(int(len(idx)))

    return np.stack(extrinsics, axis=0), {
        "mean_points_used": float(np.mean(n_used)),
        "min_points_used": int(np.min(n_used)),
        "frames_without_mask": n_fallback,
    }


def point_map_to_depth(world_points: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    """Transform the frame-0-frame point map into each camera; return its z as depth."""
    S, H, W, _ = world_points.shape
    out = np.empty((S, H, W), dtype=np.float32)
    for i in range(S):
        R = extrinsics[i][:3, :3].astype(np.float64)
        t = extrinsics[i][:3, 3].astype(np.float64)
        cam = world_points[i].reshape(-1, 3).astype(np.float64) @ R.T + t
        out[i] = cam[:, 2].reshape(H, W).astype(np.float32)
    return out


def frame0_identity_error(extrinsics: np.ndarray) -> Dict[str, float]:
    """Sanity check: world_points live in frame 0's camera frame, so PnP on frame 0
    must return identity. A large error here means the geometry or the intrinsics are
    inconsistent, and every other number in the run is suspect."""
    R = extrinsics[0][:3, :3]
    t = extrinsics[0][:3, 3]
    rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return {"frame0_rot_deg": rot_deg, "frame0_trans_norm": float(np.linalg.norm(t))}


def eval_sequence(seq: str, model, args) -> Dict[str, Any]:
    rgb_paths = load_sintel_rgb_paths(args.sintel_root, seq)
    if args.max_frames > 0:
        rgb_paths = rgb_paths[: args.max_frames]

    _, _, cam_dir = sintel_seq_paths(args.sintel_root, seq)
    gt_tum, gt_ts = load_sintel_gt_poses(cam_dir, rgb_paths)
    gt_depths = load_sintel_gt_depths(args.sintel_root, seq, rgb_paths)

    pred = infer_sequence(model, rgb_paths, device=args.device, want_point=True)
    if "world_points" not in pred:
        raise RuntimeError("checkpoint has no point_head -- nothing to PnP from")

    world_points = pred["world_points"]
    conf = pred["world_points_conf"]
    S, H, W, _ = world_points.shape

    # m_geo is derived at GT resolution; move it into model crop space to filter pixels.
    masks_gt, dyn_frac = sintel_masks_and_fraction(args.sintel_root, seq, rgb_paths, args.motion_thr)
    static_masks: List[Optional[np.ndarray]] = []
    for i, rgb_path in enumerate(rgb_paths):
        if masks_gt[i] is None:
            static_masks.append(None)
            continue
        meta = compute_preprocess_meta(rgb_path)
        dyn = resize_gt_to_pred(masks_gt[i], meta, (H, W))
        static_masks.append(dyn <= args.dyn_thr)

    out: Dict[str, Any] = {"num_frames": S, "dyn_frac": dyn_frac, "pose": {}, "depth": {}, "pnp_info": {}}

    out["pose"]["camera_head"] = eval_pose_metrics(pred["extrinsic"], gt_tum, gt_ts)

    for mode, masks in (("pnp_all", None), ("pnp_static", static_masks)):
        ext, info = pnp_trajectory(world_points, conf, pred["intrinsic"], masks, args)
        out["pnp_info"][mode] = info
        if ext is None:
            out["pose"][mode] = None
            continue
        out["pnp_info"][mode].update(frame0_identity_error(ext))
        out["pose"][mode] = eval_pose_metrics(ext, gt_tum, gt_ts)

    # Depth: both variants scored on GT resolution with the same per-frame median scaling
    # the Sintel benchmark uses, so these numbers are comparable to results_natural.md's.
    depth_variants: Dict[str, np.ndarray] = {}
    if "depth" in pred:
        d = pred["depth"]
        depth_variants["depth_head"] = d[..., 0] if d.ndim == 4 else d
    depth_variants["point_map_z"] = point_map_to_depth(world_points, pred["extrinsic"])

    metas = [compute_preprocess_meta(p) for p in rgb_paths]
    for mode, dmap in depth_variants.items():
        on_gt = [resize_pred_to_gt(dmap[i], metas[i]) for i in range(S)]
        out["depth"][mode] = eval_sequence_depth(on_gt, gt_depths, max_depth=args.max_depth)

    return out


def _mean_pose(per_seq: Dict[str, Dict[str, Any]], mode: str) -> Optional[Dict[str, float]]:
    vals = [v["pose"][mode] for v in per_seq.values() if v["pose"].get(mode) is not None]
    if not vals:
        return None
    return {k: float(np.mean([v[k] for v in vals])) for k in ("ate", "rpe_trans", "rpe_rot")}


def print_summary(results: Dict[str, Any]):
    print("\n========== PnP POSE DIAGNOSTIC ==========")
    print(f"  sequences: {results['meta']['num_ok']}/{results['meta']['num_sequences']}  "
          f"max_frames={results['meta']['max_frames']}")
    print("\n  pose (mean over sequences)")
    print(f"    {'mode':<14}{'ATE':>10}{'RPE-t':>10}{'RPE-r':>10}")
    for mode in POSE_MODES:
        m = results["pose"][mode]
        if m is None:
            print(f"    {mode:<14}{'FAILED':>10}")
        else:
            print(f"    {mode:<14}{m['ate']:>10.4f}{m['rpe_trans']:>10.4f}{m['rpe_rot']:>10.4f}")
    print("\n  depth (mean over sequences)")
    print(f"    {'mode':<14}{'AbsRel':>10}{'delta1':>10}")
    for mode in DEPTH_MODES:
        m = results["depth"].get(mode)
        if m:
            print(f"    {mode:<14}{m.get('abs_rel', 0):>10.4f}{m.get('delta_1', 0):>10.4f}")


def main():
    args = parse_args()
    # The f<max_frames> level matches the gate_sweep layout: sequence length is part of a
    # run's identity here (the whole point is that gate headroom varies with it), so two
    # lengths must not land on the same results.json.
    args.out_dir = args.out_dir or os.path.join(
        default_output_dir(args.ckpt, PNP_POSE), f"f{args.max_frames}"
    )
    args.sintel_root = resolve_sintel_root(args.sintel_root)
    print(f"Output dir: {args.out_dir}")
    print(f"Sintel root: {args.sintel_root}")

    model = load_vggt_for_eval(args.ckpt, device=args.device, force_point=bool(args.point_head_from))
    if args.point_head_from:
        graft_point_head(model, args.point_head_from)
        print(
            "⚠️  grafted point_head: it never trained on this checkpoint's trunk "
            "(gate-method S1 trains global blocks 8-23) -- indicative only."
        )
    if model.point_head is None:
        raise SystemExit(
            f"checkpoint has no point_head weights: {args.ckpt}\n"
            "Every gate-lineage config disables the point head (docs/method.md §8.2). Pass "
            "--point_head_from checkpoints/VGGT-1B.pt to borrow the pretrained one."
        )

    sequences = list_sintel_sequences(args.seq_list)
    os.makedirs(args.out_dir, exist_ok=True)
    error_log = os.path.join(args.out_dir, "_error_log.txt")

    per_seq: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    for seq in tqdm(sequences, desc="pnp_pose"):
        try:
            per_seq[seq] = eval_sequence(seq, model, args)
        except Exception as e:
            msg = f"{seq}: {e}\n{traceback.format_exc()}"
            errors.append(msg)
            with open(error_log, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    results = {
        "meta": {
            "ckpt": args.ckpt,
            "point_head_from": args.point_head_from,
            "sintel_root": args.sintel_root,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "max_frames": args.max_frames,
            "motion_thr": args.motion_thr,
            "conf_percentile": args.conf_percentile,
            "n_points": args.n_points,
            "ransac_thr": args.ransac_thr,
            "num_sequences": len(sequences),
            "num_ok": len(per_seq),
        },
        "pose": {m: _mean_pose(per_seq, m) for m in POSE_MODES},
        "depth": {
            m: average_depth_results({k: v["depth"][m] for k, v in per_seq.items() if m in v["depth"]})
            for m in DEPTH_MODES
        },
        "per_seq": per_seq,
        "errors": errors,
    }

    json_path = os.path.join(args.out_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {json_path}")
    print_summary(results)


if __name__ == "__main__":
    main()
