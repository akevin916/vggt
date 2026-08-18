#!/usr/bin/env python3
"""Per-frame gate confidence + frame-to-frame rotation jitter, for one Sintel sequence.

Purpose: test the hypothesis (from manual gate-quality review) that "medium confidence"
sequences -- where the gate detects the dynamic object but doesn't push sigma(g) low/high
enough to cleanly exclude it -- show frame-to-frame FLICKER in gate confidence, and that
this flicker lines up with frame-to-frame pose jitter (the "trajectory zigzags" observed
in diag/vis/trajectory.py), which would explain the RPE regression despite improved ATE.

Two stacked subplots per sequence, saved to outputs/gate_temporal/<exp>/<seq>.png:
  top:    mean sigma(g) over GT-dynamic patches vs GT-static patches, per frame (m_geo, §5.3a)
  bottom: relative rotation angle (deg) between consecutive frames, GT vs predicted
          (alignment-free: relative rotation between consecutive poses is invariant to
          a fixed global rotation offset, so no Sim3 alignment is needed here)
"""

from __future__ import annotations

import os
import sys

_TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO_DIR = os.path.dirname(_TRAINING_DIR)
sys.path[:0] = [_TRAINING_DIR, _REPO_DIR]

import argparse

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from data.motion_mask import DIAG_SEQUENCES, compute_ego_flow, derive_motion_mask, load_sintel_gt_flows
from eval_utils.paths import GATE_TEMPORAL, default_output_dir
from eval_utils.metrics_pose import extrinsics_w2c_to_tum
from data.sintel_io import (
    load_sintel_gt_depths,
    load_sintel_gt_poses,
    load_sintel_rgb_paths,
    matching_cam_path,
    resolve_sintel_root,
    sintel_cam_read,
    sintel_seq_paths,
)
from eval_utils.vggt_infer import infer_sequence_chunked, load_vggt_for_eval
from vggt.utils.load_fn import load_and_preprocess_images


def parse_args():
    ap = argparse.ArgumentParser(description="Per-frame gate confidence + rotation jitter")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seqs", nargs="*", default=None, help="Default: DIAG_SEQUENCES")
    ap.add_argument("--sintel_root", default=None)
    ap.add_argument("--max_frames", type=int, default=20)
    ap.add_argument("--motion_thr", type=float, default=2.0)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--chunk_size", type=int, default=0)
    ap.add_argument("--out_dir", default=None, help="Default: outputs/{}/<exp>".format(GATE_TEMPORAL))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out_dir = args.out_dir or default_output_dir(args.ckpt, GATE_TEMPORAL)
    return args


def rel_rot_angles_deg(tum: np.ndarray) -> np.ndarray:
    """Relative rotation angle (deg) between each consecutive pair of poses.

    Alignment-free: R_rel = R_{i+1} @ R_i^T is invariant to left-multiplying every pose
    by the same fixed rotation (i.e. a global reference-frame offset), so GT and predicted
    series are directly comparable without Sim(3) alignment.
    """
    quats_xyzw = tum[:, [4, 5, 6, 3]]  # wxyz -> xyzw
    mats = Rotation.from_quat(quats_xyzw).as_matrix()  # (N,3,3)
    rel = np.einsum("nij,nkj->nik", mats[1:], mats[:-1])  # R_{i+1} @ R_i^T
    tr = np.trace(rel, axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def run_seq(model, args, sintel_root: str, seq: str):
    rgb_paths = load_sintel_rgb_paths(sintel_root, seq)[: args.max_frames]
    _, _, cam_dir = sintel_seq_paths(sintel_root, seq)
    gt_tum, gt_ts = load_sintel_gt_poses(cam_dir, rgb_paths)
    gt_depths = load_sintel_gt_depths(sintel_root, seq, rgb_paths)
    gt_flows = load_sintel_gt_flows(sintel_root, seq, rgb_paths)

    intrinsics, extrinsics = [], []
    for p in rgb_paths:
        K, ext = sintel_cam_read(matching_cam_path(cam_dir, p))
        intrinsics.append(K)
        extrinsics.append(ext)

    images = load_and_preprocess_images(rgb_paths, mode="crop")
    s, _, h, w = images.shape
    ph, pw = h // args.patch_size, w // args.patch_size

    with torch.no_grad():
        pred = model(images=images[None].to(args.device))
    g_prob = torch.sigmoid(pred["gate_logits"].float()).cpu()[0].reshape(s, ph, pw).numpy()

    dyn_conf = np.full(s, np.nan)
    stat_conf = np.full(s, np.nan)
    for i in range(s):
        if gt_flows[i] is None or i + 1 >= len(extrinsics):
            continue
        ego = compute_ego_flow(gt_depths[i], intrinsics[i], extrinsics[i], extrinsics[i + 1])
        mask = derive_motion_mask(gt_flows[i], ego, threshold=args.motion_thr)
        patch_mask = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_AREA) >= 0.5
        if patch_mask.any():
            dyn_conf[i] = g_prob[i][patch_mask].mean()
        if (~patch_mask).any():
            stat_conf[i] = g_prob[i][~patch_mask].mean()

    infer_kw = {"device": args.device}
    if args.chunk_size > 0:
        infer_kw["chunk_size"] = args.chunk_size
    pred_full = infer_sequence_chunked(model, rgb_paths, **infer_kw)
    pred_tum, _ = extrinsics_w2c_to_tum(pred_full["extrinsic"])

    n = min(len(gt_tum), len(pred_tum))
    gt_rot = rel_rot_angles_deg(gt_tum[:n])
    pred_rot = rel_rot_angles_deg(pred_tum[:n])

    return dyn_conf, stat_conf, gt_rot, pred_rot


def plot_seq(seq: str, dyn_conf, stat_conf, gt_rot, pred_rot, out_path: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=False)

    frames = np.arange(len(dyn_conf))
    ax1.plot(frames, dyn_conf, "r-o", label="sigma(g) on GT-dynamic patches", markersize=4)
    ax1.plot(frames, stat_conf, "b-o", label="sigma(g) on GT-static patches", markersize=4)
    ax1.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel("frame")
    ax1.set_ylabel("gate confidence")
    ax1.set_title(f"{seq}: per-frame gate confidence (m_geo-labeled patches)")
    ax1.legend()

    trans = np.arange(len(gt_rot))
    ax2.plot(trans, gt_rot, "k-o", label="GT relative rotation", markersize=4)
    ax2.plot(trans, pred_rot, "r-o", label="predicted relative rotation", markersize=4)
    ax2.set_xlabel("frame transition (i -> i+1)")
    ax2.set_ylabel("rotation angle (deg)")
    ax2.set_title(f"{seq}: frame-to-frame rotation jitter (no alignment needed)")
    ax2.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    args = parse_args()
    sintel_root = resolve_sintel_root(args.sintel_root)
    seqs = args.seqs or DIAG_SEQUENCES

    model = load_vggt_for_eval(args.ckpt, device=args.device, require_gate=True)

    for seq in seqs:
        try:
            dyn_conf, stat_conf, gt_rot, pred_rot = run_seq(model, args, sintel_root, seq)
        except Exception as e:
            print(f"[skip] {seq}: {e}")
            continue
        out_path = os.path.join(args.out_dir, f"{seq}.png")
        plot_seq(seq, dyn_conf, stat_conf, gt_rot, pred_rot, out_path)


if __name__ == "__main__":
    main()
