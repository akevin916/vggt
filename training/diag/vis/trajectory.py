#!/usr/bin/env python3
"""Overlay GT vs predicted camera trajectories for two checkpoints on chosen Sintel sequences.

Purpose: explain *why* pose improves on some Sintel sequences and regresses on others between
two checkpoints (e.g. native VGGT-1B vs a Dyn-VGGT v3 variant) by looking at the actual
trajectory shape, not just the scalar ATE/RPE. Both trajectories are Sim(3)-aligned to GT
independently (same convention as eval/pose_metrics.eval_pose_metrics), then plotted top-down.

Images are saved under outputs/trajectory/<exp>/ (repo root), never shown interactively.
"""

from __future__ import annotations

import os
import sys

_TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO_DIR = os.path.dirname(_TRAINING_DIR)
sys.path[:0] = [_TRAINING_DIR, _REPO_DIR]

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation

from eval_utils.paths import TRAJECTORY, default_output_dir
from eval_utils.metrics_pose import _make_traj, extrinsics_w2c_to_tum
from data.sintel_io import (
    SINTEL_EVAL_SEQUENCES,
    load_sintel_gt_poses,
    load_sintel_rgb_paths,
    resolve_sintel_root,
    sintel_seq_paths,
)
from eval_utils.vggt_infer import infer_sequence_chunked, load_vggt_for_eval


def parse_args():
    ap = argparse.ArgumentParser(description="Overlay GT vs predicted trajectories for two checkpoints")
    ap.add_argument("--ckpt_a", required=True, help="e.g. checkpoints/VGGT-1B.pt")
    ap.add_argument("--ckpt_b", required=True, help="e.g. checkpoints/dyn_vggt_v3_s1_inst_photo.pt")
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--seqs", nargs="*", default=None, help="Default: all SINTEL_EVAL_SEQUENCES")
    ap.add_argument("--sintel_root", default=None)
    ap.add_argument("--max_frames", type=int, default=12)
    ap.add_argument("--chunk_size", type=int, default=0)
    ap.add_argument("--out_dir", default=None, help="Default: outputs/{}/<exp>".format(TRAJECTORY))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out_dir = args.out_dir or default_output_dir(args.ckpt, TRAJECTORY)
    return args


def aligned_traj_xyz(pred_extrinsics: np.ndarray, gt_tum: np.ndarray, gt_ts: np.ndarray):
    pred_tum, pred_ts = extrinsics_w2c_to_tum(pred_extrinsics)
    pred_traj = _make_traj(pred_tum, pred_ts)
    gt_traj = _make_traj(gt_tum, gt_ts)

    n = min(pred_traj.num_poses, gt_traj.num_poses)
    pred_traj.reduce_to_ids(list(range(n)))
    gt_traj.reduce_to_ids(list(range(n)))
    pred_traj.timestamps = gt_traj.timestamps

    result = main_ape.ape(
        gt_traj, pred_traj, pose_relation=PoseRelation.translation_part, align=True, correct_scale=True
    )
    ate = float(result.stats["rmse"])
    return gt_traj.positions_xyz, pred_traj.positions_xyz, ate  # pred_traj mutated in-place by align()


def plot_seq(seq: str, gt_xyz, pred_a_xyz, pred_b_xyz, ate_a, ate_b, label_a, label_b, out_path):
    # Pick the two axes with the largest GT spread for a top-down view (Sintel is not always Y-up).
    spread = gt_xyz.max(axis=0) - gt_xyz.min(axis=0)
    axes = np.argsort(spread)[-2:]
    ax0, ax1 = int(axes[0]), int(axes[1])
    axis_names = ["x", "y", "z"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(gt_xyz[:, ax0], gt_xyz[:, ax1], "k-o", label="GT", markersize=3, linewidth=2)
    ax.plot(pred_a_xyz[:, ax0], pred_a_xyz[:, ax1], "b--o", label=f"{label_a} (ATE={ate_a:.4f})", markersize=3)
    ax.plot(pred_b_xyz[:, ax0], pred_b_xyz[:, ax1], "r--o", label=f"{label_b} (ATE={ate_b:.4f})", markersize=3)
    ax.scatter(gt_xyz[0, ax0], gt_xyz[0, ax1], c="green", s=80, marker="*", zorder=5, label="start")
    ax.set_xlabel(axis_names[ax0])
    ax.set_ylabel(axis_names[ax1])
    ax.set_title(f"{seq}: Sim(3)-aligned camera trajectory (translation only)")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    args = parse_args()
    sintel_root = resolve_sintel_root(args.sintel_root)
    seqs = args.seqs or SINTEL_EVAL_SEQUENCES

    model_a = load_vggt_for_eval(args.ckpt_a, device=args.device)
    model_b = load_vggt_for_eval(args.ckpt_b, device=args.device)

    # chunk_size=0 -> one pass over the whole sequence. Must be an explicit
    # len(rgb_paths): infer_sequence_chunked defaults to 32 and would silently split
    # longer sequences into unaligned independent passes (see infer_sequence_chunked).
    infer_kw = {
        "device": args.device,
        "chunk_size": args.chunk_size if args.chunk_size > 0 else len(rgb_paths),
    }

    for seq in seqs:
        rgb_paths = load_sintel_rgb_paths(sintel_root, seq)[: args.max_frames]
        _, _, cam_dir = sintel_seq_paths(sintel_root, seq)
        gt_tum, gt_ts = load_sintel_gt_poses(cam_dir, rgb_paths)

        try:
            pred_a = infer_sequence_chunked(model_a, rgb_paths, **infer_kw)
            pred_b = infer_sequence_chunked(model_b, rgb_paths, **infer_kw)
            gt_xyz, pred_a_xyz, ate_a = aligned_traj_xyz(pred_a["extrinsic"], gt_tum, gt_ts)
            _, pred_b_xyz, ate_b = aligned_traj_xyz(pred_b["extrinsic"], gt_tum, gt_ts)
        except Exception as e:
            print(f"[skip] {seq}: {e}")
            continue

        out_path = os.path.join(args.out_dir, f"{seq}.png")
        plot_seq(seq, gt_xyz, pred_a_xyz, pred_b_xyz, ate_a, ate_b, args.label_a, args.label_b, out_path)


if __name__ == "__main__":
    main()
