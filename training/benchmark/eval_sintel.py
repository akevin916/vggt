#!/usr/bin/env python3
"""Sintel pose + depth benchmark: checkpoint in, metrics out."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

_TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TRAINING_DIR)
sys.path.insert(0, os.path.dirname(_TRAINING_DIR))

from eval.depth_metrics import average_depth_results, eval_sequence_depth
from eval.paths import default_eval_dir
from eval.pose_metrics import eval_pose_metrics
from eval.sintel_io import (
    compute_preprocess_meta,
    list_sintel_sequences,
    load_sintel_gt_depths,
    load_sintel_gt_poses,
    load_sintel_rgb_paths,
    resolve_sintel_root,
    resize_pred_to_gt,
    sintel_seq_paths,
)
from eval.vggt_infer import infer_sequence, infer_sequence_chunked, load_vggt_for_eval


def parse_args():
    ap = argparse.ArgumentParser(description="Sintel pose + depth benchmark")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--sintel_root", type=str, default=None, help="Auto-detected from repo data/ if omitted")
    ap.add_argument("--out_dir", type=str, default=None, help="Default: logs/<exp>/eval_sintel")
    ap.add_argument("--seq_list", type=str, nargs="*", default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--chunk_size", type=int, default=0, help="0 = full sequence; else chunk inference")
    ap.add_argument("--max_depth", type=float, default=80.0)
    return ap.parse_args()


def _mean_pose(per_seq: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    keys = ["ate", "rpe_trans", "rpe_rot"]
    if not per_seq:
        return {k: 0.0 for k in keys}
    return {k: float(np.mean([v[k] for v in per_seq.values()])) for k in keys}


def evaluate(args) -> Dict[str, Any]:
    args.out_dir = args.out_dir or default_eval_dir(args.ckpt, "eval_sintel", _TRAINING_DIR)
    args.sintel_root = resolve_sintel_root(args.sintel_root)
    print(f"Output dir: {args.out_dir}")
    print(f"Sintel root: {args.sintel_root}")
    model = load_vggt_for_eval(args.ckpt, device=args.device)

    sequences = list_sintel_sequences(args.seq_list)
    os.makedirs(args.out_dir, exist_ok=True)
    error_log = os.path.join(args.out_dir, "_error_log.txt")

    pose_per_seq: Dict[str, Dict[str, float]] = {}
    depth_per_seq: Dict[str, Dict[str, float]] = {}
    errors: List[str] = []

    infer_fn = infer_sequence_chunked if args.chunk_size > 0 else infer_sequence
    infer_kw = {"device": args.device}
    if args.chunk_size > 0:
        infer_kw["chunk_size"] = args.chunk_size

    for seq in tqdm(sequences, desc="eval_sintel"):
        try:
            rgb_paths = load_sintel_rgb_paths(args.sintel_root, seq)
            _, _, cam_dir = sintel_seq_paths(args.sintel_root, seq)
            gt_tum, gt_ts = load_sintel_gt_poses(cam_dir, rgb_paths)
            gt_depths = load_sintel_gt_depths(args.sintel_root, seq, rgb_paths)

            pred = infer_fn(model, rgb_paths, **infer_kw)
            pose_per_seq[seq] = eval_pose_metrics(pred["extrinsic"], gt_tum, gt_ts)

            pred_on_gt = []
            for i, rgb_path in enumerate(rgb_paths):
                meta = compute_preprocess_meta(rgb_path)
                frame_depth = pred["depth"][i]
                if frame_depth.ndim == 3:
                    frame_depth = frame_depth[..., 0]
                pred_on_gt.append(resize_pred_to_gt(frame_depth, meta))

            depth_per_seq[seq] = eval_sequence_depth(pred_on_gt, gt_depths, max_depth=args.max_depth)
        except Exception as e:
            msg = f"{seq}: {e}\n{traceback.format_exc()}"
            errors.append(msg)
            with open(error_log, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    results = {
        "meta": {
            "ckpt": args.ckpt,
            "sintel_root": args.sintel_root,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "num_sequences": len(sequences),
            "num_ok": len(pose_per_seq),
        },
        "pose": {"per_seq": pose_per_seq, "mean": _mean_pose(pose_per_seq)},
        "depth": {"per_seq": depth_per_seq, "mean": average_depth_results(depth_per_seq)},
        "errors": errors,
    }

    json_path = os.path.join(args.out_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {json_path}")
    print_summary(results)
    return results


def print_summary(results: Dict[str, Any]):
    pm = results["pose"]["mean"]
    dm = results["depth"]["mean"]
    print("\n========== SINTEL BENCHMARK ==========")
    print(f"  Pose  ATE={pm['ate']:.4f}  RPE-trans={pm['rpe_trans']:.4f}  RPE-rot={pm['rpe_rot']:.4f}")
    print(
        f"  Depth AbsRel={dm.get('abs_rel', 0):.4f}  "
        f"delta<1.25={dm.get('delta_1', 0):.4f}  "
        f"RMSE={dm.get('rmse', 0):.4f}"
    )


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
