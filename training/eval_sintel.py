#!/usr/bin/env python3
"""Sintel pose (ATE/RPE) + depth (AbsRel/delta) evaluation for Dyn-VGGT."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.depth_metrics import average_depth_results, eval_sequence_depth
from eval.pose_metrics import eval_pose_metrics, max_pose_depth_delta
from eval.sintel_io import (
    DEFAULT_SINTEL_ROOT,
    compute_preprocess_meta,
    list_sintel_sequences,
    load_sintel_gt_depths,
    load_sintel_gt_poses,
    load_sintel_rgb_paths,
    resize_pred_to_gt,
    sintel_seq_paths,
)
from eval.vggt_infer import infer_sequence, infer_sequence_chunked, load_dyn_vggt, variant_flags


def parse_args():
    ap = argparse.ArgumentParser(description="Sintel pose + depth eval for Dyn-VGGT")
    ap.add_argument("--ckpt", type=str, help="Checkpoint path")
    ap.add_argument("--variant", type=str, default="s0", choices=["s0", "dyn_vggt_s0", "vggt_base"])
    ap.add_argument("--sintel_root", type=str, default=DEFAULT_SINTEL_ROOT)
    ap.add_argument("--out_dir", type=str, default="logs/dyn_vggt_po_s0/eval_sintel")
    ap.add_argument("--seq_list", type=str, nargs="*", default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--chunk_size", type=int, default=0, help="0 = full sequence; else chunk inference")
    ap.add_argument("--max_depth", type=float, default=80.0)
    ap.add_argument("--compare", type=str, nargs=2, metavar=("BASE_JSON", "S0_JSON"))
    ap.add_argument("--bitwise_ckpt", type=str, default=None, help="Second ckpt for bitwise check on first seq")
    ap.add_argument("--bitwise_variant", type=str, default="vggt_base")
    return ap.parse_args()


def _mean_pose(per_seq: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    keys = ["ate", "rpe_trans", "rpe_rot"]
    if not per_seq:
        return {k: 0.0 for k in keys}
    return {k: float(np.mean([v[k] for v in per_seq.values()])) for k in keys}


def evaluate_checkpoint(args) -> Dict[str, Any]:
    temporal, motion, flow = variant_flags(args.variant)
    model = load_dyn_vggt(
        args.ckpt,
        temporal=temporal,
        motion=motion,
        flow=flow,
        device=args.device,
    )

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

    for seq in tqdm(sequences, desc=f"eval {args.variant}"):
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
            "variant": args.variant,
            "sintel_root": args.sintel_root,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "num_sequences": len(sequences),
            "num_ok": len(pose_per_seq),
        },
        "pose": {"per_seq": pose_per_seq, "mean": _mean_pose(pose_per_seq)},
        "depth": {"per_seq": depth_per_seq, "mean": average_depth_results(depth_per_seq)},
        "errors": errors,
    }

    tag = args.variant
    json_path = os.path.join(args.out_dir, f"results_{tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    write_summary_md(results, os.path.join(args.out_dir, f"summary_{tag}.md"))
    print(f"\nSaved {json_path}")
    print_summary(results)
    return results


def print_summary(results: Dict[str, Any]):
    pm = results["pose"]["mean"]
    dm = results["depth"]["mean"]
    print("\n========== SINTEL EVAL SUMMARY ==========")
    print(f"  Pose  ATE={pm['ate']:.4f}  RPE-trans={pm['rpe_trans']:.4f}  RPE-rot={pm['rpe_rot']:.4f}")
    print(
        f"  Depth AbsRel={dm.get('abs_rel', 0):.4f}  "
        f"delta<1.25={dm.get('delta_1', 0):.4f}  "
        f"RMSE={dm.get('rmse', 0):.4f}"
    )


def write_summary_md(results: Dict[str, Any], path: str):
    pm = results["pose"]["mean"]
    dm = results["depth"]["mean"]
    meta = results["meta"]
    lines = [
        f"# Sintel Eval — {meta['variant']}",
        "",
        f"- Checkpoint: `{meta['ckpt']}`",
        f"- Sequences OK: {meta['num_ok']} / {meta['num_sequences']}",
        "",
        "## Mean metrics",
        "",
        "| ATE | RPE-trans | RPE-rot | AbsRel | delta<1.25 | RMSE |",
        "|-----|-----------|---------|--------|------------|------|",
        (
            f"| {pm['ate']:.4f} | {pm['rpe_trans']:.4f} | {pm['rpe_rot']:.4f} | "
            f"{dm.get('abs_rel', 0):.4f} | {dm.get('delta_1', 0):.4f} | {dm.get('rmse', 0):.4f} |"
        ),
        "",
        "## Per-sequence pose",
        "",
        "| Seq | ATE | RPE-trans | RPE-rot |",
        "|-----|-----|-----------|---------|",
    ]
    for seq, m in sorted(results["pose"]["per_seq"].items()):
        lines.append(f"| {seq} | {m['ate']:.4f} | {m['rpe_trans']:.4f} | {m['rpe_rot']:.4f} |")

    lines += ["", "## Per-sequence depth", "", "| Seq | AbsRel | delta<1.25 | RMSE |", "|-----|--------|------------|------|"]
    for seq, m in sorted(results["depth"]["per_seq"].items()):
        lines.append(f"| {seq} | {m['abs_rel']:.4f} | {m['delta_1']:.4f} | {m['rmse']:.4f} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def compare_results(base_path: str, s0_path: str, out_dir: str):
    with open(base_path, encoding="utf-8") as f:
        base = json.load(f)
    with open(s0_path, encoding="utf-8") as f:
        s0 = json.load(f)

    bp, sp = base["pose"]["mean"], s0["pose"]["mean"]
    bd, sd = base["depth"]["mean"], s0["depth"]["mean"]

    ate_delta = abs(sp["ate"] - bp["ate"])
    ate_rel = ate_delta / max(bp["ate"], 1e-9)
    absrel_delta = abs(sd.get("abs_rel", 0) - bd.get("abs_rel", 0))

    pose_pass = ate_rel < 0.01 or ate_delta < 0.01
    depth_pass = absrel_delta < 0.001
    overall = pose_pass and depth_pass

    report = {
        "baseline": base_path,
        "s0": s0_path,
        "delta": {
            "ate_abs": ate_delta,
            "ate_rel": ate_rel,
            "abs_rel_abs": absrel_delta,
        },
        "gate": {
            "pose_pass": pose_pass,
            "depth_pass": depth_pass,
            "overall_pass": overall,
        },
        "baseline_mean": {"pose": bp, "depth": bd},
        "s0_mean": {"pose": sp, "depth": sd},
    }

    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "compare_gate.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# S0 vs VGGT-1B Regression Gate",
        "",
        f"- Overall: **{'PASS' if overall else 'FAIL'}**",
        f"- Pose gate (|dATE|/ATE<1% or |dATE|<0.01m): {'PASS' if pose_pass else 'FAIL'} "
        f"(dATE={ate_delta:.6f}, rel={ate_rel:.6f})",
        f"- Depth gate (|dAbsRel|<0.001): {'PASS' if depth_pass else 'FAIL'} (dAbsRel={absrel_delta:.6f})",
        "",
        "| | ATE | RPE-trans | RPE-rot | AbsRel | delta<1.25 |",
        "|---|-----|-----------|---------|--------|------------|",
        (
            f"| VGGT-1B | {bp['ate']:.4f} | {bp['rpe_trans']:.4f} | {bp['rpe_rot']:.4f} | "
            f"{bd.get('abs_rel', 0):.4f} | {bd.get('delta_1', 0):.4f} |"
        ),
        (
            f"| Dyn-VGGT S0 | {sp['ate']:.4f} | {sp['rpe_trans']:.4f} | {sp['rpe_rot']:.4f} | "
            f"{sd.get('abs_rel', 0):.4f} | {sd.get('delta_1', 0):.4f} |"
        ),
    ]
    out_md = os.path.join(out_dir, "compare_gate.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n========== REGRESSION GATE ==========")
    print(f"  Overall: {'PASS' if overall else 'FAIL'}")
    print(f"  Saved {out_json}")
    return report


def run_bitwise_check(args):
    seq = list_sintel_sequences(args.seq_list)[0]
    rgb_paths = load_sintel_rgb_paths(args.sintel_root, seq)[:6]

    t_a, m_a, f_a = variant_flags(args.variant)
    t_b, m_b, f_b = variant_flags(args.bitwise_variant)

    model_a = load_dyn_vggt(args.ckpt, temporal=t_a, motion=m_a, flow=f_a, device=args.device)
    model_b = load_dyn_vggt(args.bitwise_ckpt, temporal=t_b, motion=m_b, flow=f_b, device=args.device)

    pred_a = infer_sequence(model_a, rgb_paths, device=args.device)
    pred_b = infer_sequence(model_b, rgb_paths, device=args.device)
    delta = max_pose_depth_delta(pred_a, pred_b)
    print("\n========== BITWISE CHECK (first seq, 6 frames) ==========")
    print(f"  max|Δdepth|={delta['max_delta_depth']:.2e}")
    print(f"  max|Δpose_enc|={delta['max_delta_pose_enc']:.2e}")
    ok = max(delta.values()) < 1e-2
    print(f"  -> {'PASS' if ok else 'CHECK'}")
    return delta


def main():
    args = parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1], args.out_dir)
        return

    if args.bitwise_ckpt:
        if not args.ckpt:
            raise SystemExit("--ckpt required with --bitwise_ckpt")
        run_bitwise_check(args)
        return

    if not args.ckpt:
        raise SystemExit("--ckpt is required unless using --compare or --bitwise_ckpt")

    evaluate_checkpoint(args)


if __name__ == "__main__":
    main()
