#!/usr/bin/env python3
"""Unified oracle-mask gate bias ablation on Sintel / PointOdyssey.

Modes injected via ``gate_logits_override``:
  off         neutral no-gate control (uniform large-negative logit)
  predicted   model's own gate
  oracle      GT / trusted mask-derived gate
  predicted_xT temperature-scaled model gate (optional)
"""

from __future__ import annotations

import os
import sys

_TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_TRAINING_DIR, os.path.dirname(_TRAINING_DIR)]

import argparse
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from data.datasets.pointodyssey import PointOdysseyDataset
from data.paths import data_path
from eval_utils.gate_common import oracle_logits_from_masks, po_common_conf, pool_to_patch
from data.motion_mask import DIAG_SEQUENCES, sintel_masks_and_fraction
from eval_utils.paths import GATE_BIAS_ABLATION, GATE_BIAS_ABLATION_PO, default_output_dir
from eval_utils.metrics_pose import eval_pose_metrics, extrinsics_w2c_to_tum
from data.sintel_io import (
    SINTEL_EVAL_SEQUENCES,
    load_sintel_gt_poses,
    load_sintel_rgb_paths,
    resolve_sintel_root,
    sintel_seq_paths,
)
from eval_utils.vggt_infer import infer_sequence_chunked, load_vggt_for_eval
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

MODES = ["off", "predicted", "oracle"]
POSE_METRICS = ("ate", "rpe_trans", "rpe_rot")


def parse_args():
    ap = argparse.ArgumentParser(description="Oracle-mask gate bias ablation (Sintel / PointOdyssey)")
    ap.add_argument("--dataset", choices=["sintel", "po"], default="sintel")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--k", type=float, default=30.0, help="oracle/off logit magnitude (bias ~ -softplus(k))")
    ap.add_argument(
        "--scales",
        type=float,
        nargs="*",
        default=[3.0, 10.0],
        help="temperature(s) T for predicted_x{T}: bias=-softplus(T*g_predicted). Empty to skip.",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--require_gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require ckpt to contain gate_predictor weights.",
    )
    ap.add_argument(
        "--force_gate",
        action="store_true",
        help="Build the gate mechanism even if the ckpt has no gate_predictor (e.g. pretrained "
        "VGGT-1B). GatePredictor is random-init -> only off/oracle modes are meaningful.",
    )

    # Sintel-specific
    ap.add_argument("--sintel_root", default=None)
    ap.add_argument("--seqs", nargs="*", default=None, help="Sintel sequences (default: DIAG_SEQUENCES)")
    ap.add_argument(
        "--all_seqs",
        action="store_true",
        help=f"Run all {len(SINTEL_EVAL_SEQUENCES)} Sintel eval sequences",
    )
    ap.add_argument("--max_frames", type=int, default=12)
    ap.add_argument("--motion_thr", type=float, default=2.0, help="px threshold for GT flow-residual oracle mask")
    ap.add_argument("--chunk_size", type=int, default=0, help="0 = full sequence; else chunk inference")
    ap.add_argument(
        "--report_dynamic_fraction",
        action="store_true",
        help="Include and print per-sequence dynamic-pixel fraction from GT flow residual.",
    )

    # PointOdyssey-specific
    ap.add_argument("--po_dir", default=data_path("train", "point_odyssey"))
    ap.add_argument("--n_clips", type=int, default=10, help="number of PO test clips")
    ap.add_argument("--img_per_seq", type=int, default=12)
    ap.add_argument("--img_size", type=int, default=518)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def _scaled_mode_names(scales: List[float]) -> List[str]:
    return [f"predicted_x{T:g}" for T in scales]


def _mean_by_mode(items: Dict[str, Dict[str, Dict[str, float]]], scales: List[float]) -> Dict[str, Dict[str, float]]:
    all_modes = MODES + _scaled_mode_names(scales)
    return {
        mode: {
            metric: float(np.mean([items[k][mode][metric] for k in items if mode in items[k]]))
            for metric in POSE_METRICS
        }
        for mode in all_modes
        if any(mode in items[k] for k in items)
    }


def _print_summary(results: Dict[str, Any], title: str, item_label: str) -> None:
    modes = list(results["mean"].keys())
    print(f"\n========== {title} ==========")
    print(f"{'mode':<16} {'ATE':>8} {'RPE-t':>8} {'RPE-r':>8}")
    for mode in modes:
        m = results["mean"][mode]
        print(f"{mode:<16} {m['ate']:>8.4f} {m['rpe_trans']:>8.4f} {m['rpe_rot']:>8.4f}")
    print(f"\nper-{item_label}:")
    key = f"per_{item_label}"
    for name, item_modes in results[key].items():
        line = f"  {name:<20}"
        for mode in modes:
            if mode in item_modes:
                line += f" | {mode}: ATE={item_modes[mode]['ate']:.4f}"
        print(line)


def _save_results(out_dir: str, results: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {json_path}")
    return json_path


def seq_dynamic_fraction(sintel_root: str, seq: str, max_frames: int, motion_thr: float) -> float:
    rgb_paths = load_sintel_rgb_paths(sintel_root, seq)[:max_frames]
    _, mean_frac = sintel_masks_and_fraction(sintel_root, seq, rgb_paths, motion_thr)
    return mean_frac


def _infer_extrinsic(model, img, h, w, device: str, gate_override: Optional[torch.Tensor]):
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=(device == "cuda")):
            pred = model(images=img, gate_logits_override=gate_override)
    extrinsic, _ = pose_encoding_to_extri_intri(pred["pose_enc"], image_size_hw=(h, w))
    extrinsic_np = extrinsic.squeeze(0).float().cpu().numpy()
    gate_logits_np = pred["gate_logits"].squeeze(0).float().cpu().numpy() if "gate_logits" in pred else None
    return extrinsic_np, gate_logits_np


def _run_sintel_seq(model, args, sintel_root: str, seq: str) -> Dict[str, Any]:
    rgb_paths = load_sintel_rgb_paths(sintel_root, seq)[: args.max_frames]
    _, _, cam_dir = sintel_seq_paths(sintel_root, seq)
    gt_tum, gt_ts = load_sintel_gt_poses(cam_dir, rgb_paths)

    images = load_and_preprocess_images(rgb_paths, mode="crop")
    s, _, h, w = images.shape
    ph, pw = h // args.patch_size, w // args.patch_size

    masks, dyn_frac = sintel_masks_and_fraction(sintel_root, seq, rgb_paths, args.motion_thr)
    off_logits = torch.full((1, s, ph * pw), -args.k)
    oracle_logits = oracle_logits_from_masks(masks, s, ph, pw, args.k)

    infer_kw = {"device": args.device}
    if args.chunk_size > 0:
        infer_kw["chunk_size"] = args.chunk_size

    results = {}
    for mode, override in [("off", off_logits.to(args.device)), ("oracle", oracle_logits.to(args.device))]:
        pred = infer_sequence_chunked(model, rgb_paths, gate_logits_override=override, **infer_kw)
        results[mode] = eval_pose_metrics(pred["extrinsic"], gt_tum, gt_ts)

    pred = infer_sequence_chunked(model, rgb_paths, gate_logits_override=None, **infer_kw)
    results["predicted"] = eval_pose_metrics(pred["extrinsic"], gt_tum, gt_ts)

    if "gate_logits" in pred:
        g_pred = torch.from_numpy(pred["gate_logits"]).unsqueeze(0)
        for T in args.scales:
            scaled = infer_sequence_chunked(
                model, rgb_paths, gate_logits_override=(g_pred * T).to(args.device), **infer_kw
            )
            results[f"predicted_x{T:g}"] = eval_pose_metrics(scaled["extrinsic"], gt_tum, gt_ts)
    elif args.scales:
        print(f"[warn] {seq}: model returned no gate_logits, skipping predicted_x{{T}} modes")

    return {"modes": results, "dynamic_fraction": dyn_frac}


def _evaluate_sintel(args) -> Dict[str, Any]:
    args.out_dir = args.out_dir or default_output_dir(args.ckpt, GATE_BIAS_ABLATION)
    sintel_root = resolve_sintel_root(args.sintel_root)
    seqs = SINTEL_EVAL_SEQUENCES if args.all_seqs else (args.seqs or DIAG_SEQUENCES)
    print(f"Output dir: {args.out_dir}")
    print(f"Sintel root: {sintel_root}")
    print(f"Sequences: {seqs}")

    model = load_vggt_for_eval(args.ckpt, device=args.device, require_gate=args.require_gate, force_gate=args.force_gate)

    per_seq: Dict[str, Dict[str, Dict[str, float]]] = {}
    dynamic_fraction_by_seq: Dict[str, float] = {}
    errors: List[str] = []
    for seq in tqdm(seqs, desc="gate_bias_ablation_sintel"):
        try:
            out = _run_sintel_seq(model, args, sintel_root, seq)
            per_seq[seq] = out["modes"]
            dynamic_fraction_by_seq[seq] = out["dynamic_fraction"]
        except Exception as e:
            errors.append(f"{seq}: {e}")
            print(f"[skip] {seq}: {e}")

    results = {
        "meta": {
            "dataset": "sintel",
            "ckpt": args.ckpt,
            "sintel_root": sintel_root,
            "seqs": seqs,
            "max_frames": args.max_frames,
            "k": args.k,
            "scales": args.scales,
            "motion_thr": args.motion_thr,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "per_seq": per_seq,
        "mean": _mean_by_mode(per_seq, args.scales),
        "errors": errors,
    }
    if args.report_dynamic_fraction:
        valid = [v for v in dynamic_fraction_by_seq.values() if np.isfinite(v)]
        results["dynamic_fraction_by_seq"] = dynamic_fraction_by_seq
        results["dynamic_fraction_mean"] = float(np.mean(valid)) if valid else float("nan")

    _save_results(args.out_dir, results)
    _print_summary(results, "GATE BIAS ABLATION (Sintel)", item_label="seq")
    if args.report_dynamic_fraction:
        print("\nsequence dynamic_fraction (desc):")
        for seq, frac in sorted(dynamic_fraction_by_seq.items(), key=lambda x: -x[1]):
            print(f"  {seq:<14} {frac:.3f}")
    return results


def _run_po_clip(model, args, ds: PointOdysseyDataset, seq_index: int) -> Dict[str, Dict[str, float]]:
    batch = ds.get_data(seq_index=seq_index, img_per_seq=args.img_per_seq, aspect_ratio=1.0)

    img = torch.from_numpy(np.stack(batch["images"]).astype(np.float32)).permute(0, 3, 1, 2).div(255)
    img = img[None].to(args.device)
    mm = torch.from_numpy(np.stack(batch["motion_mask"]).astype(np.float32))

    gt_extri = np.stack(batch["extrinsics"]).astype(np.float64)
    gt_tum, gt_ts = extrinsics_w2c_to_tum(gt_extri)

    _, s, _, h, w = img.shape
    ph, pw = h // args.patch_size, w // args.patch_size

    off_logits = torch.full((1, s, ph * pw), -args.k, device=args.device)
    m_pool = pool_to_patch(mm, ph, pw).reshape(1, s, ph * pw)
    oracle_logits = ((m_pool * 2.0 - 1.0) * args.k).to(args.device)

    results = {}
    for mode, override in [("off", off_logits), ("oracle", oracle_logits)]:
        extrinsic_np, _ = _infer_extrinsic(model, img, h, w, args.device, override)
        results[mode] = eval_pose_metrics(extrinsic_np, gt_tum, gt_ts)

    extrinsic_np, g_pred = _infer_extrinsic(model, img, h, w, args.device, None)
    results["predicted"] = eval_pose_metrics(extrinsic_np, gt_tum, gt_ts)

    if g_pred is not None:
        g_pred_t = torch.from_numpy(g_pred).unsqueeze(0)
        for T in args.scales:
            extrinsic_np, _ = _infer_extrinsic(model, img, h, w, args.device, (g_pred_t * T).to(args.device))
            results[f"predicted_x{T:g}"] = eval_pose_metrics(extrinsic_np, gt_tum, gt_ts)
    elif args.scales:
        print("[warn] model returned no gate_logits, skipping predicted_x{T} modes")

    return results


def _evaluate_po(args) -> Dict[str, Any]:
    args.out_dir = args.out_dir or default_output_dir(args.ckpt, GATE_BIAS_ABLATION_PO)
    print(f"Output dir: {args.out_dir}")
    print(f"PO dir: {args.po_dir}")

    model = load_vggt_for_eval(args.ckpt, img_size=args.img_size, device=args.device, require_gate=args.require_gate, force_gate=args.force_gate)
    ds = PointOdysseyDataset(
        common_conf=po_common_conf(args),
        split="test",
        PO_DIR=args.po_dir,
        min_num_images=args.img_per_seq,
        dynamic_source="instance",
    )
    print(f"PO test sequences available: {ds.sequence_list_len}; using {args.n_clips} clips")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    per_clip: Dict[str, Dict[str, Dict[str, float]]] = {}
    errors: List[str] = []
    for ci in tqdm(range(args.n_clips), desc="gate_bias_ablation_po"):
        seq_index = ci % ds.sequence_list_len
        label = f"{ds.sequence_list[seq_index]}_{ci}"
        try:
            per_clip[label] = _run_po_clip(model, args, ds, seq_index)
        except Exception as e:
            errors.append(f"{label}: {e}")
            print(f"[skip] {label}: {e}")

    results = {
        "meta": {
            "dataset": "po",
            "ckpt": args.ckpt,
            "po_dir": args.po_dir,
            "n_clips": args.n_clips,
            "img_per_seq": args.img_per_seq,
            "k": args.k,
            "scales": args.scales,
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "per_clip": per_clip,
        "mean": _mean_by_mode(per_clip, args.scales),
        "errors": errors,
    }

    _save_results(args.out_dir, results)
    _print_summary(results, "GATE BIAS ABLATION (PointOdyssey, m*_inst oracle)", item_label="clip")
    return results


def evaluate(args) -> Dict[str, Any]:
    if args.dataset == "po":
        return _evaluate_po(args)
    return _evaluate_sintel(args)


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
