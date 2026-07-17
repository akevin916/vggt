#!/usr/bin/env python3
"""Measure predicted-gate QUALITY (not pose) against GT dynamic masks on Sintel.

For each sequence we run the model, take its per-patch gate probability sigma(g),
and compare it to the GT motion mask pooled to the patch grid. We report:
  AUC   -- ranking: does sigma(g) rank dynamic patches above static ones (floor-free)
  F1    -- classification at threshold 0.5 (dynamic = positive)
  gap   -- mean sigma over dynamic minus over static patches (confidence spread)
  p_dyn/p_stat -- mean sigma on each class (reads under-confidence directly)
  calibration  -- reliability curve + signed error: separates a gate that HEDGES from
                  one that is honestly uncertain (see _calibration)

Aggregates: macro = mean over sequences; micro = pool all patches then score.
Run from training/:
  python diag/gate_quality.py --ckpt logs/dyn_vggt_v3_s1_inst/ckpts/best.pt --all_seqs
"""
from __future__ import annotations

import os
import sys

_TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_TRAINING_DIR, os.path.dirname(_TRAINING_DIR)]

import argparse
import json
from datetime import datetime
from typing import Dict, List

import cv2
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

from data.motion_mask import sintel_masks_and_fraction
from eval_utils.paths import GATE_QUALITY, default_output_dir
from data.sintel_io import SINTEL_EVAL_SEQUENCES, load_sintel_rgb_paths, resolve_sintel_root
from eval_utils.vggt_infer import infer_sequence_chunked, load_vggt_for_eval
from vggt.utils.load_fn import load_and_preprocess_images


def _patch_labels_and_probs(model, args, sintel_root, seq):
    """Return (probs, labels_bin, label_frac) flattened over all patches of a seq."""
    rgb_paths = load_sintel_rgb_paths(sintel_root, seq)[: args.max_frames]
    images = load_and_preprocess_images(rgb_paths, mode="crop")
    s, _, h, w = images.shape
    ph, pw = h // args.patch_size, w // args.patch_size

    infer_kw = {"device": args.device}
    if args.chunk_size > 0:
        infer_kw["chunk_size"] = args.chunk_size
    pred = infer_sequence_chunked(model, rgb_paths, gate_logits_override=None, **infer_kw)
    if "gate_logits" not in pred:
        raise RuntimeError("ckpt returned no gate_logits")
    g = np.asarray(pred["gate_logits"], dtype=np.float32).reshape(s, -1)  # [s, ph*pw]
    probs = 1.0 / (1.0 + np.exp(-g))

    masks, _ = sintel_masks_and_fraction(sintel_root, seq, rgb_paths, args.motion_thr)
    frac = np.zeros((s, ph, pw), dtype=np.float32)
    for i in range(min(s, len(masks))):
        if masks[i] is not None:
            frac[i] = cv2.resize(masks[i], (pw, ph), interpolation=cv2.INTER_AREA)
    frac = frac.reshape(s, ph * pw)
    labels = (frac >= args.label_thr).astype(np.int8)
    return probs.reshape(-1), labels.reshape(-1), frac.reshape(-1)


def _score(probs, labels):
    pos = int(labels.sum())
    out = {
        "n_patches": int(labels.size),
        "dynamic_fraction": float(labels.mean()),
        "p_dyn": float(probs[labels == 1].mean()) if pos else float("nan"),
        "p_stat": float(probs[labels == 0].mean()) if pos < labels.size else float("nan"),
    }
    out["gap"] = out["p_dyn"] - out["p_stat"]
    # AUC/F1 need both classes present
    if 0 < pos < labels.size:
        out["auc"] = float(roc_auc_score(labels, probs))
        out["f1"] = float(f1_score(labels, (probs >= 0.5).astype(np.int8), zero_division=0))
    else:
        out["auc"] = float("nan")
        out["f1"] = float("nan")
    return out


def _calibration(probs, labels, n_bins=10):
    """Reliability curve over pooled patches: predicted sigma(g) vs empirical dynamic rate.

    p_dyn ~= 0.35 reads as "under-confident" only if patches the gate scores 0.35 are
    dynamic MORE often than 35% of the time. If they are dynamic exactly 35% of the time
    the gate is calibrated and 0.35 is the honest answer -- no amount of re-weighting or
    temperature will fix that, because the bottleneck is the feature, not the loss.

    signed_err = sum_b w_b * (empirical_b - predicted_b):
      > 0  under-confident (hedging)   |   ~ 0  calibrated   |   < 0  over-confident
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    bins, ece, signed = [], 0.0, 0.0
    for b in range(n_bins):
        m = idx == b
        cnt = int(m.sum())
        rec = {"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": cnt}
        if cnt:
            rec["mean_pred"] = float(probs[m].mean())
            rec["emp_freq"] = float(labels[m].mean())
            w = cnt / labels.size
            ece += w * abs(rec["emp_freq"] - rec["mean_pred"])
            signed += w * (rec["emp_freq"] - rec["mean_pred"])
        else:
            rec["mean_pred"] = rec["emp_freq"] = float("nan")
        bins.append(rec)
    return {"n_bins": n_bins, "bins": bins, "ece": float(ece), "signed_err": float(signed)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--all_seqs", action="store_true")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--max_frames", type=int, default=16)
    ap.add_argument("--chunk_size", type=int, default=0)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--motion_thr", type=float, default=2.0, help="GT flow-residual motion threshold (px)")
    ap.add_argument("--label_thr", type=float, default=0.5, help="patch dynamic-fraction -> binary label")
    ap.add_argument("--cal_bins", type=int, default=10, help="reliability-curve bins over sigma(g)")
    ap.add_argument("--sintel_root", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--require_gate", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    sintel_root = resolve_sintel_root(args.sintel_root)
    seqs = SINTEL_EVAL_SEQUENCES if args.all_seqs else (args.seqs or SINTEL_EVAL_SEQUENCES)
    model = load_vggt_for_eval(args.ckpt, device=args.device, require_gate=args.require_gate)

    per_seq: Dict[str, Dict] = {}
    all_probs, all_labels = [], []
    errors: List[str] = []
    for seq in tqdm(seqs, desc="gate_quality"):
        try:
            p, y, _ = _patch_labels_and_probs(model, args, sintel_root, seq)
            per_seq[seq] = _score(p, y)
            all_probs.append(p)
            all_labels.append(y)
        except Exception as e:
            errors.append(f"{seq}: {e}")
            print(f"[skip] {seq}: {e}")

    macro_keys = ("auc", "f1", "gap", "p_dyn", "p_stat")
    macro = {k: float(np.nanmean([v[k] for v in per_seq.values()])) for k in macro_keys}
    micro = _score(np.concatenate(all_probs), np.concatenate(all_labels)) if all_probs else {}
    calib = _calibration(np.concatenate(all_probs), np.concatenate(all_labels), args.cal_bins) if all_probs else {}

    results = {
        "meta": {
            "ckpt": args.ckpt, "seqs": seqs, "max_frames": args.max_frames,
            "motion_thr": args.motion_thr, "label_thr": args.label_thr,
            "cal_bins": args.cal_bins,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "per_seq": per_seq, "macro": macro, "micro": micro,
        "calibration": calib, "errors": errors,
    }

    out_dir = args.out_dir or default_output_dir(args.ckpt, GATE_QUALITY)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"quality_f{args.max_frames}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'seq':>12} {'dyn%':>6} {'AUC':>6} {'F1':>6} {'gap':>6} {'p_dyn':>6} {'p_stat':>7}")
    for seq, v in sorted(per_seq.items(), key=lambda kv: -(kv[1]["auc"] if np.isfinite(kv[1]["auc"]) else -1)):
        print(f"{seq:>12} {v['dynamic_fraction']*100:6.1f} {v['auc']:6.3f} {v['f1']:6.3f} "
              f"{v['gap']:6.3f} {v['p_dyn']:6.3f} {v['p_stat']:7.3f}")
    print(f"{'MACRO':>12} {'':>6} {macro['auc']:6.3f} {macro['f1']:6.3f} {macro['gap']:6.3f} "
          f"{macro['p_dyn']:6.3f} {macro['p_stat']:7.3f}")
    print(f"{'MICRO':>12} {micro['dynamic_fraction']*100:6.1f} {micro['auc']:6.3f} {micro['f1']:6.3f} "
          f"{micro['gap']:6.3f} {micro['p_dyn']:6.3f} {micro['p_stat']:7.3f}")

    if calib:
        print(f"\ncalibration (micro, {calib['n_bins']} bins over sigma(g))")
        print(f"{'bin':>12} {'n':>9} {'%mass':>6} {'pred':>6} {'empir':>6} {'emp-pred':>9}")
        for b in calib["bins"]:
            if not b["n"]:
                continue
            print(f"  [{b['lo']:.1f},{b['hi']:.1f}){b['n']:>9} "
                  f"{b['n']/micro['n_patches']*100:6.1f} {b['mean_pred']:6.3f} {b['emp_freq']:6.3f} "
                  f"{b['emp_freq']-b['mean_pred']:+9.3f}")
        verdict = ("UNDER-confident (hedging)" if calib["signed_err"] > 0.05 else
                   "OVER-confident" if calib["signed_err"] < -0.05 else
                   "CALIBRATED -> p_dyn is honest uncertainty, not hedging")
        print(f"  ECE={calib['ece']:.4f}  signed_err={calib['signed_err']:+.4f}  -> {verdict}")
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
