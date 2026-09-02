#!/usr/bin/env python3
"""Gate evaluation: mask QUALITY and pose ABLATION, from one set of forward passes.

Merged from ``diag/gate_quality.py`` + ``diag/gate_bias_ablation.py`` (2026-08-19). The two
asked different questions off the SAME work: quality needs sigma(g) and the GT mask pooled to
the patch grid; the ablation needs the GT mask (to build the oracle logits) and the predicted
pass's extrinsics. Run separately, the ``predicted`` forward pass and the GT-mask derivation
each happened twice. Here they happen once.

Two metric families, independently switchable via --metrics:

  quality  Is sigma(g) a good dynamic mask?  (no pose involved)
             AUC   ranking: does sigma(g) rank dynamic patches above static ones (floor-free)
             F1    classification at 0.5 (dynamic = positive)
             gap   mean sigma over dynamic minus over static (confidence spread)
             p_dyn/p_stat   mean sigma per class -- reads under-confidence directly
             calibration    reliability curve + signed error: separates a gate that HEDGES
                            from one that is honestly uncertain (see _calibration)
           Use AUC/F1, never BCE, to judge the gate: the BCE target is a soft average-pooled
           label whose boundary patches carry an irreducible floor.

  pose     Does the gate bias actually improve the camera?  Modes via ``gate_logits_override``:
             off           neutral no-gate control (uniform large-negative logit)
             predicted     the model's own gate
             oracle        GT / trusted mask-derived gate -- the headroom bound
             predicted_hard@t / predicted_topR
                           the model's own gate binarised to 0/1 (--hard_taus / --hard_topk),
                           i.e. an oracle-SHAPED signal driven by the prediction

Outputs keep the two pre-merge layouts so older results stay comparable:
    pose     -> outputs/gate_bias_ablation[_po]/<exp>/results.json
    quality  -> outputs/gate_quality/<exp>/quality_f<max_frames>.json

PointOdyssey supports ``pose`` only -- the quality metrics read Sintel's flow-residual masks.

Run from training/:
    python diag/gate_eval.py --ckpt logs/<exp>/ckpts/best.pt --all_seqs
    python diag/gate_eval.py --ckpt <ckpt> --metrics quality --all_seqs --max_frames 16 --gif
    python diag/gate_eval.py --ckpt <ckpt> --dataset po --metrics pose
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

import cv2
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

from data.datasets.pointodyssey import PointOdysseyDataset
from data.motion_mask import DIAG_SEQUENCES, sintel_masks_and_fraction
from data.paths import data_path
from data.sintel_io import (
    SINTEL_EVAL_SEQUENCES,
    load_sintel_gt_poses,
    load_sintel_rgb_paths,
    resolve_sintel_root,
    sintel_seq_paths,
)
from eval_utils.gate_common import oracle_logits_from_masks, po_common_conf, pool_to_patch
from eval_utils.gate_vis import save_gate_overlay_gif
from eval_utils.metrics_pose import eval_pose_metrics, extrinsics_w2c_to_tum
from eval_utils.paths import (
    GATE_BIAS_ABLATION,
    GATE_BIAS_ABLATION_PO,
    GATE_QUALITY,
    default_output_dir,
)
from eval_utils.vggt_infer import infer_sequence_chunked, load_vggt_for_eval
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

MODES = ["off", "predicted", "oracle"]
POSE_METRICS = ("ate", "rpe_trans", "rpe_rot")
QUALITY_MACRO_KEYS = ("auc", "f1", "gap", "p_dyn", "p_stat")


def parse_args():
    ap = argparse.ArgumentParser(description="Gate quality + pose-bias ablation (Sintel / PointOdyssey)")
    ap.add_argument("--dataset", choices=["sintel", "po"], default="sintel")
    ap.add_argument(
        "--metrics",
        default="quality,pose",
        help="comma-separated subset of {quality,pose}. PointOdyssey supports pose only.",
    )
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default=None, help="overrides the POSE output dir only")
    ap.add_argument("--quality_out_dir", default=None)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--k", type=float, default=30.0, help="oracle/off logit magnitude (bias ~ -softplus(k))")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--gate_leaky",
        type=float,
        default=0.0,
        help="model flag: >0 gives the bias clamp's flat half a slope, so patches the gate is "
             "only mildly suspicious of (g<0) stop being indistinguishable from confident-static "
             "ones. 1.0 removes the clamp entirely.",
    )
    ap.add_argument(
        "--gate_bias_zero_ref",
        action="store_true",
        help="model flag: measure the bias from 0 instead of softplus(0), i.e. drop the kink that "
             "makes the gate act only above sigma(g)=0.5. Same ordering among patches as "
             "--gate_leaky 1.0, differing only by the constant log(2) relative to the "
             "camera/register keys.",
    )
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
    # 12 was the ablation's default; the published quality table (docs/results/natural.md table 3) was
    # measured at 16 over --all_seqs. The quality json is named quality_f<max_frames>.json, so a
    # run at a different length lands in its own file rather than overwriting that table's source.
    ap.add_argument("--max_frames", type=int, default=12)
    ap.add_argument("--motion_thr", type=float, default=2.0, help="px threshold for GT flow-residual mask")
    ap.add_argument("--label_thr", type=float, default=0.5, help="patch dynamic-fraction -> binary label")
    ap.add_argument("--cal_bins", type=int, default=10, help="reliability-curve bins over sigma(g)")
    ap.add_argument(
        "--gif",
        action="store_true",
        help="quality only: also write one animated GIF per sequence (sigma(g) heat-overlaid on "
             "the RGB, GT outlined in green, AUC/F1 burned into every frame) next to the json",
    )
    ap.add_argument(
        "--hard_taus",
        type=float,
        nargs="*",
        default=[],
        help="pose only: binarise the PREDICTED gate at each absolute tau on sigma(g) and feed it "
             "as a 0/1 override (+k / -k), the same shape of signal as `oracle`. Temperature "
             "scaling cannot reach these operating points, which is why that mode was dropped: it is "
             "multiplicative, so a gate whose logits are all negative stays under the bias "
             "clamp at every temperature. Re-centring g additively is the only knob that moves it.",
    )
    ap.add_argument(
        "--hard_topk",
        type=float,
        nargs="*",
        default=[],
        help="pose only: same 0/1 override, but the threshold is the PER-FRAME (1-rho) quantile of "
             "g instead of an absolute value -- adapts to sequences whose confidence sits at a "
             "different level, at the cost of masking rho of the patches even in a static scene.",
    )
    ap.add_argument("--overlay_alpha", type=float, default=0.45, help="heatmap blend weight")
    ap.add_argument("--duration_ms", type=int, default=200, help="ms per frame in the GIF")
    ap.add_argument("--chunk_size", type=int, default=0, help="0 = full sequence; else chunk inference")
    ap.add_argument(
        "--report_dynamic_fraction",
        action="store_true",
        help="Include and print per-sequence dynamic-pixel fraction from the GT flow residual.",
    )

    # PointOdyssey-specific
    ap.add_argument("--po_dir", default=data_path("train", "point_odyssey"))
    ap.add_argument("--n_clips", type=int, default=10, help="number of PO test clips")
    ap.add_argument("--img_per_seq", type=int, default=12)
    ap.add_argument("--img_size", type=int, default=518)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    args.metric_set = {m.strip() for m in args.metrics.split(",") if m.strip()}
    unknown = args.metric_set - {"quality", "pose"}
    if unknown:
        ap.error(f"unknown --metrics entries: {sorted(unknown)}")
    if not args.metric_set:
        ap.error("--metrics selected nothing")
    if args.dataset == "po" and "quality" in args.metric_set:
        print("[warn] PointOdyssey supports pose metrics only; dropping 'quality'")
        args.metric_set.discard("quality")
        if not args.metric_set:
            ap.error("nothing left to compute for --dataset po")
    return args


# ---------------------------------------------------------------------------
# quality scoring
# ---------------------------------------------------------------------------


def _score(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
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


def _calibration(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
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


def _patch_labels(masks, s: int, ph: int, pw: int, label_thr: float) -> np.ndarray:
    """GT pixel masks -> binary per-patch labels, flattened. INTER_AREA = average pooling."""
    frac = np.zeros((s, ph, pw), dtype=np.float32)
    for i in range(min(s, len(masks))):
        if masks[i] is not None:
            frac[i] = cv2.resize(masks[i], (pw, ph), interpolation=cv2.INTER_AREA)
    return (frac.reshape(-1) >= label_thr).astype(np.int8)


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------


def _hard_mode_names(args) -> List[str]:
    return ([f"predicted_hard@{t:g}" for t in getattr(args, "hard_taus", [])]
            + [f"predicted_top{r:g}" for r in getattr(args, "hard_topk", [])])


def _extra_mode_names(args) -> List[str]:
    return _hard_mode_names(args)


def _mean_by_mode(items: Dict[str, Dict[str, Dict[str, float]]], extra: List[str]) -> Dict[str, Dict[str, float]]:
    all_modes = MODES + list(extra)
    return {
        mode: {
            metric: float(np.mean([items[k][mode][metric] for k in items if mode in items[k]]))
            for metric in POSE_METRICS
        }
        for mode in all_modes
        if any(mode in items[k] for k in items)
    }


def _print_pose_summary(results: Dict[str, Any], title: str, item_label: str) -> None:
    modes = list(results["mean"].keys())
    print(f"\n========== {title} ==========")
    print(f"{'mode':<16} {'ATE':>8} {'RPE-t':>8} {'RPE-r':>8}")
    for mode in modes:
        m = results["mean"][mode]
        print(f"{mode:<16} {m['ate']:>8.4f} {m['rpe_trans']:>8.4f} {m['rpe_rot']:>8.4f}")
    print(f"\nper-{item_label}:")
    for name, item_modes in results[f"per_{item_label}"].items():
        line = f"  {name:<20}"
        for mode in modes:
            if mode in item_modes:
                line += f" | {mode}: ATE={item_modes[mode]['ate']:.4f}"
        print(line)


def _print_quality_summary(results: Dict[str, Any]) -> None:
    per_seq, macro, micro, calib = (
        results["per_seq"], results["macro"], results["micro"], results["calibration"],
    )
    print("\n========== GATE QUALITY (Sintel) ==========")
    print(f"{'seq':>12} {'dyn%':>6} {'AUC':>6} {'F1':>6} {'gap':>6} {'p_dyn':>6} {'p_stat':>7}")
    for seq, v in sorted(per_seq.items(), key=lambda kv: -(kv[1]["auc"] if np.isfinite(kv[1]["auc"]) else -1)):
        print(f"{seq:>12} {v['dynamic_fraction']*100:6.1f} {v['auc']:6.3f} {v['f1']:6.3f} "
              f"{v['gap']:6.3f} {v['p_dyn']:6.3f} {v['p_stat']:7.3f}")
    print(f"{'MACRO':>12} {'':>6} {macro['auc']:6.3f} {macro['f1']:6.3f} {macro['gap']:6.3f} "
          f"{macro['p_dyn']:6.3f} {macro['p_stat']:7.3f}")
    if micro:
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


def _save_json(out_dir: str, name: str, payload: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {path}")
    return path


# ---------------------------------------------------------------------------
# Sintel
# ---------------------------------------------------------------------------


def _run_sintel_seq(model, args, sintel_root: str, seq: str) -> Dict[str, Any]:
    """One sequence: the GT masks and the `predicted` forward pass feed BOTH metric families."""
    want_pose = "pose" in args.metric_set
    want_quality = "quality" in args.metric_set

    rgb_paths = load_sintel_rgb_paths(sintel_root, seq)[: args.max_frames]
    images = load_and_preprocess_images(rgb_paths, mode="crop")
    s, _, h, w = images.shape
    ph, pw = h // args.patch_size, w // args.patch_size

    # Derived once, used twice: oracle logits (pose) and patch labels (quality).
    masks, dyn_frac = sintel_masks_and_fraction(sintel_root, seq, rgb_paths, args.motion_thr)

    # chunk_size=0 means "one pass over the whole sequence". It must be translated to an
    # explicit len(rgb_paths), NOT left unset: infer_sequence_chunked defaults to 32, which
    # silently splits any >32-frame sequence into INDEPENDENT passes that are concatenated
    # without alignment. The resulting ATE is dominated by the arbitrary pose jump at the
    # seam, so it explodes AND stops depending on the model -- on Sintel f50 this produced
    # temple_2 ATE 2.53 (vs 0.057 correct) and near-identical numbers across every ckpt and
    # every gate mode. The same warning lives on infer_sequence_chunked itself.
    infer_kw = {
        "device": args.device,
        "chunk_size": args.chunk_size if args.chunk_size > 0 else len(rgb_paths),
    }

    out: Dict[str, Any] = {"dynamic_fraction": dyn_frac}
    pose: Dict[str, Dict[str, float]] = {}

    gt_tum = gt_ts = None
    if want_pose:
        _, _, cam_dir = sintel_seq_paths(sintel_root, seq)
        gt_tum, gt_ts = load_sintel_gt_poses(cam_dir, rgb_paths)
        off_logits = torch.full((1, s, ph * pw), -args.k)
        oracle_logits = oracle_logits_from_masks(masks, s, ph, pw, args.k)
        for mode, override in [("off", off_logits), ("oracle", oracle_logits)]:
            pred = infer_sequence_chunked(
                model, rgb_paths, gate_logits_override=override.to(args.device), **infer_kw
            )
            pose[mode] = eval_pose_metrics(pred["extrinsic"], gt_tum, gt_ts)

    # THE shared pass: its extrinsics are the `predicted` pose row and its gate_logits are the
    # quality scores. Running the two tools separately paid for this twice.
    pred = infer_sequence_chunked(model, rgb_paths, gate_logits_override=None, **infer_kw)
    if want_pose:
        pose["predicted"] = eval_pose_metrics(pred["extrinsic"], gt_tum, gt_ts)

    if "gate_logits" not in pred:
        if want_quality:
            raise RuntimeError("ckpt returned no gate_logits -- quality metrics need them")
        if args.hard_taus or args.hard_topk:
            print(f"[warn] {seq}: model returned no gate_logits, skipping predicted_hard modes")
    else:
        g = np.asarray(pred["gate_logits"], dtype=np.float32).reshape(s, -1)
        if want_quality:
            labels = _patch_labels(masks, s, ph, pw, args.label_thr)
            probs = (1.0 / (1.0 + np.exp(-g))).reshape(-1)
            out["quality"] = (probs, labels)
            if args.gif:
                # GIF inputs are the SCORED arrays reshaped back to the patch grid --
                # never a re-derived mask, so the picture cannot disagree with the number.
                out["overlay"] = (
                    images.cpu().numpy(),
                    probs.reshape(s, ph, pw),
                    labels.reshape(s, ph, pw),
                )
        if want_pose and (args.hard_taus or args.hard_topk):
            # 0/1 override built from the model's OWN gate: keep the ranking (AUC is the healthy
            # part), discard the absolute values (calibration is the broken part). This is an
            # additive re-centring of g, which is what the bias clamp actually needs -- see
            # --hard_taus for why a multiplicative temperature cannot do it.
            probs_g = 1.0 / (1.0 + np.exp(-g))                    # [s, P_patch]
            hard_frac: Dict[str, float] = {}

            def _run_hard(sel: np.ndarray, name: str) -> None:
                override = torch.from_numpy(
                    np.where(sel, args.k, -args.k).astype(np.float32)
                ).reshape(1, s, -1)
                hp = infer_sequence_chunked(
                    model, rgb_paths, gate_logits_override=override.to(args.device), **infer_kw
                )
                pose[name] = eval_pose_metrics(hp["extrinsic"], gt_tum, gt_ts)
                # How much of the frame each mode actually suppressed. Without it an ATE change
                # is unattributable: "masked the right patches" and "masked almost nothing" look
                # identical in the pose column alone.
                hard_frac[name] = float(sel.mean())

            for tau in args.hard_taus:
                _run_hard(probs_g >= tau, f"predicted_hard@{tau:g}")
            for rho in args.hard_topk:
                # Per-frame quantile: each frame masks exactly rho of its patches.
                kth = np.quantile(g, 1.0 - rho, axis=1, keepdims=True)
                _run_hard(g >= kth, f"predicted_top{rho:g}")
            out["hard_frac"] = hard_frac

    if want_pose:
        out["pose"] = pose
    return out


def _evaluate_sintel(args) -> Dict[str, Any]:
    sintel_root = resolve_sintel_root(args.sintel_root)
    seqs = SINTEL_EVAL_SEQUENCES if args.all_seqs else (args.seqs or DIAG_SEQUENCES)
    print(f"Sintel root: {sintel_root}")
    print(f"Sequences: {seqs}")
    print(f"Metrics: {sorted(args.metric_set)}")

    model = load_vggt_for_eval(
        args.ckpt, device=args.device, require_gate=args.require_gate, force_gate=args.force_gate,
        gate_leaky=args.gate_leaky, gate_bias_zero_ref=args.gate_bias_zero_ref,
    )

    pose_per_seq: Dict[str, Dict[str, Dict[str, float]]] = {}
    quality_per_seq: Dict[str, Dict[str, float]] = {}
    overlay_payload: Dict[str, Any] = {}
    hard_frac_by_seq: Dict[str, Dict[str, float]] = {}
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    dynamic_fraction_by_seq: Dict[str, float] = {}
    errors: List[str] = []

    for seq in tqdm(seqs, desc="gate_eval_sintel"):
        try:
            out = _run_sintel_seq(model, args, sintel_root, seq)
        except Exception as e:
            errors.append(f"{seq}: {e}")
            print(f"[skip] {seq}: {e}")
            continue
        dynamic_fraction_by_seq[seq] = out["dynamic_fraction"]
        if "pose" in out:
            pose_per_seq[seq] = out["pose"]
            if "hard_frac" in out:
                hard_frac_by_seq[seq] = out["hard_frac"]
        if "quality" in out:
            probs, labels = out["quality"]
            quality_per_seq[seq] = _score(probs, labels)
            all_probs.append(probs)
            all_labels.append(labels)
            if "overlay" in out:
                overlay_payload[seq] = out["overlay"]

    meta = {
        "dataset": "sintel",
        "ckpt": args.ckpt,
        "sintel_root": sintel_root,
        "seqs": seqs,
        "max_frames": args.max_frames,
        # 0 = one pass per sequence. Recorded because a wrong chunk_size silently inflates ATE
        # (see _run_sintel_seq) and the json is otherwise indistinguishable from a correct run.
        "chunk_size": args.chunk_size,
        "motion_thr": args.motion_thr,
        # The bias formulation is part of the model, not the override, so it silently applies to
        # every mode -- recorded here or two runs become indistinguishable in the json.
        "gate_leaky": args.gate_leaky,
        "gate_bias_zero_ref": args.gate_bias_zero_ref,
        "metrics": sorted(args.metric_set),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    out_all: Dict[str, Any] = {}

    if "pose" in args.metric_set:
        results = {
            "meta": {**meta, "k": args.k,
                     "hard_taus": args.hard_taus, "hard_topk": args.hard_topk},
            "per_seq": pose_per_seq,
            "mean": _mean_by_mode(pose_per_seq, _extra_mode_names(args)),
            "errors": errors,
        }
        if args.report_dynamic_fraction:
            valid = [v for v in dynamic_fraction_by_seq.values() if np.isfinite(v)]
            results["dynamic_fraction_by_seq"] = dynamic_fraction_by_seq
            results["dynamic_fraction_mean"] = float(np.mean(valid)) if valid else float("nan")
        if hard_frac_by_seq:
            results["hard_frac_by_seq"] = hard_frac_by_seq
            results["hard_frac_mean"] = {
                mode: float(np.mean([v[mode] for v in hard_frac_by_seq.values() if mode in v]))
                for mode in _hard_mode_names(args)
            }
        out_dir = args.out_dir or default_output_dir(args.ckpt, GATE_BIAS_ABLATION)
        _save_json(out_dir, "results.json", results)
        _print_pose_summary(results, "GATE BIAS ABLATION (Sintel)", item_label="seq")
        if hard_frac_by_seq:
            print("\nfraction of patches suppressed (mean over seqs):")
            for mode, f in results["hard_frac_mean"].items():
                print(f"  {mode:<22} {f*100:5.1f}%")
        if args.report_dynamic_fraction:
            print("\nsequence dynamic_fraction (desc):")
            for seq, frac in sorted(dynamic_fraction_by_seq.items(), key=lambda x: -x[1]):
                print(f"  {seq:<14} {frac:.3f}")
        out_all["pose"] = results

    if "quality" in args.metric_set:
        macro = {
            k: float(np.nanmean([v[k] for v in quality_per_seq.values()])) if quality_per_seq else float("nan")
            for k in QUALITY_MACRO_KEYS
        }
        pooled_p = np.concatenate(all_probs) if all_probs else None
        pooled_y = np.concatenate(all_labels) if all_labels else None
        results = {
            "meta": {**meta, "label_thr": args.label_thr, "cal_bins": args.cal_bins},
            "per_seq": quality_per_seq,
            "macro": macro,
            "micro": _score(pooled_p, pooled_y) if pooled_p is not None else {},
            "calibration": _calibration(pooled_p, pooled_y, args.cal_bins) if pooled_p is not None else {},
            "errors": errors,
        }
        out_dir = args.quality_out_dir or default_output_dir(args.ckpt, GATE_QUALITY)
        _save_json(out_dir, f"quality_f{args.max_frames}.json", results)
        _print_quality_summary(results)
        if overlay_payload:
            ck = os.path.basename(args.ckpt)
            vis_dir = os.path.join(out_dir, f"overlay_f{args.max_frames}")
            print(f"\noverlay GIFs -> {vis_dir}")
            for seq, (imgs, p_grid, y_grid) in overlay_payload.items():
                q = quality_per_seq[seq]
                head = (f"{seq}   AUC {q['auc']:.3f}   F1 {q['f1']:.3f}   "
                        f"gap {q['gap']:+.3f}   dyn {q['dynamic_fraction']*100:.1f}%")
                sub = (f"ckpt={ck}  label_thr={args.label_thr}  motion_thr={args.motion_thr}px  "
                       f"frames={imgs.shape[0]}")
                fp = save_gate_overlay_gif(
                    os.path.join(vis_dir, f"{seq}.gif"), imgs, p_grid, y_grid,
                    header=head, subheader=sub,
                    alpha=args.overlay_alpha, duration_ms=args.duration_ms,
                )
                print(f"  {fp}")
        out_all["quality"] = results

    return out_all


# ---------------------------------------------------------------------------
# PointOdyssey (pose only)
# ---------------------------------------------------------------------------


def _infer_extrinsic(model, img, h, w, device: str, gate_override: Optional[torch.Tensor]):
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=(device == "cuda")):
            pred = model(images=img, gate_logits_override=gate_override)
    extrinsic, _ = pose_encoding_to_extri_intri(pred["pose_enc"], image_size_hw=(h, w))
    extrinsic_np = extrinsic.squeeze(0).float().cpu().numpy()
    gate_logits_np = pred["gate_logits"].squeeze(0).float().cpu().numpy() if "gate_logits" in pred else None
    return extrinsic_np, gate_logits_np


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

    return results


def _evaluate_po(args) -> Dict[str, Any]:
    print(f"PO dir: {args.po_dir}")

    model = load_vggt_for_eval(
        args.ckpt, img_size=args.img_size, device=args.device,
        require_gate=args.require_gate, force_gate=args.force_gate,
        gate_leaky=args.gate_leaky, gate_bias_zero_ref=args.gate_bias_zero_ref,
    )
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
    for ci in tqdm(range(args.n_clips), desc="gate_eval_po"):
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
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "per_clip": per_clip,
        "mean": _mean_by_mode(per_clip, _extra_mode_names(args)),
        "errors": errors,
    }

    out_dir = args.out_dir or default_output_dir(args.ckpt, GATE_BIAS_ABLATION_PO)
    _save_json(out_dir, "results.json", results)
    _print_pose_summary(results, "GATE BIAS ABLATION (PointOdyssey, m*_inst oracle)", item_label="clip")
    return {"pose": results}


def evaluate(args) -> Dict[str, Any]:
    if args.dataset == "po":
        return _evaluate_po(args)
    return _evaluate_sintel(args)


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
