#!/usr/bin/env python3
"""Measure the residual that L_ego_flow actually minimises: |ego_flow(pred) - ego_flow(GT)|.

WHY THIS EXISTS, given flow_loss_probe.py already probed "the flow loss": that probe scored
every variant against RAFT optical flow, because it was evaluating a port of MonST3R's loss.
L_ego_flow uses a GT-DERIVED target instead, which changes the quantity completely:

  * the floor is exactly 0, not RAFT's ~0.09 px noise level -- so "PO has no signal" (pred
    0.16 vs floor 0.09, docs/table.md table 5) says nothing about this loss;
  * the residual is a monotone function of pose/depth error, so it cannot be adversarial to
    ATE the way the RAFT residual was (flow_pose_headroom.py) -- that -5.7% ceiling does not
    bound this loss;
  * MonST3R's outlier machinery was defending against RAFT failures that do not exist here.

None of the published numbers transfer. This script measures the new quantity directly, by
calling `loss.compute_ego_flow_loss` itself rather than reimplementing it -- so what is
reported here is exactly what training will optimise, with no risk of the probe and the loss
drifting apart.

Reports per dataset:
  * loss_ego_flow, the training quantity itself -- its MAGNITUDE. It does not set the loss
    weight: that needs every term's magnitude at the same point in training, which only a real
    training step has (see config/*_smoke.yaml)
  * per-pixel |dflow| percentiles, in pixels
  * keep rate under a per_pixel_thre sweep, i.e. how much the outlier rejection is biting
  * the same with use_dynamic_mask off, which is the direct test of the claim that the mask
    is inert once the target is GT-derived

Run from training/:
  python diag/ego_flow_residual.py --ckpt logs/<run>/ckpts/epoch_30.pt
  python diag/ego_flow_residual.py --ckpt <...> --datasets po sintel
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

import numpy as np
import torch

from data.paths import data_path
from diag.flow_loss_probe import build_train_dataset, load_sintel_clip, load_train_clip
from ego_flow import ego_flow_from_disp, pixel_grid, relative_w2c
from eval_utils.paths import EGO_FLOW_RESIDUAL as TOOL
from eval_utils.paths import default_output_dir
from eval_utils.vggt_infer import load_vggt_for_eval
from loss import compute_ego_flow_loss
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def _batch_from_clip(clip, device) -> Dict[str, torch.Tensor]:
    """Clip (model input space) -> the batch keys the loss functions read."""
    img = clip["img_u8"].float().div(255).to(device)                    # (S,3,H,W)
    return {
        "images": img[None],
        "extrinsics": clip["gt_extri"][None].to(device),
        "intrinsics": clip["gt_intri"][None].to(device),
        "depths": clip["gt_depth"][None].to(device),
        "point_masks": clip["pmask"][None].to(device),
        "motion_mask": clip["motion"][None].to(device),
        "ids": torch.as_tensor(np.asarray(clip["ids"]))[None].to(device),
    }


@torch.no_grad()
def _residual_percentiles(predictions, batch, dyn_thresh, use_dynamic_mask, max_dt):
    """Per-pixel |ego_flow(pred) - ego_flow(GT)| over the same pixels the loss scores.

    Mirrors the loss's masking but reports the raw pixel distribution, which the scalar loss
    value hides -- a smooth-L1 mean cannot distinguish "uniformly 0.5 px off" from "mostly
    exact with a bad tail", and those imply different weights.
    """
    pose_enc = predictions["pose_enc_list"][-1].float()
    B, S, _, H, W = batch["images"].shape
    pr_extri, pr_intri = pose_encoding_to_extri_intri(pose_enc, (H, W), build_intrinsics=True)
    pr_depth = predictions["depth"][..., 0].float()
    gt_extri, gt_intri = batch["extrinsics"].float(), batch["intrinsics"].float()
    gt_depth = batch["depths"].float()
    coord = pixel_grid(H, W, batch["images"].device, torch.float32)
    ids, eps = batch["ids"], 1e-6

    errs = []
    for t in range(S - 1):
        dt = float(ids[0, t + 1] - ids[0, t])
        if dt <= 0 or dt > max_dt:
            continue
        for s_idx, t_idx in [(t, t + 1), (t + 1, t)]:
            R_pr, t_pr = relative_w2c(pr_extri[:, s_idx], pr_extri[:, t_idx])
            R_gt, t_gt = relative_w2c(gt_extri[:, s_idx], gt_extri[:, t_idx])
            f_pr, z_pr = ego_flow_from_disp(
                R_pr, t_pr, (1.0 / pr_depth[:, s_idx].clamp(min=eps))[:, None],
                pr_intri[:, t_idx], torch.linalg.inv(pr_intri[:, s_idx]), coord)
            f_gt, z_gt = ego_flow_from_disp(
                R_gt, t_gt, (1.0 / gt_depth[:, s_idx].clamp(min=eps))[:, None],
                gt_intri[:, t_idx], torch.linalg.inv(gt_intri[:, s_idx]), coord)
            m = batch["point_masks"][:, s_idx] & (z_pr > eps) & (z_gt > eps)
            if use_dynamic_mask:
                m = m & (batch["motion_mask"][:, s_idx] < dyn_thresh)
            if m.sum() < 100:
                continue
            errs.append((f_pr - f_gt).abs()[m[:, None].expand(-1, 2, -1, -1)].flatten())
    if not errs:
        return {}
    e = torch.cat(errs).float()
    q = torch.quantile(e[torch.randperm(e.numel(), device=e.device)[:2_000_000]],
                       torch.tensor([0.5, 0.9, 0.99], device=e.device))
    return {"p50": float(q[0]), "p90": float(q[1]), "p99": float(q[2]), "mean": float(e.mean())}


@torch.no_grad()
def score_clip(model, clip, args) -> Dict:
    batch = _batch_from_clip(clip, args.device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        predictions = model(batch["images"])
    if "depth" not in predictions:
        raise SystemExit("checkpoint has no depth head — L_ego_flow needs predicted depth")

    row: Dict[str, float] = {}
    for tag, use_mask in [("masked", True), ("nomask", False)]:
        out = compute_ego_flow_loss(
            predictions, batch, per_pixel_thre=args.per_pixel_thre, max_dt=args.max_dt,
            dyn_thresh=args.dyn_thresh, use_dynamic_mask=use_mask,
        )
        row[f"loss_{tag}"] = float(out["loss_ego_flow"])
        row[f"kept_{tag}"] = float(out["loss_ego_flow_kept"])
        row[f"pairs_{tag}"] = float(out["loss_ego_flow_pairs"])
        if tag == "masked":
            row.update({f"px_{k}": v for k, v in _residual_percentiles(
                predictions, batch, args.dyn_thresh, True, args.max_dt).items()})

    # Rejection sweep: how much of the residual sits above each threshold. per_pixel_thre is
    # in smooth-L1 units (beta=1 => ~ |d| - 0.5 for large |d|), not pixels.
    for thre in args.thre_sweep:
        out = compute_ego_flow_loss(predictions, batch, per_pixel_thre=thre,
                                    max_dt=args.max_dt, dyn_thresh=args.dyn_thresh)
        row[f"kept@{thre:g}"] = float(out["loss_ego_flow_kept"])

    # Far-pixel cap sweep. Distant pixels carry almost no translation signal (ego-flow's
    # translation term is disparity * t) while being where predicted depth is worst, so they
    # inflate this term most on far-heavy sets like Waymo. The cut is in units of the frame's
    # own median GT depth, which is scale-free and therefore comparable across datasets.
    for ratio in args.depth_ratio_sweep:
        out = compute_ego_flow_loss(predictions, batch, per_pixel_thre=args.per_pixel_thre,
                                    max_dt=args.max_dt, dyn_thresh=args.dyn_thresh,
                                    max_depth_ratio=ratio)
        row[f"loss_cap{ratio:g}"] = float(out["loss_ego_flow"])

    # NO loss_camera here, deliberately. An earlier version reported it as the reference for
    # setting loss.ego_flow.weight and was wrong by three orders of magnitude: clips loaded
    # straight from a dataset never pass through trainer._process_batch, which rescales GT
    # extrinsics and depths in lock-step to unit average distance. Without that, GT translations
    # are in metres while the model predicts in its normalised scale, and compute_camera_loss
    # measures the unit mismatch (it reported 5.67 on PO and 36.8 on Sintel against a real
    # training value of 0.0087 — the values sorted by scene scale in metres, which was the tell).
    # ego_flow itself is unaffected: it is scale-invariant and each side is self-consistent.
    # Calibrate the weight from a real training step instead (config/*_smoke.yaml).
    return row


def _agg(rows: List[Dict]) -> Dict[str, float]:
    keys = sorted({k for r in rows for k in r})
    return {k: float(np.median([r[k] for r in rows if k in r])) for k in keys}


def main():
    ap = argparse.ArgumentParser(description="Measure the GT-target ego-flow residual")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--datasets", nargs="*", default=["po", "tartanair", "waymo", "spring", "sintel"])
    ap.add_argument("--n_clips", type=int, default=10)
    ap.add_argument("--img_per_seq", type=int, default=16)
    ap.add_argument("--max_dt", type=int, default=5)
    ap.add_argument("--dyn_thresh", type=float, default=0.5)
    # Defaults to the value the training config uses, not MonST3R's 50: at 50 nothing is ever
    # rejected (measured keep rate 1.000 everywhere), so every share reported at 50 includes a
    # tail that training will not actually see.
    ap.add_argument("--per_pixel_thre", type=float, default=5.0)
    ap.add_argument("--thre_sweep", type=float, nargs="*", default=[1.0, 5.0, 20.0, 50.0])
    ap.add_argument("--depth_ratio_sweep", type=float, nargs="*", default=[10.0, 5.0, 3.0, 2.0],
                    help="max_depth_ratio values to sweep (x the frame's median GT depth)")
    ap.add_argument("--mix", nargs="*", default=["po=10000", "tartanair=5000", "waymo=4000",
                                                 "spring=1000", "sintel=0"],
                    help="training sampling counts per dataset (len_train in the config); the "
                         "weight suggestion is mixed by these. sintel is val-only -> 0")
    # dataset locations / probe args, reused by build_train_dataset & load_*_clip
    ap.add_argument("--po_dir", default=data_path("train", "point_odyssey"))
    ap.add_argument("--tartanair_dir", default=data_path("train", "tartanair"))
    ap.add_argument("--waymo_dir", default=data_path("train", "waymo_processed"))
    ap.add_argument("--spring_dir", default=data_path("train", "spring"))
    ap.add_argument("--sintel_root", default=None)
    ap.add_argument("--motion_thr", type=float, default=2.0)
    ap.add_argument("--max_depth", type=float, default=80.0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--img_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = load_vggt_for_eval(args.ckpt, device=args.device)

    results = {}
    for name in args.datasets:
        if name == "sintel":
            from data.sintel_io import SINTEL_EVAL_SEQUENCES, resolve_sintel_root
            root = resolve_sintel_root(args.sintel_root)
            jobs = [(s, lambda s=s: load_sintel_clip(root, s, args))
                    for s in SINTEL_EVAL_SEQUENCES[: args.n_clips]]
        else:
            ds = build_train_dataset(name, args)
            jobs = [(str(i), lambda i=i: load_train_clip(ds, i, args)) for i in range(args.n_clips)]

        rows = []
        for key, loader in jobs:
            try:
                rows.append(score_clip(model, loader(), args))
            except Exception as exc:                     # one bad clip must not kill the sweep
                print(f"  [{name}/{key}] skipped: {exc}")
        if not rows:
            continue
        results[name] = {"n_clips": len(rows), "median": _agg(rows)}
        m = results[name]["median"]
        print(f"\n== {name}  (n={len(rows)})")
        print(f"   loss_ego_flow  masked {m['loss_masked']:.4f}   nomask {m['loss_nomask']:.4f}"
              f"   (mask effect {100*(m['loss_nomask']-m['loss_masked'])/max(m['loss_masked'],1e-9):+.1f}%)")
        if "px_p50" in m:
            print(f"   |dflow| px     p50 {m['px_p50']:.3f}  p90 {m['px_p90']:.3f}  p99 {m['px_p99']:.3f}")
        print(f"   kept           " + "  ".join(f"@{t:g} {m.get(f'kept@{t:g}', float('nan')):.3f}"
                                                for t in args.thre_sweep))
        print("   depth cap      " + "  ".join(
            f"x{r:g} {m.get(f'loss_cap{r:g}', float('nan')):.3f}"
            for r in args.depth_ratio_sweep))

    if results:
        # Mix by the TRAINING sampling ratios, so the number describes what an average step
        # actually sees. Sintel is validation-only and contributes 0.
        mix = {}
        for spec in args.mix:
            name, _, cnt = spec.partition("=")
            mix[name] = float(cnt)
        used = {k: mix.get(k, 0.0) for k in results}
        tot = sum(used.values())
        if tot > 0:
            ego = sum(used[k] * results[k]["median"]["loss_masked"] for k in results) / tot
            results["_mix_weighted"] = {"mix": used, "loss_ego_flow": ego}
            print(f"\nmix-weighted loss_ego_flow = {ego:.4f}")
            print("  This is the term's MAGNITUDE, not a weight. Setting loss.ego_flow.weight "
                  "needs the other\n  terms' magnitudes at the same moment of training, which "
                  "only a real training step has:\n  run config/*_smoke.yaml and read the "
                  "weighted contributions off its log.")

    out_dir = args.out_dir or default_output_dir(args.ckpt, TOOL)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results.json")
    with open(path, "w") as f:
        json.dump({"created": datetime.now().isoformat(timespec="seconds"),
                   "args": vars(args), "results": results}, f, indent=1)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
