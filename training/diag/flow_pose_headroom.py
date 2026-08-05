#!/usr/bin/env python3
"""Upper-bound probe: if the MonST3R flow residual were driven to its floor, what ATE?

`flow_loss_probe.py` established two facts on Sintel: the model's per-pair ego-flow
disagrees with RAFT by 4.6x the achievable floor (0.46 vs 0.10 px), and NO current loss
touches that residual -- it is identical for VGGT-1B and for the ep30 checkpoint whose ATE
is 21.6% better. So the residual is a large, measured, completely un-targeted error.

Un-targeted is not the same as useful. Before writing a training loss for it we need to
know whether reducing it moves ATE at all. This script answers that by *directly
minimising* it: start from the model's predicted poses on a Sintel sequence, optimise them
(and optionally depth) against RAFT flow, and score ATE at every step.

This is test-time optimisation, but used the same way the oracle-mask modes are used in
gate_bias_ablation.py -- as an UPPER BOUND, not as the method. It answers "if we could
perfectly minimise this, where would we land", which bounds anything a training loss for
it could ever achieve.

Reading the result:
  ATE 0.134 -> ~0.11 or below   the axis is productive AND closes the MonST3R gap
                                -> worth building a training loss for it
  ATE barely moves              the axis is inert for ATE; a flow loss would only move
                                RPE-trans -> the claim degrades to a secondary metric
  ATE gets worse                flow residual and ATE are adversarial -> close the line

Only the FINAL predictions are optimised; the network is never touched. Run from training/:
  python diag/flow_pose_headroom.py --ckpt logs/<run>/ckpts/epoch_30.pt
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
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from tqdm import tqdm

# Reuse the probe's verified geometry/loss ports so the quantity being minimised here is
# byte-identical to the one that was measured there.
from diag.flow_loss_probe import (
    _pixel_grid,
    _raft_flow,
    _relative_w2c,
    ego_flow_disparity_form,
    load_sintel_clip,
    smooth_L1_loss_fn,
)
from eval_utils.metrics_pose import eval_pose_metrics
from eval_utils.paths import default_output_dir
from eval_utils.vggt_infer import load_vggt_for_eval
from data.sintel_io import (
    SINTEL_EVAL_SEQUENCES,
    load_sintel_gt_poses,
    load_sintel_rgb_paths,
    resolve_sintel_root,
    sintel_seq_paths,
)
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

FLOW_POSE_HEADROOM = "flow_pose_headroom"


def relative_pose_loss_w2c(R_rel, t_rel, translation_weight=0.1):
    """monst3r optimizer.py:1014 (`relative_pose_loss`), expressed on the relative transform
    we already compute. First-order: it penalises the inter-frame motion ITSELF, so the
    prior is "the camera barely moves" -- not the second-order/acceleration form used by
    our training-side compute_camera_smooth_loss."""
    I = torch.eye(3, device=R_rel.device, dtype=R_rel.dtype).expand_as(R_rel)
    rot = torch.norm(R_rel - I, dim=(1, 2))
    trans = torch.norm(t_rel[..., 0], dim=1)
    return (rot + translation_weight * trans).sum()


def depth_regularization_si_weighted(depth_pred, depth_init, pixel_wise_weight=None,
                                     pixel_wise_weight_scale=1.0, pixel_wise_weight_bias=1.0,
                                     eps=1e-6):
    """monst3r goem_opt.py:16, verbatim. Scale-invariant pull back toward the INITIAL depth:
    fits the best global log-offset first, so it constrains structure without pinning scale."""
    depth_pred = depth_pred.clamp(min=eps)
    depth_init = depth_init.clamp(min=eps)
    log_p, log_i = torch.log(depth_pred), torch.log(depth_init)
    B, _, H, W = depth_pred.shape
    scale = torch.sum(log_i - log_p, dim=[1, 2, 3], keepdim=True) / (H * W)
    if pixel_wise_weight is not None:
        pixel_wise_weight = pixel_wise_weight * pixel_wise_weight_scale + pixel_wise_weight_bias
    else:
        pixel_wise_weight = 1.0
    si = torch.sum(pixel_wise_weight * (log_p - log_i + scale) ** 2, dim=[1, 2, 3]) / (H * W)
    return si.mean()


def _so3_exp(w: torch.Tensor) -> torch.Tensor:
    """Rodrigues: (N,3) axis-angle -> (N,3,3). Used as a DELTA on the predicted rotation so
    that w=0 starts the optimisation exactly at the model's output (the ATE at step 0 must
    reproduce the benchmark number, which is the script's built-in sanity check)."""
    theta = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = w / theta
    K = torch.zeros(w.shape[0], 3, 3, device=w.device, dtype=w.dtype)
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(K)
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    return I + s * K + (1 - c) * (K @ K)


@torch.no_grad()
def _predict(model, clip, args):
    """One forward pass; returns the initial extrinsics / intrinsics / depth to optimise."""
    images = clip["img_u8"].float().div(255)[None].to(args.device)
    _, _, _, H, W = images.shape
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = model(images)
    pose_enc = pred["pose_enc_list"][-1].float() if "pose_enc_list" in pred else pred["pose_enc"].float()
    extri, intri = pose_encoding_to_extri_intri(pose_enc, (H, W))
    if "depth" not in pred:
        raise RuntimeError("checkpoint predicts no depth; ego-flow needs it")
    return extri[0], intri[0], pred["depth"][0, ..., 0].float()


def optimise_sequence(model, raft, tf, sintel_root, seq, args) -> Dict:
    dev = args.device
    clip = load_sintel_clip(sintel_root, seq, args)
    extri0, intri, depth0 = _predict(model, clip, args)
    S, H, W = depth0.shape

    rgb_paths = load_sintel_rgb_paths(sintel_root, seq)[: args.img_per_seq]
    _, _, cam_dir = sintel_seq_paths(sintel_root, seq)
    gt_tum, gt_ts = load_sintel_gt_poses(cam_dir, rgb_paths)

    # Consecutive pairs only: flow_loss_probe showed the RAFT floor blows up past dt=1
    # (TartanAir 2.0 -> 11.3 px, Waymo 0.54 -> 3.8 px), so wider pairs would optimise noise.
    ia = torch.arange(S - 1)
    ib = torch.arange(1, S)
    flow_fwd = _raft_flow(raft, tf, clip["img_u8"], ia, ib, dev).detach()
    flow_bwd = _raft_flow(raft, tf, clip["img_u8"], ib, ia, dev).detach()

    dyn = (clip["motion"] >= args.dyn_thresh)
    if args.mask_dilate > 0:
        # Sintel's m_geo IS already GT-derived (|gt_flow - ego_flow(GT)| > motion_thr), so
        # there is no better mask to swap in. What can still be wrong is its EXTENT: a
        # threshold tuned for the object interior misses the penumbra (motion boundaries,
        # shadows, slow limbs). Dilating tests exactly that -- if cave_2/temple_3 stop
        # degrading, under-masking was the cause.
        k = 2 * args.mask_dilate + 1
        dyn = F.max_pool2d(dyn.float()[:, None], k, stride=1, padding=args.mask_dilate)[:, 0] > 0
    static = (~dyn) & clip["pmask"]
    m_fwd = static[ia].float()[:, None]
    m_bwd = static[ib].float()[:, None]
    dyn_f = dyn.float()[:, None]
    coord = _pixel_grid(H, W, dev, torch.float32)

    # Parameters: a delta on top of the prediction, so step 0 == the model's own output.
    dw = torch.zeros(S, 3, device=dev, requires_grad=True)
    dt = torch.zeros(S, 3, device=dev, requires_grad=True)
    params = [dw, dt]
    log_ds = torch.zeros(S, H, W, device=dev, requires_grad=args.optimise_depth)
    if args.optimise_depth:
        params.append(log_ds)
    opt = torch.optim.Adam(params, lr=args.lr)

    R0, T0 = extri0[:, :3, :3], extri0[:, :3, 3]
    inv_intri = torch.linalg.inv(intri)

    def current_extrinsics():
        R = _so3_exp(dw) @ R0
        return R, T0 + dt

    # Anchor scale: dw is in radians, dt in VGGT world units, so the translation part is
    # normalised by the trajectory extent to make w_anchor dimensionless and comparable
    # across sequences.
    t_scale = T0.norm(dim=-1).mean().clamp(min=1e-6).detach()

    def objective():
        R, T = current_extrinsics()
        ext = torch.cat([R, T[..., None]], dim=-1)
        depth = depth0 * torch.exp(log_ds)
        disp = 1.0 / depth.clamp(min=1e-6)

        flow_term, smooth_term, n = 0.0, 0.0, 0
        for src, tgt, flow, m in [(ia, ib, flow_fwd, m_fwd), (ib, ia, flow_bwd, m_bwd)]:
            R_rel, t_rel = _relative_w2c(ext[src], ext[tgt])
            ego = ego_flow_disparity_form(
                R_rel, t_rel, disp[src][:, None], intri[tgt], inv_intri[src], coord
            )
            # Same masked smooth-L1 as monst3r, but differentiable end-to-end here.
            raw = F.smooth_l1_loss(ego * m, flow * m, beta=1.0, reduction="none")
            keep = ((raw < args.per_pixel_thre) * m).detach()
            flow_term = flow_term + (raw * keep).sum() / keep.sum().clamp(min=1.0)
            if n == 0:  # smoothing is defined on consecutive pairs once, not per direction
                smooth_term = relative_pose_loss_w2c(R_rel, t_rel)
            n += 1
        flow_term = flow_term / n

        # Replaces monst3r's (li + lj): that term stitches PAIRWISE pointmaps into a global
        # solution -- machinery VGGT does not need, since one forward pass already yields a
        # globally consistent one. Its second role, anchoring the solution near the network's
        # output, is what must be carried over, and that is this term.
        anchor = dw.pow(2).mean() + (dt / t_scale).pow(2).mean()

        dprior = (
            depth_regularization_si_weighted(depth[:, None], depth0[:, None].detach(), dyn_f)
            if args.w_dprior > 0 else torch.zeros((), device=dev)
        )
        total = (args.w_flow * flow_term + args.w_anchor * anchor
                 + args.w_smooth * smooth_term + args.w_dprior * dprior)
        return total, flow_term

    def score():
        R, T = current_extrinsics()
        ext = torch.cat([R, T[..., None]], dim=-1).detach().cpu().numpy().astype(np.float64)
        return eval_pose_metrics(ext, gt_tum[: len(ext)], gt_ts[: len(ext)])

    hist = []
    for it in range(args.iters + 1):
        loss, flow_term = objective()
        if it % args.eval_every == 0 or it == args.iters:
            hist.append({"iter": it, "flow_loss": float(flow_term.detach()),
                         "total": float(loss.detach()), **score()})
        if it == args.iters:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()

    return {"seq": seq, "n_frames": S, "history": hist,
            "ate_init": hist[0]["ate"], "ate_final": hist[-1]["ate"],
            "flow_init": hist[0]["flow_loss"], "flow_final": hist[-1]["flow_loss"]}


def main():
    ap = argparse.ArgumentParser(description="ATE headroom of the MonST3R flow residual")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sintel_root", default=None)
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--img_per_seq", type=int, default=50, help="frames per sequence (Sintel mean 45.9)")
    ap.add_argument("--iters", type=int, default=300, help="monst3r uses 300")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--optimise_depth", action="store_true",
                    help="also free the depth (monst3r does); pose-only is the default so the "
                         "result attributes cleanly to pose")
    # Objective weights. Defaults reproduce the flow-only run (no anchor / smoothing / prior),
    # which is the w_anchor -> 0 end of the sweep. monst3r's own ratios put flow and temporal
    # smoothing at 0.01 against a backbone of 1.0; here the backbone is `anchor`, so sweeping
    # w_anchor upward from 0 walks the same axis.
    ap.add_argument("--w_flow", type=float, default=1.0)
    ap.add_argument("--w_anchor", type=float, default=0.0,
                    help="pull pose back to the VGGT prediction; replaces monst3r's (li+lj)")
    ap.add_argument("--w_smooth", type=float, default=0.0,
                    help="monst3r relative_pose_loss (first-order 'camera barely moves')")
    ap.add_argument("--w_dprior", type=float, default=0.0,
                    help="scale-invariant pull of depth back to the prediction; needs --optimise_depth")
    ap.add_argument("--mask_dilate", type=int, default=0,
                    help="dilate the dynamic mask by N px before excluding it (tests under-masking)")
    ap.add_argument("--per_pixel_thre", type=float, default=50.0)
    ap.add_argument("--dyn_thresh", type=float, default=0.5)
    ap.add_argument("--motion_thr", type=float, default=2.0)
    ap.add_argument("--max_depth", type=float, default=80.0)
    ap.add_argument("--img_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    args.out_dir = args.out_dir or os.path.join(
        default_output_dir(args.ckpt, FLOW_POSE_HEADROOM),
        os.path.splitext(os.path.basename(args.ckpt))[0],
    )
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output dir: {args.out_dir}")

    root = resolve_sintel_root(args.sintel_root)
    seqs = args.seqs or SINTEL_EVAL_SEQUENCES
    model = load_vggt_for_eval(args.ckpt, img_size=args.img_size, device=args.device)
    weights = Raft_Large_Weights.DEFAULT
    raft = raft_large(weights=weights, progress=False).to(args.device).eval()
    tf = weights.transforms()

    results: List[Dict] = []
    for seq in tqdm(seqs, desc="sequences"):
        try:
            results.append(optimise_sequence(model, raft, tf, root, seq, args))
        except Exception as e:
            tqdm.write(f"  {seq}: skipped ({type(e).__name__}: {e})")

    if not results:
        print("nothing optimised")
        return

    print(f"\n{'='*74}\nPER-SEQUENCE  (pose{'+depth' if args.optimise_depth else ' only'}, {args.iters} iters)\n{'='*74}")
    print(f"{'seq':<14}{'n':>4}{'ATE init':>10}{'ATE final':>11}{'Δ%':>9}{'flow init':>11}{'flow final':>11}")
    for r in results:
        d = 100 * (r["ate_final"] - r["ate_init"]) / max(r["ate_init"], 1e-9)
        print(f"{r['seq']:<14}{r['n_frames']:>4}{r['ate_init']:>10.4f}{r['ate_final']:>11.4f}"
              f"{d:>8.1f}%{r['flow_init']:>11.3f}{r['flow_final']:>11.3f}")

    # ATE(12) -- the project's outlier-robust pose metric (drops cave_2 / temple_3), which is
    # where the flow term's effect is legible; the 14-seq mean is dominated by those two.
    OUT = ("cave_2", "temple_3")
    keep = [r for r in results if r["seq"] not in OUT]
    if keep:
        k_i = float(np.mean([r["ate_init"] for r in keep]))
        k_f = float(np.mean([r["ate_final"] for r in keep]))
        print(f"\n{'ATE(12)':<14}{'':>4}{k_i:>10.4f}{k_f:>11.4f}{100*(k_f-k_i)/k_i:>8.1f}%")

    ai = float(np.mean([r["ate_init"] for r in results]))
    af = float(np.mean([r["ate_final"] for r in results]))
    fi = float(np.mean([r["flow_init"] for r in results]))
    ff = float(np.mean([r["flow_final"] for r in results]))
    print(f"\n{'MEAN':<14}{'':>4}{ai:>10.4f}{af:>11.4f}{100*(af-ai)/ai:>8.1f}%{fi:>11.3f}{ff:>11.3f}")
    print(f"\nflow residual reduced by {100*(ff-fi)/max(fi,1e-9):.1f}%  ->  ATE moved {100*(af-ai)/ai:+.1f}%")
    print("MonST3R reference (本機實測, 14 seq): ATE 0.1110")

    out = {"created": datetime.now().isoformat(timespec="seconds"), "args": vars(args),
           "mean": {"ate_init": ai, "ate_final": af, "flow_init": fi, "flow_final": ff},
           "sequences": results}
    # Encode the objective in the filename so a weight sweep does not overwrite itself.
    tag = "pose_depth" if args.optimise_depth else "pose"
    for k, v in [("a", args.w_anchor), ("s", args.w_smooth), ("d", args.w_dprior)]:
        if v > 0:
            tag += f"_{k}{v:g}"
    if args.mask_dilate:
        tag += f"_dil{args.mask_dilate}"
    path = os.path.join(args.out_dir, f"results_{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
