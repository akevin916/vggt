#!/usr/bin/env python3
"""How observable is camera pose from optical flow, per sequence?

Motivating puzzle (docs/table.md table 5 + flow_pose_headroom.py): minimising a flow
residual improved `market_5`/`market_6` by ~36% ATE but did essentially nothing for
`cave_2` -- even though `cave_2` has the WORST RPE of all 14 sequences (rot 1.224,
trans 0.322) and a comparable dynamic fraction. Its flow residual is small at every
percentile (p50 0.50, p90 1.38) while `market_5`'s is 1.30 / 11.03.

Four indirect explanations were tried against the data and all failed or came out weak:
accumulated drift (contradicted by RPE, which is a LOCAL metric and is the worst there),
depth diversity (cave_2's static p90/p10 is 13.0, better than most), absolute disparity
(corr -0.25), and texture (cave_2's static-pixel gradient is HIGHER than market_5's).

Each of those was a proxy for the same underlying claim, so this script measures the claim
itself:

    if the pose is perturbed by a fixed amount, how much does the ego-flow change?

Low sensitivity means a wrong pose produces almost the same flow field -- the residual is
flat along that direction, so no flow-based objective can recover it, and the sequence's
pose error is invisible to the loss no matter how it is weighted.

Deliberately target-free: it perturbs the pose and compares ego-flow to ego-flow. No RAFT,
no GT flow. The conclusion therefore applies to BOTH the RAFT-target port
(flow_pose_headroom.py) and the GT-derived L_ego_flow -- an unobservable direction is
unobservable regardless of what the residual is measured against.

Rotation and translation are perturbed separately, because the geometry predicts they
behave differently: the rotation term of the ego-flow carries no depth, while the
translation term is scaled by disparity, so a far-away static reference weakens translation
observability specifically. `cave_2`'s static-pixel disparity is 0.076 vs `market_5`'s
0.204 -- if that is the mechanism, its translation sensitivity should be the outlier while
its rotation sensitivity is ordinary.

Run from training/:
  python diag/flow_observability.py --ckpt logs/<run>/ckpts/epoch_30.pt
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
from tqdm import tqdm

# Shared differentiable geometry -- this script publishes no frozen numbers, so unlike
# flow_loss_probe.py it should track ego_flow.py rather than keep a private copy.
from ego_flow import ego_flow_from_disp, pixel_grid, relative_w2c
from diag.flow_loss_probe import load_sintel_clip
from eval_utils.paths import default_output_dir
from eval_utils.vggt_infer import load_vggt_for_eval
from data.sintel_io import SINTEL_EVAL_SEQUENCES, resolve_sintel_root
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

FLOW_OBSERVABILITY = "flow_observability"


def _so3_exp(w: torch.Tensor) -> torch.Tensor:
    """Rodrigues: (N,3) axis-angle -> (N,3,3)."""
    theta = w.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    k = w / theta
    K = torch.zeros(w.shape[0], 3, 3, device=w.device, dtype=w.dtype)
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(K)
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    return I + s * K + (1 - c) * (K @ K)


def _rand_unit(n: int, device, gen) -> torch.Tensor:
    v = torch.randn(n, 3, device=device, generator=gen)
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12)


@torch.no_grad()
def observability(model, sintel_root, seq, args) -> Dict:
    dev = args.device
    clip = load_sintel_clip(sintel_root, seq, args)

    images = clip["img_u8"].float().div(255)[None].to(dev)
    _, _, _, H, W = images.shape
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = model(images)
    pose_enc = pred["pose_enc_list"][-1].float() if "pose_enc_list" in pred else pred["pose_enc"].float()
    extri, intri = pose_encoding_to_extri_intri(pose_enc, (H, W))
    extri, intri = extri[0], intri[0]
    if "depth" not in pred:
        raise RuntimeError("checkpoint predicts no depth")
    depth = pred["depth"][0, ..., 0].float()
    S = depth.shape[0]
    del pred, images
    torch.cuda.empty_cache()

    disp = (1.0 / depth.clamp(min=1e-6))[:, None]
    inv_intri = torch.linalg.inv(intri)
    coord = pixel_grid(H, W, torch.device(dev))
    ia, ib = torch.arange(S - 1), torch.arange(1, S)

    # Same pixel set the loss sees: static AND valid GT depth, on the source frame.
    m = ((clip["motion"] < args.dyn_thresh) & clip["pmask"])[ia]

    def mean_dflow(ext_pert) -> float:
        """Mean |flow(perturbed) - flow(base)| over the masked pixels.

        Chunked over pairs and reducing to a running sum: holding both flow fields for a
        50-frame sequence is ~140 MB, which is enough to OOM when a training run shares the
        GPU. The base flow is recomputed per chunk rather than cached -- it is a few matmuls
        and buys back the memory.
        """
        tot, cnt = 0.0, 0
        for i in range(0, len(ia), args.pair_chunk):
            sa, sb = ia[i : i + args.pair_chunk], ib[i : i + args.pair_chunk]
            R0, t0 = relative_w2c(extri[sa], extri[sb])
            f0, z0 = ego_flow_from_disp(R0, t0, disp[sa], intri[sb], inv_intri[sa], coord)
            R1, t1 = relative_w2c(ext_pert[sa], ext_pert[sb])
            f1, z1 = ego_flow_from_disp(R1, t1, disp[sa], intri[sb], inv_intri[sa], coord)
            v = m[i : i + args.pair_chunk] & (z0 > 1e-3) & (z1 > 1e-3)
            if v.sum() == 0:
                continue
            d = (f1 - f0).norm(dim=1)
            tot += float(d[v].sum())
            cnt += int(v.sum())
            del f0, f1, z0, z1, d
        return tot / cnt if cnt else float("nan")

    # Translation perturbations are sized relative to the trajectory extent so the number is
    # comparable across sequences (VGGT's scene scale is per-sequence normalised).
    t_scale = extri[:, :3, 3].norm(dim=-1).mean().clamp(min=1e-6)
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    def sensitivity(kind: str) -> float:
        acc = []
        for _ in range(args.n_draws):
            ext = extri.clone()
            if kind == "rot":
                w = _rand_unit(S, dev, gen) * args.eps_rot
                ext[:, :3, :3] = _so3_exp(w) @ extri[:, :3, :3]
            else:
                ext[:, :3, 3] = extri[:, :3, 3] + _rand_unit(S, dev, gen) * (args.eps_trans * t_scale)
            v = mean_dflow(ext)
            if np.isfinite(v):
                acc.append(v)
        return float(np.mean(acc)) if acc else float("nan")

    stat_disp = disp[ia][:, 0][m]
    disp_med = float(stat_disp.median()) if stat_disp.numel() else float("nan")
    return {
        "seq": seq,
        "n_frames": S,
        "sens_rot": sensitivity("rot"),
        "sens_trans": sensitivity("trans"),
        "static_frac": float(m.float().mean()),
        "disp_med": disp_med,
        "t_scale": float(t_scale),
        # disp_med alone is NOT comparable across sequences: VGGT normalises scene scale per
        # sequence, so it is in arbitrary per-sequence units. disp * t_scale is the
        # dimensionless parallax the trajectory actually produces, and IS comparable --
        # scaling the scene by s divides disp by s and multiplies t_scale by s.
        "parallax": disp_med * float(t_scale),
    }


def main():
    ap = argparse.ArgumentParser(description="Per-sequence pose observability from ego-flow")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sintel_root", default=None)
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--img_per_seq", type=int, default=50)
    ap.add_argument("--eps_rot", type=float, default=0.005, help="radians per frame")
    ap.add_argument("--eps_trans", type=float, default=0.01, help="fraction of trajectory extent")
    ap.add_argument("--n_draws", type=int, default=20, help="random perturbation directions to average")
    ap.add_argument("--pair_chunk", type=int, default=8, help="frame pairs per chunk (lower if OOM)")
    ap.add_argument("--dyn_thresh", type=float, default=0.5)
    ap.add_argument("--motion_thr", type=float, default=2.0)
    ap.add_argument("--max_depth", type=float, default=80.0)
    ap.add_argument("--img_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    args.out_dir = args.out_dir or os.path.join(
        default_output_dir(args.ckpt, FLOW_OBSERVABILITY),
        os.path.splitext(os.path.basename(args.ckpt))[0],
    )
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output dir: {args.out_dir}")

    root = resolve_sintel_root(args.sintel_root)
    seqs = args.seqs or SINTEL_EVAL_SEQUENCES
    model = load_vggt_for_eval(args.ckpt, img_size=args.img_size, device=args.device)

    res: List[Dict] = []
    for s in tqdm(seqs, desc="sequences"):
        try:
            res.append(observability(model, root, s, args))
        except Exception as e:
            tqdm.write(f"  {s}: skipped ({type(e).__name__}: {e})")
    if not res:
        print("nothing measured")
        return

    print(f"\n{'='*78}")
    print(f"FLOW SENSITIVITY  (mean |dflow| in px; rot {args.eps_rot} rad, "
          f"trans {args.eps_trans:.0%} of trajectory, {args.n_draws} draws)")
    print("="*78)
    print(f"{'seq':<13}{'旋轉':>9}{'平移':>9}{'平移/旋轉':>11}{'視差量':>10}{'靜態%':>8}")
    for r in sorted(res, key=lambda x: x["sens_trans"]):
        ratio = r["sens_trans"] / r["sens_rot"] if r["sens_rot"] else float("nan")
        print(f"{r['seq']:<13}{r['sens_rot']:>9.3f}{r['sens_trans']:>9.3f}{ratio:>11.3f}"
              f"{r['parallax']:>10.3f}{r['static_frac']*100:>7.1f}%")

    d = np.array([r["parallax"] for r in res])
    st = np.array([r["sens_trans"] for r in res])
    sr = np.array([r["sens_rot"] for r in res])
    ok = np.isfinite(d) & np.isfinite(st)
    if ok.sum() >= 4:  # a correlation over 2-3 points is not evidence of anything
        print(f"\ncorr( 視差量 , 平移敏感度 ) = {np.corrcoef(d[ok], st[ok])[0,1]:+.3f}"
              "   （幾何預測：強正相關，平移項 ∝ 視差）")
        print(f"corr( 視差量 , 旋轉敏感度 ) = {np.corrcoef(d[ok], sr[ok])[0,1]:+.3f}"
              "   （幾何預測：≈0，旋轉項不含深度）")
    else:
        print(f"\n（序列數 {int(ok.sum())} < 4，不印相關係數）")

    out = {"created": datetime.now().isoformat(timespec="seconds"), "args": vars(args), "sequences": res}
    path = os.path.join(args.out_dir, "results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
