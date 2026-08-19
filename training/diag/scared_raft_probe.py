#!/usr/bin/env python3
"""Probe whether a RAFT flow-residual dynamic mask is viable on SCARED (endoscopy).

Question this answers (docs/scared_dataset.md §8): SCARED ships no dynamic annotation,
and the cheap option is the residual recipe the other datasets use --

    m*_raft = 1[ || f_RAFT - f_ego || > thr ],  f_ego from GT depth + GT w2c pose

Two reasons to doubt it here, both of which this script measures rather than assumes:

  1. RAFT is trained on natural imagery. Endoscopy (textureless mucosa, specular
     highlights, smoke, blood) is out of domain, so f_RAFT itself may be unusable --
     in which case a large residual says nothing about motion.
  2. Parallax is tiny: per-frame camera translation is ~0.25 mm against ~100 mm scene
     depth. f_ego is then nearly zero and the residual degenerates into "whatever RAFT
     did", which is why the frame GAP is swept instead of fixed at 1.

The discriminator between (1) and a working setup is the agreement between f_RAFT and
f_ego on the STATIC part of the scene. If the mask is meaningful, the two agree over
most pixels and disagree on a compact blob (the instrument); if RAFT is simply broken,
they disagree everywhere and the "dynamic fraction" is high at every threshold.

Read-only: writes nothing next to the data, only stats + panels under
outputs/scared_raft_probe/<seq>/.

Run (from training/):
  python diag/scared_raft_probe.py --seq ../data/train/scared/sample/test_stride1/dataset6/keyframe4
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))  # training/
import cv2
import numpy as np
import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from data.motion_mask import compute_ego_flow
from eval_utils.paths import output_dir_for_exp

TOOL = "scared_raft_probe"
DEPTH_SCALE = 100.0  # uint16 counts per mm (scared_convert.py)


def load_seq(seq_dir: str):
    fids = np.loadtxt(osp.join(seq_dir, "cam_data", "frames.txt"), dtype=np.int64, ndmin=1)
    E = np.loadtxt(osp.join(seq_dir, "cam_data", "extrinsics.txt")).reshape(-1, 3, 4)
    fx, fy, cx, cy = np.loadtxt(osp.join(seq_dir, "cam_data", "intrinsics.txt"))
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    assert len(fids) == len(E), f"{len(fids)} frames vs {len(E)} extrinsics"
    return fids, E, K


def read_rgb(seq_dir, fid):
    return cv2.imread(osp.join(seq_dir, "image_left", f"{fid:06d}.png"))  # BGR uint8


def read_depth(seq_dir, fid):
    d16 = cv2.imread(osp.join(seq_dir, "depth_left", f"{fid:06d}.png"), cv2.IMREAD_UNCHANGED)
    return d16.astype(np.float32) / DEPTH_SCALE  # mm, 0 = invalid


def scaled_intrinsics(seq_dir, fids, K, procW, procH):
    natH, natW = read_rgb(seq_dir, int(fids[0])).shape[:2]
    sx, sy = procW / natW, procH / natH
    Kp = K.copy()
    Kp[0] *= sx
    Kp[1] *= sy
    return Kp, sx, sy


@torch.no_grad()
def pair_residual(seq_dir, fids, E, Kp, model, tf, device, t, gap, procW, procH, sx, sy):
    """RAFT flow, ego flow and their residual for the pair (t, t+gap). None if unusable."""
    t2 = t + gap
    if t2 >= len(fids):
        return None
    fid, fid2 = int(fids[t]), int(fids[t2])

    def proc_rgb(f):
        img = cv2.cvtColor(read_rgb(seq_dir, f), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (procW, procH), interpolation=cv2.INTER_LINEAR)
        # keep uint8: weights.transforms() expects it (float [0,255] silently yields garbage)
        return torch.from_numpy(img).permute(2, 0, 1).contiguous()

    b1, b2 = tf(proc_rgb(fid)[None], proc_rgb(fid2)[None])
    flow = model(b1.to(device), b2.to(device))[-1][0].cpu().numpy().transpose(1, 2, 0)

    depth = read_depth(seq_dir, fid)
    depth = cv2.resize(depth, (procW, procH), interpolation=cv2.INTER_NEAREST)
    ego = compute_ego_flow(depth, Kp, E[t], E[t2])

    valid = depth > 0
    if valid.sum() < 1000:
        return None
    res = flow - ego
    # -> native px so thresholds are comparable with the other datasets' recipes
    res = np.stack([res[..., 0] / sx, res[..., 1] / sy], axis=-1)
    return np.linalg.norm(res, axis=-1), valid, flow, ego, fid, fid2


def build_panel(seq_dir, fid, fid2, gap, resid, valid, thr, procW, procH):
    mask = ((resid > thr) & valid).astype(np.uint8) * 255
    rgb = cv2.resize(read_rgb(seq_dir, fid), (procW, procH))
    vis_valid = cv2.cvtColor(valid.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    heat = cv2.applyColorMap(
        np.clip(resid / (thr * 3) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat[~valid] = 0
    panel = np.concatenate([rgb, vis_valid, heat, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)], axis=1)
    cv2.putText(panel, f"gap={gap} fid={fid}->{fid2}  RGB | valid-depth | "
                f"residual(0..{thr*3:.0f}px) | mask@{thr}px",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return panel


def run_gap(seq_dir, fids, E, K, model, tf, device, gap, anchors, procW, procH,
            thrs, vis_dir, vis_n):
    Kp, sx, sy = scaled_intrinsics(seq_dir, fids, K, procW, procH)

    rows, n_vis = [], 0
    for t in anchors:
        out = pair_residual(seq_dir, fids, E, Kp, model, tf, device, t, gap, procW, procH, sx, sy)
        if out is None:
            continue
        resid, valid, flow, ego, fid, fid2 = out

        nf = np.linalg.norm(flow, axis=-1)[valid] / max(sx, sy)
        ne = np.linalg.norm(ego, axis=-1)[valid] / max(sx, sy)
        # cosine between the two flow fields where BOTH are big enough to have a direction;
        # near-zero vectors have arbitrary angle and would only add noise.
        big = valid & (np.linalg.norm(flow, axis=-1) > 0.5) & (np.linalg.norm(ego, axis=-1) > 0.5)
        if big.sum() > 100:
            cos = float(np.median(
                (flow[big] * ego[big]).sum(-1)
                / (np.linalg.norm(flow[big], axis=-1) * np.linalg.norm(ego[big], axis=-1) + 1e-9)))
        else:
            cos = float("nan")

        rec = dict(t=int(t), fid=fid, fid2=fid2, valid_frac=float(valid.mean()),
                   flow_med=float(np.median(nf)), ego_med=float(np.median(ne)),
                   resid_med=float(np.median(resid[valid])),
                   resid_p90=float(np.percentile(resid[valid], 90)), cos_med=cos)
        for thr in thrs:
            rec[f"frac@{thr}"] = float(((resid > thr) & valid).sum() / max(valid.sum(), 1))
        rows.append(rec)

        if vis_dir and n_vis < vis_n:
            panel = build_panel(seq_dir, fid, fid2, gap, resid, valid, thrs[1], procW, procH)
            cv2.imwrite(osp.join(vis_dir, f"gap{gap:02d}_{fid:06d}.png"), panel)
            n_vis += 1
    return rows


def run_gif(seq_dir, fids, E, K, model, tf, device, gap, t0, t1, procW, procH, thr,
            out_path, scale, duration_ms):
    """Same panel as run_gap, but over CONSECUTIVE anchors so the mask can be scrubbed."""
    from PIL import Image

    Kp, sx, sy = scaled_intrinsics(seq_dir, fids, K, procW, procH)
    frames = []
    for t in range(t0, min(t1, len(fids) - gap)):
        out = pair_residual(seq_dir, fids, E, Kp, model, tf, device, t, gap,
                            procW, procH, sx, sy)
        if out is None:
            continue
        resid, valid, _, _, fid, fid2 = out
        panel = build_panel(seq_dir, fid, fid2, gap, resid, valid, thr, procW, procH)
        if scale != 1.0:
            panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)))
    if not frames:
        return None
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)
    return len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="a converted SCARED keyframe directory")
    ap.add_argument("--gaps", nargs="*", type=int, default=[1, 5, 10, 20, 30])
    ap.add_argument("--thrs", nargs="*", type=float, default=[1.0, 2.0, 3.0, 5.0, 10.0])
    ap.add_argument("--n_anchors", type=int, default=40, help="frames probed per gap")
    ap.add_argument("--proc_scale", type=float, default=0.5, help="RAFT input scale")
    ap.add_argument("--vis_n", type=int, default=6, help="panels saved per gap")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--gif", action="store_true",
                    help="also write an animated panel over CONSECUTIVE frames (one gap)")
    ap.add_argument("--gif_gap", type=int, default=5)
    ap.add_argument("--gif_thr", type=float, default=2.0)
    ap.add_argument("--gif_range", default="0:200", help="'a:b' in frame-list index space")
    ap.add_argument("--gif_scale", type=float, default=0.5, help="panel downscale for the GIF")
    ap.add_argument("--gif_duration_ms", type=int, default=120)
    ap.add_argument("--no_sweep", action="store_true", help="skip the gap sweep (GIF only)")
    args = ap.parse_args()

    seq_dir = osp.abspath(args.seq)
    tag = "_".join(seq_dir.rstrip("/").split("/")[-3:])  # split_dataset_keyframe
    out_dir = args.out_dir or output_dir_for_exp(tag, TOOL)
    vis_dir = osp.join(out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    fids, E, K = load_seq(seq_dir)
    natH, natW = read_rgb(seq_dir, int(fids[0])).shape[:2]
    procW = int(round(natW * args.proc_scale)) // 8 * 8
    procH = int(round(natH * args.proc_scale)) // 8 * 8

    device = "cuda"
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False).to(device).eval()
    tf = weights.transforms()

    print(f"{seq_dir}\n  {len(fids)} frames  native {natW}x{natH} -> RAFT {procW}x{procH}"
          f"  gaps={args.gaps}  anchors/gap={args.n_anchors}")

    summary = {}
    for gap in (args.gaps if not args.no_sweep else []):
        anchors = np.linspace(0, max(len(fids) - gap - 1, 0), args.n_anchors).astype(int)
        anchors = np.unique(anchors)
        rows = run_gap(seq_dir, fids, E, K, model, tf, device, gap, anchors,
                       procW, procH, args.thrs, vis_dir, args.vis_n)
        if not rows:
            continue
        agg = {k: float(np.nanmedian([r[k] for r in rows]))
               for k in rows[0] if k not in ("t", "fid", "fid2")}
        summary[gap] = dict(n=len(rows), **agg)
        print(f"\ngap={gap:3d}  n={len(rows)}")
        print(f"  |f_RAFT| med {agg['flow_med']:7.2f} px | |f_ego| med {agg['ego_med']:7.2f} px"
              f" | cos(f_RAFT,f_ego) med {agg['cos_med']:6.3f}")
        print(f"  residual med {agg['resid_med']:6.2f} px  p90 {agg['resid_p90']:7.2f} px"
              f"  (valid-depth frac {agg['valid_frac']:.2f})")
        print("  dynamic fraction: " + "  ".join(
            f"@{t:g}px {agg[f'frac@{t}']:.3f}" for t in args.thrs))

    if summary:
        with open(osp.join(out_dir, "summary.json"), "w") as f:
            json.dump(dict(seq=seq_dir, proc=[procW, procH], thrs=args.thrs, per_gap=summary),
                      f, indent=2)
        print(f"\nsummary -> {osp.join(out_dir, 'summary.json')}\npanels  -> {vis_dir}")

    if args.gif:
        a, b = (int(x) for x in args.gif_range.split(":"))
        gif_path = osp.join(out_dir, f"mask_gap{args.gif_gap:02d}_thr{args.gif_thr:g}_{a}-{b}.gif")
        n = run_gif(seq_dir, fids, E, K, model, tf, device, args.gif_gap, a, b,
                    procW, procH, args.gif_thr, gif_path, args.gif_scale, args.gif_duration_ms)
        print(f"\ngif ({n} frames) -> {gif_path}")


if __name__ == "__main__":
    main()
