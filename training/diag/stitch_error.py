#!/usr/bin/env python3
"""What does Sim3 stitching cost, in ATE?

benchmark/eval_scared.py can only score SCARED's 411/834-frame pose trajectories by cutting
them into overlapping chunks and joining each seam with a similarity transform
(eval_utils.vggt_infer.infer_sequence_stitched). That join is not free, and a stitched ATE
is only publishable once we know how much of it is the model and how much is the seams.

So: take a CONTIGUOUS window short enough for one forward pass, score it both ways against
the same GT, and report the gap. Contiguous matters -- an evenly-subsampled window has larger
inter-frame motion than the chunks the real run will actually form, which would flatter or
punish the stitch for the wrong reason.

Run (from training/):
  python diag/stitch_error.py --ckpt logs/scared_cam_b16_gg/ckpts/best_ate.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_TRAINING_DIR, os.path.dirname(_TRAINING_DIR)]

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from data.paths import data_path
from eval_utils.metrics_pose import eval_pose_metrics, snippet_pose_metrics
from eval_utils.paths import default_output_dir
from eval_utils.vggt_infer import (infer_sequence, infer_sequence_stitched,
                                   umeyama_sim3, _extrinsic_to_c2w)

TOOL = "stitch_error"


def gt_tum(E: np.ndarray, fids: np.ndarray):
    tum, ts = [], []
    for k in range(len(E)):
        c2w = np.linalg.inv(np.vstack([E[k], [0, 0, 0, 1]]))
        q = Rotation.from_matrix(c2w[:3, :3]).as_quat()
        tum.append(np.concatenate([c2w[:3, 3], [q[3], q[0], q[1], q[2]]]))
        ts.append(float(fids[k]))
    tum = np.stack(tum)
    tum[:, :3] -= tum[:, :3].mean(0, keepdims=True)
    return tum, np.array(ts)[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/scared_cam_b16_gg/ckpts/best_ate.pt")
    ap.add_argument("--seq", default="pose_seq/dataset3/keyframe4")
    ap.add_argument("--scared_root", default=data_path("train", "scared"))
    ap.add_argument("--n_frames", type=int, default=80, help="contiguous window, one pass")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--configs", nargs="*", default=["32:16", "32:8", "40:20", "48:16"],
                    help="chunk:overlap pairs to compare against the single pass")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    seq_dir = os.path.join(args.scared_root, args.seq)
    fids = np.loadtxt(os.path.join(seq_dir, "cam_data", "frames.txt"), dtype=np.int64, ndmin=1)
    E = np.loadtxt(os.path.join(seq_dir, "cam_data", "extrinsics.txt")).reshape(-1, 3, 4)
    sl = slice(args.start, args.start + args.n_frames)
    fids, E = fids[sl], E[sl]
    paths = [os.path.join(seq_dir, "image_left", f"{int(f):06d}.png") for f in fids]
    tum, ts = gt_tum(E, fids)
    C_gt = _extrinsic_to_c2w(E)[:, :3, 3]
    extent = float(np.linalg.norm(C_gt.max(0) - C_gt.min(0)))
    print(f"{args.seq} frames {args.start}..{args.start + len(paths) - 1} "
          f"({len(paths)} contiguous), GT bbox {extent:.2f} mm")

    from eval_utils.vggt_infer import load_vggt_for_eval
    model = load_vggt_for_eval(args.ckpt, device=args.device)

    def score(pred, label):
        m = eval_pose_metrics(pred["extrinsic"], tum, ts)
        m.update(snippet_pose_metrics(pred["extrinsic"], E))
        m["ate_rel_pct"] = 100 * m["ate"] / extent
        print(f"{label:<24}ATE {m['ate']:8.4f} mm ({m['ate_rel_pct']:5.2f}% of extent)   "
              f"snipATE {m['snippet_ate']:8.4f}")
        return m

    with torch.no_grad():
        ref = infer_sequence(model, paths, device=args.device)
    out = {"single_pass": score(ref, "single pass (reference)")}
    C_ref = _extrinsic_to_c2w(ref["extrinsic"])[:, :3, 3]
    pred_extent = float(np.linalg.norm(C_ref.max(0) - C_ref.min(0)))
    out["single_pass"]["pred_extent_mm"] = pred_extent
    print(f"{'':24}predicted extent {pred_extent:.3f} mm vs GT {extent:.2f} mm "
          f"-> ATE's scale alignment magnifies prediction-space error {extent / pred_extent:.0f}x")

    for cfg in args.configs:
        c, o = (int(x) for x in cfg.split(":"))
        torch.cuda.empty_cache()
        with torch.no_grad():
            st = infer_sequence_stitched(model, paths, device=args.device,
                                         chunk_size=c, overlap=o)
        m = score(st, f"stitched {c}:{o}")
        # Compare the two trajectories directly, independent of GT. The normaliser must be
        # the PREDICTED trajectory's own extent, not the GT's: monocular VGGT has no absolute
        # scale and here predicts ~0.37 mm of motion where GT travels 17 mm, so ATE's
        # correct_scale alignment multiplies every prediction-space error by ~46 on the way
        # into the ATE column. Dividing by the GT extent understates the damage by that same
        # factor -- which is exactly how this looked like a 0.05% effect at first.
        C_st = _extrinsic_to_c2w(st["extrinsic"])[:, :3, 3]
        s, R, t = umeyama_sim3(C_st, C_ref)
        dev = np.linalg.norm((s * (R @ C_st.T)).T + t - C_ref, axis=1)
        m["dev_from_single_max"] = float(dev.max())
        m["dev_from_single_mean"] = float(dev.mean())
        m["dev_rel_pred_pct"] = float(100 * dev.mean() / pred_extent)
        m["ate_delta"] = float(m["ate"] - out["single_pass"]["ate"])
        m["ate_delta_pct"] = float(100 * m["ate_delta"] / out["single_pass"]["ate"])
        m["n_seams"] = int(len(st["stitch_scales"]))
        m["stitch_scales"] = [round(float(x), 5) for x in st["stitch_scales"]]
        print(f"{'':24}vs single: dev mean {dev.mean():.4f} mm = "
              f"{m['dev_rel_pred_pct']:5.2f}% of PREDICTED extent ({pred_extent:.3f} mm)"
              f"  ->  ΔATE {m['ate_delta']:+.4f} ({m['ate_delta_pct']:+.1f}%)"
              f"   {m['n_seams']} seams, scales {m['stitch_scales']}")
        out[f"stitched_{c}_{o}"] = m

    d = default_output_dir(args.ckpt, TOOL)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{os.path.basename(args.seq).replace('/', '_')}"
                        f"_{args.start}_{len(paths)}f.json")
    with open(p, "w") as f:
        json.dump(dict(meta=vars(args) | {"gt_extent_mm": extent, "frames": len(paths)},
                       results=out), f, indent=2, default=str)
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
