"""Reconstruct the private gastric sequences and dump point clouds + video.

Three pre-processed segments live under ``data/eval/gastric/prep/seq_*/``,
each containing an ``image/`` folder of cropped mono frames, a ``mask.png``,
a ``segment.mp4`` preview, and a ``meta.json``.

There is no stereo baseline, no calibration, and no ground-truth depth, so
**no PSNR is reported here**. This script is purely a reconstruction sanity
check and a source of material for the temporal timeline visualisation.

Two main uses:

  1. **Qualitative sanity check** -- does the model produce plausible geometry
     on continuous endoscopic footage?
  2. **Diagnostic for the lesion sparsity question** -- if the gastric clouds
     fuse cleanly (neighbouring frames share a surface), the lesion scatter is
     a data artefact (sparse shot-to-shot baseline), not a pose failure. If
     they do not fuse, the pose head is the culprit.

MEMORY WARNING
--------------
The 82-frame segment (seq_000189_000432) needs ~1.8× the attention budget of
the 50-frame SCARED eval. If a CUDA OOM occurs, fall back to the 54-frame
segment (seq_000984_001143) as the primary and skip the 82-frame one or pass
``--segs seq_000984_001143 seq_000606_000723``.

Usage::

    cd training
    python benchmark/eval_gastric.py \\
        --ckpts checkpoints/VGGT-1B.pt \\
                logs/scared_cam_b16/ckpts/best_ate.pt
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.paths import data_path
from eval_utils.media_io import write_video
from eval_utils.paths import default_output_dir
from eval_utils.ply_io import write_ply
from eval_utils.vggt_infer import infer_sequence, load_vggt_for_eval
from eval_utils.warp_psnr import unproject_to_world
from vggt.utils.load_fn import load_and_preprocess_images

TOOL = "eval_gastric"

# All three segments from gastric_prep.py, ordered by confidence.
# 82-frame first (richest); swap order via --segs if OOM.
DEFAULT_SEGS = [
    "seq_000189_000432",
    "seq_000984_001143",
    "seq_000606_000723",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def list_images(seg_dir):
    img_dir = os.path.join(seg_dir, "image")
    return sorted(
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.endswith(".png")
    )


def preprocessed(paths):
    t = load_and_preprocess_images(paths, mode="crop")
    return t.permute(0, 2, 3, 1).numpy().astype(np.float32)


def save_cloud(out_stem, depth, intrinsic, extrinsic, images, max_points, depth_pct):
    """Point cloud from depth unproject. Writes .ply + .npz with frame_id."""
    per_frame = max(1, max_points // max(1, len(depth)))
    rng = np.random.default_rng(0)
    pts, cols, fids = [], [], []
    for i in range(len(depth)):
        d = depth[i]
        w = unproject_to_world(d, intrinsic[i], extrinsic[i]).reshape(-1, 3)
        c = images[i].reshape(-1, 3)
        keep = (d.reshape(-1) > 0) & np.isfinite(w).all(axis=1)
        if depth_pct < 100 and (d > 0).any():
            keep &= d.reshape(-1) <= np.percentile(d[d > 0], depth_pct)
        wk, ck = w[keep], c[keep]
        if len(wk) > per_frame:
            sel = rng.choice(len(wk), per_frame, replace=False)
            wk, ck = wk[sel], ck[sel]
        pts.append(wk)
        cols.append(ck)
        fids.append(np.full(len(wk), i, np.int16))

    pts = np.concatenate(pts) if pts else np.zeros((0, 3), np.float32)
    cols = np.concatenate(cols) if cols else np.zeros((0, 3), np.float32)
    fids = np.concatenate(fids) if fids else np.zeros((0,), np.int16)

    np.savez_compressed(
        out_stem + ".npz",
        points=pts.astype(np.float32),
        colors=(np.clip(cols, 0, 1) * 255).astype(np.uint8),
        frame_id=fids,
        n_frames=np.int32(len(depth)),
    )
    return write_ply(out_stem + ".ply", pts, cols)


def dump_arrays(path, pred, images=None):
    """Everything the polished renderer needs, in one file beside cloud.npz.

    ``images`` is the preprocessed tensor the model actually saw, stored as uint8 so
    diag/vis/cloud_polish.py (and timeline.py --blend) can colour a blended cloud without
    re-deriving the crop. eval_lesion.py stores the same key for the same reason.
    """
    payload = {k: v for k, v in pred.items() if isinstance(v, np.ndarray)}
    if "depth" in payload:
        payload["depth"] = payload["depth"].astype(np.float16)
    if images is not None:
        payload["images"] = (np.clip(images, 0, 1) * 255).astype(np.uint8)
    np.savez_compressed(path, **payload)


# ---------------------------------------------------------------------------
# per-segment
# ---------------------------------------------------------------------------

def run_segment(model, seg_dir, out_dir, args):
    paths = list_images(seg_dir)
    if not paths:
        return dict(error="no images found")
    if args.max_frames and args.max_frames > 0:
        paths = paths[:args.max_frames]

    seg_name = os.path.basename(seg_dir)
    meta_path = os.path.join(seg_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    print(f"  {seg_name}: {len(paths)} frames  "
          f"({meta['motion_median']:.1f} px/step median motion)")

    pred = infer_sequence(model, paths, device=args.device)
    imgs = preprocessed(paths)

    os.makedirs(out_dir, exist_ok=True)
    dump_arrays(os.path.join(out_dir, "seq.npz"), pred, images=imgs)
    write_video(os.path.join(out_dir, "input.mp4"), imgs, fps=args.fps)

    info = dict(
        seg=seg_name,
        n_frames=len(paths),
        meta=meta,
    )

    if args.ply and "depth" in pred:
        stem = os.path.join(out_dir, "cloud")
        n_pts = save_cloud(
            stem, pred["depth"], pred["intrinsic"],
            pred["extrinsic"], imgs, args.max_points, args.depth_pct,
        )
        info["ply_points"] = n_pts
        info["ply"] = stem + ".ply"
        print(f"    -> {n_pts:,} points  {stem}.ply")

    return info


# ---------------------------------------------------------------------------
# per-checkpoint
# ---------------------------------------------------------------------------

def eval_ckpt(ckpt, args):
    model = load_vggt_for_eval(ckpt, device=args.device)
    base = args.out_dir or default_output_dir(ckpt, TOOL)
    results = {}

    gastric_root = args.gastric_root
    for seg in args.segs:
        seg_dir = os.path.join(gastric_root, "prep", seg)
        if not os.path.isdir(seg_dir):
            print(f"  [skip] {seg_dir} not found")
            results[seg] = dict(error="not found")
            continue
        out_dir = os.path.join(base, seg)
        try:
            results[seg] = run_segment(model, seg_dir, out_dir, args)
        except Exception:
            traceback.print_exc()
            results[seg] = dict(error=True)

    del model
    torch.cuda.empty_cache()

    payload = dict(
        meta=dict(
            ckpt=ckpt,
            gastric_root=gastric_root,
            segs=args.segs,
            note="mono sequence, no calibration; no PSNR -- reconstruction only",
        ),
        results=results,
    )
    os.makedirs(base, exist_ok=True)
    summary_path = os.path.join(base, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"-> {summary_path}")
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="+", default=[
        "checkpoints/VGGT-1B.pt",
        "logs/scared_cam_b16/ckpts/best_ate.pt",
    ])
    ap.add_argument("--gastric_root", default=data_path("eval", "gastric"))
    ap.add_argument("--segs", nargs="*", default=None,
                    help="segment names under prep/; default all three in confidence order")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ply", action="store_true", default=True)
    ap.add_argument("--no_ply", dest="ply", action="store_false")
    ap.add_argument("--max_points", type=int, default=8_000_000)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--depth_pct", type=float, default=99.0)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--max_frames", type=int, default=0,
                    help="if >0, truncate each segment to the first N frames")
    args = ap.parse_args()

    args.segs = args.segs or DEFAULT_SEGS
    print(f"{len(args.ckpts)} ckpts x {len(args.segs)} segments")
    print(f"  segments: {args.segs}")

    for ckpt in args.ckpts:
        print(f"\n=== {ckpt}")
        eval_ckpt(ckpt, args)


if __name__ == "__main__":
    main()
