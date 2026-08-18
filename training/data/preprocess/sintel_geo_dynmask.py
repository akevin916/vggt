#!/usr/bin/env python
"""Precompute + cache Sintel m_geo dynamic masks (and a per-sequence preview GIF).

m_geo = |GT optical flow - ego flow| > threshold, with ego flow reprojected from
GT depth + GT relative pose (see data/motion_mask.py). The masks are deterministic
functions of the GT fields, so we compute them once and cache them; eval/vis then
load the cache via ``motion_mask.load_masks`` instead of recomputing every run.

Output layout (default ``<sintel_root>/mask/``):
    mask/meta.json                       # {threshold, version, timestamp}
    mask/<seq>/<frame_stem>.png          # uint8 {0,255} pixel mask
    mask/<seq>/<seq>.gif                 # RGB | m_geo | overlay preview

Run from training/:
    python data/preprocess/sintel_geo_dynmask.py                 # all sequences
    python data/preprocess/sintel_geo_dynmask.py --seqs cave_2   # subset
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.motion_mask import (  # noqa: E402
    MASK_CACHE_VERSION,
    compute_ego_flow,
    derive_motion_mask,
    load_sintel_gt_flows,
    mask_cache_dir,
    mask_cache_meta_path,
)
from data.paths import data_path  # noqa: E402
from data.sintel_io import (  # noqa: E402
    load_sintel_gt_depths,
    load_sintel_rgb_paths,
    matching_cam_path,
    sintel_cam_read,
    sintel_seq_paths,
)

DEFAULT_ROOT = data_path("eval", "sintel")


def list_sequences(sintel_root: str):
    final_dir = os.path.join(sintel_root, "final")
    return sorted(d for d in os.listdir(final_dir) if os.path.isdir(os.path.join(final_dir, d)))


def build_preview_frame(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """RGB | m_geo(red) | overlay, returned as an RGB uint8 strip."""
    h, w = mask.shape
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    red = np.zeros_like(rgb)
    red[..., 0] = (mask * 255).astype(np.uint8)  # R channel
    overlay = cv2.addWeighted(rgb, 0.6, red, 0.4, 0)
    strip = np.concatenate([rgb, red, overlay], axis=1)
    cv2.putText(strip, "RGB | m_geo | overlay", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return strip


def process_seq(sintel_root: str, seq: str, threshold: float, rel_threshold: float,
                out_root: str, gif: bool) -> dict:
    rgb_paths = load_sintel_rgb_paths(sintel_root, seq)
    _, _, cam_dir = sintel_seq_paths(sintel_root, seq)
    gt_flows = load_sintel_gt_flows(sintel_root, seq, rgb_paths)
    gt_depths = load_sintel_gt_depths(sintel_root, seq, rgb_paths)
    intrinsics, extrinsics = [], []
    for p in rgb_paths:
        K, ext = sintel_cam_read(matching_cam_path(cam_dir, p))
        intrinsics.append(K)
        extrinsics.append(ext)

    seq_dir = os.path.join(out_root, seq)
    os.makedirs(seq_dir, exist_ok=True)

    fracs = []
    preview = []
    for i, p in enumerate(rgb_paths):
        stem = os.path.splitext(os.path.basename(p))[0]
        if gt_flows[i] is None or i + 1 >= len(extrinsics):
            continue  # last frame has no forward flow -> no mask
        ego = compute_ego_flow(gt_depths[i], intrinsics[i], extrinsics[i], extrinsics[i + 1])
        mask = derive_motion_mask(gt_flows[i], ego, threshold=threshold, rel_threshold=rel_threshold)
        cv2.imwrite(os.path.join(seq_dir, f"{stem}.png"), (mask * 255).astype(np.uint8))
        fracs.append(float(mask.mean()))
        if gif:
            rgb = np.array(Image.open(p).convert("RGB"))
            preview.append(build_preview_frame(rgb, mask))

    if gif and preview:
        gif_path = os.path.join(seq_dir, f"{seq}.gif")
        pil = [Image.fromarray(f) for f in preview]
        pil[0].save(gif_path, save_all=True, append_images=pil[1:], duration=120, loop=0)

    mean_frac = float(np.mean(fracs)) if fracs else float("nan")
    print(f"  {seq:12s} {len(fracs):3d} masks  mean_dyn_frac={mean_frac:.3f}"
          + (f"  gif -> {seq}/{seq}.gif" if gif else ""))
    return {"n_masks": len(fracs), "mean_dyn_frac": mean_frac}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sintel_root", default=DEFAULT_ROOT)
    ap.add_argument("--out_root", default=None, help="default <sintel_root>/mask")
    ap.add_argument("--threshold", type=float, default=2.0, help="absolute residual floor (px)")
    ap.add_argument("--rel_threshold", type=float, default=0.5,
                    help="relative bar = max(threshold, rel*|gt_flow|); tames fast-camera "
                         "frames (e.g. cave_2 front half). 0 = pure absolute.")
    ap.add_argument("--seqs", nargs="*", default=None, help="subset (default: all)")
    ap.add_argument("--no_gif", action="store_true")
    args = ap.parse_args()

    out_root = args.out_root or mask_cache_dir(args.sintel_root)
    seqs = args.seqs or list_sequences(args.sintel_root)
    os.makedirs(out_root, exist_ok=True)
    print(f"m_geo precompute: threshold={args.threshold} rel_threshold={args.rel_threshold} -> {out_root}")

    per_seq = {}
    for seq in seqs:
        per_seq[seq] = process_seq(args.sintel_root, seq, args.threshold, args.rel_threshold,
                                   out_root, gif=not args.no_gif)

    meta = {
        "threshold": args.threshold,
        "rel_threshold": args.rel_threshold,
        "version": MASK_CACHE_VERSION,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "sequences": per_seq,
    }
    # meta path is fixed under <sintel_root>/mask; only write it for a full/default build.
    meta_path = mask_cache_meta_path(args.sintel_root) if out_root == mask_cache_dir(args.sintel_root) \
        else os.path.join(out_root, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
