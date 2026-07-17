#!/usr/bin/env python3
"""Animated RGB | m_geo_patch | g | overlay GIF, one per Sintel sequence, all frames.

Same panel layout as eval_utils.gate_vis.save_sintel_panel (used by diag/vis/gate.py), but stitched
into a GIF across the WHOLE sequence instead of a handful of saved PNG frames -- lets you scrub
through gate behaviour over time and spot exactly which frames correspond to the ATE spikes found
by vis_error_growth.py (e.g. cave_4's last few frames, temple_3's frame ~30-38).

Uses a single whole-sequence forward pass (no chunking) for gate_logits, consistent with the
chunk_size fix from vis_error_growth.py.
"""

from __future__ import annotations

import os
import sys

_TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO_DIR = os.path.dirname(_TRAINING_DIR)
sys.path[:0] = [_TRAINING_DIR, _REPO_DIR]

import argparse

import cv2
import numpy as np
import torch
from PIL import Image

from eval_utils.paths import GATE_GIF, default_output_dir
from eval_utils.gate_vis import colorize_map
from data.motion_mask import DIAG_SEQUENCES, compute_ego_flow, derive_motion_mask, load_sintel_gt_flows
from data.sintel_io import (
    load_sintel_gt_depths,
    load_sintel_rgb_paths,
    matching_cam_path,
    resolve_sintel_root,
    sintel_cam_read,
    sintel_seq_paths,
)
from eval_utils.vggt_infer import load_vggt_for_eval
from vggt.utils.load_fn import load_and_preprocess_images


def parse_args():
    ap = argparse.ArgumentParser(description="Per-sequence RGB/mask/gate GIF")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seqs", nargs="*", default=None, help="Default: DIAG_SEQUENCES")
    ap.add_argument("--sintel_root", default=None)
    ap.add_argument("--max_frames", type=int, default=0, help="0 = full sequence")
    ap.add_argument("--motion_thr", type=float, default=2.0)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--duration_ms", type=int, default=200, help="ms per frame in the GIF")
    ap.add_argument("--out_dir", default=None, help="Default: outputs/<exp>/{}".format(GATE_GIF))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out_dir = args.out_dir or default_output_dir(args.ckpt, GATE_GIF)
    return args


def build_panel(rgb: np.ndarray, m_geo_patch: np.ndarray, g_prob: np.ndarray, h: int, w: int, frame_idx: int) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    pseudo_col = colorize_map(m_geo_patch, h, w, "nearest")
    g_col = colorize_map(g_prob, h, w, "nearest")
    overlay = cv2.addWeighted(bgr, 0.55, g_col, 0.45, 0)
    row = np.concatenate([bgr, pseudo_col, g_col, overlay], axis=1)
    label = f"frame {frame_idx:03d}  RGB | m_geo_patch | g=sigma(gate_logits) | overlay"
    cv2.putText(row, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(row, cv2.COLOR_BGR2RGB)


def run_seq(model, args, sintel_root: str, seq: str) -> list[np.ndarray]:
    rgb_paths = load_sintel_rgb_paths(sintel_root, seq)
    if args.max_frames > 0:
        rgb_paths = rgb_paths[: args.max_frames]
    _, _, cam_dir = sintel_seq_paths(sintel_root, seq)
    gt_depths = load_sintel_gt_depths(sintel_root, seq, rgb_paths)
    gt_flows = load_sintel_gt_flows(sintel_root, seq, rgb_paths)

    intrinsics, extrinsics = [], []
    for p in rgb_paths:
        K, ext = sintel_cam_read(matching_cam_path(cam_dir, p))
        intrinsics.append(K)
        extrinsics.append(ext)

    images = load_and_preprocess_images(rgb_paths, mode="crop")
    s, _, h, w = images.shape
    ph, pw = h // args.patch_size, w // args.patch_size

    with torch.no_grad():
        pred = model(images=images[None].to(args.device))
    g_prob = torch.sigmoid(pred["gate_logits"].float()).cpu()[0].reshape(s, ph, pw).numpy()

    frames = []
    for i in range(s):
        if gt_flows[i] is not None and i + 1 < len(extrinsics):
            ego = compute_ego_flow(gt_depths[i], intrinsics[i], extrinsics[i], extrinsics[i + 1])
            mask = derive_motion_mask(gt_flows[i], ego, threshold=args.motion_thr)
            m_geo_patch = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_AREA)
        else:
            m_geo_patch = np.zeros((ph, pw), dtype=np.float32)

        rgb = (images[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frames.append(build_panel(rgb, m_geo_patch, g_prob[i], h, w, i))

    return frames


def main():
    args = parse_args()
    sintel_root = resolve_sintel_root(args.sintel_root)
    seqs = args.seqs or DIAG_SEQUENCES

    model = load_vggt_for_eval(args.ckpt, device=args.device, require_gate=True)

    for seq in seqs:
        try:
            frames = run_seq(model, args, sintel_root, seq)
        except Exception as e:
            print(f"[skip] {seq}: {e}")
            continue
        out_path = os.path.join(args.out_dir, f"{seq}.gif")
        os.makedirs(args.out_dir, exist_ok=True)
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            out_path, save_all=True, append_images=pil_frames[1:], duration=args.duration_ms, loop=0
        )
        print(f"saved {out_path}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
