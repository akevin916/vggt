#!/usr/bin/env python3
"""v3 gate evaluation on PointOdyssey (in-domain) and Sintel (cross-domain derived masks)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _TRAINING_DIR)
sys.path.insert(0, os.path.dirname(_TRAINING_DIR))

from data.datasets.pointodyssey import PointOdysseyDataset
from eval.gate_metrics import summarize_gate
from eval.motion_mask import DIAG_SEQUENCES, compute_ego_flow, derive_motion_mask, load_sintel_gt_flows
from eval.paths import default_eval_dir
from eval.sintel_io import (
    load_sintel_gt_depths,
    load_sintel_rgb_paths,
    matching_cam_path,
    resolve_sintel_root,
    sintel_cam_read,
    sintel_seq_paths,
)
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def parse_args():
    ap = argparse.ArgumentParser(description="v3 gate evaluation (PO + Sintel)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument(
        "--dataset",
        choices=["po", "sintel", "all"],
        default="all",
        help="po=in-domain PO test; sintel=cross-domain derived masks; all=both",
    )
    ap.add_argument("--out_dir", default=None, help="Default: logs/<exp>/eval_gate")
    ap.add_argument("--n_clips", type=int, default=30)
    ap.add_argument("--img_per_seq", type=int, default=6)
    ap.add_argument("--img_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--gate_block_iter", type=int, default=7)
    ap.add_argument("--dynamic_source", default="native", choices=["native", "raft", "instance"])
    ap.add_argument("--sintel_root", default=None, help="Auto-detected from repo data/ if omitted")
    ap.add_argument("--seqs", nargs="*", default=None, help="Sintel sequences (default: DIAG_SEQUENCES)")
    ap.add_argument("--max_frames", type=int, default=12)
    ap.add_argument("--motion_thr", type=float, default=2.0)
    ap.add_argument("--n_vis", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def build_gate_model(ckpt: str, args) -> VGGT:
    model = VGGT(
        img_size=args.img_size,
        enable_camera=True,
        enable_depth=True,
        enable_point=False,
        enable_track=False,
        enable_temporal=True,
        enable_motion=False,
        enable_flow=False,
        enable_gate=True,
        gate_block_iter=args.gate_block_iter,
    )
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    miss, unexp = model.load_state_dict(sd, strict=False)
    gp = [k for k in sd if "gate_predictor" in k]
    print(f"loaded {ckpt}: gate_predictor keys={len(gp)} missing={len(miss)} unexpected={len(unexp)}")
    if not gp:
        raise SystemExit("checkpoint has no gate_predictor weights")
    return model.to(args.device).eval()


def pool_to_patch(mask_shw: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
    s, h, w = mask_shw.shape
    pooled = F.adaptive_avg_pool2d(mask_shw.reshape(s, 1, h, w).float(), (patch_h, patch_w))
    return pooled.reshape(s, patch_h * patch_w)


def save_po_vis(out_dir: str, rgb: np.ndarray, gt_mask: np.ndarray, gate_prob: np.ndarray, tag: str):
    os.makedirs(out_dir, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gt_col = cv2.applyColorMap((gt_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    pr_col = cv2.applyColorMap((gate_prob * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(bgr, 0.55, pr_col, 0.45, 0)
    row = np.concatenate([bgr, gt_col, pr_col, overlay], axis=1)
    cv2.imwrite(os.path.join(out_dir, tag), row)


def save_sintel_vis(out_dir: str, rgb: np.ndarray, gt_soft: np.ndarray, gate_prob: np.ndarray, tag: str):
    os.makedirs(out_dir, exist_ok=True)
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gt_up = cv2.resize(gt_soft, (w, h), interpolation=cv2.INTER_NEAREST)
    pr_up = cv2.resize(gate_prob, (w, h), interpolation=cv2.INTER_LINEAR)
    gt_col = cv2.applyColorMap((gt_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
    pr_col = cv2.applyColorMap((pr_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(bgr, 0.55, pr_col, 0.45, 0)
    row = np.concatenate([bgr, gt_col, pr_col, overlay], axis=1)
    cv2.imwrite(os.path.join(out_dir, tag), row)


def eval_po(model: VGGT, args) -> Dict[str, Any]:
    common = SimpleNamespace(
        img_size=args.img_size,
        patch_size=args.patch_size,
        augs=SimpleNamespace(scales=None),
        rescale=True,
        rescale_aug=False,
        landscape_check=False,
        debug=False,
        training=False,
        get_nearby=True,
        load_depth=True,
        inside_random=False,
        allow_duplicate_img=False,
    )
    ds = PointOdysseyDataset(
        common_conf=common,
        split="test",
        min_num_images=args.img_per_seq,
        dynamic_source=args.dynamic_source,
    )
    print(f"PO test sequences: {ds.sequence_list_len}")

    vis_dir = os.path.join(args.out_dir, "po", "vis")
    probs_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    vis_saved = 0

    torch.manual_seed(0)
    np.random.seed(0)
    for ci in range(args.n_clips):
        batch = ds.get_data(seq_index=ci % ds.sequence_list_len, img_per_seq=args.img_per_seq, aspect_ratio=1.0)
        img = torch.from_numpy(np.stack(batch["images"]).astype(np.float32)).permute(0, 3, 1, 2).div(255)[None]
        mm = torch.from_numpy(np.stack(batch["motion_mask"]).astype(np.float32))

        _, s, _, h, w = img.shape
        patch_h, patch_w = h // args.patch_size, w // args.patch_size

        with torch.no_grad():
            pred = model(images=img.to(args.device))
        gate_prob = torch.sigmoid(pred["gate_logits"].float()).cpu()[0]

        soft = pool_to_patch(mm, patch_h, patch_w)
        lab = (soft >= 0.5).numpy().astype(np.int8)

        probs_all.append(gate_prob.flatten().numpy())
        labels_all.append(lab.flatten())

        if vis_saved < args.n_vis:
            rgb = (img[0].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            gp_img = gate_prob.reshape(s, patch_h, patch_w)
            for frame_idx in range(s):
                if vis_saved >= args.n_vis:
                    break
                pr = F.interpolate(
                    gp_img[frame_idx][None, None],
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].numpy()
                save_po_vis(vis_dir, rgb[frame_idx], mm[frame_idx].numpy(), pr, f"clip{ci}_f{frame_idx}.png")
                vis_saved += 1

    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    summary = summarize_gate(probs, labels)
    summary["dataset"] = "pointodyssey"
    summary["dynamic_source"] = args.dynamic_source
    return summary


def eval_sintel(model: VGGT, args) -> Dict[str, Any]:
    seqs = args.seqs or DIAG_SEQUENCES
    vis_dir = os.path.join(args.out_dir, "sintel", "vis")
    probs_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    per_seq: Dict[str, Dict[str, Any]] = {}

    for seq in seqs:
        rgb_paths = load_sintel_rgb_paths(args.sintel_root, seq)[: args.max_frames + 1]
        _, _, cam_dir = sintel_seq_paths(args.sintel_root, seq)
        gt_depths = load_sintel_gt_depths(args.sintel_root, seq, rgb_paths)
        gt_flows = load_sintel_gt_flows(args.sintel_root, seq, rgb_paths)

        intrinsics, extrinsics = [], []
        for path in rgb_paths:
            k, ext = sintel_cam_read(matching_cam_path(cam_dir, path))
            intrinsics.append(k)
            extrinsics.append(ext)

        masks: List[np.ndarray | None] = []
        for i in range(len(rgb_paths)):
            if gt_flows[i] is not None and i + 1 < len(extrinsics):
                ego = compute_ego_flow(gt_depths[i], intrinsics[i], extrinsics[i], extrinsics[i + 1])
                masks.append(derive_motion_mask(gt_flows[i], ego, threshold=args.motion_thr))
            else:
                masks.append(None)

        images = load_and_preprocess_images(rgb_paths[: args.max_frames], mode="crop")
        s, _, h, w = images.shape
        patch_h, patch_w = h // args.patch_size, w // args.patch_size

        with torch.no_grad():
            pred = model(images=images[None].to(args.device))
        gate = torch.sigmoid(pred["gate_logits"].float()).cpu()[0].reshape(s, patch_h, patch_w)

        seq_probs: List[np.ndarray] = []
        seq_labels: List[np.ndarray] = []
        vis = 0
        for i in range(s):
            if masks[i] is None:
                continue
            soft = cv2.resize(masks[i], (patch_w, patch_h), interpolation=cv2.INTER_AREA)
            lab = (soft >= 0.5).astype(np.int8)
            gp = gate[i].numpy()
            seq_probs.append(gp.flatten())
            seq_labels.append(lab.flatten())

            if vis < min(3, args.n_vis):
                rgb = (images[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                save_sintel_vis(vis_dir, rgb, soft, gp, f"{seq}_f{i}.png")
                vis += 1

        if not seq_probs:
            print(f"  {seq}: no valid frames")
            continue

        p = np.concatenate(seq_probs)
        l = np.concatenate(seq_labels)
        probs_all.append(p)
        labels_all.append(l)
        per_seq[seq] = summarize_gate(p, l)
        print(
            f"  {seq:12s}  AUC={per_seq[seq].get('roc_auc', float('nan')):.3f}  "
            f"AP={per_seq[seq].get('ap', float('nan')):.3f}  dyn_frac={per_seq[seq]['dynamic_fraction']:.3f}"
        )

    if not probs_all:
        return {"dataset": "sintel", "error": "no valid sequences"}

    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    summary = summarize_gate(probs, labels)
    summary["dataset"] = "sintel"
    summary["motion_thr"] = args.motion_thr
    summary["per_seq"] = per_seq
    return summary


def print_gate_summary(name: str, summary: Dict[str, Any]):
    print(f"\n========== GATE — {name} ==========")
    if "error" in summary:
        print(f"  {summary['error']}")
        return
    print(f"  patches={summary['n_patches']}  dyn_frac={summary['dynamic_fraction']:.3f}")
    print(
        f"  mean σ(g): dynamic={summary['mean_prob_dynamic']:.3f}  "
        f"static={summary['mean_prob_static']:.3f}"
    )
    print(
        f"  ROC-AUC={summary.get('roc_auc', float('nan')):.3f}  "
        f"AP={summary.get('ap', float('nan')):.3f}  "
        f"F1@0.5={summary['f1_at_0_5']:.3f}  "
        f"best-F1={summary['best_f1']:.3f}@{summary['best_f1_threshold']:.2f}"
    )


def main():
    args = parse_args()
    args.out_dir = args.out_dir or default_eval_dir(args.ckpt, "eval_gate", _TRAINING_DIR)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output dir: {args.out_dir}")
    if args.dataset in ("sintel", "all"):
        args.sintel_root = resolve_sintel_root(args.sintel_root)
        print(f"Sintel root: {args.sintel_root}")

    model = build_gate_model(args.ckpt, args)
    report: Dict[str, Any] = {
        "meta": {
            "ckpt": args.ckpt,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": args.dataset,
        }
    }

    if args.dataset in ("po", "all"):
        report["po"] = eval_po(model, args)
        print_gate_summary("PointOdyssey (in-domain)", report["po"])

    if args.dataset in ("sintel", "all"):
        report["sintel"] = eval_sintel(model, args)
        print_gate_summary("Sintel (cross-domain)", report["sintel"])

    json_path = os.path.join(args.out_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved {json_path}")
    print(f"visualizations -> {os.path.join(args.out_dir, 'po', 'vis')}/  and  {os.path.join(args.out_dir, 'sintel', 'vis')}/")


if __name__ == "__main__":
    main()
