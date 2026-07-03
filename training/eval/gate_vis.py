"""Gate diagnostic visualization for PointOdyssey and Sintel."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from data.datasets.pointodyssey import PointOdysseyDataset
from eval.gate_metrics import summarize_gate_diag
from eval.motion_mask import DIAG_SEQUENCES, compute_ego_flow, derive_motion_mask, load_sintel_gt_flows
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


def po_common_conf(args) -> SimpleNamespace:
    return SimpleNamespace(
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


def pool_to_patch(mask_shw: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
    s, h, w = mask_shw.shape
    pooled = F.adaptive_avg_pool2d(mask_shw.reshape(s, 1, h, w).float(), (patch_h, patch_w))
    return pooled.reshape(s, patch_h, patch_w)


def colorize_map(x01: np.ndarray, h: int, w: int, mode: str = "nearest") -> np.ndarray:
    t = torch.from_numpy(x01)[None, None].float()
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    up = F.interpolate(t, size=(h, w), mode=mode, **kwargs)[0, 0].numpy()
    return cv2.applyColorMap((np.clip(up, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)


def save_po_panel(
    out_path: str,
    rgb: np.ndarray,
    m_gt: np.ndarray,
    m_star: np.ndarray,
    g_prob: np.ndarray,
    h: int,
    w: int,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gt_col = cv2.applyColorMap((m_gt * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mstar_col = colorize_map(m_star, h, w, "nearest")
    g_col = colorize_map(g_prob, h, w, "nearest")
    row = np.concatenate([bgr, gt_col, mstar_col, g_col], axis=1)
    label = "RGB | m_gt | m_star (L_gate target) | g=sigma(gate_logits)"
    cv2.putText(row, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, row)


def save_sintel_panel(
    out_path: str,
    rgb: np.ndarray,
    pseudo_patch: np.ndarray,
    g_prob: np.ndarray,
    h: int,
    w: int,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    pseudo_col = colorize_map(pseudo_patch, h, w, "nearest")
    g_col = colorize_map(g_prob, h, w, "nearest")
    overlay = cv2.addWeighted(bgr, 0.55, g_col, 0.45, 0)
    row = np.concatenate([bgr, pseudo_col, g_col, overlay], axis=1)
    label = "RGB | pseudo dynamic (flow residual) | g | overlay"
    cv2.putText(row, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, row)


def print_diag_summary(name: str, summary: Dict[str, Any]) -> None:
    print(f"\n--- gate diag: {name} ---")
    if "error" in summary:
        print(f"  {summary['error']}")
        return
    print(
        f"  patches={summary['n_patches']}  dyn_frac={summary['dynamic_fraction']:.3f}  "
        f"σ(g) dyn={summary['mean_prob_dynamic']:.3f}  stat={summary['mean_prob_static']:.3f}  "
        f"gap={summary['separation_gap']:.3f}  acc@0.5={summary['accuracy_at_0_5']:.3f}"
    )


def run_po_vis(model: VGGT, args, out_dir: str) -> Dict[str, Any]:
    ds = PointOdysseyDataset(
        common_conf=po_common_conf(args),
        split="test",
        min_num_images=args.img_per_seq,
        dynamic_source=args.dynamic_source,
    )
    print(f"PO test sequences: {ds.sequence_list_len}")

    vis_dir = os.path.join(out_dir, "po")
    probs_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    saved = 0

    torch.manual_seed(0)
    np.random.seed(0)
    for ci in range(args.n_clips):
        batch = ds.get_data(seq_index=ci % ds.sequence_list_len, img_per_seq=args.img_per_seq, aspect_ratio=1.0)
        img = torch.from_numpy(np.stack(batch["images"]).astype(np.float32)).permute(0, 3, 1, 2).div(255)[None]
        mm = torch.from_numpy(np.stack(batch["motion_mask"]).astype(np.float32))

        _, s, _, h, w = img.shape
        ph, pw = h // args.patch_size, w // args.patch_size

        with torch.no_grad():
            pred = model(images=img.to(args.device))
        g_prob = torch.sigmoid(pred["gate_logits"].float()).cpu()[0].reshape(s, ph, pw).numpy()
        m_star = pool_to_patch(mm, ph, pw).numpy()
        labels = (m_star >= 0.5).astype(np.int8)

        probs_all.append(g_prob.reshape(-1))
        labels_all.append(labels.reshape(-1))

        rgb = (img[0].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        for f in range(s):
            out_path = os.path.join(vis_dir, f"clip{ci}_f{f}.png")
            save_po_panel(out_path, rgb[f], mm[f].numpy(), m_star[f], g_prob[f], h, w)
            print(f"saved {out_path}")
            saved += 1

    summary = summarize_gate_diag(np.concatenate(probs_all), np.concatenate(labels_all))
    summary["dataset"] = "pointodyssey"
    summary["dynamic_source"] = args.dynamic_source
    summary["num_images"] = saved
    return summary


def run_sintel_vis(model: VGGT, args, out_dir: str, sintel_root: str) -> Dict[str, Any]:
    seqs = args.seqs or DIAG_SEQUENCES
    vis_dir = os.path.join(out_dir, "sintel")
    probs_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    per_seq: Dict[str, Dict[str, float]] = {}
    saved = 0

    for seq in seqs:
        rgb_paths = load_sintel_rgb_paths(sintel_root, seq)[: args.max_frames + 1]
        _, _, cam_dir = sintel_seq_paths(sintel_root, seq)
        gt_depths = load_sintel_gt_depths(sintel_root, seq, rgb_paths)
        gt_flows = load_sintel_gt_flows(sintel_root, seq, rgb_paths)

        intrinsics, extrinsics = [], []
        for path in rgb_paths:
            k, ext = sintel_cam_read(matching_cam_path(cam_dir, path))
            intrinsics.append(k)
            extrinsics.append(ext)

        masks: List[Optional[np.ndarray]] = []
        for i in range(len(rgb_paths)):
            if gt_flows[i] is not None and i + 1 < len(extrinsics):
                ego = compute_ego_flow(gt_depths[i], intrinsics[i], extrinsics[i], extrinsics[i + 1])
                masks.append(derive_motion_mask(gt_flows[i], ego, threshold=args.motion_thr))
            else:
                masks.append(None)

        images = load_and_preprocess_images(rgb_paths[: args.max_frames], mode="crop")
        s, _, h, w = images.shape
        ph, pw = h // args.patch_size, w // args.patch_size

        with torch.no_grad():
            pred = model(images=images[None].to(args.device))
        g_prob = torch.sigmoid(pred["gate_logits"].float()).cpu()[0].reshape(s, ph, pw).numpy()

        seq_probs: List[np.ndarray] = []
        seq_labels: List[np.ndarray] = []
        vis_count = 0
        for i in range(s):
            if masks[i] is None:
                continue
            pseudo = cv2.resize(masks[i], (pw, ph), interpolation=cv2.INTER_AREA)
            lab = (pseudo >= 0.5).astype(np.int8)
            seq_probs.append(g_prob[i].reshape(-1))
            seq_labels.append(lab.reshape(-1))

            if vis_count >= args.n_vis_per_seq:
                continue
            rgb = (images[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            out_path = os.path.join(vis_dir, f"{seq}_f{i}.png")
            save_sintel_panel(out_path, rgb, pseudo, g_prob[i], h, w)
            print(f"saved {out_path}")
            saved += 1
            vis_count += 1

        if not seq_probs:
            print(f"  {seq}: no valid frames")
            continue

        p = np.concatenate(seq_probs)
        l = np.concatenate(seq_labels)
        probs_all.append(p)
        labels_all.append(l)
        per_seq[seq] = summarize_gate_diag(p, l)
        ps = per_seq[seq]
        print(
            f"  {seq:12s}  gap={ps['separation_gap']:.3f}  "
            f"σ(g) dyn={ps['mean_prob_dynamic']:.3f} stat={ps['mean_prob_static']:.3f}"
        )

    if not probs_all:
        return {"dataset": "sintel", "error": "no valid sequences"}

    summary = summarize_gate_diag(np.concatenate(probs_all), np.concatenate(labels_all))
    summary["dataset"] = "sintel"
    summary["motion_thr"] = args.motion_thr
    summary["per_seq"] = per_seq
    summary["num_images"] = saved
    return summary
