#!/usr/bin/env python3
"""Probe MonST3R's flow_loss on a candidate warm-start checkpoint WITHOUT training it.

Question this answers: if we port MonST3R's flow_loss verbatim (optimizer.py:780-802) into
our training loop, would it actually produce gradient, or would the whole-term fuse
(``if flow_loss > flow_loss_thre: flow_loss = 0``) zero it out at every step?

MonST3R's two thresholds were tuned for a CONVERGED test-time optimization, not for a
training loop starting from a warm-start checkpoint. If they don't transfer, the loss is a
silent no-op: the log shows the term, its value is 0.0, and it never contributed a gradient.
This script measures the distribution the thresholds would sit on, so they can be chosen
from data instead of inherited.

What it computes, per adjacent frame pair of a PointOdyssey clip:
  ego_flow  = the 2D displacement implied by (pose, depth, intrinsics), disparity form,
              a port of monst3r warp_by_disp (use_depth=False)
  flow_raft = RAFT on the same preprocessed image pair (same coordinate frame -- no
              vector rescaling needed, which is the main reason to measure on-the-fly)
  loss      = smooth_L1_loss_fn(ego_flow, flow_raft, mask=static & valid_depth)
              verbatim from monst3r optimizer.py:18-24

Two variants are computed for every pair:
  pred -- ego_flow from the MODEL's predicted pose/depth/intrinsics  (what training sees)
  gt   -- ego_flow from GT pose/depth/intrinsics                     (the CONTROL)

The control is the point of this script. `gt` is the floor: it is what remains when the
pose is perfect, so it isolates RAFT error + mask error + geometric misalignment from
model error. If `gt` is already above the fuse threshold, the problem is not the model
and no amount of training will bring the loss under the fuse.

Reported: percentiles of the raw per-pixel disagreement, per-pixel rejection rate, the
resulting loss distribution, and the fuse rate over a threshold sweep -- broken down by
frame gap, since occlusion and mask staleness both grow with it.

Run from training/:
  python diag/flow_loss_probe.py --ckpt logs/dyn_vggt_v3_s1_inst_smooth_temporal/ckpts/epoch_40.pt
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

from eval_utils.gate_common import po_common_conf
from eval_utils.paths import FLOW_LOSS_PROBE, default_output_dir
from eval_utils.vggt_infer import load_vggt_for_eval
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# --------------------------------------------------------------------------------------- #
# monst3r ports (kept byte-faithful so the numbers are comparable to their paper setting)
#
# NOTE: `training/ego_flow.py` now holds a differentiable copy of this geometry for the
# L_ego_flow training loss. These copies are deliberate: docs/table.md's numbers must stay
# reproducible from the code that produced them, so this file does not import that module.
# If the two ever disagree, fix ego_flow.py — not this file.
# --------------------------------------------------------------------------------------- #

def smooth_L1_loss_fn(estimate, gt, mask, beta=1.0, per_pixel_thre=50.0):
    """Verbatim port of monst3r optimizer.py:18-24, plus rejection-rate reporting.

    Note the returned value is bounded by per_pixel_thre by construction: every surviving
    pixel has raw loss < per_pixel_thre, and the result is their mean. So the whole-term
    fuse can only fire on a genuinely bad *average*, never on a few outliers.
    """
    loss_raw_shape = F.smooth_l1_loss(estimate * mask, gt * mask, beta=beta, reduction="none")
    if per_pixel_thre > 0:
        per_pixel_mask = (loss_raw_shape < per_pixel_thre) * mask
    else:
        per_pixel_mask = mask.expand_as(loss_raw_shape)
    denom = torch.sum(per_pixel_mask)
    # mask is (P,1,H,W) and broadcasts against the (P,2,H,W) residual -- monst3r does the
    # same, so the loss is faithful, but the kept-fraction denominator must count both
    # channels or it reports 200%.
    kept_frac = float(denom / torch.clamp(torch.sum(mask) * loss_raw_shape.shape[1], min=1.0))
    if denom < 1:
        # monst3r divides by zero here (-> NaN) and its `> thre` fuse does NOT catch NaN,
        # since `nan > x` is False in Python. Reported explicitly instead of propagated.
        return float("nan"), kept_frac
    return float(torch.sum(loss_raw_shape * per_pixel_mask) / denom), kept_frac


def ego_flow_disparity_form(R_rel, t_rel, disp, K_tgt, K_src_inv, coord):
    """Port of monst3r warp_by_disp (goem_opt.py:196, use_depth=False).

        tgt = (K2 R_rel K1^-1) u  +  disp * (K2 t_rel)
        tgt = tgt / tgt[2]
        flow = tgt - u

    Rotation contributes independently of depth; translation scales with disparity. Written
    in disparity rather than depth so z -> inf degenerates to a pure rotational homography
    instead of dividing by a large z.

    Args:
        R_rel: (P,3,3) cam_src -> cam_tgt rotation
        t_rel: (P,3,1) cam_src -> cam_tgt translation
        disp:  (P,1,H,W) inverse depth of the SOURCE frame
        K_tgt: (P,3,3);  K_src_inv: (P,3,3)
        coord: (1,3,H*W) homogeneous pixel grid
    Returns:
        (P,2,H,W) flow in pixels
    """
    P, _, H, W = disp.shape
    H_mat = K_tgt.matmul(R_rel.matmul(K_src_inv))          # (P,3,3)
    flat_disp = disp.view(P, 1, H * W)
    tgt_coord = torch.matmul(H_mat, coord) + flat_disp * torch.matmul(K_tgt, t_rel)
    tgt_coord = tgt_coord / (tgt_coord[:, -1:, :] + 1e-6)
    return (tgt_coord - coord).view(P, 3, H, W)[:, :2]


# --------------------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------------------- #

def _pixel_grid(H, W, device, dtype):
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xx, yy, torch.ones_like(xx)], 0).reshape(1, 3, H * W)


def _relative_w2c(extri_src, extri_tgt):
    """Relative transform from world-to-cam extrinsics (VGGT/OpenCV convention).

    X_cam2 = R2 (R1^T (X_cam1 - t1)) + t2  =>  R_rel = R2 R1^T,  t_rel = t2 - R_rel t1.
    monst3r's get_relative_transform takes cam-to-world instead; converting here rather than
    inverting the poses keeps the numerics in one place.
    """
    R1, t1 = extri_src[:, :3, :3], extri_src[:, :3, 3:]
    R2, t2 = extri_tgt[:, :3, :3], extri_tgt[:, :3, 3:]
    R_rel = R2.matmul(R1.transpose(-1, -2))
    t_rel = t2 - R_rel.matmul(t1)
    return R_rel, t_rel


@torch.no_grad()
def _raft_flow(raft, tf, img_u8, idx_a, idx_b, device, chunk=8):
    """RAFT flow for frame pairs (idx_a -> idx_b) of one clip.

    img_u8: (S,3,H,W) uint8, already at the model's input resolution. 518 is not divisible
    by 8, so it is zero-padded up to the next multiple and the flow cropped back; padding
    (rather than resizing) means the flow vectors need no rescaling.
    """
    S, _, H, W = img_u8.shape
    ph, pw = (-H) % 8, (-W) % 8
    # Chunked: RAFT's all-pairs correlation volume is the memory hog, and a 50-frame Sintel
    # sequence is ~98 pairs per direction -- enough to OOM alongside VGGT's activations.
    out = []
    for i in range(0, len(idx_a), chunk):
        a, b = img_u8[idx_a[i : i + chunk]], img_u8[idx_b[i : i + chunk]]
        if ph or pw:
            a = F.pad(a, (0, pw, 0, ph))
            b = F.pad(b, (0, pw, 0, ph))
        ta, tb = tf(a, b)
        out.append(raft(ta.to(device), tb.to(device), num_flow_updates=20)[-1][..., :H, :W])
    return torch.cat(out, dim=0)


# --------------------------------------------------------------------------------------- #
# probe core (dataset-agnostic) + one loader per dataset
# --------------------------------------------------------------------------------------- #

@torch.no_grad()
def probe_pairs(model, raft, tf, clip, key, args) -> List[Dict]:
    """Score every usable frame pair of one clip.

    ``clip`` holds tensors already in the MODEL's input space (518-crop): img_u8 (S,3,H,W)
    uint8, gt_extri (S,3,4) world-to-cam, gt_intri (S,3,3), gt_depth (S,H,W), motion
    (S,H,W) 1=dynamic, pmask (S,H,W) bool valid-depth, ids (S,) chronological frame index.
    Keeping this dataset-agnostic is what makes the PointOdyssey and Sintel numbers
    comparable -- only the loader differs.
    """
    dev = args.device
    img_u8 = clip["img_u8"]
    images = img_u8.float().div(255)[None].to(dev)                        # (1,S,3,H,W)
    S, _, H, W = img_u8.shape

    ids = clip["ids"]
    gt_extri, gt_intri = clip["gt_extri"], clip["gt_intri"]
    gt_depth, motion, pmask = clip["gt_depth"], clip["motion"], clip["pmask"]

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = model(images)
    pose_enc = pred["pose_enc_list"][-1].float() if "pose_enc_list" in pred else pred["pose_enc"].float()
    pr_extri, pr_intri = pose_encoding_to_extri_intri(pose_enc, (H, W))
    pr_extri, pr_intri = pr_extri[0], pr_intri[0]
    pr_depth = pred["depth"][0, ..., 0].float() if "depth" in pred else None

    # Adjacent pairs of the clip. Duplicate frames (dt == 0) have zero flow by construction
    # and are dropped; dt > max_dt is where the gap-5 instance mask stops being a superset
    # of the true dynamics (see docs -- under-masking is the dangerous direction).
    src, tgt, dts = [], [], []
    for t in range(S - 1):
        dt = int(ids[t + 1] - ids[t])
        if dt <= 0 or dt > args.max_dt:
            continue
        src.append(t)
        tgt.append(t + 1)
        dts.append(dt)
    if not src:
        return []
    idx_a = torch.tensor(src)
    idx_b = torch.tensor(tgt)

    flow_fwd = _raft_flow(raft, tf, img_u8, idx_a, idx_b, dev)
    flow_bwd = _raft_flow(raft, tf, img_u8, idx_b, idx_a, dev)

    coord = _pixel_grid(H, W, dev, torch.float32)
    rows = []

    for direction, (ia, ib, raft_flow) in [
        ("fwd", (idx_a, idx_b, flow_fwd)),
        ("bwd", (idx_b, idx_a, flow_bwd)),
    ]:
        # mask: static AND valid GT depth, on the SOURCE frame. Same mask for both variants
        # so pred/gt differ only in the geometry that generates ego_flow.
        m = ((motion[ia] < args.dyn_thresh) & pmask[ia]).float()[:, None]  # (P,1,H,W)

        variants = {}
        row_scale = float("nan")
        gt_disp = 1.0 / (gt_depth[ia].clamp(min=1e-6))[:, None]
        R_rel, t_rel = _relative_w2c(gt_extri[ia], gt_extri[ib])
        variants["gt"] = ego_flow_disparity_form(
            R_rel, t_rel, gt_disp, gt_intri[ib], torch.linalg.inv(gt_intri[ia]), coord
        )
        if pr_depth is not None:
            pr_disp = 1.0 / (pr_depth[ia].clamp(min=1e-6))[:, None]
            Rp, tp = _relative_w2c(pr_extri[ia], pr_extri[ib])
            variants["pred"] = ego_flow_disparity_form(
                Rp, tp, pr_disp, pr_intri[ib], torch.linalg.inv(pr_intri[ia]), coord
            )
            # Mixing GT depth with predicted translation is INVALID unless the scale is
            # reconciled first: ego_flow's translation term is disparity * t, and VGGT
            # predicts depth and translation in its own (normalised) scale while GT is in
            # metres. Without this factor the mixed variants measure scale mismatch, not
            # pose or depth error. s = metres per VGGT unit, from the median depth ratio.
            sel = pmask[ia] & (pr_depth[ia] > 1e-6) & (gt_depth[ia] > 1e-6)
            s = (
                torch.median(gt_depth[ia][sel] / pr_depth[ia][sel])
                if sel.any() else torch.tensor(1.0, device=dev)
            )
            # pose_only: predicted pose (rescaled to metres) + GT depth/intrinsics.
            variants["pose_only"] = ego_flow_disparity_form(
                Rp, tp * s, gt_disp, gt_intri[ib], torch.linalg.inv(gt_intri[ia]), coord
            )
            # depth_only: GT pose (rescaled to VGGT units) + predicted depth/intrinsics.
            variants["depth_only"] = ego_flow_disparity_form(
                R_rel, t_rel / s, pr_disp, pr_intri[ib], torch.linalg.inv(pr_intri[ia]), coord
            )
            row_scale = float(s)

        for p in range(len(ia)):
            # pair_key ties the fwd and bwd row of the SAME frame pair together, so the
            # whole-term fuse can be evaluated on (fwd + bwd) as monst3r sums them.
            row = {"dt": dts[p], "direction": direction, "pair_key": f"{key}_{p}", "clip": key}
            mp = m[p : p + 1]
            valid = mp.bool().expand(-1, 2, -1, -1)
            for name, ego in variants.items():
                loss, kept = smooth_L1_loss_fn(
                    ego[p : p + 1], raft_flow[p : p + 1], mp, per_pixel_thre=args.per_pixel_thre
                )
                err = (ego[p : p + 1] - raft_flow[p : p + 1]).abs()[valid]
                q = torch.quantile(err.float(), torch.tensor([0.5, 0.9, 0.99], device=dev))
                row[f"{name}_loss"] = loss
                row[f"{name}_kept"] = kept
                row[f"{name}_p50"] = float(q[0])
                row[f"{name}_p90"] = float(q[1])
                row[f"{name}_p99"] = float(q[2])
            row["static_frac"] = float(mp.mean())
            row["scale_gt_per_pred"] = row_scale  # metres per VGGT depth unit, this clip
            rows.append(row)
    return rows


def load_train_clip(ds, seq_index, args) -> Dict:
    """Any of the four training datasets -- they share get_data() and its batch keys, and
    already return everything in model input space.

    motion_mask is absent on Waymo/Spring when built without a dynamic_source; an all-static
    fallback is used, which for the probe only widens the pixel set (safe direction).
    """
    b = ds.get_data(seq_index=seq_index, img_per_seq=args.img_per_seq, aspect_ratio=1.0)
    dev = args.device
    t = lambda x, d: torch.from_numpy(np.stack(x).astype(d)).to(dev)
    pmask = t(b["point_masks"], np.bool_)
    motion = t(b["motion_mask"], np.float32) if "motion_mask" in b else torch.zeros_like(pmask, dtype=torch.float32)
    return {
        "img_u8": torch.from_numpy(np.stack(b["images"]).astype(np.uint8)).permute(0, 3, 1, 2).contiguous(),
        "ids": np.asarray(b["ids"]),
        "gt_extri": t(b["extrinsics"], np.float32),
        "gt_intri": t(b["intrinsics"], np.float32),
        "gt_depth": t(b["depths"], np.float32),
        "motion": motion,
        "pmask": pmask,
    }


def build_train_dataset(name, args):
    """Mirrors the 4-dataset S1 training config (dyn_vggt_v3_s1_inst_smooth_temporal.yaml),
    including each set's dynamic_source, so the probe scores what training actually sees."""
    conf = po_common_conf(args)
    if name == "po":
        from data.datasets.pointodyssey import PointOdysseyDataset
        return PointOdysseyDataset(common_conf=conf, split=args.split, PO_DIR=args.po_dir,
                                   min_num_images=args.img_per_seq, dynamic_source="instance")
    if name == "tartanair":
        from data.datasets.tartanair import TartanAirDataset
        return TartanAirDataset(common_conf=conf, TARTANAIR_DIR=args.tartanair_dir,
                                min_num_images=args.img_per_seq)
    if name == "waymo":
        from data.datasets.waymo import WaymoDataset
        return WaymoDataset(common_conf=conf, WAYMO_DIR=args.waymo_dir,
                            min_num_images=args.img_per_seq,
                            dynamic_source="raft", dynamic_max_frac=0.5)
    if name == "spring":
        from data.datasets.spring import SpringDataset
        return SpringDataset(common_conf=conf, split="train", SPRING_DIR=args.spring_dir,
                             min_num_images=args.img_per_seq,
                             dynamic_source="raft", dynamic_max_frac=0.5)
    raise ValueError(name)


def load_sintel_clip(sintel_root, seq, args) -> Dict:
    """Sintel: GT lives at native 1024x436, so every field is mapped into the model's
    518-crop space -- including the intrinsics, which must be scaled AND shifted by the
    crop offset or the ego-flow is silently wrong by a constant."""
    from data.motion_mask import load_masks
    from data.sintel_io import (
        compute_preprocess_meta,
        load_sintel_gt_depths,
        load_sintel_rgb_paths,
        matching_cam_path,
        resize_gt_to_pred,
        sintel_cam_read,
        sintel_seq_paths,
    )
    from vggt.utils.load_fn import load_and_preprocess_images

    dev = args.device
    rgb_paths = load_sintel_rgb_paths(sintel_root, seq)[: args.img_per_seq]
    images = load_and_preprocess_images(rgb_paths, mode="crop")            # (S,3,H,W) 0..1
    S, _, H, W = images.shape
    meta = compute_preprocess_meta(rgb_paths[0], target_size=args.img_size, mode="crop")
    sx, sy = meta.new_w / meta.orig_w, meta.new_h / meta.orig_h

    _, _, cam_dir = sintel_seq_paths(sintel_root, seq)
    extri, intri = [], []
    for p in rgb_paths:
        K, ext_w2c = sintel_cam_read(matching_cam_path(cam_dir, p))
        K = K.astype(np.float32).copy()
        K[0] *= sx
        K[1] *= sy
        K[1, 2] -= meta.crop_y0
        intri.append(K)
        extri.append(ext_w2c.astype(np.float32)[:3])

    depths = [resize_gt_to_pred(d, meta, (H, W)) for d in load_sintel_gt_depths(sintel_root, seq, rgb_paths)]
    masks = load_masks(sintel_root, seq, rgb_paths, threshold=args.motion_thr)
    motion = [
        resize_gt_to_pred(m, meta, (H, W)) if m is not None else np.zeros((H, W), np.float32)
        for m in masks
    ]
    d = np.stack(depths)
    return {
        "img_u8": images.mul(255).round().clamp(0, 255).to(torch.uint8),
        "ids": np.arange(S),
        "gt_extri": torch.from_numpy(np.stack(extri)).to(dev),
        "gt_intri": torch.from_numpy(np.stack(intri)).to(dev),
        "gt_depth": torch.from_numpy(d).to(dev),
        "motion": torch.from_numpy(np.stack(motion)).to(dev),
        "pmask": torch.from_numpy(np.isfinite(d) & (d > 0) & (d < args.max_depth)).to(dev),
    }


# --------------------------------------------------------------------------------------- #
# aggregation / report
# --------------------------------------------------------------------------------------- #

def _fuse_rates(rows, key, thresholds):
    """Fuse rate for the WHOLE term (fwd + bwd of the same pair), per monst3r's `_i + _j`.

    A NaN total counts as fired here even though monst3r's `nan > thre` is False and would
    let it through -- reporting the hole rather than reproducing it.
    """
    by_pair: Dict[str, Dict[str, float]] = {}
    for r in rows:
        by_pair.setdefault(r["pair_key"], {})[r["direction"]] = r[key]
    totals = np.array([
        d["fwd"] + d["bwd"] for d in by_pair.values() if "fwd" in d and "bwd" in d
    ])
    out = {}
    for th in thresholds:
        out[str(th)] = float(np.mean(np.isnan(totals) | (totals > th))) if len(totals) else float("nan")
    return out, totals


def _fmt(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)])
    if not len(v):
        return "  n/a"
    return f"{np.median(v):7.2f} {np.percentile(v, 90):8.2f} {v.mean():8.2f}"


def main():
    ap = argparse.ArgumentParser(description="Probe monst3r flow_loss on a warm-start ckpt")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="po",
                    choices=["po", "tartanair", "waymo", "spring", "sintel"],
                    help="the four training sets (in-domain) vs sintel (where ATE is bad)")
    ap.add_argument("--po_dir", default="/media/cvml-75/ssd2t1/data/point_odyssey")
    ap.add_argument("--tartanair_dir", default="/media/cvml-75/ssd2t1/data/tartanair")
    ap.add_argument("--waymo_dir", default="/media/cvml-75/ssd2t1/data/waymo_processed")
    ap.add_argument("--spring_dir", default="/media/cvml-75/ssd2t1/data/spring")
    ap.add_argument("--sintel_root", default=None)
    ap.add_argument("--motion_thr", type=float, default=2.0, help="sintel m_geo mask threshold")
    ap.add_argument("--max_depth", type=float, default=80.0, help="sintel GT depth validity cap")
    ap.add_argument("--split", default="train", choices=["train", "test"],
                    help="train = what the training loop would actually see (po only)")
    ap.add_argument("--n_clips", type=int, default=20)
    ap.add_argument("--img_per_seq", type=int, default=16)
    ap.add_argument("--max_dt", type=int, default=5,
                    help="drop pairs wider than the instance mask's gap-5 basis")
    ap.add_argument("--dyn_thresh", type=float, default=0.5)
    ap.add_argument("--per_pixel_thre", type=float, default=50.0, help="monst3r default")
    ap.add_argument("--fuse_sweep", type=float, nargs="*",
                    default=[5, 10, 20, 50, 100, 200])
    ap.add_argument("--img_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    # <exp>/flow_loss_probe/<ckpt_stem>/ -- paths.py resolves only to <exp>, so two ckpts of
    # the SAME run (epoch_20 vs epoch_30) would silently overwrite each other without the
    # stem level. Same fix as training/run.sh applies for eval_sintel.
    args.out_dir = args.out_dir or os.path.join(
        default_output_dir(args.ckpt, FLOW_LOSS_PROBE),
        os.path.splitext(os.path.basename(args.ckpt))[0],
    )
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output dir: {args.out_dir}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = load_vggt_for_eval(args.ckpt, img_size=args.img_size, device=args.device)

    if args.dataset != "sintel":
        ds = build_train_dataset(args.dataset, args)
        n_seq = ds.sequence_list_len
        print(f"{args.dataset} sequences: {n_seq}; probing {args.n_clips} clips")
        jobs = [(str(i), lambda i=i: load_train_clip(ds, i % n_seq, args))
                for i in range(args.n_clips)]
    else:
        from data.sintel_io import SINTEL_EVAL_SEQUENCES, resolve_sintel_root
        root = resolve_sintel_root(args.sintel_root)
        seqs = SINTEL_EVAL_SEQUENCES[: args.n_clips] if args.n_clips > 0 else SINTEL_EVAL_SEQUENCES
        print(f"Sintel root: {root}; probing {len(seqs)} sequences "
              f"(first {args.img_per_seq} frames each)")
        jobs = [(s, lambda s=s: load_sintel_clip(root, s, args)) for s in seqs]

    weights = Raft_Large_Weights.DEFAULT
    raft = raft_large(weights=weights, progress=False).to(args.device).eval()
    tf = weights.transforms()

    rows: List[Dict] = []
    for key, loader in tqdm(jobs, desc="clips"):
        try:
            rows.extend(probe_pairs(model, raft, tf, loader(), key, args))
        except Exception as e:  # one bad sequence must not kill the whole probe
            tqdm.write(f"  clip {key}: skipped ({type(e).__name__}: {e})")

    if not rows:
        print("no usable pairs -- nothing to report")
        return

    variants = [k[: -len("_loss")] for k in rows[0] if k.endswith("_loss")]

    print(f"\n{'='*78}\nPER-PIXEL DISAGREEMENT |ego_flow - raft_flow|, static+valid px (pixels)\n{'='*78}")
    print(f"{'variant':<11}{'p50':>8}{'p90':>9}{'p99':>9}   {'kept@%g' % args.per_pixel_thre:>9}")
    for v in variants:
        p50 = np.nanmedian([r[f"{v}_p50"] for r in rows])
        p90 = np.nanmedian([r[f"{v}_p90"] for r in rows])
        p99 = np.nanmedian([r[f"{v}_p99"] for r in rows])
        kept = np.nanmean([r[f"{v}_kept"] for r in rows])
        print(f"{v:<11}{p50:8.2f}{p90:9.2f}{p99:9.2f}   {kept*100:8.1f}%")

    print(f"\n{'='*78}\nLOSS VALUE (one direction)     median      p90     mean\n{'='*78}")
    for v in variants:
        print(f"{v:<11}{_fmt([r[f'{v}_loss'] for r in rows])}")

    print(f"\n{'='*78}\nWHOLE-TERM FUSE RATE  (fwd+bwd > threshold -> term dropped that step)\n{'='*78}")
    header = "threshold  " + "".join(f"{v:>12}" for v in variants)
    print(header)
    fuse_all = {}
    for v in variants:
        fuse_all[v], _ = _fuse_rates(rows, f"{v}_loss", args.fuse_sweep)
    for th in args.fuse_sweep:
        line = f"{th:>9g}  "
        for v in variants:
            line += f"{fuse_all[v][str(th)]*100:11.1f}%"
        print(line)

    print(f"\n{'='*78}\nBY FRAME GAP (dt)  --  median loss per direction\n{'='*78}")
    print(f"{'dt':>4}{'n':>7}" + "".join(f"{v:>12}" for v in variants))
    for dt in sorted({r["dt"] for r in rows}):
        sub = [r for r in rows if r["dt"] == dt]
        line = f"{dt:>4}{len(sub):>7}"
        for v in variants:
            vals = [r[f"{v}_loss"] for r in sub if np.isfinite(r[f"{v}_loss"])]
            line += f"{(np.median(vals) if vals else float('nan')):11.2f} "
        print(line)

    nan_rate = float(np.mean([not np.isfinite(r[f"{variants[0]}_loss"]) for r in rows]))
    print(f"\nNaN loss rate (all pixels rejected -> monst3r's fuse would NOT catch it): {nan_rate*100:.1f}%")
    print(f"mean static+valid fraction: {np.mean([r['static_frac'] for r in rows])*100:.1f}%")
    sc = np.asarray([r.get("scale_gt_per_pred", np.nan) for r in rows])
    sc = sc[np.isfinite(sc)]
    if len(sc):
        print(f"GT/pred depth scale: median {np.median(sc):.3f}  "
              f"[p10 {np.percentile(sc,10):.3f}, p90 {np.percentile(sc,90):.3f}]  "
              f"(spread across clips is what pose cannot compensate for)")

    out = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "n_pairs": len(rows),
        "fuse_rate": fuse_all,
        "nan_rate": nan_rate,
        "rows": rows,
    }
    # one file per dataset so a Sintel run never clobbers the PointOdyssey baseline
    path = os.path.join(args.out_dir, f"results_{args.dataset}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
