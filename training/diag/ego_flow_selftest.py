#!/usr/bin/env python3
"""Correctness self-test for `loss.compute_ego_flow_loss` — no model, no checkpoint.

The loss compares an ego-flow built from PREDICTED geometry against one built from GT
geometry. So the decisive test is the identity case: feed the GT extrinsics / intrinsics /
depth in as if they were the predictions, and the loss must be ~0. Any error in the
coordinate convention (row- vs column-vector), the relative-transform direction, the
disparity form, or the pixel grid produces a residual of many pixels here.

  1a. identity              pred == GT, loss defaults (reachable target)  -> ~0
  1b. raw-GT target         same, target_from_pose_encoding off           -> shows the floor
  1c. floor decomposition   which round-trip step costs what, in pixels
  2.  perturbed pose        GT pose + an offset scaled to the trajectory extent -> grows
  3.  duplicate frames      ids with dt == 0                                   -> inert

Why 1a and 1b are separate. Scored against the RAW GT, the identity case does not reach zero
on real data (0.061 px on PointOdyssey): the dataset's GT rotation matrices have drifted off
SO(3) by ~5e-4 through crop/resize/rotate in float32, and the model's quaternion can only
express a proper rotation, so the decode silently re-orthonormalises. The prediction is being
charged for a camera it cannot represent. `target_from_pose_encoding` (on by default)
projects the GT onto the representable set and removes it — 1a checks the loss as configured,
1b measures what the flag is worth. 1c attributes the gap: on PointOdyssey the rotation
accounts for it and the dropped principal point costs nothing, because shifting the principal
point translates the whole flow field rather than changing its values.

Case 2 matters as much as case 1: a loss that is zero because the mask killed every pixel
would also pass case 1. The offset is scaled by the camera-trajectory extent because these
clips are NOT scale-normalised the way trainer._process_batch normalises them — an absolute
offset means something completely different in a 2-metre indoor clip and a 200-metre outdoor
one (early versions of this test read as "no signal" purely from drawing a distant scene).

Run from training/:
    python diag/ego_flow_selftest.py                       # synthetic geometry (no data needed)
    python diag/ego_flow_selftest.py --dataset po          # real PointOdyssey clip
"""
from __future__ import annotations

import os
import sys

_TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_TRAINING_DIR, os.path.dirname(_TRAINING_DIR)]

import argparse

import numpy as np
import torch

from data.paths import data_path
from loss import compute_ego_flow_loss
from vggt.utils.pose_enc import extri_intri_to_pose_encoding
from vggt.utils.rotation import mat_to_quat, quat_to_mat


def _as_predictions(extri, intri, depth, hw):
    """Wrap GT geometry in the shape `compute_ego_flow_loss` expects from the model."""
    pose_enc = extri_intri_to_pose_encoding(extri, intri, hw)          # (B,S,9)
    return {"pose_enc_list": [pose_enc], "depth": depth[..., None]}


def synthetic_batch(B=2, S=5, H=70, W=98, device="cpu", seed=0):
    """A small forward-moving camera looking at a slanted plane.

    Deliberately not square and not a multiple of the patch size: a bug that transposes H/W
    or assumes a square image shows up as a large residual rather than silently cancelling.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    f = 0.9 * W
    K = torch.tensor([[f, 0.0, W / 2], [0.0, f, H / 2], [0.0, 0.0, 1.0]])
    intri = K[None, None].expand(B, S, 3, 3).contiguous()

    extri = torch.zeros(B, S, 3, 4)
    for b in range(B):
        for s in range(S):
            ang = 0.03 * s + 0.01 * b
            ca, sa = float(np.cos(ang)), float(np.sin(ang))
            extri[b, s, :3, :3] = torch.tensor([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
            extri[b, s, :3, 3] = torch.tensor([0.05 * s, 0.0, 0.02 * s])

    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    plane = 3.0 + 0.004 * yy + 0.002 * xx                              # slanted, positive everywhere
    depth = plane[None, None].expand(B, S, H, W).contiguous()
    depth = depth + 0.05 * torch.rand(depth.shape, generator=g)

    images = torch.rand(B, S, 3, H, W, generator=g)
    point_masks = torch.ones(B, S, H, W, dtype=torch.bool)
    motion_mask = torch.zeros(B, S, H, W)
    ids = torch.arange(S)[None].expand(B, S).contiguous()

    batch = {
        "images": images, "extrinsics": extri, "intrinsics": intri, "depths": depth,
        "point_masks": point_masks, "motion_mask": motion_mask, "ids": ids,
    }
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def real_batch(args):
    """One clip from a real training dataset, through the same loaders training uses."""
    from diag.flow_loss_probe import build_train_dataset, load_train_clip

    ds = build_train_dataset(args.dataset, args)
    clip = load_train_clip(ds, seq_index=args.seq_index, args=args)
    S, _, H, W = clip["img_u8"].shape
    dev = args.device
    stack = lambda x: x[None].to(dev)
    return {
        "images": stack(clip["img_u8"].float().div(255).permute(0, 1, 2, 3)),
        "extrinsics": stack(clip["gt_extri"]),
        "intrinsics": stack(clip["gt_intri"]),
        "depths": stack(clip["gt_depth"]),
        "point_masks": stack(clip["pmask"]),
        "motion_mask": stack(clip["motion"]),
        "ids": torch.as_tensor(np.asarray(clip["ids"]))[None].to(dev),
    }


def _centre_principal_point(intri, H, W):
    """GT intrinsics with the principal point forced to the image centre.

    Matches what pose_encoding_to_extri_intri can express, so an identity test is not
    charged for information the pose encoding cannot carry.
    """
    K = intri.clone()
    K[..., 0, 2] = W / 2.0
    K[..., 1, 2] = H / 2.0
    return K


def _trajectory_extent(extri):
    """Spread of the camera centres, as a length in the clip's own units.

    C = -R^T t is the camera position in world coordinates; its bounding-box diagonal is a
    scale every clip has, whatever units its depths happen to be in.
    """
    R, t = extri[..., :3, :3], extri[..., :3, 3:]
    C = -R.transpose(-1, -2).matmul(t)[..., 0]                          # (B,S,3)
    extent = (C.amax(dim=1) - C.amin(dim=1)).norm(dim=-1)               # (B,)
    return float(extent.mean().clamp(min=1e-6))


def _decompose_floor(extri, intri, H, W):
    """Attribute a non-zero identity residual to a specific step of the round trip.

    The identity case sends GT geometry through `extri_intri_to_pose_encoding` and back. Any
    of three steps can lose something, and they are worth telling apart rather than guessing:

      rotation   R -> quaternion -> R. Exact for a proper rotation, but NOT if the stored R
                 has drifted off SO(3) (float32 accumulation through the dataset's crop /
                 resize / 90-degree-rotation path): the quaternion path silently
                 re-orthonormalises, so the "prediction" gets a cleaner R than the target.
      focal      fx -> FoV -> fx, via atan/tan. Only float round-off.
      principal  cx, cy are NOT stored at all; they come back as (W/2, H/2).

    Printed as an equivalent pixel error, using err_pixels ~ err_radians * focal, so the
    numbers can be compared against the measured floor directly.
    """
    R = extri[..., :3, :3].reshape(-1, 3, 3).double()
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    ortho = float((R.matmul(R.transpose(-1, -2)) - eye).abs().max())
    R_rt = quat_to_mat(mat_to_quat(R.float())).double()
    rot_rt = float((R - R_rt).abs().max())

    fx = intri[..., 0, 0].reshape(-1).double()
    fy = intri[..., 1, 1].reshape(-1).double()
    fov_w = 2 * torch.atan((W / 2) / fx)
    fx_rt = float(((W / 2) / torch.tan(fov_w / 2) - fx).abs().max())
    focal = float(fx.mean())

    dcx = float((intri[..., 0, 2] - W / 2.0).abs().max())
    dcy = float((intri[..., 1, 2] - H / 2.0).abs().max())

    print("[1c] floor decomposition (equivalent pixel error at focal "
          f"{focal:.1f}):")
    print(f"     R off SO(3)        {ortho:.2e}          ~{ortho * focal:.4f} px")
    print(f"     R quat round-trip  {rot_rt:.2e}          ~{rot_rt * focal:.4f} px")
    print(f"     focal round-trip   {fx_rt:.2e} px focal  ~{fx_rt:.4f} px")
    print(f"     principal point    dropped, off by       {max(dcx, dcy):.4f} px")
    return {"ortho": ortho, "rot_round_trip": rot_rt, "focal_round_trip": fx_rt,
            "pp_offset": max(dcx, dcy), "focal": focal}


def _to_px(loss):
    """Smooth-L1 (beta=1) value -> the per-pixel error that would produce it, in pixels.

    Below the transition smooth-L1 is x^2/2, so x = sqrt(2*loss). Only a rough read, but it
    turns an opaque loss value into the unit the residual is actually measured in.
    """
    return float(np.sqrt(2.0 * max(loss, 0.0))) if loss < 0.5 else float(loss + 0.5)


def run(batch, device, tol=1e-6, floor_warn=0.05):
    extri, intri = batch["extrinsics"], batch["intrinsics"]
    depth = batch["depths"]
    H, W = batch["images"].shape[-2:]
    hw = (H, W)
    ok = True

    # --- 1a. identity with the loss's own defaults (the real correctness assertion) -------
    # target_from_pose_encoding defaults to True, so the GT side is projected onto what the
    # pose encoding can express and the identity case is genuinely reachable.
    intri_c = _centre_principal_point(intri, H, W)
    batch_c = dict(batch); batch_c["intrinsics"] = intri_c
    out = compute_ego_flow_loss(_as_predictions(extri, intri_c, depth, hw), batch_c)
    ident = float(out["loss_ego_flow"])
    pairs = float(out["loss_ego_flow_pairs"])
    kept = float(out["loss_ego_flow_kept"])
    print(f"[1a] identity (K matched)  loss={ident:.3e}  pairs={pairs:.0f}  kept={kept:.3f}")
    if not (ident < tol):
        print(f"     FAIL: expected < {tol} — the geometry is inconsistent somewhere"); ok = False
    if pairs < 1:
        print("     FAIL: no usable pairs — the term is inert, so [1a] passed vacuously"); ok = False
    if kept < 0.9:
        print(f"     WARN: keep rate {kept:.3f} is low for an identity case")

    # --- 1b. the floor that target_from_pose_encoding removes ---------------------------
    # Same identity case scored against the RAW GT. The gap between [1a] and [1b] is exactly
    # what the flag buys, and it is worth printing rather than trusting: it is the part of the
    # target the model is structurally unable to reach.
    out_f = compute_ego_flow_loss(_as_predictions(extri, intri, depth, hw), batch,
                                  target_from_pose_encoding=False)
    floor = float(out_f["loss_ego_flow"])
    dcx = float((intri[..., 0, 2] - W / 2.0).abs().mean())
    dcy = float((intri[..., 1, 2] - H / 2.0).abs().mean())
    print(f"[1b] raw-GT target         loss={floor:.3e}  (~{_to_px(floor):.3f} px)"
          f"  pp offset {dcx:.2f},{dcy:.2f} px")
    print("     ^ the floor if target_from_pose_encoding were off. [1a] should be far below it.")
    _decompose_floor(extri, intri, H, W)
    if floor > floor_warn:
        print(f"     NOTE: floor {floor:.3e} is large; keep target_from_pose_encoding on.")

    # --- 2. perturbed pose, scaled to the clip's own scale ------------------------------
    scale = _trajectory_extent(extri)
    print(f"[2] trajectory extent {scale:.3f} (clip units) — offsets are fractions of it")
    prev, biggest = 0.0, 0.0
    for frac in (0.01, 0.05, 0.2):
        pert = extri.clone()
        pert[:, 1:, :3, 3] += frac * scale                    # shift every frame but the first
        val = float(compute_ego_flow_loss(_as_predictions(pert, intri, depth, hw), batch)["loss_ego_flow"])
        print(f"[2] pose +{frac:<5g} extent   loss={val:.4f}  (~{_to_px(val):.3f} px)")
        if not (val > prev):
            print("    FAIL: the loss did not grow with the perturbation"); ok = False
        prev, biggest = val, val
    # A perturbation of a fifth of the trajectory is a gross error; if it does not clear the
    # floor by a wide margin, the term cannot see pose error on this clip at all.
    if biggest < 20 * max(ident, 1e-12):
        print(f"    FAIL: largest perturbation ({biggest:.3e}) is not clear of the identity"
              f" residual ({ident:.3e})"); ok = False

    # --- 3. duplicate frames -----------------------------------------------------------
    dup = dict(batch)
    dup["ids"] = torch.zeros_like(batch["ids"])                         # every dt == 0
    out_d = compute_ego_flow_loss(_as_predictions(extri, intri, depth, hw), dup)
    dpairs = float(out_d["loss_ego_flow_pairs"])
    print(f"[3] all dt==0              pairs={dpairs:.0f}  loss={float(out_d['loss_ego_flow']):.3e}")
    if dpairs != 0:
        print("    FAIL: duplicate frames were scored"); ok = False

    return ok


def main():
    ap = argparse.ArgumentParser(description="Self-test compute_ego_flow_loss")
    ap.add_argument("--dataset", default=None, choices=["po", "tartanair", "waymo", "spring"],
                    help="omit for synthetic geometry (needs no data on disk)")
    ap.add_argument("--seq_index", type=int, default=0)
    ap.add_argument("--img_per_seq", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="tolerance for [1a], the principal-point-matched identity case")
    ap.add_argument("--floor_warn", type=float, default=0.05,
                    help="fail if the [1b] principal-point floor exceeds this")
    # accepted so build_train_dataset/load_train_clip can be reused unchanged
    ap.add_argument("--po_dir", default=data_path("train", "point_odyssey"))
    ap.add_argument("--tartanair_dir", default=data_path("train", "tartanair"))
    ap.add_argument("--waymo_dir", default=data_path("train", "waymo_processed"))
    ap.add_argument("--spring_dir", default=data_path("train", "spring"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--img_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    batch = real_batch(args) if args.dataset else synthetic_batch(device=args.device)
    src = args.dataset or "synthetic"
    print(f"ego_flow self-test on {src}  ({tuple(batch['images'].shape)})")
    ok = run(batch, args.device, tol=args.tol, floor_warn=args.floor_warn)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
