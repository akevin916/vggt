#!/usr/bin/env python3
"""Render an accumulating reconstruction with the frames blended, not fought over.

The clouds dumped by ``benchmark/eval_{lesion,gastric}.py`` and ``diag/dump_recon.py``
used to render as a patchwork: each screen pixel took its colour from whichever single
frame happened to be nearest, that winner flipped abruptly across the image (sub-mm depth
differences decide it), and every brightness difference between frames therefore showed up
as a hard seam.

This draws them the other way round. Every frame that lands on a pixel contributes to it,
weighted by a feather that fades out towards that frame's own borders, and only
contributions within ``--blend_tol`` of the nearest depth are averaged so a genuinely
occluded background never bleeds through. The colour of a spot no longer depends on who
won, so the seams have nothing to attach to.

Tried and dropped (2026-08-20): per-frame photometric harmonisation (1/d^2 + Lab lightness
matching) and voxel colour averaging both made things worse -- the first applies a
different constant to each frame, which is precisely what creates a step at a frame
boundary; the second cannot help where frames do not overlap in space, which is the lesion
set's whole problem. Outlier culling helped on its own but is not needed once frames are
averaged rather than picked between.

Everything here is CPU and re-runnable; nothing is re-inferred.

Usage (from training/):
  python diag/vis/cloud_polish.py \
      --seq_npz ../outputs/eval_lesion/scared_cam_b16/seq/病灶3/seq.npz --exp lesion_病灶3
  python diag/vis/cloud_polish.py \
      --seq_npz ../outputs/eval_gastric/scared_cam_b16/seq_000189_000432/seq.npz \
      --image_dir ../data/eval/gastric/prep/seq_000189_000432/image --exp gastric_seq189
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval_utils.media_io import write_video
from eval_utils.paths import output_dir_for_exp
from eval_utils.warp_psnr import to_homogeneous_extrinsic, unproject
from vggt.utils.load_fn import load_and_preprocess_images

TOOL = "cloud_polish"


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def load_images(image_dir, n, hw):
    """The exact tensor the model saw, as [S,H,W,3] float in [0,1]."""
    names = sorted(f for f in os.listdir(image_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))[:n]
    t = load_and_preprocess_images([os.path.join(image_dir, f) for f in names], mode="crop")
    imgs = t.permute(0, 2, 3, 1).numpy().astype(np.float32)
    if imgs.shape[1:3] != tuple(hw):
        raise SystemExit(f"image size {imgs.shape[1:3]} != depth size {tuple(hw)}")
    return imgs


def load_seq(seq_npz, image_dir):
    z = np.load(seq_npz)
    depth = z["depth"].astype(np.float32)
    K = z["intrinsic"].astype(np.float64)
    E = z["extrinsic"].astype(np.float64)
    if "images" in z.files:
        imgs = z["images"].astype(np.float32) / 255.0
    else:
        if not image_dir:
            raise SystemExit("seq.npz carries no images; pass --image_dir")
        imgs = load_images(image_dir, len(depth), depth.shape[1:3])
    return depth, K, E, imgs


# ---------------------------------------------------------------------------
# cloud assembly
# ---------------------------------------------------------------------------

def cam_to_world(cam_pts, E):
    M = np.linalg.inv(to_homogeneous_extrinsic(E))               # camera-to-world
    flat = cam_pts.reshape(-1, 3).astype(np.float64)
    return (flat @ M[:3, :3].T + M[:3, 3]).reshape(cam_pts.shape).astype(np.float32)


def feather_weights(hw, margin_frac, floor):
    """Per-pixel confidence for a frame's own contribution, fading to ~0 at its borders.

    A frame's edge pixels are where its depth is worst and where its footprint on the
    surface simply ends; letting them vote as loudly as its centre is what turns the
    boundary between two frames into a visible line. Fading them out means the handover
    between frames happens gradually instead of at a hard edge.
    """
    H, W = hw
    def ramp(n, margin):
        i = np.arange(n, dtype=np.float32)
        t = np.clip(np.minimum(i, n - 1 - i) / max(margin, 1e-6), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)                    # smoothstep
    wy = ramp(H, margin_frac * H)[:, None]
    wx = ramp(W, margin_frac * W)[None, :]
    return np.maximum(wy * wx, floor).astype(np.float32)


def build_cloud(depth, K, E, imgs, args):
    """[N,3] points + [N,3] colours + [N] blend weight + [N] frame_id, per-frame budgeted
    (same rule as benchmark save_cloud: an even share per frame, so accumulation reads as
    accumulation rather than arriving in fits and starts)."""
    S, H, W = depth.shape
    per_frame = max(1, args.max_points // max(1, S))
    rng = np.random.default_rng(0)
    fmap = feather_weights((H, W), args.feather_margin, args.feather_floor)

    pts, cols, wts, fids = [], [], [], []
    for i in range(S):
        d = depth[i]
        world = cam_to_world(unproject(d, K[i]), E[i])
        keep = (d > 0) & np.isfinite(world).all(axis=-1)
        if args.depth_pct < 100 and (d > 0).any():
            keep &= d <= np.percentile(d[d > 0], args.depth_pct)

        wk, ck, fk = world[keep], imgs[i][keep], fmap[keep]
        if len(wk) > per_frame:
            sel = rng.choice(len(wk), per_frame, replace=False)
            wk, ck, fk = wk[sel], ck[sel], fk[sel]
        pts.append(wk); cols.append(ck); wts.append(fk)
        fids.append(np.full(len(wk), i, np.int32))

    pts = np.concatenate(pts).astype(np.float64)
    cols = np.clip(np.concatenate(cols), 0, 1) * 255.0
    wts = np.concatenate(wts).astype(np.float32)
    fids = np.concatenate(fids)
    print(f"  {len(pts):,} points over {S} frames")
    return pts, cols.astype(np.uint8), wts, fids, S


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def look_at(eye, target, up=(0.0, -1.0, 0.0)):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f) + 1e-12
    up = np.asarray(up, float)
    if abs(float(f @ (up / (np.linalg.norm(up) + 1e-12)))) > 0.99:
        up = np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up); r /= np.linalg.norm(r) + 1e-12
    u = np.cross(r, f)
    # -u, not u. ``up`` is the world's UP direction (-Y under the Y-down convention these
    # clouds live in), but the camera's y row must point the way image v grows, i.e. DOWN.
    # Stacking [r, u, f] made det(R) = -1 -- a reflection, not a rotation -- so every render
    # came out vertically flipped (a point above the centre projected below it). Verified
    # numerically: for f=+Z, up=-Y, the point (0,-1,5) gave cam_y=+1 before and -1 after.
    R = np.stack([r, -u, f])
    return np.hstack([R, (-R @ np.asarray(eye, float))[:, None]])


def frame_cloud(points, hw, fov_deg, dist_scale, azim_deg, elev_deg):
    """A camera that frames the *whole* cloud, so the view never rescales as points are
    added -- accumulation must read as accumulation, not as a zoom-out."""
    H, W = hw
    lo, hi = np.percentile(points, [1, 99], axis=0)
    centre = (lo + hi) / 2.0
    extent = float(np.linalg.norm(hi - lo)) + 1e-9
    f = 0.5 * W / np.tan(np.deg2rad(fov_deg) / 2.0)
    K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
    a, e = np.deg2rad(azim_deg), np.deg2rad(elev_deg)
    direction = np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])
    eye = centre - direction * extent * dist_scale
    return K, look_at(eye, centre)


def blend_pixels(pix, z, col, w, hw, tol, bg):
    """Every frame that sees a pixel contributes to it, weighted -- instead of the single
    nearest point winning outright.

    Only contributions within ``tol`` of the nearest depth are averaged, so a genuinely
    occluded background never bleeds through the surface in front of it.
    """
    H, W = hw
    img = np.full((H * W, 3), float(bg), np.float32)
    if len(pix) == 0:
        return img.reshape(H, W, 3).astype(np.uint8)

    order = np.argsort(pix, kind="stable")
    pix_s, z_s = pix[order], z[order]
    starts = np.flatnonzero(np.r_[True, pix_s[1:] != pix_s[:-1]])
    znear = np.minimum.reduceat(z_s, starts)
    counts = np.diff(np.r_[starts, len(pix_s)])
    keep = z_s <= np.repeat(znear, counts) * (1.0 + tol)

    idx = pix_s[keep]
    ws = w[order][keep]
    cs = col[order][keep]
    denom = np.bincount(idx, weights=ws, minlength=H * W)
    hit = denom > 0
    for c in range(3):
        num = np.bincount(idx, weights=ws * cs[:, c], minlength=H * W)
        img[hit, c] = num[hit] / denom[hit]
    return np.clip(img, 0, 255).reshape(H, W, 3).astype(np.uint8)


def render(points, colors, weights, K, E, hw, radius=1, bg=12, blend_tol=0.02):
    H, W = hw
    if len(points) == 0:
        return np.full((H, W, 3), bg, np.uint8)

    cam = points @ E[:3, :3].T + E[:3, 3]
    z = cam[:, 2]
    m = z > 1e-6
    if not m.any():
        return np.full((H, W, 3), bg, np.uint8)
    cam, col, z, wgt = cam[m], colors[m].astype(np.float32), z[m], weights[m]

    proj = cam @ K.T
    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not inb.any():
        return np.full((H, W, 3), bg, np.uint8)
    u = u[inb].astype(np.int32); v = v[inb].astype(np.int32)
    col, z, wgt = col[inb], z[inb], wgt[inb]

    d = np.arange(-radius, radius + 1, dtype=np.int32)
    dy, dx = np.meshgrid(d, d, indexing="ij")
    dy = dy.ravel(); dx = dx.ravel()
    vv = (v[:, None] + dy[None, :]).ravel()
    uu = (u[:, None] + dx[None, :]).ravel()
    zz = np.repeat(z, len(dy))
    cc = np.repeat(col, len(dy), axis=0)
    ww = np.repeat(wgt, len(dy))
    clip = (vv >= 0) & (vv < H) & (uu >= 0) & (uu < W)
    return blend_pixels((vv[clip] * W + uu[clip]).astype(np.int64), zz[clip],
                        cc[clip], ww[clip], (H, W), blend_tol, bg)


def label(img, text, y=8, x=8):
    a = np.ascontiguousarray(img)
    cv2.putText(a, text, (x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return a


def progress_bar(img, frac, h=5):
    img[-h:, :, :] = 40
    img[-h:, :int(img.shape[1] * frac), :] = (90, 200, 120)
    return img


# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq_npz", required=True)
    ap.add_argument("--image_dir", default=None, help="needed when seq.npz has no images")
    ap.add_argument("--exp", default="blend", help="<exp> level under outputs/cloud_polish/")
    ap.add_argument("--out_dir", default=None)
    # cloud
    ap.add_argument("--max_points", type=int, default=1_200_000)
    ap.add_argument("--depth_pct", type=float, default=98.0)
    # blending
    ap.add_argument("--feather_margin", type=float, default=0.25,
                    help="fraction of the frame over which its weight fades in from the edge")
    ap.add_argument("--feather_floor", type=float, default=0.02)
    ap.add_argument("--blend_tol", type=float, default=0.02,
                    help="depth window, relative to the nearest hit, that still counts as "
                         "the same surface rather than as an occluded one behind it")
    # camera / video
    ap.add_argument("--size", nargs=2, type=int, default=[540, 720], metavar=("H", "W"))
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--dist", type=float, default=1.1)
    ap.add_argument("--azim", type=float, default=0.0)
    ap.add_argument("--elev", type=float, default=12.0)
    ap.add_argument("--orbit", type=float, default=40.0)
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--hold", type=int, default=10)
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or output_dir_for_exp(args.exp, TOOL)
    os.makedirs(out_dir, exist_ok=True)

    depth, K, E, imgs = load_seq(args.seq_npz, args.image_dir)
    print(f"{args.seq_npz}: {len(depth)} frames {depth.shape[1]}x{depth.shape[2]}")
    pts, cols, wts, fids, n = build_cloud(depth, K, E, imgs, args)

    H, W = args.size
    total = n + args.hold
    frames = []
    for t in range(total):
        step = min(t, n - 1)
        frac = step / max(1, n - 1)
        K_r, E_r = frame_cloud(pts, (H, W), args.fov, args.dist,
                               args.azim + args.orbit * frac, args.elev)
        keep = fids <= step
        img = render(pts[keep], cols[keep], wts[keep], K_r, E_r, (H, W),
                     radius=args.radius, blend_tol=args.blend_tol)
        img = label(img, f"frame {step+1}/{n}  {int(keep.sum()):,} pts")
        frames.append(progress_bar(img, (step + 1) / n))
        if t % max(1, total // 5) == 0:
            print(f"  rendering {t+1}/{total}", flush=True)

    path = os.path.join(out_dir, "blend.mp4")
    write_video(path, frames, fps=args.fps)
    still = os.path.join(out_dir, "blend_final.png")
    cv2.imwrite(still, cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))
    print(f"  -> {path}\n  -> {still}")


if __name__ == "__main__":
    main()
