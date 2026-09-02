#!/usr/bin/env python3
"""Convert the raw C3VD release into the layout data/datasets/c3vd.py expects.

C3VD (Colonoscopy 3D Video Dataset, Bobrow et al.) is real Olympus CF-HQ190L footage of
silicone colon phantoms, registered against a CT-derived mesh. 22 sequences, 10,015 frames,
each with GT depth and a GT camera pose. Everything below was read out of the official
loader at ``reference/C3VD/utils/exampleDataLoader.py`` -- none of it is guesswork, but
``--verify`` re-derives the two conventions that would silently corrupt training if wrong.

RAW LAYOUT (one flat directory per sequence, as the zips unpack):

    <raw>/cecum_t1_a/
        0000_color.png       1350x1080 uint8  (fisheye, black corners)
        0000_depth.tiff      1350x1080 uint16
        pose.txt             one COMMA-separated 16-vector per frame
        coverage_mesh.obj    (unused here)
        ... plus *_normals.tiff, *_flow.tiff, *_occlusion.png which we never read

FIVE THINGS THE RAW RELEASE GETS "WRONG" FOR US, all absorbed here:

  1. THE CAMERA IS A FISHEYE, not a pinhole. Intrinsics follow Scaramuzza's omnidirectional
     model (a polynomial mapping image radius -> ray z), so there is no K. VGGT's whole
     geometry stack -- depth, extrinsics, point maps -- assumes a pinhole. Feeding raw
     fisheye frames with an invented K makes depth, pose and point-map supervision mutually
     inconsistent. We therefore RESAMPLE every frame onto a real pinhole camera and emit the
     matching K. EndoSfM3D skips this step; it can afford to, because it is self-supervised
     at 256x320 with learn_intrinsics=True. We cannot.

  2. POSE.TXT IS TRANSPOSED. The official loader does ``p_row.dot(M)``, i.e. it treats
     points as ROW vectors. Under the column-vector convention everyone else uses, the
     stored matrix is ``c2w.T``. So the chain is  M -> transpose -> invert -> world-to-cam.
     Skipping the transpose still yields a valid-looking rigid matrix, which is exactly why
     ``--verify`` scores both conventions instead of trusting this comment.

  3. DEPTH ENCODES INVALID AS TWO DIFFERENT VALUES. 0 (no surface) and 65535 (clipped past
     100 mm) both mean "no data"; only the open interval carries signal. Scale is
     ``v / 65535 * 100`` millimetres. We re-encode to the SCARED convention (uint16 counts of
     0.01 mm, 0 = invalid), which tops out at 10000 counts here -- no clipping.

  4. FISHEYE CORNERS ARE BLACK. They survive undistortion as garbage pixels. Any destination
     pixel whose source ray falls outside the calibrated radius, or outside the image, is
     written as black with depth 0, so the point loss never sees it.

  5. FRAME INDICES RESTART AT 0 PER SEQUENCE and are already gapless, so frames.txt is just
     0..N-1. It is written anyway to keep the loader identical in shape to SCARED's.

FOV / RESOLUTION. ``--fov_deg`` picks how much of the fisheye survives. A colonoscope sees
~140-170 degrees; a pinhole cannot represent that (the projection diverges at 180), so the
periphery has to be either dropped or stretched. Small FOV = crisp but throws away the
sidewall the endoscope relies on and shrinks frame-to-frame overlap (worse pose). Large FOV
= keeps coverage but smears the edge, where one source pixel is spread over many. Output
size only has to be large enough that the CENTRE is not resampled below native detail;
beyond ~1024 it is wasted disk, since training resizes to 518 anyway. Use ``--fov_sweep`` to
dump one sample frame per FOV and pick by eye instead of by argument.

Usage::

    # 0. after unzipping ONE sequence -- confirm the layout matches the docstring
    python -m pipeline.data.preprocess.c3vd_convert --raw_dir ../data/train/c3vd_raw --inspect cecum_t1_a
    # 1. prove the pose convention on real numbers before converting 10k frames
    python -m pipeline.data.preprocess.c3vd_convert --raw_dir ../data/train/c3vd_raw --verify cecum_t1_a
    # 2. look at what each FOV costs
    python -m pipeline.data.preprocess.c3vd_convert --raw_dir ../data/train/c3vd_raw --fov_sweep cecum_t1_a
    # 3. convert everything
    python -m pipeline.data.preprocess.c3vd_convert --raw_dir ../data/train/c3vd_raw
"""
from __future__ import annotations

import argparse
import json
import math
import os
import os.path as osp
import re

import cv2
import numpy as np

from pipeline.data.paths import data_path

# Scaramuzza omnidirectional intrinsics for the CF-HQ190L, copied verbatim from the official
# loader (reference/C3VD/utils/exampleDataLoader.py). a1 is 0 in that model and unused.
CALIB = {
    "width": 1350, "height": 1080,
    "cx": 678.544839263292, "cy": 542.975887548343,
    "a0": 769.243600037458, "a2": -0.000812770624150226,
    "a3": 6.25674244578925e-07, "a4": -1.19662182144280e-09,
    "c": 0.999986882249990, "d": 0.00288273829525059, "e": -0.00296316513429569,
}

DEPTH_SCALE = 100.0        # output uint16 counts per millimetre (0.01 mm resolution)
C3VD_DEPTH_MAX_MM = 100.0  # the release clamps here; 100 mm * 100 = 10000 fits in uint16

SEQ_RE = re.compile(r"^(?P<ana>[a-z]+)_t(?P<tex>\d+)_(?P<vid>[a-z])$")

# Split by TEXTURE, not by frame. Texture 4 is held out whole, which also removes the only
# descending-colon sequence -- so the test set is unseen texture AND unseen anatomy. val is
# one sequence per remaining anatomy, again whole, so no val frame is a train frame's
# neighbour (adjacent frames of an endoscope are near-duplicates; a frame-level split would
# leak outright).
SPLIT_MAP = {
    "test": ["cecum_t4_a", "cecum_t4_b", "trans_t4_a", "trans_t4_b", "desc_t4_a"],
    "val":  ["cecum_t2_c", "trans_t3_b", "sigmoid_t3_b"],
    # everything else -> train
}


# ---------------------------------------------------------------------------
# the omnidirectional model
# ---------------------------------------------------------------------------

def _poly_z(rho, k=CALIB):
    """Scaramuzza's radius->axial polynomial. Positive near the axis, falls with radius."""
    return k["a0"] + k["a2"] * rho ** 2 + k["a3"] * rho ** 3 + k["a4"] * rho ** 4


def ray_map(k=CALIB):
    """Per-source-pixel normalised ray direction, [H, W, 3]. Mirrors the official loader."""
    ix, iy = np.meshgrid(np.arange(k["width"]), np.arange(k["height"]))
    uvp = np.stack([ix - k["cx"], iy - k["cy"]], axis=-1)
    inv_stretch = np.linalg.inv(np.array([[k["c"], k["d"]], [k["e"], 1.0]]))
    uvpp = np.einsum("ij,...j->...i", inv_stretch, uvp)
    rho = np.linalg.norm(uvpp, axis=-1)
    rays = np.stack([uvpp[..., 0], uvpp[..., 1], _poly_z(rho, k)], axis=-1)
    n = np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays / np.where(n == 0, 1.0, n)


def pinhole_maps(fov_deg, out_w, out_h, k=CALIB):
    """Build cv2.remap tables taking a pinhole destination grid back to fisheye source pixels.

    Returns ``(map_x, map_y, valid, K)``. The inversion is the only fiddly part: a
    destination pixel has ray (x, y, 1), whose in-plane/axial ratio is r = hypot(x, y); a
    source pixel at radius rho has ratio rho / z(rho). That ratio rises monotonically with
    rho until z crosses zero (the model's horizon), so it inverts by a 1-D lookup over the
    monotone stretch. Anything past the last usable rho has no source pixel and is dropped.
    """
    f = (out_w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    cx_o, cy_o = (out_w - 1) / 2.0, (out_h - 1) / 2.0
    K = np.array([[f, 0.0, cx_o], [0.0, f, cy_o], [0.0, 0.0, 1.0]], dtype=np.float64)

    rho_max = math.hypot(max(k["cx"], k["width"] - k["cx"]),
                         max(k["cy"], k["height"] - k["cy"]))
    rho_lut = np.linspace(0.0, rho_max, 8192)
    z_lut = _poly_z(rho_lut, k)
    ratio = np.divide(rho_lut, z_lut, out=np.zeros_like(rho_lut), where=z_lut > 0)
    # keep only the leading strictly-increasing, z>0 stretch
    ok = z_lut > 0
    ok[1:] &= np.diff(ratio) > 0
    last = int(np.argmin(ok)) if not ok.all() else len(ok)
    rho_lut, ratio = rho_lut[:last], ratio[:last]

    u, v = np.meshgrid(np.arange(out_w), np.arange(out_h))
    x, y = (u - cx_o) / f, (v - cy_o) / f
    r = np.hypot(x, y)
    valid = r <= ratio[-1]
    rho = np.interp(r, ratio, rho_lut)

    scale = np.divide(rho, r, out=np.zeros_like(r), where=r > 0)
    uvpp_x, uvpp_y = x * scale, y * scale
    map_x = k["c"] * uvpp_x + k["d"] * uvpp_y + k["cx"]
    map_y = k["e"] * uvpp_x + 1.0 * uvpp_y + k["cy"]
    valid &= (map_x >= 0) & (map_x <= k["width"] - 1) & (map_y >= 0) & (map_y <= k["height"] - 1)
    return map_x.astype(np.float32), map_y.astype(np.float32), valid, K


# ---------------------------------------------------------------------------
# raw IO
# ---------------------------------------------------------------------------

def frame_stems(seq_dir, suffix="_color.png"):
    """Map frame index -> the filename stem that actually exists on disk, for one modality.

    NOT reconstructible from the index, and NOT shared between modalities. ``cecum_t1_a``
    ships UNPADDED colour names (``9_color.png``) but PADDED depth names (``0009_depth.tiff``)
    in the same directory; the other 21 sequences pad both. The release's 2023 renaming
    missed that directory's colour frames only. So each modality's stem has to be read off
    disk rather than formatted from the index -- and the two dicts must be kept separate.
    """
    rx = re.compile(r"^(?P<stem>\d+)" + re.escape(suffix) + r"$")
    stems = {}
    for f in os.listdir(seq_dir):
        m = rx.match(f)
        if m:
            stems[int(m["stem"])] = m["stem"]
    return dict(sorted(stems.items()))


def frame_ids(seq_dir):
    return list(frame_stems(seq_dir))


def read_depth_mm(seq_dir, stem):
    """Raw 16-bit depth -> float millimetres, with BOTH invalid codes mapped to 0."""
    p = osp.join(seq_dir, f"{stem}_depth.tiff")
    d = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(p)
    if d.ndim == 3:                      # defensive: a 3-channel read would silently reorder
        d = d[..., 0]
    bad = (d == 0) | (d == 65535)
    mm = d.astype(np.float32) / 65535.0 * C3VD_DEPTH_MAX_MM
    mm[bad] = 0.0
    return mm


def read_poses(seq_dir, transpose=True):
    """pose.txt -> [N, 4, 4] camera-to-world under the COLUMN-vector convention.

    ``transpose`` exists so --verify can score the alternative; production always wants True
    (see docstring item 2).
    """
    out = []
    with open(osp.join(seq_dir, "pose.txt")) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            m = np.fromstring(line.replace(",", " "), dtype=np.float64, sep=" ")
            if m.size != 16:
                raise ValueError(f"pose line with {m.size} elements in {seq_dir}")
            m = m.reshape(4, 4)
            out.append(m.T if transpose else m)
    return np.stack(out)


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------

def convert_one(seq, raw_dir, out_root, split, args):
    seq_dir = osp.join(raw_dir, seq)
    m = SEQ_RE.match(seq)
    if m is None:
        raise ValueError(f"unrecognised sequence name {seq!r}")
    out_dir = osp.join(out_root, split, m["ana"], f"t{m['tex']}_{m['vid']}")
    img_dir, dep_dir, cam_dir = (osp.join(out_dir, d) for d in ("image_left", "depth_left", "cam_data"))
    for d in (img_dir, dep_dir, cam_dir):
        os.makedirs(d, exist_ok=True)

    map_x, map_y, valid, K = pinhole_maps(args.fov_deg, args.out_w, args.out_h)
    cstems = frame_stems(seq_dir, "_color.png")
    dstems = frame_stems(seq_dir, "_depth.tiff")
    fids = list(cstems)
    if list(dstems) != fids:
        raise ValueError(f"{seq}: colour and depth frame indices disagree")
    c2w = read_poses(seq_dir)
    if len(c2w) != len(fids):
        raise ValueError(f"{seq}: {len(fids)} frames but {len(c2w)} poses")

    valid_frac, extri = [], []
    for i, fid in enumerate(fids):
        if args.skip_existing and osp.exists(osp.join(dep_dir, f"{fid:06d}.png")):
            pass
        else:
            color = cv2.imread(osp.join(seq_dir, f"{cstems[fid]}_color.png"), cv2.IMREAD_COLOR)
            color = cv2.remap(color, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
            color[~valid] = 0
            cv2.imwrite(osp.join(img_dir, f"{fid:06d}.png"), color)

            # NEAREST, not LINEAR: interpolating across a depth discontinuity invents surface
            # that is in front of nothing, and 0 means invalid so it would also bleed holes.
            mm = cv2.remap(read_depth_mm(seq_dir, dstems[fid]), map_x, map_y, cv2.INTER_NEAREST,
                           borderValue=0)
            mm[~valid] = 0.0
            d16 = np.clip(np.rint(mm * DEPTH_SCALE), 0, 65535).astype(np.uint16)
            cv2.imwrite(osp.join(dep_dir, f"{fid:06d}.png"), d16)

        d16 = cv2.imread(osp.join(dep_dir, f"{fid:06d}.png"), cv2.IMREAD_UNCHANGED)
        valid_frac.append(float((d16 > 0).mean()))
        w2c = np.linalg.inv(c2w[i])
        extri.append(w2c[:3, :4].reshape(-1))
        if args.progress and i % 50 == 0:
            print(f"  {seq} {i}/{len(fids)}", flush=True)

    np.savetxt(osp.join(cam_dir, "intrinsics.txt"),
               np.array([[K[0, 0], K[1, 1], K[0, 2], K[1, 2]]]), fmt="%.8f")
    np.savetxt(osp.join(cam_dir, "extrinsics.txt"), np.stack(extri), fmt="%.8f")
    np.savetxt(osp.join(cam_dir, "frames.txt"), np.array(fids, dtype=np.int64), fmt="%d")
    np.savetxt(osp.join(cam_dir, "valid_frac.txt"), np.array(valid_frac), fmt="%.6f")
    with open(osp.join(out_dir, "meta.json"), "w") as f:
        json.dump({"source_seq": seq, "split": split, "num_frames": len(fids),
                   "fov_deg": args.fov_deg, "out_size": [args.out_w, args.out_h],
                   "K_fx_fy_cx_cy": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
                   "pinhole_pixel_frac": float(valid.mean()),
                   "depth_scale_counts_per_mm": DEPTH_SCALE,
                   "depth_max_mm": C3VD_DEPTH_MAX_MM,
                   "units": "millimetres", "extrinsics": "world-to-cam 3x4 row-major",
                   "calib": CALIB}, f, indent=2)
    print(f"[{split}] {seq} -> {out_dir}  ({len(fids)} frames, "
          f"median valid depth {np.median(valid_frac):.1%})")


# ---------------------------------------------------------------------------
# pre-flight modes
# ---------------------------------------------------------------------------

def inspect(seq_dir):
    print(f"=== {seq_dir}")
    names = sorted(os.listdir(seq_dir))
    print("files:", len(names), "| sample:", names[:6])
    stems = frame_stems(seq_dir, "_color.png")
    dstems = frame_stems(seq_dir, "_depth.tiff")
    fids = list(stems)
    print(f"padding: colour {'padded' if len(stems[fids[-1]]) == 4 else 'UNPADDED'}, "
          f"depth {'padded' if len(dstems[fids[-1]]) == 4 else 'UNPADDED'}")
    print(f"frames: {len(fids)}  range {fids[0]}..{fids[-1]}  "
          f"gapless={fids == list(range(fids[0], fids[-1] + 1))}")
    c = cv2.imread(osp.join(seq_dir, f"{stems[fids[0]]}_color.png"), cv2.IMREAD_UNCHANGED)
    print(f"color: shape {c.shape} dtype {c.dtype}   (docstring expects 1080x1350x3 uint8)")
    d = cv2.imread(osp.join(seq_dir, f"{dstems[fids[0]]}_depth.tiff"), cv2.IMREAD_UNCHANGED)
    print(f"depth: shape {d.shape} dtype {d.dtype} min {d.min()} max {d.max()} "
          f"zeros {np.mean(d == 0):.1%} clipped {np.mean(d == 65535):.1%}")
    mm = read_depth_mm(seq_dir, dstems[fids[0]])
    v = mm[mm > 0]
    print(f"depth mm: p1 {np.percentile(v, 1):.2f}  median {np.median(v):.2f}  "
          f"p99 {np.percentile(v, 99):.2f}")
    P = read_poses(seq_dir, transpose=False)
    print(f"pose.txt: {P.shape}  last row of frame0 {P[0, 3]}  last col {P[0, :, 3]}")
    step = np.linalg.norm(np.diff(P[:, 3, :3] if abs(P[0, 3, 3] - 1) < 1e-6 else P[:, :3, 3],
                                  axis=0), axis=1)
    print(f"per-frame translation: median {np.median(step):.3f}  p95 {np.percentile(step, 95):.3f}"
          f"  total path {step.sum():.1f}   (units should match depth mm)")


def verify(seq_dir, gap, cull):
    """Prove the pose convention on real geometry instead of trusting the docstring.

    Two independent questions, answered separately:

      (a) TRANSPOSE. Settled structurally, not by a metric: a homogeneous transform stores
          its translation either in the last column (column-vector convention) or the last
          row (row-vector). Whichever slot is [0,0,0,1] is the one that is NOT translation.
          Scoring this with a point metric is a trap -- reading the matrix under the wrong
          convention yields translation 0, which parks every frame's cloud on top of every
          other and therefore SCORES WELL for a camera that barely moves.

      (b) DIRECTION. Genuinely ambiguous, so it is measured: unproject two frames' GT depth,
          push both to world under c2w = T and under c2w = inv(T), and compare. Both are
          proper rigid transforms, so this comparison is fair. The correct one should land
          sub-millimetre; the wrong one is off by roughly the camera's travel.
    """
    from scipy.spatial import cKDTree
    rm = ray_map()
    dstems = frame_stems(seq_dir, "_depth.tiff")
    fids = list(dstems)
    a, b = fids[0], fids[min(gap, len(fids) - 1)]
    print(f"=== verify {osp.basename(seq_dir)}: frame {a} vs {b}")

    raw = read_poses(seq_dir, transpose=False)
    last_row, last_col = raw[:, 3, :], raw[:, :, 3]
    row_is_pad = np.allclose(last_row, [0, 0, 0, 1])
    col_is_pad = np.allclose(last_col, [0, 0, 0, 1])
    print(f"  (a) transpose: last row [0,0,0,1] in every frame? {row_is_pad};  "
          f"last col? {col_is_pad}")
    if col_is_pad == row_is_pad:
        raise RuntimeError("pose.txt matches neither convention cleanly -- inspect by hand")
    transpose = col_is_pad          # translation sits in the last ROW -> stored matrix is T.T
    print(f"      -> translation lives in the last {'ROW' if transpose else 'COLUMN'}; "
          f"{'transposing' if transpose else 'using as-is'}.")

    T = read_poses(seq_dir, transpose=transpose)
    R = T[:, :3, :3]
    orth = np.abs(R @ R.transpose(0, 2, 1) - np.eye(3)).max()
    print(f"      rotation block orthonormal to {orth:.2e}, det {np.linalg.det(R).mean():.6f}")

    def cloud(fid, c2w):
        mm = read_depth_mm(seq_dir, dstems[fid])
        pc = np.stack([mm * rm[..., 0] / rm[..., 2], mm * rm[..., 1] / rm[..., 2], mm], -1)
        pc = pc.reshape(-1, 3)[(mm > 0).reshape(-1)][::cull]
        return pc @ c2w[:3, :3].T + c2w[:3, 3]

    print("  (b) direction:")
    for name, mats in (("c2w = T          (official)", T),
                       ("c2w = inv(T)     (stored is w2c)", np.linalg.inv(T))):
        ca = cloud(a, mats[fids.index(a)])
        cb = cloud(b, mats[fids.index(b)])
        d, _ = cKDTree(ca).query(cb)
        print(f"      {name:34s} median NN {np.median(d):7.3f} mm   p90 {np.percentile(d, 90):7.3f} mm")
    print("      -> the smaller one is correct; a right answer is a fraction of a millimetre.")


def fov_sweep(seq_dir, out_dir, fovs, out_w, out_h):
    """Dump one undistorted sample per FOV, plus the three numbers the choice turns on.

      source kept   -- fraction of the fisheye's LIT pixels (depth>0, i.e. excluding the
                       black corners) that survive into the pinhole image. This is the
                       coverage you give up.
      centre px/px  -- source pixels consumed per output pixel at the optical axis. Below 1
                       means the output is upsampling, i.e. inventing resolution.
      edge px/px    -- the same at the image edge. The ratio edge/centre IS the distortion
                       cost: it says how much harder the periphery is being stretched than
                       the middle.
    """
    os.makedirs(out_dir, exist_ok=True)
    stems = frame_stems(seq_dir, "_color.png")
    dstems = frame_stems(seq_dir, "_depth.tiff")
    f0 = list(stems)[0]
    src = cv2.imread(osp.join(seq_dir, f"{stems[f0]}_color.png"), cv2.IMREAD_COLOR)
    lit = read_depth_mm(seq_dir, dstems[f0]) > 0
    seq = osp.basename(seq_dir)

    p = osp.join(out_dir, f"{seq}_fov_source.png")
    cv2.imwrite(p, src)
    print(f"source: {p}\n")
    print(f"{'FOV':>5} {'f px':>7} {'grid used':>10} {'source kept':>12} "
          f"{'centre px/px':>13} {'edge px/px':>11} {'edge/centre':>12}")

    tiles = []
    for fov in fovs:
        mx, my, valid, K = pinhole_maps(fov, out_w, out_h)
        img = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderValue=0)
        img[~valid] = 0

        # local resampling scale: |Jacobian| of the dest->source map, in source px per dest px
        gy_x, gx_x = np.gradient(mx.astype(np.float64))
        gy_y, gx_y = np.gradient(my.astype(np.float64))
        jac = np.sqrt(np.abs(gx_x * gy_y - gy_x * gx_y))
        cy_i, cx_i = out_h // 2, out_w // 2
        centre = jac[cy_i - 8:cy_i + 8, cx_i - 8:cx_i + 8].mean()
        edge = jac[cy_i - 8:cy_i + 8, :16].mean()

        # forward direction: which lit source pixels land inside the pinhole frame
        rm = ray_map()
        z = rm[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * rm[..., 0] / z + K[0, 2]
            v = K[1, 1] * rm[..., 1] / z + K[1, 2]
        inside = (z > 0) & (u >= 0) & (u <= out_w - 1) & (v >= 0) & (v <= out_h - 1)
        kept = float((inside & lit).sum()) / float(lit.sum())

        print(f"{fov:5.0f} {K[0,0]:7.1f} {valid.mean():9.1%} {kept:11.1%} "
              f"{centre:13.2f} {edge:11.2f} {edge/centre:12.2f}")
        p = osp.join(out_dir, f"{seq}_fov_{int(fov):03d}.png")
        cv2.imwrite(p, img)

        t = cv2.resize(img, (360, 360))
        cv2.rectangle(t, (0, 0), (360, 34), (0, 0, 0), -1)
        cv2.putText(t, f"FOV {fov:.0f}  kept {kept:.0%}  edge x{edge/centre:.1f}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)

    ncol = min(4, len(tiles))
    rows = [np.hstack(tiles[i:i + ncol] + [np.zeros_like(tiles[0])] * (ncol - len(tiles[i:i + ncol])))
            for i in range(0, len(tiles), ncol)]
    grid = osp.join(out_dir, f"{seq}_fov_grid.png")
    cv2.imwrite(grid, np.vstack(rows))
    print(f"\ngrid: {grid}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir", default=data_path("train", "c3vd_raw"),
                    help="directory holding the unzipped <seq>/ folders")
    ap.add_argument("--out_dir", default=data_path("train", "c3vd"))
    ap.add_argument("--seqs", nargs="*", default=None, help="default: every folder in --raw_dir")
    ap.add_argument("--fov_deg", type=float, default=100.0)
    ap.add_argument("--out_w", type=int, default=1024)
    ap.add_argument("--out_h", type=int, default=1024)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--inspect", metavar="SEQ", default=None)
    ap.add_argument("--verify", metavar="SEQ", default=None)
    ap.add_argument("--verify_gap", type=int, default=10,
                    help="frames between the two clouds. Keep it SMALL: at a large gap the two "
                         "views stop overlapping and the chamfer scores noise, which makes "
                         "both conventions look equally wrong (measured: gap 200 on "
                         "sigmoid_t2_a inverts the verdict).")
    ap.add_argument("--verify_cull", type=int, default=50)
    ap.add_argument("--fov_sweep", metavar="SEQ", default=None)
    ap.add_argument("--fov_list", nargs="*", type=float, default=[60, 70, 80, 90, 100, 110, 120, 130])
    ap.add_argument("--sweep_out", default=None,
                    help="default: outputs/c3vd_convert/fov_sweep/")
    args = ap.parse_args()

    if args.inspect:
        inspect(osp.join(args.raw_dir, args.inspect)); return
    if args.verify:
        verify(osp.join(args.raw_dir, args.verify), args.verify_gap, args.verify_cull); return
    if args.fov_sweep:
        out = args.sweep_out or osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))),
                                         "outputs", "c3vd_convert", "fov_sweep")
        fov_sweep(osp.join(args.raw_dir, args.fov_sweep), osp.abspath(out),
                  args.fov_list, args.out_w, args.out_h); return

    seqs = args.seqs or sorted(d for d in os.listdir(args.raw_dir)
                               if osp.isdir(osp.join(args.raw_dir, d)) and SEQ_RE.match(d))
    where = {s: sp for sp, ss in SPLIT_MAP.items() for s in ss}
    for s in seqs:
        convert_one(s, args.raw_dir, args.out_dir, where.get(s, "train"), args)


if __name__ == "__main__":
    main()
