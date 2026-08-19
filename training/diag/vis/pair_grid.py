"""2×2 grid video comparing three models on each stereo pair.

Layout (each cell = size // 2):
  ┌──────────────┬──────────────┐
  │  Left image  │  MonST3R     │
  ├──────────────┼──────────────┤
  │  VGGT-1B     │  Dyn-VGGT    │
  └──────────────┴──────────────┘

Each stereo pair contributes ``--frames_per_pair`` frames to the video
(a short orbit sweep so depth structure is visible).

Usage::

    cd training
    python diag/vis/pair_grid.py --folder 病灶3
    python diag/vis/pair_grid.py --folder 病灶1 --frames_per_pair 5 --size 720 960
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval_utils.media_io import write_video
from eval_utils.paths import output_dir_for_exp

TOOL = "pair_grid"

OUTPUTS = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "outputs")
)
LESION_EVAL = os.path.join(OUTPUTS, "eval_lesion")


# ---------------------------------------------------------------------------
# rendering (reuse the same painter from timeline.py)
# ---------------------------------------------------------------------------

def look_at(eye, target, up=(0.0, -1.0, 0.0)):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f) + 1e-12
    up = np.asarray(up, float)
    if abs(float(f @ (up / (np.linalg.norm(up) + 1e-12)))) > 0.99:
        up = np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up); r /= np.linalg.norm(r) + 1e-12
    u = np.cross(r, f)
    R = np.stack([r, u, f])
    return np.hstack([R, (-R @ np.asarray(eye, float))[:, None]])


def frame_cloud(points, hw, fov_deg, dist_scale, azim_deg, elev_deg):
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


def render(points, colors, K, E, hw, radius=1, bg=12):
    H, W = hw
    img = np.full((H, W, 3), bg, np.uint8)
    if len(points) == 0:
        return img
    cam = points @ E[:3, :3].T + E[:3, 3]
    z = cam[:, 2]
    m = z > 1e-6
    if not m.any():
        return img
    cam, col, z = cam[m], colors[m], z[m]
    proj = cam @ K.T
    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]
    inb = (u >= radius) & (u < W - radius) & (v >= radius) & (v < H - radius)
    if not inb.any():
        return img
    u = u[inb].astype(np.int32); v = v[inb].astype(np.int32)
    col = col[inb]; z = z[inb]
    order = np.argsort(-z)
    u, v, col = u[order], v[order], col[order]
    if radius <= 0:
        img[v, u] = col
    else:
        d = np.arange(-radius, radius + 1, dtype=np.int32)
        dy, dx = np.meshgrid(d, d, indexing="ij")
        dy = dy.ravel(); dx = dx.ravel()
        vs = (v[:, None] + dy[None, :]).ravel()
        us = (u[:, None] + dx[None, :]).ravel()
        cs = np.repeat(col, len(dy), axis=0)
        clip = (vs >= 0) & (vs < H) & (us >= 0) & (us < W)
        img[vs[clip], us[clip]] = cs[clip]
    return img


def label(img, text, y=8, x=8):
    a = np.ascontiguousarray(img)
    cv2.putText(a, text, (x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return a


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_cloud(path):
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return z["points"].astype(np.float64), z["colors"].astype(np.uint8)


def load_left_image(folder_dir, stem, hw):
    H, W = hw
    p = os.path.join(folder_dir, stem + "_L.png")
    img = cv2.imread(p)
    if img is None:
        return np.zeros((H, W, 3), np.uint8)
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (W, H))
    return img


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", required=True, help="e.g. 病灶3")
    ap.add_argument("--lesion_root", default=None,
                    help="path to the lesion data folder (default: data/eval/lesion)")
    ap.add_argument("--monst3r_dir", default=None)
    ap.add_argument("--vggt1b_dir",  default=None)
    ap.add_argument("--dynvggt_dir", default=None)
    ap.add_argument("--size", nargs=2, type=int, default=[720, 960], metavar=("H", "W"))
    ap.add_argument("--fov",  type=float, default=60.0)
    ap.add_argument("--dist", type=float, default=1.1)
    ap.add_argument("--azim", type=float, default=50.0)
    ap.add_argument("--orbit", type=float, default=-100.0)
    ap.add_argument("--elev", type=float, default=12.0)
    ap.add_argument("--pingpong", action="store_true", default=True)
    ap.add_argument("--frames_per_pair", type=int, default=32,
                    help="total frames per pair; divided evenly across 4 swing directions "
                         "(right/left/up/down, each returning to centre)")
    ap.add_argument("--azim_swing", type=float, default=35.0,
                    help="max horizontal swing in degrees")
    ap.add_argument("--elev_swing", type=float, default=20.0,
                    help="max vertical swing in degrees")
    ap.add_argument("--images", action="store_true",
                    help="output one PNG per pair instead of a video")
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--fps",  type=float, default=3.0)
    ap.add_argument("--exp",  default="private", help="sub-dir under outputs/pair_grid/")
    ap.add_argument("--out",  default=None)
    args = ap.parse_args()

    folder = args.folder
    H, W = args.size
    pH, pW = H // 2, W // 2

    # resolve dirs
    from data.paths import data_path
    lesion_root = args.lesion_root or data_path("eval", "lesion")
    folder_dir  = os.path.join(lesion_root, folder)

    def pair_ply(model_dir, stem):
        # MonST3R saves under .../pair/<folder>/ply/stem.npz;
        # VGGT-1B/scared_cam_b16 save under .../pair/<folder>/stem.npz directly.
        with_ply = os.path.join(model_dir, "pair", folder, "ply", stem + ".npz")
        without_ply = os.path.join(model_dir, "pair", folder, stem + ".npz")
        return with_ply if os.path.exists(with_ply) else without_ply

    monst3r_dir = args.monst3r_dir or os.path.join(LESION_EVAL, "MonST3R")
    vggt1b_dir  = args.vggt1b_dir  or os.path.join(LESION_EVAL, "VGGT-1B")
    dynvggt_dir = args.dynvggt_dir or os.path.join(LESION_EVAL, "scared_cam_b16")

    stems = sorted(
        n[:-6] for n in os.listdir(folder_dir) if n.endswith("_L.png")
    )
    print(f"{folder}: {len(stems)} pairs")

    # pre-load all clouds (so framing can be computed globally)
    all_pts = []
    clouds = {}
    for stem in stems:
        row = {}
        for name, base in [("MonST3R", monst3r_dir),
                            ("VGGT-1B", vggt1b_dir),
                            ("Dyn-VGGT", dynvggt_dir)]:
            c = load_cloud(pair_ply(base, stem))
            row[name] = c
            if c is not None:
                all_pts.append(c[0])
        clouds[stem] = row

    if not all_pts:
        print("No clouds found -- check paths.")
        return
    all_pts_arr = np.concatenate(all_pts)

    # shared framing across ALL pairs and ALL models
    K_ref, _ = frame_cloud(all_pts_arr, (pH, pW), args.fov, args.dist,
                            args.azim, args.elev)

    def render_grid(pair_idx, stem, a, e=None):
        e = e if e is not None else args.elev
        left_img = load_left_image(folder_dir, stem, (pH, pW))
        row = clouds[stem]
        panels = []
        for name in ("MonST3R", "VGGT-1B", "Dyn-VGGT"):
            c = row[name]
            if c is None:
                panels.append(np.zeros((pH, pW, 3), np.uint8))
                continue
            pts, cols = c
            _, E = frame_cloud(pts, (pH, pW), args.fov, args.dist, a, e)
            img = render(pts, cols, K_ref, E, (pH, pW), radius=args.radius)
            img = label(img, f"{name}  {stem}")
            panels.append(img)
        left_panel = label(left_img.copy(), f"input L  {stem}")
        top    = np.concatenate([left_panel, panels[0]], axis=1)
        bottom = np.concatenate([panels[1],  panels[2]], axis=1)
        return np.concatenate([top, bottom], axis=0)

    if args.images:
        # One PNG per pair at the frontal (azim) angle.
        out_dir = args.out or os.path.join(output_dir_for_exp(args.exp, TOOL), folder)
        os.makedirs(out_dir, exist_ok=True)
        for pair_idx, stem in enumerate(stems):
            grid = render_grid(pair_idx, stem, args.azim)
            path = os.path.join(out_dir, f"{stem}.png")
            cv2.imwrite(path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            print(f"  {stem}.png", flush=True)
        print(f"\n{len(stems)} images -> {out_dir}")
    else:
        fpp   = args.frames_per_pair
        # 4 swings: +azim, -azim, +elev, -elev — each swing out and back
        seg   = max(1, fpp // 4)

        def swing(n, base_a, base_e, da, de):
            """n frames: centre → peak → centre."""
            frames = []
            for i in range(n):
                t = i / max(1, n - 1)           # 0→1
                s = np.sin(t * np.pi)            # 0→1→0 (smooth, returns to 0)
                frames.append((base_a + da * s, base_e + de * s))
            return frames

        base_a, base_e = args.azim, args.elev
        swing_seq = (
            swing(seg, base_a, base_e,  args.azim_swing,  0) +
            swing(seg, base_a, base_e, -args.azim_swing,  0) +
            swing(seg, base_a, base_e,  0,  args.elev_swing) +
            swing(seg, base_a, base_e,  0, -args.elev_swing)
        )

        out_frames = []
        for pair_idx, stem in enumerate(stems):
            for a, e in swing_seq:
                out_frames.append(render_grid(pair_idx, stem, a, e))
        out = args.out or os.path.join(output_dir_for_exp(args.exp, TOOL), f"{folder}.mp4")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        n = write_video(out, out_frames, fps=args.fps)
        print(f"\n{n} frames -> {out}")


if __name__ == "__main__":
    main()
