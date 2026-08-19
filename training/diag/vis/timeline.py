"""Scrub through a reconstruction as it accumulates over time.

"The reconstruction at time t" is just the slice ``frame_id <= t`` of a cloud dumped by
``save_cloud`` -- every frame already sits in one common coordinate frame, so nothing is
re-inferred and the whole thing is CPU-only and re-runnable at will.

Two modes, for two different audiences:

  ``--mode video``  offscreen render to mp4, fixed or orbiting camera. This is what goes
                    in a slide deck. Several clouds can be rendered side by side, which is
                    the point of the exercise: a pair reconstruction has two frames and
                    its timeline barely moves, while a sequence keeps growing.
  ``--mode viser``  live server with a real slider, free camera. For demoing in person.

The renderer is a hand-rolled painter rather than a 3D library: it is a projection, a
far-to-near sort, and an indexed write, which is a few lines, has no extra dependency, and
keeps the framing under our control (a viewer's auto-fit would rescale between timesteps
and destroy the sense of accumulation).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval_utils.media_io import write_video
from eval_utils.paths import output_dir_for_exp

TOOL = "timeline"


def load_cloud(path):
    z = np.load(path)
    return (z["points"].astype(np.float64), z["colors"].astype(np.uint8),
            z["frame_id"].astype(np.int32), int(z["n_frames"]))


def look_at(eye, target, up=(0.0, -1.0, 0.0)):
    """World-to-camera [3,4] for a camera at ``eye`` looking at ``target``."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f) + 1e-12
    up = np.asarray(up, float)
    if abs(float(f @ (up / (np.linalg.norm(up) + 1e-12)))) > 0.99:   # degenerate up
        up = np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up); r /= np.linalg.norm(r) + 1e-12
    u = np.cross(r, f)
    R = np.stack([r, u, f])                       # rows: camera x, y, z axes in world
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
    u, v, col, z = u[inb].astype(np.int32), v[inb].astype(np.int32), col[inb], z[inb]

    order = np.argsort(-z)                        # paint far first; near overwrites
    u, v, col = u[order], v[order], col[order]

    if radius <= 0:
        img[v, u] = col
    else:
        # Vectorized splat: expand each point to (2r+1)^2 pixels at once.
        d = np.arange(-radius, radius + 1, dtype=np.int32)
        dy, dx = np.meshgrid(d, d, indexing="ij")   # (2r+1, 2r+1)
        dy = dy.ravel(); dx = dx.ravel()
        vs = (v[:, None] + dy[None, :]).ravel()      # (N * k^2,)
        us = (u[:, None] + dx[None, :]).ravel()
        cs = np.repeat(col, len(dy), axis=0)
        clip = (vs >= 0) & (vs < H) & (us >= 0) & (us < W)
        img[vs[clip], us[clip]] = cs[clip]
    return img


def label(img, text, y=8, x=8):
    """Caption without a font dependency -- cv2 ships with the repo already."""
    import cv2
    a = np.ascontiguousarray(img)
    cv2.putText(a, text, (x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return a


def progress_bar(img, frac, h=5):
    img[-h:, :, :] = 40
    img[-h:, :int(img.shape[1] * frac), :] = (90, 200, 120)
    return img


def load_video_frames(path, n_frames, hw):
    """Read up to ``n_frames`` from a video file and resize to (H,W,3) uint8.
    If the video is shorter it is looped; if longer it is truncated."""
    import cv2
    H, W = hw
    cap = cv2.VideoCapture(path)
    raw = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        raw.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (W, H)))
    cap.release()
    if not raw:
        return [np.zeros((H, W, 3), np.uint8)] * n_frames
    # loop if shorter
    out = []
    for i in range(n_frames):
        out.append(raw[i % len(raw)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clouds", nargs="+", required=True, help="cloud.npz from save_cloud")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--mode", default="video", choices=["video", "viser"])
    ap.add_argument("--size", nargs=2, type=int, default=[720, 960], metavar=("H", "W"))
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--dist", type=float, default=1.1, help="camera distance in cloud extents")
    ap.add_argument("--azim", type=float, default=0.0)
    ap.add_argument("--elev", type=float, default=12.0)
    ap.add_argument("--orbit", type=float, default=40.0, help="degrees swept over the clip")
    ap.add_argument("--pingpong", action="store_true",
                    help="sweep azim from --azim to --azim+--orbit and back again")
    ap.add_argument("--radius", type=int, default=1, help="splat radius in px")
    ap.add_argument("--per_cloud_camera", action="store_true",
                    help="frame each panel to its own cloud instead of to all of them; "
                         "makes a small reconstruction look as large as a big one, so it "
                         "is off by default whenever panels are meant to be compared")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--hold", type=int, default=12, help="extra frames on the final state")
    ap.add_argument("--exp", default="private", help="<exp> level under outputs/<tool>/")
    ap.add_argument("--out", default=None)
    ap.add_argument("--input_video", default=None,
                    help="path to original input mp4; when given, arranges all panels "
                         "in a 2×2 grid with the input video in the top-left cell")
    args = ap.parse_args()

    clouds = [load_cloud(p) for p in args.clouds]
    labels = args.labels or [os.path.basename(os.path.dirname(p)) for p in args.clouds]
    for p, (pts, _, _, n) in zip(args.clouds, clouds):
        print(f"{p}: {len(pts)} points, {n} frames")

    if args.mode == "viser":
        return run_viser(clouds[0], labels[0])

    grid_2x2 = args.input_video is not None
    H, W = args.size
    # In 2×2 mode each cell is half the requested output size so the overall
    # video stays at HxW.
    pH, pW = (H // 2, W // 2) if grid_2x2 else (H, W)

    n_steps = max(n for _, _, _, n in clouds)
    frame_ref = ([pts for pts, _, _, _ in clouds] if args.per_cloud_camera
                 else [np.concatenate([pts for pts, _, _, _ in clouds])] * len(clouds))
    cams = [frame_cloud(ref, (pH, pW), args.fov, args.dist, args.azim, args.elev)
            for ref in frame_ref]

    total_steps = n_steps + args.hold
    input_frames = (load_video_frames(args.input_video, total_steps, (pH, pW))
                    if grid_2x2 else None)

    out_frames = []
    for t in range(total_steps):
        if t % max(1, total_steps // 10) == 0:
            print(f"  rendering {t+1}/{total_steps} ...", flush=True)
        step = min(t, n_steps - 1)
        panels = []
        for (pts, cols, fid, n), (K, _), ref, name in zip(clouds, cams, frame_ref, labels):
            frac = step / max(1, n_steps - 1)
            orbit_frac = (2 * frac if frac < 0.5 else 2 * (1 - frac)) if args.pingpong else frac
            a = args.azim + args.orbit * orbit_frac
            _, E = frame_cloud(ref, (pH, pW), args.fov, args.dist, a, args.elev)
            keep = fid <= min(step, n - 1)
            img = render(pts[keep], cols[keep], K, E, (pH, pW), radius=args.radius)
            img = label(img, f"{name}  frame {min(step, n - 1) + 1}/{n}  "
                             f"{int(keep.sum()):,} pts")
            panels.append(progress_bar(img, min(step + 1, n) / n))

        if grid_2x2:
            # Pad to 4 panels: [input_video, cloud0, cloud1, cloud2, ...]
            # Top row: input | panels[0]
            # Bottom row: panels[1] | panels[2]  (or black if missing)
            vid_panel = label(input_frames[t].copy(), f"input  frame {min(step,n_steps-1)+1}/{n_steps}")
            cells = [vid_panel] + panels
            while len(cells) < 4:
                cells.append(np.zeros((pH, pW, 3), np.uint8))
            top    = np.concatenate(cells[:2], axis=1)
            bottom = np.concatenate(cells[2:4], axis=1)
            out_frames.append(np.concatenate([top, bottom], axis=0))
        else:
            out_frames.append(np.concatenate(panels, axis=1))

    out = args.out or os.path.join(output_dir_for_exp(args.exp, TOOL), "timeline.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n_written = write_video(out, out_frames, fps=args.fps)
    print(f"\n{n_written} frames -> {out}")


def run_viser(cloud, name):
    """Live viewer: a real slider, and the camera stays wherever the user put it."""
    import time
    import viser

    pts, cols, fid, n = cloud
    server = viser.ViserServer()
    handle = server.scene.add_point_cloud("cloud", points=pts[fid <= 0].astype(np.float32),
                                          colors=cols[fid <= 0], point_size=0.002)
    slider = server.gui.add_slider(f"{name}: frame", min=0, max=n - 1, step=1,
                                   initial_value=n - 1)
    cumulative = server.gui.add_checkbox("cumulative", initial_value=True)

    def update(_=None):
        t = int(slider.value)
        keep = (fid <= t) if cumulative.value else (fid == t)
        handle.points = pts[keep].astype(np.float32)
        handle.colors = cols[keep]

    slider.on_update(update)
    cumulative.on_update(update)
    update()
    print("viser running; ctrl-C to stop")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
