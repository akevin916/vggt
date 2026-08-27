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

TWO RENDERERS, picked with ``--blend``:

  default   nearest-point painter. Each screen pixel takes the colour of whichever point is
            closest to the camera. Cheap, but the winner flips abruptly across the image --
            sub-mm depth differences decide it -- so every brightness difference between
            frames shows up as a hard seam and the reconstruction reads as a patchwork of
            plates. Those plates are a RENDERING artefact and must not be read as pose error.
  --blend   the weighted renderer from cloud_polish.py: every frame that lands on a pixel
            contributes, feathered towards its own borders, averaging only contributions
            within ``--blend_tol`` of the nearest depth so occluded background cannot bleed
            through. Needs ``seq.npz`` (depth + intrinsics + extrinsics + images) beside each
            ``cloud.npz``; --blend finds it automatically, so the command line is unchanged.

The default renderer is a hand-rolled painter rather than a 3D library: it is a projection, a
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
    # -u, not u. ``up`` is the world's UP direction (-Y under the Y-down convention these
    # clouds live in), but the camera's y row must point the way image v grows, i.e. DOWN.
    # Stacking [r, u, f] made det(R) = -1 -- a reflection, not a rotation -- so every render
    # came out vertically flipped (a point above the centre projected below it). Verified
    # numerically: for f=+Z, up=-Y, the point (0,-1,5) gave cam_y=+1 before and -1 after.
    R = np.stack([r, -u, f])                      # rows: camera x, y, z axes in world
    return np.hstack([R, (-R @ np.asarray(eye, float))[:, None]])


def robust_bounds(points, pct=1.0):
    """Centre and extent with ``pct``% trimmed off each end of every axis."""
    lo, hi = np.percentile(points, [pct, 100.0 - pct], axis=0)
    return (lo + hi) / 2.0, float(np.linalg.norm(hi - lo)) + 1e-9


def normalise_scale(points, pct=1.0):
    """Centre a cloud on its robust centre and divide by its robust extent.

    Turns "up to an arbitrary similarity transform" into one fixed choice, so panels can
    share a camera. Returns the transformed points.
    """
    centre, extent = robust_bounds(points, pct)
    return (points - centre) / extent


def frame_cloud(points, hw, fov_deg, dist_scale, azim_deg, elev_deg, pct=1.0):
    """A camera that frames the *whole* cloud, so the view never rescales as points are
    added -- accumulation must read as accumulation, not as a zoom-out."""
    H, W = hw
    centre, extent = robust_bounds(points, pct)

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


def build_shot_list(n_steps, args):
    """The whole clip as a list of ``(step, azim, elev)``.

    ``step`` is the accumulation index -- which frames of the cloud are visible -- and is
    -1 for the intro, where nothing has been reconstructed yet. Separating the shot list
    from the render loop is what lets the camera and the accumulation run at different
    rates: the build can be slowed with --build_hold, and the swings that follow all sit
    on the finished cloud (step = n_steps-1) while only the camera moves.

    Structure (each phase is skipped when its count is 0, so the defaults reproduce the
    old plain accumulate-while-orbiting behaviour):

        intro -> build -> pause -> right -> pause -> left -> pause -> up -> pause
              -> down -> pause -> hold
    """
    a0, e0 = args.azim, args.elev
    shots = [(-1, a0, e0)] * max(0, args.intro)

    for t in range(n_steps):
        frac = t / max(1, n_steps - 1)
        orbit_frac = (2 * frac if frac < 0.5 else 2 * (1 - frac)) if args.pingpong else frac
        a = a0 + args.orbit * orbit_frac
        shots += [(t, a, e0)] * max(1, args.build_hold)

    last = n_steps - 1
    pause = [(last, a0, e0)] * max(0, args.pause)
    shots += pause

    if args.swing_frames > 0:
        for da, de in ((args.azim_swing, 0.0), (-args.azim_swing, 0.0),
                       (0.0, args.elev_swing), (0.0, -args.elev_swing)):
            for i in range(args.swing_frames):
                # sin, not a ramp: starts and ends at the centre with zero velocity, so the
                # four swings read as one continuous move instead of four jump cuts.
                s = float(np.sin(np.pi * i / max(1, args.swing_frames - 1)))
                shots.append((last, a0 + da * s, e0 + de * s))
            shots += pause

    shots += [(last, a0, e0)] * max(0, args.hold)
    return shots


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
    ap.add_argument("--scale_norm", action="store_true",
                    help="rescale every cloud to the same robust size before framing. "
                         "Each arm reconstructs up to its own arbitrary global scale "
                         "(MonST3R's is ~1/3 of VGGT's on these clips), so a shared camera "
                         "otherwise frames the union of three incompatible unit systems and "
                         "every panel comes out small and off-centre. This does NOT change "
                         "the shape of any reconstruction -- a cloud that is spread out "
                         "stays spread out, it is just drawn at a comparable size")
    ap.add_argument("--frame_pct", type=float, default=1.0,
                    help="percentile trimmed off each end when measuring a cloud's centre "
                         "and extent; raise it (e.g. 5) when stray points are pushing the "
                         "framing out and leaving the surface small in the middle")
    ap.add_argument("--no_progress", action="store_true",
                    help="drop the green progress bar along the bottom of each panel")
    ap.add_argument("--per_cloud_camera", action="store_true",
                    help="frame each panel to its own cloud instead of to all of them; "
                         "makes a small reconstruction look as large as a big one, so it "
                         "is off by default whenever panels are meant to be compared")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--hold", type=int, default=12, help="extra frames on the final state")
    # --- shot list (presentation cut). All default to 0/1, i.e. no change in behaviour.
    ap.add_argument("--intro", type=int, default=0,
                    help="frames held before anything is reconstructed (empty panels)")
    ap.add_argument("--build_hold", type=int, default=1,
                    help="repeat each accumulation step this many frames -- the knob that "
                         "slows the build down without lowering --fps for the whole clip")
    ap.add_argument("--pause", type=int, default=0,
                    help="frames held, back at the centre view, between shots")
    ap.add_argument("--swing_frames", type=int, default=0,
                    help="frames per swing; 4 swings (right/left/up/down) run after the "
                         "build finishes, each leaving and returning to the centre view. "
                         "0 disables the whole swing phase")
    ap.add_argument("--azim_swing", type=float, default=35.0,
                    help="peak horizontal swing in degrees")
    ap.add_argument("--elev_swing", type=float, default=20.0,
                    help="peak vertical swing in degrees")
    ap.add_argument("--exp", default="private", help="<exp> level under outputs/<tool>/")
    ap.add_argument("--out", default=None)
    ap.add_argument("--blend", action="store_true",
                    help="use cloud_polish's weighted renderer; reads seq.npz beside each "
                         "--clouds entry instead of the cloud.npz itself")
    ap.add_argument("--blend_tol", type=float, default=0.02,
                    help="blend only: relative depth window that counts as the same surface")
    ap.add_argument("--feather_margin", type=float, default=0.25, help="blend only")
    ap.add_argument("--feather_floor", type=float, default=0.02, help="blend only")
    ap.add_argument("--depth_pct", type=float, default=98.0, help="blend only")
    ap.add_argument("--max_points", type=int, default=1_200_000, help="blend only")
    ap.add_argument("--image_dir", default=None,
                    help="blend only: source frames, for seq.npz files written without an "
                         "`images` key (every panel of one segment saw the same frames, so "
                         "one directory covers them all)")
    ap.add_argument("--input_video", default=None,
                    help="path to original input mp4; when given, arranges all panels "
                         "in a 2×2 grid with the input video in the top-left cell")
    args = ap.parse_args()

    if args.blend:
        # cloud.npz holds points+colours only; the weighted renderer needs the per-frame
        # depth/pose/images, which live in seq.npz written beside it by eval_{lesion,gastric}.
        from types import SimpleNamespace

        from diag.vis.cloud_polish import build_cloud, load_seq
        from diag.vis.cloud_polish import render as blend_render

        ns = SimpleNamespace(max_points=args.max_points, depth_pct=args.depth_pct,
                             feather_margin=args.feather_margin,
                             feather_floor=args.feather_floor)
        clouds = []
        for p in args.clouds:
            seq = os.path.join(os.path.dirname(p), "seq.npz")
            if not os.path.exists(seq):
                raise SystemExit(f"--blend needs {seq}, which does not exist")
            depth, K_, E_, imgs = load_seq(seq, args.image_dir)
            pts, cols, wts, fids, S = build_cloud(depth, K_, E_, imgs, ns)
            clouds.append((pts, cols, fids, S, wts))
    else:
        blend_render = None
        clouds = [load_cloud(p) + (None,) for p in args.clouds]

    if args.scale_norm:
        # Each arm's global scale is arbitrary and they disagree by a lot -- MonST3R's median
        # depth is about a third of VGGT's on these clips -- so put every cloud in the same
        # units before anything measures it. Purely a viewing transform: the geometry of a
        # reconstruction, including how spread out it is, is unchanged.
        clouds = [(normalise_scale(pts, args.frame_pct), cols, fid, n, wts)
                  for pts, cols, fid, n, wts in clouds]

    labels = args.labels or [os.path.basename(os.path.dirname(p)) for p in args.clouds]
    for p, (pts, _, _, n, _w) in zip(args.clouds, clouds):
        print(f"{p}: {len(pts)} points, {n} frames")

    if args.mode == "viser":
        return run_viser(clouds[0], labels[0])

    grid_2x2 = args.input_video is not None
    H, W = args.size
    # In 2×2 mode each cell is half the requested output size so the overall
    # video stays at HxW.
    pH, pW = (H // 2, W // 2) if grid_2x2 else (H, W)

    n_steps = max(n for _, _, _, n, _w in clouds)
    frame_ref = ([pts for pts, _, _, _, _w in clouds] if args.per_cloud_camera
                 else [np.concatenate([pts for pts, _, _, _, _w in clouds])] * len(clouds))
    cams = [frame_cloud(ref, (pH, pW), args.fov, args.dist, args.azim, args.elev,
                        pct=args.frame_pct)
            for ref in frame_ref]

    shots = build_shot_list(n_steps, args)
    total_steps = len(shots)
    input_frames = (load_video_frames(args.input_video, n_steps, (pH, pW))
                    if grid_2x2 else None)

    out_frames = []
    for t, (step, a_shot, e_shot) in enumerate(shots):
        if t % max(1, total_steps // 10) == 0:
            print(f"  rendering {t+1}/{total_steps} ...", flush=True)
        panels = []
        for (pts, cols, fid, n, wts), (K, _), ref, name in zip(clouds, cams, frame_ref, labels):
            a, e = a_shot, e_shot
            _, E = frame_cloud(ref, (pH, pW), args.fov, args.dist, a, e,
                               pct=args.frame_pct)
            keep = fid <= min(step, n - 1)
            img = (blend_render(pts[keep], cols[keep], wts[keep], K, E, (pH, pW),
                                radius=args.radius, blend_tol=args.blend_tol)
                   if args.blend else
                   render(pts[keep], cols[keep], K, E, (pH, pW), radius=args.radius))
            img = label(img, f"{name}  frame {min(step, n - 1) + 1}/{n}  "
                             f"{int(keep.sum()):,} pts")
            if not args.no_progress:
                img = progress_bar(img, min(step + 1, n) / n)
            panels.append(img)

        if grid_2x2:
            # Pad to 4 panels: [input_video, cloud0, cloud1, cloud2, ...]
            # Top row: input | panels[0]
            # Bottom row: panels[1] | panels[2]  (or black if missing)
            # indexed by the accumulation step, not by output frame: with --build_hold /
            # --pause the two no longer run at the same rate, and the input clip must stay
            # locked to the reconstruction it is being compared against.
            vid_step = min(max(step, 0), n_steps - 1)
            vid_panel = label(input_frames[vid_step].copy(),
                              f"input  frame {max(step, -1) + 1}/{n_steps}")
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

    pts, cols, fid, n = cloud[:4]
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
