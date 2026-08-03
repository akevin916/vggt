#!/usr/bin/env python3
"""Overlay a checkpoint's predicted point cloud on the Sintel GT point cloud.

Purpose: see *where* the predicted geometry departs from GT, not just the scalar depth /
ATE numbers -- e.g. whether a v3 gate variant fixes the drift on the dynamic actors while
leaving the static background alone.

VGGT predicts up to an unknown similarity: its world frame is frame-0's camera and its
scale is arbitrary, while Sintel GT lives in its own metric world frame. The two clouds are
therefore brought together by a Sim(3) fitted on the *camera centres* (Umeyama, scale+R+t),
then applied to the predicted cloud. Note what that implies: the residual you see is the
geometry error that survives pose alignment, which is exactly the quantity the pose metrics
in eval_utils/metrics_pose.py summarise into one number.

The Sim(3) is fitted here rather than taken from evo because extrinsics_w2c_to_tum() mean-
centres the translations, so evo's alignment lives in a shifted frame and cannot be applied
to points as-is.

Three outputs, all optional:
  --port N        interactive viser server with a pred/GT visibility + gizmo toggle
  --save_views    fixed-viewpoint PNGs under outputs/<exp>/pointcloud_gt/ (never shown)
  --export_ply    gt.ply + pred.ply in the shared frame, for a local viewer (CloudCompare/
                  MeshLab) -- the offline path when the network makes viser too laggy
"""

from __future__ import annotations

import os
import sys

_TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO_DIR = os.path.dirname(_TRAINING_DIR)
sys.path[:0] = [_TRAINING_DIR, _REPO_DIR]

import argparse
import time
from typing import Dict, List, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.sintel_io import (
    SINTEL_EVAL_SEQUENCES,
    compute_preprocess_meta,
    load_sintel_rgb_paths,
    matching_cam_path,
    matching_depth_path,
    read_sintel_depth,
    resize_gt_to_pred,
    resolve_sintel_root,
    sintel_cam_read,
    sintel_seq_paths,
)
from scipy.spatial import cKDTree

from eval_utils.paths import OUTPUTS_DIR, POINTCLOUD_GT, default_output_dir
from eval_utils.ply_io import write_ply
from eval_utils.vggt_infer import infer_sequence, infer_sequence_chunked, load_vggt_for_eval
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images


def parse_args():
    ap = argparse.ArgumentParser(description="Overlay predicted vs GT Sintel point clouds")
    ap.add_argument("--ckpt", required=True, help="e.g. checkpoints/VGGT-1B.pt")
    ap.add_argument("--seqs", nargs="*", default=None, help="Sintel sequences (default: all SINTEL_EVAL_SEQUENCES)")
    ap.add_argument("--sintel_root", default=None)
    ap.add_argument("--max_frames", type=int, nargs="+", default=[12], help="One or more frame counts, e.g. 12 50")
    ap.add_argument("--chunk_size", type=int, default=0, help="0 = full sequence in one pass (keep it 0 for geometry)")
    ap.add_argument("--max_depth", type=float, default=80.0, help="Drop points beyond this GT-scale depth (sky)")
    ap.add_argument("--stride", type=int, default=2, help="Pixel subsampling stride for both clouds")
    ap.add_argument(
        "--geom_align",
        action="store_true",
        help="After the camera-centre Sim(3), refine pred->GT with a scaled ICP on the points so "
        "the pose/orientation error is factored out and only the geometry (shape) difference remains.",
    )
    ap.add_argument(
        "--method",
        default=None,
        help="Method name (e.g. base, inst). When set, export uses the comparison layout "
        "outputs/point_cloud/<scene>/<method>.ply + gt.ply instead of outputs/<exp>/pointcloud_gt/.",
    )
    ap.add_argument(
        "--per_frame",
        action="store_true",
        help="Also emit single-frame point maps (GT downsampled to the model's grid, so GT and pred "
        "are pixel-aligned per frame) under outputs/point_cloud/<scene>/frame_XXXX/. Needs --method.",
    )
    ap.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=None,
        help="Frame indices for --per_frame (default: first, middle, last).",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--port", type=int, default=0, help="0 = no viser server; else serve on this port")
    ap.add_argument("--save_views", action="store_true", help="Save fixed-viewpoint PNGs")
    ap.add_argument("--export_ply", action="store_true", help="Write gt.ply + pred.ply for a local viewer (CloudCompare/MeshLab)")
    ap.add_argument("--max_plot_points", type=int, default=60000, help="Per-cloud point budget for the PNGs")
    ap.add_argument("--max_viser_points", type=int, default=400000, help="Per-cloud point budget streamed to viser (lag control)")
    ap.add_argument("--out_dir", default=None, help=f"Default: outputs/<exp>/{POINTCLOUD_GT}")
    args = ap.parse_args()
    args.out_dir = args.out_dir or default_output_dir(args.ckpt, POINTCLOUD_GT)
    return args


# --------------------------------------------------------------------------------------
# Sim(3)


def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (s, R, t) with s * R @ src + t ~= dst. Both (N, 3)."""
    n = src.shape[0]
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_src, dst - mu_dst

    cov = (dc.T @ sc) / n
    u, d, vt = np.linalg.svd(cov)

    s_fix = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:  # reflection -> flip the smallest axis
        s_fix[2, 2] = -1.0

    rot = u @ s_fix @ vt
    var_src = (sc**2).sum() / n
    scale = float(np.trace(np.diag(d) @ s_fix) / var_src)
    trans = mu_dst - scale * rot @ mu_src
    return scale, rot, trans


def apply_sim3(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return scale * points @ rot.T + trans


def camera_centres(extrinsics: np.ndarray) -> np.ndarray:
    """(S, 3, 4) world-to-cam -> (S, 3) camera centres in world coordinates."""
    return closed_form_inverse_se3(extrinsics)[:, :3, 3]


def scaled_icp(
    src: np.ndarray, dst: np.ndarray, n_sample: int = 40000, iters: int = 30, trim: float = 0.8,
    max_step_scale: float = 1.25, total_scale_band: Tuple[float, float] = (0.2, 5.0),
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """Refine a similarity src->dst by trimmed, scaled point-to-point ICP.

    Fits on a random ``n_sample`` subset of ``src`` against a KD-tree over all of ``dst``. Each
    iteration keeps only the closest ``trim`` fraction of correspondences, so moving objects and
    non-overlapping regions (which never find a good match) don't drag the fit. Assumes ``src`` is
    already roughly aligned (we seed it with the camera-centre Sim(3)). Returns the cumulative
    (scale, R, t) plus the trimmed RMS residual after the last step.

    Two guards stop the classic scaled-ICP failure where the cloud collapses toward a point to
    trivially minimise point-to-point distance: each step's scale is clamped to
    ``[1/max_step_scale, max_step_scale]`` (no single catastrophic jump), and the *cumulative*
    scale is held inside ``total_scale_band`` relative to the seed (so many small shrink steps
    can't compound into a collapse either). The band is wide enough for the genuine ~0.3x rescales
    some scenes need after the camera-centre seed.
    """
    tree = cKDTree(dst)
    rng = np.random.default_rng(0)
    idx = rng.choice(src.shape[0], size=min(n_sample, src.shape[0]), replace=False)
    cur = src[idx].copy()
    lo, hi = total_scale_band

    s_tot, r_tot, t_tot = 1.0, np.eye(3), np.zeros(3)
    rms = float("nan")
    for _ in range(iters):
        d, nn = tree.query(cur, workers=-1)
        keep = d <= np.quantile(d, trim)
        rms = float(np.sqrt((d[keep] ** 2).mean()))
        s_i, r_i, t_i = umeyama_sim3(cur[keep], dst[nn[keep]])
        if not np.isfinite(s_i) or s_i <= 0:  # degenerate step -> stop refining
            break
        s_i = float(np.clip(s_i, 1.0 / max_step_scale, max_step_scale))
        s_i = float(np.clip(s_tot * s_i, lo, hi)) / s_tot  # keep cumulative scale in-band
        # Recompute translation for the clamped scale so the centroids still line up.
        mu_src, mu_dst = cur[keep].mean(0), dst[nn[keep]].mean(0)
        t_i = mu_dst - s_i * r_i @ mu_src
        cur = apply_sim3(cur, s_i, r_i, t_i)
        s_tot, r_tot, t_tot = s_i * s_tot, r_i @ r_tot, s_i * (r_i @ t_tot) + t_i
    return s_tot, r_tot, t_tot, rms


# --------------------------------------------------------------------------------------
# Clouds


def _flatten(points: np.ndarray, colors: np.ndarray, valid: np.ndarray, stride: int):
    points = points[:, ::stride, ::stride].reshape(-1, 3)
    colors = colors[:, ::stride, ::stride].reshape(-1, 3)
    valid = valid[:, ::stride, ::stride].reshape(-1)
    return points[valid], colors[valid]


def gt_cloud(sintel_root: str, seq: str, rgb_paths: List[str], max_depth: float, stride: int):
    """GT cloud at native resolution, in Sintel's own world frame."""
    _, depth_dir, cam_dir = sintel_seq_paths(sintel_root, seq)

    depths, intrinsics, extrinsics, colors = [], [], [], []
    for path in rgb_paths:
        depth = read_sintel_depth(matching_depth_path(depth_dir, path))
        k, w2c = sintel_cam_read(matching_cam_path(cam_dir, path))
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depths.append(depth)
        intrinsics.append(k)
        extrinsics.append(w2c)
        colors.append(rgb)

    depths = np.stack(depths).astype(np.float32)
    extrinsics = np.stack(extrinsics).astype(np.float32)
    intrinsics = np.stack(intrinsics).astype(np.float32)
    colors = np.stack(colors)

    world = unproject_depth_map_to_point_map(depths[..., None], extrinsics, intrinsics)
    valid = (depths > 0) & (depths < max_depth)
    points, colors = _flatten(world, colors, valid, stride)
    return points, colors, camera_centres(extrinsics)


def pred_cloud(pred: Dict[str, np.ndarray], rgb_paths: List[str], stride: int):
    """Predicted cloud in VGGT's frame-0 world frame, at the model's cropped resolution."""
    if "depth" not in pred:
        raise SystemExit("This checkpoint has no depth head, so there is no predicted cloud to draw.")

    depth = pred["depth"].astype(np.float32)  # (S, H, W)
    world = unproject_depth_map_to_point_map(depth[..., None], pred["extrinsic"], pred["intrinsic"])

    # The same crop the model saw, so colours line up with the predicted depth pixel-for-pixel.
    images = load_and_preprocess_images(rgb_paths, mode="crop").numpy()  # (S, 3, H, W) in [0, 1]
    colors = images.transpose(0, 2, 3, 1)

    valid = depth > 0
    points, colors = _flatten(world, colors, valid, stride)
    return points, colors, camera_centres(pred["extrinsic"])


# --------------------------------------------------------------------------------------
# Per-frame point maps (GT on the model's grid, so GT and pred are pixel-aligned per frame)


def pred_frame_cloud(pred: Dict[str, np.ndarray], images: np.ndarray, i: int):
    """Frame ``i`` of the predicted cloud, in VGGT's frame-0 world frame (not yet GT-aligned)."""
    depth = pred["depth"][i].astype(np.float32)  # (H, W)
    world = unproject_depth_map_to_point_map(
        depth[None, ..., None], pred["extrinsic"][i : i + 1], pred["intrinsic"][i : i + 1]
    )[0]
    colors = images[i].transpose(1, 2, 0)  # (H, W, 3) in [0, 1], same 518 crop the model saw
    valid = depth > 0
    return world[valid], colors[valid]


def gt_frame_cloud(sintel_root: str, seq: str, rgb_path: str, images_i: np.ndarray, max_depth: float):
    """Frame's GT cloud resampled onto the model's grid, so it is pixel-aligned with pred_frame_cloud.

    GT depth is area-downsampled to the 518 crop and the GT intrinsic is scaled+shifted to match,
    then unprojected with the GT extrinsic (result stays in the GT world frame). Colours reuse the
    model's own 518 crop so GT and pred points share identical RGB per pixel.
    """
    _, depth_dir, cam_dir = sintel_seq_paths(sintel_root, seq)
    depth_native = read_sintel_depth(matching_depth_path(depth_dir, rgb_path))
    k, w2c = sintel_cam_read(matching_cam_path(cam_dir, rgb_path))

    model_h, model_w = images_i.shape[1], images_i.shape[2]
    meta = compute_preprocess_meta(rgb_path)
    depth = resize_gt_to_pred(depth_native, meta, (model_h, model_w))  # (model_h, model_w)

    s = meta.new_w / meta.orig_w  # crop resize is isotropic, then a vertical crop
    k2 = k.astype(np.float64).copy()
    k2[0, 0] *= s
    k2[1, 1] *= s
    k2[0, 2] *= s
    k2[1, 2] = k[1, 2] * s - meta.crop_y0

    world = unproject_depth_map_to_point_map(
        depth[None, ..., None].astype(np.float32), w2c[None].astype(np.float32), k2[None].astype(np.float32)
    )[0]
    colors = images_i.transpose(1, 2, 0)
    valid = (depth > 0) & (depth < max_depth)
    return world[valid], colors[valid]


# --------------------------------------------------------------------------------------
# PLY export (offline viewing in CloudCompare / MeshLab)


def _write_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    n = write_ply(path, points, colors)
    print(f"saved {path}  ({n} points)")


def export_ply(name: str, gt_pts, gt_cols, pred_pts, pred_cols, out_dir: str) -> None:
    """Two clouds in the shared GT world frame: open both together in CloudCompare/MeshLab.

    Colours are the real RGB, so also drop a flat-shaded pred (all red) that reads the
    geometry difference at a glance when overlaid on the grey-ish GT.
    """
    os.makedirs(out_dir, exist_ok=True)
    _write_ply(os.path.join(out_dir, f"{name}_gt.ply"), gt_pts, gt_cols)
    _write_ply(os.path.join(out_dir, f"{name}_pred.ply"), pred_pts, pred_cols)
    red = np.broadcast_to(np.array([0.9, 0.2, 0.2], np.float32), pred_cols.shape)
    _write_ply(os.path.join(out_dir, f"{name}_pred_flat.ply"), pred_pts, red)


def export_method(scene: str, method: str, gt_pts, gt_cols, pred_pts, pred_cols) -> None:
    """Comparison layout: outputs/point_cloud/<scene>/{<method>.ply, gt.ply}.

    All methods for one scene land in the same folder, so CloudCompare opens the whole scene's
    clouds in one shot. gt.ply is identical across methods (never transformed), so re-writing it
    per run is harmless.
    """
    out_dir = os.path.join(OUTPUTS_DIR, "point_cloud", scene)
    os.makedirs(out_dir, exist_ok=True)
    _write_ply(os.path.join(out_dir, "gt.ply"), gt_pts, gt_cols)
    _write_ply(os.path.join(out_dir, f"{method}.ply"), pred_pts, pred_cols)


# --------------------------------------------------------------------------------------
# Static views


def _subsample(points: np.ndarray, colors: np.ndarray, budget: int, rng: np.random.Generator):
    if points.shape[0] <= budget:
        return points, colors
    idx = rng.choice(points.shape[0], size=budget, replace=False)
    return points[idx], colors[idx]


def save_views(name: str, title: str, gt_pts, pred_pts, out_dir: str, budget: int):
    """Three fixed viewpoints, pred in red over GT in grey, plus an RGB-coloured pair."""
    rng = np.random.default_rng(0)
    gt_s, _ = _subsample(gt_pts, gt_pts, budget, rng)
    pred_s, _ = _subsample(pred_pts, pred_pts, budget, rng)

    # Clip to the GT bounding box so a few blown-up predicted points cannot squash the view.
    lo, hi = np.percentile(gt_s, 1, axis=0), np.percentile(gt_s, 99, axis=0)
    span = (hi - lo).max()
    centre = (lo + hi) / 2

    views = [("top-down", 90, -90), ("front", 0, -90), ("side", 15, 0)]
    fig = plt.figure(figsize=(18, 6))
    for i, (view_name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(gt_s[:, 0], gt_s[:, 1], gt_s[:, 2], s=0.4, c="0.6", alpha=0.35, linewidths=0, label="GT")
        ax.scatter(pred_s[:, 0], pred_s[:, 1], pred_s[:, 2], s=0.4, c="tab:red", alpha=0.35, linewidths=0, label="pred")
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(centre[0] - span / 2, centre[0] + span / 2)
        ax.set_ylim(centre[1] - span / 2, centre[1] + span / 2)
        ax.set_zlim(centre[2] - span / 2, centre[2] + span / 2)
        ax.set_title(f"{view_name}")
        if i == 0:
            leg = ax.legend(loc="upper right", markerscale=20)
            for handle in leg.legend_handles:
                handle.set_alpha(1.0)

    fig.suptitle(f"{title}: predicted vs GT point cloud (Sim(3)-aligned on camera centres)")
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}_pred_vs_gt.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")
    return out_path


# --------------------------------------------------------------------------------------
# Viser


def serve(seq: str, gt_pts, gt_cols, pred_pts, pred_cols, port: int):
    import viser

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Scene graph: /root -> {/root/gt, /root/pred} -> clouds, where each of those three nodes
    # *is* a transform gizmo. Dragging /root moves both clouds together; dragging a child moves
    # that cloud alone, in its parent's frame, so the two modes compose instead of fighting.
    # All three stay in the scene and are only hidden, so switching mode never drops a pose.
    spread = float(np.linalg.norm(gt_pts.max(axis=0) - gt_pts.min(axis=0)))
    scale = max(spread * 0.15, 1e-3)

    root_gizmo = server.scene.add_transform_controls("/root", scale=scale)
    gt_gizmo = server.scene.add_transform_controls("/root/gt", scale=scale * 0.6)
    pred_gizmo = server.scene.add_transform_controls("/root/pred", scale=scale * 0.6)

    gt_handle = server.scene.add_point_cloud("/root/gt/cloud", points=gt_pts, colors=gt_cols, point_size=scale * 0.01)
    pred_handle = server.scene.add_point_cloud(
        "/root/pred/cloud", points=pred_pts, colors=pred_cols, point_size=scale * 0.01
    )

    gui_mode = server.gui.add_dropdown("Gizmo", options=("locked together", "move separately"))
    gui_show_gt = server.gui.add_checkbox("Show GT", initial_value=True)
    gui_show_pred = server.gui.add_checkbox("Show pred", initial_value=True)
    gui_flat = server.gui.add_checkbox("Flat colours (grey/red)", initial_value=False)
    gui_size = server.gui.add_slider(
        "Point size", min=scale * 0.001, max=scale * 0.05, step=scale * 0.001, initial_value=scale * 0.01
    )
    gui_reset = server.gui.add_button("Reset transforms")

    grey = np.broadcast_to(np.array([0.6, 0.6, 0.6], np.float32), gt_cols.shape).copy()
    red = np.broadcast_to(np.array([0.9, 0.2, 0.2], np.float32), pred_cols.shape).copy()

    def apply_mode() -> None:
        linked = gui_mode.value == "locked together"
        root_gizmo.visible = linked
        gt_gizmo.visible = not linked
        pred_gizmo.visible = not linked

    @gui_mode.on_update
    def _(_) -> None:
        apply_mode()

    @gui_show_gt.on_update
    def _(_) -> None:
        gt_handle.visible = gui_show_gt.value

    @gui_show_pred.on_update
    def _(_) -> None:
        pred_handle.visible = gui_show_pred.value

    @gui_flat.on_update
    def _(_) -> None:
        gt_handle.colors = grey if gui_flat.value else gt_cols
        pred_handle.colors = red if gui_flat.value else pred_cols

    @gui_size.on_update
    def _(_) -> None:
        gt_handle.point_size = gui_size.value
        pred_handle.point_size = gui_size.value

    @gui_reset.on_click
    def _(_) -> None:
        for gizmo in (root_gizmo, gt_gizmo, pred_gizmo):
            gizmo.position = (0.0, 0.0, 0.0)
            gizmo.wxyz = (1.0, 0.0, 0.0, 0.0)

    apply_mode()
    print(f"viser serving {seq} on port {port} -- ctrl-c to stop")
    while True:
        time.sleep(0.1)


# --------------------------------------------------------------------------------------


def process(model, sintel_root, seq, n_frames, args):
    """One (sequence, frame-count) cell: infer, align, and emit the requested artifacts.

    Returns the aligned clouds so a single-cell run can hand them to viser afterwards.
    """
    rgb_paths = load_sintel_rgb_paths(sintel_root, seq)[:n_frames]
    if not rgb_paths:
        print(f"[{seq} f{n_frames}] no frames -- skipped")
        return None

    name = f"{seq}_f{len(rgb_paths)}"
    print(f"[{name}] {len(rgb_paths)} frames")
    infer_fn = infer_sequence_chunked if args.chunk_size > 0 else infer_sequence
    infer_kw = {"device": args.device}
    if args.chunk_size > 0:
        infer_kw["chunk_size"] = args.chunk_size
    pred = infer_fn(model, rgb_paths, **infer_kw)

    gt_pts, gt_cols, gt_centres = gt_cloud(sintel_root, seq, rgb_paths, args.max_depth, args.stride)
    pred_pts, pred_cols, pred_centres = pred_cloud(pred, rgb_paths, args.stride)

    scale, rot, trans = umeyama_sim3(pred_centres, gt_centres)
    pred_pts = apply_sim3(pred_pts, scale, rot, trans)
    residual = np.linalg.norm(apply_sim3(pred_centres, scale, rot, trans) - gt_centres, axis=1)
    print(f"  Sim(3): scale={scale:.4f}  cam-centre RMSE after align={np.sqrt((residual**2).mean()):.4f}")

    # The full pred->GT transform, as an ordered list of Sim(3)s to replay on the per-frame clouds.
    transforms = [(scale, rot, trans)]

    # Sky and blown-up predictions survive `depth > 0`; clip them in the now-shared GT scale.
    keep = np.linalg.norm(pred_pts - gt_centres.mean(axis=0), axis=1) < args.max_depth
    pred_pts, pred_cols = pred_pts[keep], pred_cols[keep]
    print(f"  points: GT={gt_pts.shape[0]}  pred={pred_pts.shape[0]}")

    if args.geom_align:
        # Factor out the residual pose/orientation so only the shape difference remains.
        s_g, r_g, t_g, rms = scaled_icp(pred_pts, gt_pts)
        pred_pts = apply_sim3(pred_pts, s_g, r_g, t_g)
        transforms.append((s_g, r_g, t_g))
        print(f"  geom ICP: extra scale={s_g:.4f}  trimmed C2C RMS after refine={rms:.4f}")

    if args.save_views:
        save_views(name, f"{name} ({os.path.basename(args.ckpt)})", gt_pts, pred_pts, args.out_dir, args.max_plot_points)
    if args.export_ply:
        if args.method:
            export_method(seq, args.method, gt_pts, gt_cols, pred_pts, pred_cols)
        else:
            export_ply(name, gt_pts, gt_cols, pred_pts, pred_cols, args.out_dir)

    if args.per_frame and args.method:
        export_per_frame(sintel_root, seq, rgb_paths, pred, args.method, transforms, args)

    return name, gt_pts, gt_cols, pred_pts, pred_cols


def export_per_frame(sintel_root, seq, rgb_paths, pred, method, transforms, args):
    """Single-frame point maps: GT on the model grid + pred replaying the fused alignment, so
    frame i of GT and pred are pixel-aligned in the shared GT frame."""
    n = len(rgb_paths)
    frames = args.frames if args.frames is not None else sorted({0, n // 2, n - 1})
    images = load_and_preprocess_images(rgb_paths, mode="crop").numpy()  # (S, 3, H, W)

    for i in frames:
        if not (0 <= i < n):
            print(f"  per-frame: index {i} out of range 0..{n - 1} -- skipped")
            continue
        gp, gc = gt_frame_cloud(sintel_root, seq, rgb_paths[i], images[i], args.max_depth)
        pp, pc = pred_frame_cloud(pred, images, i)
        for s, r, t in transforms:  # bring this frame's pred into the GT frame, same as the fused cloud
            pp = apply_sim3(pp, s, r, t)

        out_dir = os.path.join(OUTPUTS_DIR, "point_cloud", seq, f"frame_{i:04d}")
        os.makedirs(out_dir, exist_ok=True)
        _write_ply(os.path.join(out_dir, "gt.ply"), gp, gc)
        _write_ply(os.path.join(out_dir, f"{method}.ply"), pp, pc)


def main():
    args = parse_args()
    sintel_root = resolve_sintel_root(args.sintel_root)
    seqs = args.seqs or SINTEL_EVAL_SEQUENCES
    frame_counts = args.max_frames
    cells = [(seq, n) for seq in seqs for n in frame_counts]

    if not (args.save_views or args.export_ply or args.port):
        raise SystemExit("Nothing to do: pass --save_views, --export_ply, and/or --port N")
    if args.port and len(cells) > 1:
        raise SystemExit(
            f"--port serves one cloud, but {len(seqs)} seq x {len(frame_counts)} frame-count = {len(cells)} cells "
            "were requested. Narrow to a single --seqs and single --max_frames to use viser, or drop --port for a batch."
        )

    print(f"{len(cells)} cell(s): seqs={list(seqs)}  frames={frame_counts}")
    model = load_vggt_for_eval(args.ckpt, device=args.device)

    last = None
    for seq, n in cells:
        result = process(model, sintel_root, seq, n, args)
        if result is not None:
            last = result

    if args.port and last is not None:
        # viser renders every point in WebGL; cap each cloud so the browser stays interactive.
        name, gt_pts, gt_cols, pred_pts, pred_cols = last
        rng = np.random.default_rng(0)
        gt_v, gt_cv = _subsample(gt_pts, gt_cols, args.max_viser_points, rng)
        pred_v, pred_cv = _subsample(pred_pts, pred_cols, args.max_viser_points, rng)
        print(f"viser points (capped at {args.max_viser_points}): GT={gt_v.shape[0]}  pred={pred_v.shape[0]}")
        serve(name, gt_v, gt_cv, pred_v, pred_cv, args.port)


if __name__ == "__main__":
    main()
