#!/usr/bin/env python3
"""Gradio UI dedicated to exporting VGGT point clouds from folders of images.

This is the export-focused sibling of demo_gradio.py. The workflow it is built around:
point it at one or more image folders on disk (each folder = one scene), pick a checkpoint,
and for each scene choose the frame segment whose point cloud you want, preview it in the
browser, then export it to a .ply for CloudCompare/MeshLab. Unlike a batch script it keeps a
human in the loop -- you look at each scene and decide what to keep.

Fixed choices baked in on purpose:
  * stride is always 1 -- every pixel of the model's output resolution becomes a point.
  * the checkpoint is chosen in the UI (base VGGT-1B or any dyn-vggt ckpt under training/),
    loaded through eval_utils.load_vggt_for_eval so gate/temporal/etc. flags are inferred.

Run:  python demo_pointcloud_export.py            # then open the printed URL
      (tunnel the port like viser if you are on a remote box: ssh -L 7860:localhost:7860 ...)
"""

from __future__ import annotations

import gc
import glob
import os
import sys
import tempfile
from datetime import datetime

import numpy as np
import torch
import gradio as gr

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_TRAINING_DIR = os.path.join(_REPO_DIR, "training")
sys.path[:0] = [_TRAINING_DIR, _REPO_DIR]

from eval_utils.paths import POINTCLOUD_EXPORT, default_output_dir
from eval_utils.ply_io import write_ply
from eval_utils.vggt_infer import load_vggt_for_eval
from visual_util import predictions_to_glb
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# Keep at most one model resident: switching checkpoints frees the previous one's VRAM.
_LOADED: dict = {"ckpt": None, "model": None}


# --------------------------------------------------------------------------------------
# Checkpoints


def list_checkpoints() -> list[str]:
    """Every .pt the UI offers: training/checkpoints/*.pt plus each run's best/last.pt."""
    found = sorted(glob.glob(os.path.join(_TRAINING_DIR, "checkpoints", "*.pt")))
    for tag in ("best.pt", "last.pt"):
        found += sorted(glob.glob(os.path.join(_TRAINING_DIR, "logs", "*", "ckpts", tag)))
    # Show paths relative to training/ so the dropdown stays readable.
    return [os.path.relpath(p, _TRAINING_DIR) for p in found]


def ensure_model(ckpt_rel: str):
    if not ckpt_rel:
        raise gr.Error("Pick a checkpoint first.")
    ckpt = os.path.join(_TRAINING_DIR, ckpt_rel)
    if _LOADED["ckpt"] == ckpt and _LOADED["model"] is not None:
        return _LOADED["model"]
    if _LOADED["model"] is not None:  # free the previous checkpoint before loading another
        _LOADED["model"] = None
        gc.collect()
        torch.cuda.empty_cache()
    _LOADED["model"] = load_vggt_for_eval(ckpt, device=DEVICE)
    _LOADED["ckpt"] = ckpt
    return _LOADED["model"]


# --------------------------------------------------------------------------------------
# Scenes (folders of images)


def scene_images(folder: str) -> list[str]:
    if not folder or not os.path.isdir(folder):
        return []
    files = [p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(IMAGE_EXTS)]
    return sorted(files)


def collect_scenes(folders_text: str, parent_dir: str) -> list[str]:
    """A scene is any folder that directly contains images. From the textbox (one path per
    line) plus, if given, every image-bearing subfolder of ``parent_dir``."""
    scenes: list[str] = []
    for line in (folders_text or "").splitlines():
        line = line.strip()
        if line and os.path.isdir(line) and scene_images(line):
            scenes.append(os.path.abspath(line))
    if parent_dir and os.path.isdir(parent_dir):
        if scene_images(parent_dir):  # the parent itself may be a single scene
            scenes.append(os.path.abspath(parent_dir))
        for sub in sorted(glob.glob(os.path.join(parent_dir, "*"))):
            if os.path.isdir(sub) and scene_images(sub):
                scenes.append(os.path.abspath(sub))
    # De-dup, preserve order.
    seen, unique = set(), []
    for s in scenes:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def on_scan(folders_text: str, parent_dir: str):
    scenes = collect_scenes(folders_text, parent_dir)
    if not scenes:
        return gr.update(choices=[], value=None), "No image folders found. Check the paths."
    msg = f"Found {len(scenes)} scene(s):\n" + "\n".join(f"  - {s} ({len(scene_images(s))} imgs)" for s in scenes)
    return gr.update(choices=scenes, value=scenes[0]), msg


def on_select_scene(folder: str):
    imgs = scene_images(folder)
    n = len(imgs)
    if n == 0:
        empty = gr.update(minimum=0, maximum=1, value=0)
        return [], empty, empty, "empty scene"
    last = n - 1
    hi = max(last, 1)  # gradio needs minimum < maximum even for a 1-frame scene
    start = gr.update(minimum=0, maximum=hi, value=0, step=1)
    end = gr.update(minimum=0, maximum=hi, value=last, step=1)
    return imgs, start, end, f"{os.path.basename(folder)}: {n} frames (0..{last})"


# --------------------------------------------------------------------------------------
# Inference + selection


@torch.no_grad()
def run_inference(model, image_paths: list[str]) -> dict:
    images = load_and_preprocess_images(image_paths).to(DEVICE)  # (S, 3, H, W), stride-1 crop res
    dtype = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.amp.autocast("cuda", dtype=dtype, enabled=(DEVICE == "cuda")):
        pred = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    pred["extrinsic"] = extrinsic
    pred["intrinsic"] = intrinsic
    out = {}
    for k, v in pred.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.squeeze(0).cpu().numpy()  # drop batch dim
    out["images"] = images.squeeze(0).cpu().numpy()  # (S, 3, H, W) in [0, 1]
    out["world_points_from_depth"] = unproject_depth_map_to_point_map(
        out["depth"], out["extrinsic"], out["intrinsic"]
    )
    torch.cuda.empty_cache()
    return out


def select_points(pred: dict, conf_thres: float, mask_black_bg: bool, mask_white_bg: bool):
    """Points + [0,1] colours after the same conf/background filtering the glb preview uses,
    so the exported .ply is exactly what you see in the browser."""
    pts = pred["world_points_from_depth"].reshape(-1, 3)
    conf = pred.get("depth_conf", np.ones(pred["world_points_from_depth"].shape[:-1])).reshape(-1)
    colors = np.transpose(pred["images"], (0, 2, 3, 1)).reshape(-1, 3)  # NCHW -> NHWC -> (N,3), [0,1]

    thr = 0.0 if conf_thres <= 0.0 else np.percentile(conf, conf_thres)
    mask = (conf >= thr) & (conf > 1e-5)
    rgb255 = colors * 255.0
    if mask_black_bg:
        mask &= rgb255.sum(axis=1) >= 16
    if mask_white_bg:
        mask &= ~((rgb255[:, 0] > 240) & (rgb255[:, 1] > 240) & (rgb255[:, 2] > 240))
    return pts[mask], colors[mask]


def build_glb(pred: dict, conf_thres, mask_black_bg, mask_white_bg, show_cam) -> str:
    scene = predictions_to_glb(
        pred,
        conf_thres=conf_thres,
        filter_by_frames="all",
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=False,
        target_dir=None,
        prediction_mode="Depthmap and Camera Branch",
    )
    tmp = os.path.join(_REPO_DIR, "outputs", "_preview")
    os.makedirs(tmp, exist_ok=True)
    path = os.path.join(tmp, f"preview_{datetime.now().strftime('%H%M%S_%f')}.glb")
    scene.export(file_obj=path)
    return path


# --------------------------------------------------------------------------------------
# Callbacks


def reconstruct(ckpt_rel, folder, start, end, conf_thres, mask_black_bg, mask_white_bg, show_cam):
    imgs = scene_images(folder)
    if not imgs:
        raise gr.Error("Select a scene with images first (Scan, then pick from the dropdown).")
    start, end = int(start), int(end)
    if end < start:
        start, end = end, start
    segment = imgs[start : end + 1]

    model = ensure_model(ckpt_rel)
    pred = run_inference(model, segment)
    meta = {"ckpt": ckpt_rel, "scene": os.path.basename(folder.rstrip("/")), "start": start, "end": end}

    glb = build_glb(pred, conf_thres, mask_black_bg, mask_white_bg, show_cam)
    log = f"Reconstructed {meta['scene']} frames {start}..{end} ({len(segment)} imgs) at {pred['images'].shape[-1]}px width."
    return glb, log, (pred, meta)


def update_preview(state, conf_thres, mask_black_bg, mask_white_bg, show_cam):
    if not state:
        return None, "Nothing reconstructed yet."
    pred, _ = state
    return build_glb(pred, conf_thres, mask_black_bg, mask_white_bg, show_cam), "Preview updated."


def export_ply(state, ckpt_rel, conf_thres, mask_black_bg, mask_white_bg):
    if not state:
        raise gr.Error("Reconstruct a scene before exporting.")
    pred, meta = state
    pts, cols = select_points(pred, conf_thres, mask_black_bg, mask_white_bg)
    if pts.shape[0] == 0:
        raise gr.Error("No points survive the current confidence/background filters.")

    ckpt = os.path.join(_TRAINING_DIR, ckpt_rel)
    out_dir = default_output_dir(ckpt, POINTCLOUD_EXPORT)
    os.makedirs(out_dir, exist_ok=True)
    name = f"{meta['scene']}_f{meta['start']}-{meta['end']}.ply"
    path = os.path.join(out_dir, name)
    n = write_ply(path, pts, cols)
    return path, f"Saved {n} points -> {path}"


# --------------------------------------------------------------------------------------
# UI


def build_ui():
    with gr.Blocks(title="VGGT Point-Cloud Export") as demo:
        gr.Markdown(
            "# VGGT Point-Cloud Export\n"
            "Pick a checkpoint, point at one or more image folders, choose a frame segment per "
            "scene, preview, then export a full-resolution `.ply` (stride 1)."
        )
        state = gr.State(None)  # (predictions dict, meta) for the last reconstruction

        with gr.Row():
            with gr.Column(scale=2):
                ckpt = gr.Dropdown(choices=list_checkpoints(), label="Checkpoint (under training/)")
                folders_text = gr.Textbox(
                    label="Scene folders (one path per line)", lines=3, placeholder="/path/to/scene_a\n/path/to/scene_b"
                )
                parent_dir = gr.Textbox(label="...or a parent dir to scan for scene subfolders", placeholder="/path/to/scenes")
                scan_btn = gr.Button("Scan folders", variant="secondary")
                scene = gr.Dropdown(choices=[], label="Scene")
                gallery = gr.Gallery(label="Scene frames", columns=6, height=200)
                with gr.Row():
                    # min<max required at construction; real bounds are set on scene select.
                    start = gr.Slider(0, 1, value=0, step=1, label="Start frame")
                    end = gr.Slider(0, 1, value=1, step=1, label="End frame")
                info = gr.Markdown()

            with gr.Column(scale=4):
                preview = gr.Model3D(height=520, label="Point-cloud preview")
                with gr.Row():
                    reconstruct_btn = gr.Button("Reconstruct segment", variant="primary")
                    export_btn = gr.Button("Export PLY", variant="primary")
                with gr.Row():
                    conf_thres = gr.Slider(0, 100, value=50, step=0.1, label="Confidence filter (%)")
                    show_cam = gr.Checkbox(label="Show cameras", value=True)
                    mask_black_bg = gr.Checkbox(label="Filter black bg", value=False)
                    mask_white_bg = gr.Checkbox(label="Filter white bg", value=False)
                log = gr.Markdown()
                download = gr.File(label="Exported PLY")

        scan_btn.click(on_scan, [folders_text, parent_dir], [scene, info])
        scene.change(on_select_scene, [scene], [gallery, start, end, info])

        recon_inputs = [ckpt, scene, start, end, conf_thres, mask_black_bg, mask_white_bg, show_cam]
        reconstruct_btn.click(reconstruct, recon_inputs, [preview, log, state])

        # Changing a display-only control rebuilds the glb from the cached predictions -- no re-inference.
        for ctrl in (conf_thres, mask_black_bg, mask_white_bg, show_cam):
            ctrl.change(update_preview, [state, conf_thres, mask_black_bg, mask_white_bg, show_cam], [preview, log])

        export_btn.click(export_ply, [state, ckpt, conf_thres, mask_black_bg, mask_white_bg], [download, log])

    return demo


def allowed_read_roots() -> list[str]:
    """Scene folders live anywhere on this box (often behind the data/ symlink to /media),
    but gradio 6 only serves files under CWD/temp/allowed_paths. Allow the real dataset mount
    points so the gallery can show source frames. Bound to localhost + an SSH tunnel, this is a
    single-user tool, so opening these read-only roots to the served UI is acceptable."""
    roots = {tempfile.gettempdir(), _REPO_DIR}
    data_link = os.path.join(_REPO_DIR, "data")
    if os.path.exists(data_link):
        roots.add(os.path.realpath(data_link))
    for mount in ("/media", "/mnt", "/data", "/home"):
        if os.path.isdir(mount):
            roots.add(mount)
    return sorted(roots)


if __name__ == "__main__":
    if DEVICE != "cuda":
        print("WARNING: CUDA not available; inference will be very slow / may fail.")
    build_ui().queue(max_size=8).launch(
        server_name="127.0.0.1",  # reach it via `ssh -L 7860:localhost:7860 ...`
        server_port=7860,
        show_error=True,
        allowed_paths=allowed_read_roots(),
    )
