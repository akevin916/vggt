"""Sintel dataset I/O for Dyn-VGGT evaluation."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

TAG_FLOAT = 202021.25

# MonST3R standard Sintel training eval split (14 sequences).
SINTEL_EVAL_SEQUENCES = [
    "alley_2",
    "ambush_4",
    "ambush_5",
    "ambush_6",
    "cave_2",
    "cave_4",
    "market_2",
    "market_5",
    "market_6",
    "shaman_3",
    "sleeping_1",
    "sleeping_2",
    "temple_2",
    "temple_3",
]

DEFAULT_SINTEL_ROOT = "/home/cvml-75/Desktop/3D-repo/data/sintel/training"


@dataclass
class PreprocessMeta:
    """Maps model input crop back to original image coordinates."""

    orig_h: int
    orig_w: int
    new_w: int
    new_h: int
    crop_y0: int  # top of center crop in resized image (0 if no crop)


def list_sintel_sequences(seq_list: Optional[List[str]] = None) -> List[str]:
    return list(seq_list or SINTEL_EVAL_SEQUENCES)


def sintel_seq_paths(sintel_root: str, seq: str) -> Tuple[str, str, str]:
    rgb_dir = os.path.join(sintel_root, "final", seq)
    depth_dir = os.path.join(sintel_root, "depth", seq)
    cam_dir = os.path.join(sintel_root, "camdata_left", seq)
    return rgb_dir, depth_dir, cam_dir


def load_sintel_rgb_paths(sintel_root: str, seq: str) -> List[str]:
    rgb_dir, _, _ = sintel_seq_paths(sintel_root, seq)
    paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No RGB frames in {rgb_dir}")
    return paths


def frame_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def matching_depth_path(depth_dir: str, rgb_path: str) -> str:
    stem = frame_stem(rgb_path)
    path = os.path.join(depth_dir, f"{stem}.dpt")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def matching_cam_path(cam_dir: str, rgb_path: str) -> str:
    stem = frame_stem(rgb_path)
    path = os.path.join(cam_dir, f"{stem}.cam")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def read_sintel_depth(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        check = np.fromfile(f, dtype=np.float32, count=1)[0]
        if check != TAG_FLOAT:
            raise ValueError(f"Bad depth tag in {path}: {check}")
        width = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        height = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth.astype(np.float32)


def sintel_cam_read(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        check = np.fromfile(f, dtype=np.float32, count=1)[0]
        if check != TAG_FLOAT:
            raise ValueError(f"Bad cam tag in {path}: {check}")
        intrinsic = np.fromfile(f, dtype=np.float64, count=9).reshape((3, 3))
        extrinsic = np.fromfile(f, dtype=np.float64, count=12).reshape((3, 4))
    return intrinsic, extrinsic


def load_sintel_gt_poses(cam_dir: str, rgb_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return TUM-style (N,7) xyz+wxyz and timestamps (N,1), centered like MonST3R."""
    tstamps, tum_poses = [], []
    for rgb_path in rgb_paths:
        cam_path = matching_cam_path(cam_dir, rgb_path)
        _, ext_w2c = sintel_cam_read(cam_path)
        frame_id = float(frame_stem(rgb_path).split("_")[-1])
        w2c = np.vstack([ext_w2c, np.array([0, 0, 0, 1], dtype=np.float64)])
        c2w = np.linalg.inv(w2c)
        xyz = c2w[:3, 3]
        quat_xyzw = Rotation.from_matrix(c2w[:3, :3]).as_quat()
        wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        tum_poses.append(np.concatenate([xyz, wxyz]))
        tstamps.append(frame_id)

    tum = np.stack(tum_poses, axis=0)
    tum[:, :3] -= tum[:, :3].mean(axis=0, keepdims=True)
    tt = np.expand_dims(np.array(tstamps, dtype=np.float64), -1)
    return tum, tt


def load_sintel_gt_depths(sintel_root: str, seq: str, rgb_paths: List[str]) -> List[np.ndarray]:
    _, depth_dir, _ = sintel_seq_paths(sintel_root, seq)
    return [read_sintel_depth(matching_depth_path(depth_dir, p)) for p in rgb_paths]


def compute_preprocess_meta(image_path: str, target_size: int = 518, mode: str = "crop") -> PreprocessMeta:
    """Mirror vggt.utils.load_fn crop logic for depth remapping."""
    with Image.open(image_path) as img:
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")
        width, height = img.size

    if mode == "crop":
        new_w = target_size
        new_h = round(height * (new_w / width) / 14) * 14
        crop_y0 = max(0, (new_h - target_size) // 2) if new_h > target_size else 0
    else:
        raise NotImplementedError("Only crop mode is supported for Sintel eval")

    return PreprocessMeta(orig_h=height, orig_w=width, new_w=new_w, new_h=new_h, crop_y0=crop_y0)


def resize_pred_to_gt(pred: np.ndarray, meta: PreprocessMeta) -> np.ndarray:
    """Resize model depth (H_model,W_model) to original GT resolution."""
    import cv2

    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred[..., 0]

    # pred is on the cropped 518x518 (or new_w x min(new_h,518)) tensor space
    model_h, model_w = pred.shape
    resized = cv2.resize(pred, (meta.new_w, meta.new_h), interpolation=cv2.INTER_LINEAR)
    if meta.new_h > model_h:
        y0 = meta.crop_y0
        resized = resized[y0 : y0 + model_h, :]
    out = cv2.resize(resized, (meta.orig_w, meta.orig_h), interpolation=cv2.INTER_LINEAR)
    return out.astype(np.float32)
