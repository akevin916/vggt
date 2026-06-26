"""VGGT / Dyn-VGGT inference helpers for evaluation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def load_dyn_vggt(
    ckpt: str,
    img_size: int = 518,
    temporal: bool = True,
    motion: bool = True,
    flow: bool = True,
    device: str = "cuda",
) -> VGGT:
    model = VGGT(
        img_size=img_size,
        enable_camera=True,
        enable_depth=True,
        enable_point=True,
        enable_track=False,
        enable_temporal=temporal,
        enable_motion=motion,
        enable_flow=flow,
    )
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def variant_flags(variant: str) -> Tuple[bool, bool, bool]:
    if variant == "vggt_base":
        return False, False, False
    if variant in ("s0", "dyn_vggt_s0"):
        return True, True, True
    raise ValueError(f"Unknown variant: {variant}")


@torch.no_grad()
def infer_sequence(
    model: VGGT,
    image_paths: List[str],
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, np.ndarray]:
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    images = load_and_preprocess_images(image_paths, mode="crop").to(device)
    if images.dim() == 4:
        images = images.unsqueeze(0)

    with torch.cuda.amp.autocast(dtype=dtype, enabled=(device == "cuda")):
        pred = model(images=images)

    h, w = images.shape[-2], images.shape[-1]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], image_size_hw=(h, w))

    depth = pred["depth"]
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    return {
        "extrinsic": extrinsic.squeeze(0).float().cpu().numpy(),
        "intrinsic": intrinsic.squeeze(0).float().cpu().numpy(),
        "depth": depth.squeeze(0).float().cpu().numpy(),
        "pose_enc": pred["pose_enc"].squeeze(0).float().cpu().numpy(),
        "input_hw": np.array([h, w], dtype=np.int32),
    }


def infer_sequence_chunked(
    model: VGGT,
    image_paths: List[str],
    device: str = "cuda",
    chunk_size: int = 32,
) -> Dict[str, np.ndarray]:
    if len(image_paths) <= chunk_size:
        return infer_sequence(model, image_paths, device=device)

    parts = []
    for start in range(0, len(image_paths), chunk_size):
        chunk_paths = image_paths[start : start + chunk_size]
        parts.append(infer_sequence(model, chunk_paths, device=device))

    return {
        "extrinsic": np.concatenate([p["extrinsic"] for p in parts], axis=0),
        "intrinsic": np.concatenate([p["intrinsic"] for p in parts], axis=0),
        "depth": np.concatenate([p["depth"] for p in parts], axis=0),
        "pose_enc": np.concatenate([p["pose_enc"] for p in parts], axis=0),
        "input_hw": parts[0]["input_hw"],
    }
