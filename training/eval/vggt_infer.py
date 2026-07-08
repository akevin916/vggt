"""VGGT / Dyn-VGGT inference helpers for evaluation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def _load_state_dict(ckpt: str) -> dict:
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    return sd


def load_vggt_for_eval(
    ckpt: str,
    img_size: int = 518,
    gate_block_iter: int = 7,
    device: str = "cuda",
    require_gate: bool = False,
    verbose: bool = True,
) -> VGGT:
    """Build VGGT with architecture inferred from checkpoint keys.

    Each head/block family is detected independently from its own key prefix, so any
    combination (e.g. v3's pose-only oracle-gate ablation: temporal + camera, no gate,
    no depth/point/motion/flow) reconstructs correctly -- not just the three presets
    (plain VGGT / v1-v2 dyn / v3 gate) this used to special-case.
    """
    sd = _load_state_dict(ckpt)
    keys = list(sd.keys())
    has_gate = any("gate_predictor" in k for k in keys)
    has_temporal = any("temporal" in k for k in keys)
    has_depth = any(k.startswith("depth_head") for k in keys)
    has_point = any(k.startswith("point_head") for k in keys)
    has_motion = any(k.startswith("motion_head") for k in keys)
    has_flow = any(k.startswith("flow_head") for k in keys)

    if require_gate and not has_gate:
        raise SystemExit(f"checkpoint has no gate_predictor weights: {ckpt}")

    model = VGGT(
        img_size=img_size,
        enable_camera=True,
        enable_depth=has_depth,
        enable_point=has_point,
        enable_track=False,
        enable_temporal=has_temporal,
        enable_motion=has_motion,
        enable_flow=has_flow,
        enable_gate=has_gate,
        gate_block_iter=gate_block_iter,
    )
    miss, unexp = model.load_state_dict(sd, strict=False)
    if verbose:
        if has_gate:
            n_gate = sum(1 for k in sd if "gate_predictor" in k)
            print(
                f"loaded {ckpt}: gate_predictor keys={n_gate} "
                f"missing={len(miss)} unexpected={len(unexp)}"
            )
        else:
            print(f"loaded {ckpt}: missing={len(miss)} unexpected={len(unexp)}")
    return model.to(device).eval()


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
    sd = _load_state_dict(ckpt)
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
    gate_logits_override: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    images = load_and_preprocess_images(image_paths, mode="crop").to(device)
    if images.dim() == 4:
        images = images.unsqueeze(0)

    with torch.cuda.amp.autocast(dtype=dtype, enabled=(device == "cuda")):
        pred = model(images=images, gate_logits_override=gate_logits_override)

    h, w = images.shape[-2], images.shape[-1]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], image_size_hw=(h, w))

    out = {
        "extrinsic": extrinsic.squeeze(0).float().cpu().numpy(),
        "intrinsic": intrinsic.squeeze(0).float().cpu().numpy(),
        "pose_enc": pred["pose_enc"].squeeze(0).float().cpu().numpy(),
        "input_hw": np.array([h, w], dtype=np.int32),
    }
    # Pose-only checkpoints (e.g. the v3 oracle-camera-only ablation) have no depth_head.
    if "depth" in pred:
        depth = pred["depth"]
        if depth.ndim == 5 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        out["depth"] = depth.squeeze(0).float().cpu().numpy()
    if "gate_logits" in pred:
        # [S, P_patch] fp32 — the model's own predicted gate logits (pre-bias), for
        # reuse as a (possibly rescaled) gate_logits_override in follow-up ablations.
        out["gate_logits"] = pred["gate_logits"].squeeze(0).float().cpu().numpy()
    return out


def infer_sequence_chunked(
    model: VGGT,
    image_paths: List[str],
    device: str = "cuda",
    chunk_size: int = 32,
    gate_logits_override: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    if len(image_paths) <= chunk_size:
        return infer_sequence(model, image_paths, device=device, gate_logits_override=gate_logits_override)

    parts = []
    for start in range(0, len(image_paths), chunk_size):
        chunk_paths = image_paths[start : start + chunk_size]
        chunk_override = (
            gate_logits_override[:, start : start + chunk_size] if gate_logits_override is not None else None
        )
        parts.append(infer_sequence(model, chunk_paths, device=device, gate_logits_override=chunk_override))

    out = {
        "extrinsic": np.concatenate([p["extrinsic"] for p in parts], axis=0),
        "intrinsic": np.concatenate([p["intrinsic"] for p in parts], axis=0),
        "pose_enc": np.concatenate([p["pose_enc"] for p in parts], axis=0),
        "input_hw": parts[0]["input_hw"],
    }
    if "depth" in parts[0]:
        out["depth"] = np.concatenate([p["depth"] for p in parts], axis=0)
    return out
