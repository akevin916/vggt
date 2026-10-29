"""VGGT / Dyn-VGGT inference helpers for evaluation."""

from __future__ import annotations

from typing import Sequence, Dict, List, Optional

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


# Key prefixes of features removed in the slim-down (tag archive-pre-slim). A checkpoint carrying
# any of them would load here with those weights dropped as "unexpected" under strict=False and be
# scored as a different model -- silently -- so refuse it instead.
_REMOVED_FEATURE_KEYS = {
    "gate": ("aggregator.gate_predictor.", "aggregator.gate_bias_scale_p"),
    "dual_stream": ("aggregator.dual_stream_",),
    "alt_stream": ("aggregator.alt_stream_",),
    "tokdino": ("aggregator.tokdino_",),
    "3D RoPE": ("aggregator.rope_3d_dims",),
    "time-phase RoPE": ("aggregator.rope_t.", "aggregator.rope_time_base"),
    "shared temporal weights": ("aggregator.temporal_ls.",),
    "illu token": ("aggregator.illu_token", "illu_head."),
    # removed 2026-10-29: only the Gaussian mask (3-10) stays in slim
    "colmask": ("aggregator.colmask_",),
    "relmask": ("aggregator.relmask_",),
}


def load_vggt_for_eval(
    ckpt: str,
    img_size: int = 518,
    device: str = "cuda",
    force_point: bool = False,
    verbose: bool = True,
) -> VGGT:
    """Build VGGT with architecture inferred from checkpoint keys.

    Each head/block family is detected independently from its own key prefix: temporal blocks,
    depth/point heads. The Gaussian mask is NOT rebuilt here: a learned one is exported to json and
    attached by the caller (Aggregator.set_gaussmask_fixed, eval_scared --gaussmask).
    """
    sd = _load_state_dict(ckpt)
    keys = list(sd.keys())
    found = sorted({name for name, prefixes in _REMOVED_FEATURE_KEYS.items()
                    for k in keys if k.startswith(prefixes)})
    if any("lora_" in k for k in keys):
        found.append("LoRA")
    if found:
        raise SystemExit(f"{ckpt} uses features removed from this build: {found}. "
                         "Evaluate it from tag archive-pre-slim or branch illu-vggt (or p3-cross-gate / mask-grid-wip for gate).")
    has_temporal = any("temporal" in k for k in keys)
    has_depth = any(k.startswith("depth_head") for k in keys)
    # force_point: build the point head even when the checkpoint has none. The head is then
    # random-init and only useful once real weights are grafted in -- see graft_point_head.
    has_point = any(k.startswith("point_head") for k in keys) or force_point

    model = VGGT(
        img_size=img_size,
        enable_camera=True,
        enable_depth=has_depth,
        enable_point=has_point,
        enable_track=False,
        enable_temporal=has_temporal,
    )
    miss, unexp = model.load_state_dict(sd, strict=False)
    if any(k.startswith("aggregator.gaussmask.") for k in keys):
        print(f"WARNING: {ckpt} carries a learned Gaussian mask (aggregator.gaussmask.*) that this "
              "loader drops; pass its exported json (eval_scared --gaussmask) or the model runs "
              "WITHOUT the mask.")
    if verbose:
        print(f"loaded {ckpt}: missing={len(miss)} unexpected={len(unexp)}")
    return model.to(device).eval()


def graft_point_head(model: VGGT, donor_ckpt: str, verbose: bool = True) -> VGGT:
    """Load ``point_head`` weights from ``donor_ckpt`` into an already-built model.

    Lets a gate-method checkpoint -- which never builds a point head -- borrow the pretrained
    VGGT-1B one. Build ``model`` with ``force_point=True`` first.

    ⚠️ The grafted head reads trunk features it was never trained on: the gate-method S1 runs
    train global blocks 8-23, so the aggregator has drifted from what the donor head saw.
    Treat its output as indicative, not as the point head's true quality on that trunk.
    (The frozen depth_head survives the same drift -- AbsRel 0.2747 -> 0.2136 on run1 --
    which is why this is worth trying at all, but it is not proof.)
    """
    if model.point_head is None:
        raise RuntimeError("model has no point_head; build it with force_point=True")

    sd = _load_state_dict(donor_ckpt)
    prefix = "point_head."
    point_sd = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    if not point_sd:
        raise SystemExit(f"donor checkpoint has no point_head weights: {donor_ckpt}")

    model.point_head.load_state_dict(point_sd, strict=True)
    if verbose:
        print(f"grafted point_head from {donor_ckpt}: {len(point_sd)} keys")
    return model


@torch.no_grad()
def infer_sequence(
    model: VGGT,
    image_paths: List[str],
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
    want_point: bool = False,
    images: Optional[torch.Tensor] = None,
    frame_pos: Optional[Sequence[int]] = None,
) -> Dict[str, np.ndarray]:
    """Run VGGT on one sequence.

    ``frame_pos`` (length S) is each frame's index in its source sequence; only the temporal
    blocks read it (offset from the first frame). None = treat the frames as contiguous.

    ``want_point`` opts in to returning the point head's ``world_points`` /
    ``world_points_conf``. It is off by default because those are [S,H,W,3] fp32
    arrays (~160 MB for a 50-frame Sintel sequence) that only diag/pnp_pose.py wants.

    ``images`` bypasses loading and runs on an already-preprocessed [S,3,H,W] (or
    [1,S,3,H,W]) tensor instead, so a caller can perturb the pixels first --
    diag/washout_impact.py injects saturated blobs this way. ``image_paths`` is then
    only used for its length, and passing an image tensor whose geometry differs from
    what those paths would have produced silently invalidates every GT correspondence.
    """
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    if images is None:
        images = load_and_preprocess_images(image_paths, mode="crop")
    images = images.to(device)
    if images.dim() == 4:
        images = images.unsqueeze(0)

    with torch.cuda.amp.autocast(dtype=dtype, enabled=(device == "cuda")):
        fp = None if frame_pos is None else torch.as_tensor(np.asarray(frame_pos), dtype=torch.long,
                                                            device=images.device).view(1, -1)
        pred = model(images=images, frame_pos=fp)

    h, w = images.shape[-2], images.shape[-1]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], image_size_hw=(h, w))

    out = {
        "extrinsic": extrinsic.squeeze(0).float().cpu().numpy(),
        "intrinsic": intrinsic.squeeze(0).float().cpu().numpy(),
        "pose_enc": pred["pose_enc"].squeeze(0).float().cpu().numpy(),
        "input_hw": np.array([h, w], dtype=np.int32),
    }
    # Pose-only checkpoints (e.g. the oracle-camera-only ablation) have no depth_head.
    if "depth" in pred:
        depth = pred["depth"]
        if depth.ndim == 5 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        out["depth"] = depth.squeeze(0).float().cpu().numpy()
    if want_point and "world_points" in pred:
        # [S, H, W, 3] in the frame-0 camera frame, plus its [S, H, W] confidence.
        out["world_points"] = pred["world_points"].squeeze(0).float().cpu().numpy()
        out["world_points_conf"] = pred["world_points_conf"].squeeze(0).float().cpu().numpy()
    return out


def infer_sequence_chunked(
    model: VGGT,
    image_paths: List[str],
    device: str = "cuda",
    chunk_size: int = 32,
) -> Dict[str, np.ndarray]:
    """Chunked inference. ``chunk_size <= 0`` means one pass over the whole sequence.

    ⚠️ Chunks are inferred INDEPENDENTLY and concatenated without any alignment, so each
    chunk carries its own arbitrary reference frame. Any pose metric computed across a
    seam is dominated by that jump: on Sintel f50 the default 32 split temple_2 into
    32+18 and gave ATE 2.53 instead of 0.057 -- and, because the seam dwarfs everything
    else, the number stopped depending on the model at all. Callers that want whole-
    sequence geometry must pass 0 (or len(image_paths)); leaving it unset silently picks
    the 32 default. This docstring is the authoritative statement of that trap.
    """
    if chunk_size <= 0 or len(image_paths) <= chunk_size:
        return infer_sequence(model, image_paths, device=device)

    parts = []
    for start in range(0, len(image_paths), chunk_size):
        chunk_paths = image_paths[start : start + chunk_size]
        parts.append(infer_sequence(model, chunk_paths, device=device))

    out = {
        "extrinsic": np.concatenate([p["extrinsic"] for p in parts], axis=0),
        "intrinsic": np.concatenate([p["intrinsic"] for p in parts], axis=0),
        "pose_enc": np.concatenate([p["pose_enc"] for p in parts], axis=0),
        "input_hw": parts[0]["input_hw"],
    }
    if "depth" in parts[0]:
        out["depth"] = np.concatenate([p["depth"] for p in parts], axis=0)
    return out


# ---------------------------------------------------------------------------
# Overlapped, Sim3-stitched inference
#
# ``infer_sequence_chunked`` above concatenates chunks blind, which is fatal for any
# cross-seam pose metric. The SCARED pose trajectories are 411 and 834 frames, far past what
# one VGGT forward pass fits, so whole-sequence ATE needs the chunks joined properly instead:
# consecutive chunks share ``overlap`` frames, and each new chunk is mapped into the running
# trajectory by the similarity transform that best aligns its camera centres on that shared
# span. This is the standard long-sequence submap stitch, not a shortcut -- but it does add
# its own error, so measure it (run a sequence short enough for one pass both ways and
# compare) before putting stitched numbers in a table.
# ---------------------------------------------------------------------------


def umeyama_sim3(src: np.ndarray, dst: np.ndarray):
    """Least-squares similarity (s, R, t) with ``dst ≈ s * R @ src + t``. Points are (N,3)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xs, Xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(Xd.T @ Xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    var_s = (Xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 0 else 1.0
    return s, R, mu_d - s * R @ mu_s


def _extrinsic_to_c2w(E: np.ndarray) -> np.ndarray:
    bottom = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(E), 1, 1))
    return np.linalg.inv(np.concatenate([np.asarray(E, dtype=np.float64), bottom], axis=1))


def _c2w_to_extrinsic(M: np.ndarray) -> np.ndarray:
    return np.linalg.inv(M)[:, :3, :]


def infer_sequence_stitched(
    model: VGGT,
    image_paths: List[str],
    device: str = "cuda",
    chunk_size: int = 64,
    overlap: int = 16,
    want_depth: bool = False,
) -> Dict[str, np.ndarray]:
    """Whole-sequence poses from overlapping chunks joined by a per-seam Sim3.

    Returns ``extrinsic`` (S,3,4) in one common frame plus ``intrinsic``/``pose_enc`` and
    ``chunk_seams`` / ``stitch_scales`` for diagnostics.

    ``want_depth`` additionally returns each frame's depth multiplied by its own chunk's
    seam scale. That is exactly the transform the poses already got -- unprojection is
    linear in depth and the chunk's c2w has been rotated and translated by the same Sim3,
    so ``depth * s`` with the stitched extrinsic puts every chunk's surface in the ONE
    frame the poses live in. It is what a long-sequence point cloud needs and it is off by
    default anyway, because the resulting depth is expressed in chunk 0's arbitrary units
    with every later chunk rescaled into them: FOR VIEWING AND FUSION ONLY, never for a
    depth metric, which must come from a single unstitched forward pass.
    """
    n = len(image_paths)
    if chunk_size <= 0 or n <= chunk_size:
        return infer_sequence(model, image_paths, device=device)
    if not 0 < overlap < chunk_size:
        raise ValueError(f"overlap must be in (0, chunk_size); got {overlap} vs {chunk_size}")

    step = chunk_size - overlap
    starts = list(range(0, max(n - overlap, 1), step))
    if starts[-1] + chunk_size < n:
        starts.append(n - chunk_size)

    c2w_out = np.zeros((n, 4, 4))
    intr_out = [None] * n
    penc_out = [None] * n
    depth_out = None
    filled = np.zeros(n, dtype=bool)
    scales, seams = [], []

    for ci, st in enumerate(starts):
        sl = slice(st, min(st + chunk_size, n))
        pred = infer_sequence(model, image_paths[sl], device=device)
        c2w = _extrinsic_to_c2w(pred["extrinsic"])
        s_chunk = 1.0

        if ci > 0:
            shared = np.arange(sl.start, sl.stop)[filled[sl]]
            if len(shared) < 3:
                raise RuntimeError(f"seam {ci}: only {len(shared)} shared frames, need >=3 "
                                   f"for a Sim3 fit (raise --overlap)")
            local = shared - sl.start
            s, R, t = umeyama_sim3(c2w[local, :3, 3], c2w_out[shared, :3, 3])
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, t
            c2w[:, :3, 3] = (s * (R @ c2w[:, :3, 3].T)).T + t   # centres: scaled+rotated
            c2w[:, :3, :3] = R @ c2w[:, :3, :3]                  # orientations: rotated only
            s_chunk = float(s)
            scales.append(s_chunk)
            seams.append(int(sl.start))

        # Keep the earlier chunk's estimate on shared frames: it was fitted, not extrapolated.
        new = np.arange(sl.start, sl.stop)[~filled[sl]]
        c2w_out[new] = c2w[new - sl.start]
        for j in new:
            intr_out[j] = pred["intrinsic"][j - sl.start]
            penc_out[j] = pred["pose_enc"][j - sl.start]
        if want_depth:
            if "depth" not in pred:
                raise RuntimeError("want_depth=True but the model has no depth head")
            d = pred["depth"]
            if depth_out is None:
                depth_out = np.zeros((n,) + d.shape[1:], dtype=np.float32)
            depth_out[new] = d[new - sl.start] * s_chunk
        filled[sl] = True

    if not filled.all():
        raise RuntimeError(f"{int((~filled).sum())} frames never covered by a chunk")

    out = {
        "extrinsic": _c2w_to_extrinsic(c2w_out).astype(np.float32),
        "intrinsic": np.stack(intr_out),
        "pose_enc": np.stack(penc_out),
        "input_hw": np.array([0, 0], dtype=np.int32),
        "chunk_starts": np.array(starts, dtype=np.int32),
        "chunk_seams": np.array(seams, dtype=np.int32),
        "stitch_scales": np.array(scales, dtype=np.float64),
    }
    if want_depth:
        out["depth"] = depth_out
        out["input_hw"] = np.array(depth_out.shape[1:3], dtype=np.int32)
    return out
