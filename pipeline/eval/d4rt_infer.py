"""OpenD4RT inference adapter: video frames -> per-frame depth + world-to-cam extrinsics.

OpenD4RT (reference/Open-d4rt, the unofficial RHOS reproduction of D4RT, CVPR'26) only
exposes a point-query decoder: a query (u, v, t_src, t_tgt, t_cam) returns the 3D position
of pixel (u, v) of frame t_src, at time t_tgt, in camera t_cam's coordinates. Neither the
official code (unreleased) nor OpenD4RT ships a pose/depth readout, so both follow the
D4RT paper, Sec. 2.3:

  * depth of frame i  = z of (u, v, i, i, i) on the full model-resolution grid;
  * pose of frame i   = rigid Umeyama from {(u, v, i, i, i)} to {(u, v, i, i, 0)} on a coarse
                        grid -- the same points in camera i and in camera 0 -- which is
                        cam_i -> cam_0, i.e. c2w with world = camera 0.

The paper does not give the coarse grid size; POSE_GRID is our choice.

Preprocessing copies OpenD4RT's own eval (eval_track3d_in_worldtrack.py): every frame is
squashed to the model's image_size (256x256) with cv2 INTER_AREA, RGB / 255, and the native
W/H goes in as the aspect-ratio token. Inference runs in fp32 as theirs does.

Clip length is capped by the 48-entry time embeddings. The encoder patchifies time in pairs
(tubelet 2), so an odd clip would silently drop its last frame from the scene tokens; such
clips get the last frame repeated and the duplicate's outputs discarded.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OPEN_D4RT_DIR = os.path.join(REPO_DIR, "reference", "Open-d4rt")
DEFAULT_CKPT = os.path.join(OPEN_D4RT_DIR, "checkpoints", "OpenD4RT_48CLIP_9Mix_NoCropAUG",
                            "opend4rt.ckpt")
POSE_GRID = 32          # 32x32 = 1024 points per frame for the Umeyama fit
QUERY_CHUNK = 8192      # queries per decoder call


def _import_open_d4rt():
    if OPEN_D4RT_DIR not in sys.path:
        sys.path.insert(0, OPEN_D4RT_DIR)
    from src.core import load_checkpoint, load_yaml_config   # noqa: E402  (OpenD4RT's `src`)
    from src.eval.tasks import _umeyama_rigid
    from src.model import build_model
    return load_checkpoint, load_yaml_config, build_model, _umeyama_rigid


def _state_dict(payload):
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "module", "network", "net"):
            if isinstance(payload.get(key), dict):
                return payload[key]
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return payload
    raise RuntimeError("no model weights found in checkpoint")


def load_d4rt_for_eval(ckpt: str = DEFAULT_CKPT, device: str = "cuda", model_cfg: str | None = None):
    """Build OpenD4RT from the model.yaml next to the ckpt and load its weights (strict)."""
    load_checkpoint, load_yaml_config, build_model, _ = _import_open_d4rt()
    cfg = load_yaml_config(model_cfg or os.path.join(os.path.dirname(ckpt), "model.yaml"))
    model = build_model(cfg["model"])
    sd = _state_dict(load_checkpoint(ckpt, map_location="cpu"))
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    if res.missing_keys:
        raise RuntimeError(f"{len(res.missing_keys)} missing keys, e.g. {res.missing_keys[:5]}")
    if res.unexpected_keys:
        print(f"[d4rt_infer] ignoring {len(res.unexpected_keys)} unexpected keys")
    model.image_hw = tuple(int(x) for x in cfg.get_path("model.input.image_size", [256, 256]))
    model.clip_frames = int(model.query_embedder.max_frames)
    return model.to(device).eval()


def load_video(image_paths, image_hw):
    """-> video [1,T,3,H,W] in [0,1], native W/H aspect ratio."""
    h, w = image_hw
    frames, aspect = [], None
    for p in image_paths:
        img = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if aspect is None:
            aspect = img.shape[1] / img.shape[0]
        interp = cv2.INTER_AREA if img.shape[0] >= h else cv2.INTER_LINEAR
        frames.append(cv2.resize(img, (w, h), interpolation=interp))
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div(255.0)
    return video.unsqueeze(0), float(aspect)


def _grid_uv(n_rows: int, n_cols: int, image_hw):
    """Pixel-centre grid in OpenD4RT's convention u = x / (W-1), v = y / (H-1)."""
    h, w = image_hw
    ys = np.linspace(0, h - 1, n_rows) / max(h - 1, 1)
    xs = np.linspace(0, w - 1, n_cols) / max(w - 1, 1)
    uu, vv = np.meshgrid(xs, ys, indexing="xy")
    return uu.reshape(-1).astype(np.float32), vv.reshape(-1).astype(np.float32)


@torch.no_grad()
def _decode_xyz(model, video, memory, u, v, t_src, t_tgt, t_cam):
    out = []
    dev = video.device
    for s in range(0, len(u), QUERY_CHUNK):
        sl = slice(s, s + QUERY_CHUNK)
        q = {"u": torch.from_numpy(u[sl]).to(dev).view(1, -1),
             "v": torch.from_numpy(v[sl]).to(dev).view(1, -1),
             "t_src": torch.from_numpy(t_src[sl]).to(dev).view(1, -1),
             "t_tgt": torch.from_numpy(t_tgt[sl]).to(dev).view(1, -1),
             "t_cam": torch.from_numpy(t_cam[sl]).to(dev).view(1, -1)}
        out.append(model.decode_queries(video=video, query=q, memory=memory)["xyz_3d"][0].float().cpu())
    return torch.cat(out).numpy().astype(np.float64)


def umeyama_extrinsics(p_local: np.ndarray, p_ref: np.ndarray) -> np.ndarray:
    """p_local, p_ref: [T, K, 3] -- the same points in camera t and in camera 0.
    Returns world-to-cam [T, 3, 4] with world = camera 0 (paper Sec. 2.3)."""
    _, _, _, umeyama_rigid = _import_open_d4rt()
    out = []
    for t in range(p_local.shape[0]):
        ok = np.isfinite(p_local[t]).all(1) & np.isfinite(p_ref[t]).all(1)
        fit = umeyama_rigid(p_local[t][ok], p_ref[t][ok])
        if fit is None:
            raise RuntimeError(f"Umeyama failed on frame {t}")
        R, tr = fit                       # p_ref = R p_local + tr  ->  cam_t -> cam_0 (c2w)
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = R, tr
        out.append(np.linalg.inv(c2w)[:3])
    return np.stack(out)


@torch.no_grad()
def infer_sequence_d4rt(model, image_paths, device="cuda", want_depth=True, **_ignored):
    """One OpenD4RT pass over image_paths (<= clip_frames). Returns the same keys as
    pipeline.eval.vggt_infer.infer_sequence: ``extrinsic`` [T,3,4] w2c and ``depth``
    [T,H,W] at model resolution (full field of view, squashed to image_hw)."""
    T = len(image_paths)
    if T > model.clip_frames:
        raise ValueError(f"{T} frames > OpenD4RT clip_frames={model.clip_frames}; chunk the sequence")
    paths = list(image_paths) + ([image_paths[-1]] if T % 2 else [])   # tubelet 2: keep T even
    video, aspect = load_video(paths, model.image_hw)
    video = video.to(device)
    aspect_t = torch.tensor([[aspect]], dtype=torch.float32, device=device)
    memory = model.encode_video(video=video, aspect_ratio=aspect_t)

    # pose: coarse grid, each frame's points in its own camera and in camera 0
    gu, gv = _grid_uv(POSE_GRID, POSE_GRID, model.image_hw)
    K = len(gu)
    t_idx = np.repeat(np.arange(T, dtype=np.int64), K)
    u, v = np.tile(gu, T), np.tile(gv, T)
    p_local = _decode_xyz(model, video, memory, u, v, t_idx, t_idx, t_idx).reshape(T, K, 3)
    p_ref = _decode_xyz(model, video, memory, u, v, t_idx, t_idx,
                        np.zeros_like(t_idx)).reshape(T, K, 3)
    out = {"extrinsic": umeyama_extrinsics(p_local, p_ref)}

    if want_depth:
        h, w = model.image_hw
        du, dv = _grid_uv(h, w, model.image_hw)
        depth = []
        for t in range(T):
            ti = np.full(len(du), t, dtype=np.int64)
            depth.append(_decode_xyz(model, video, memory, du, dv, ti, ti, ti)[:, 2].reshape(h, w))
        out["depth"] = np.stack(depth).astype(np.float32)
    return out
