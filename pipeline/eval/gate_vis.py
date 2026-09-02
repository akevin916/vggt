"""Shared gate-visualization primitive.

Trimmed on 2026-08-19: the PO/Sintel panel builders and their driver functions
(``save_po_panel`` / ``save_sintel_panel`` / ``run_po_vis`` / ``run_sintel_vis`` /
``print_diag_summary``) went out with ``diag/vis/gate.py``. ``diag/vis/gate_gif.py``
composes its own panel and needs only the colormap step, which is what remains.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def colorize_map(x01: np.ndarray, h: int, w: int, mode: str = "nearest") -> np.ndarray:
    """Upsample a patch-resolution map in [0,1] to (h,w) and colorize it.

    The clip is ABSOLUTE -- 0 and 1 always map to the same colors regardless of the input's
    own range -- so panels stay comparable across frames, sequences and checkpoints. Anything
    wanting a stretched/relative scale must rescale before calling.
    """
    t = torch.from_numpy(x01)[None, None].float()
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    up = F.interpolate(t, size=(h, w), mode=mode, **kwargs)[0, 0].numpy()
    return cv2.applyColorMap((np.clip(up, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)


def _to_bgr_u8(img_chw: np.ndarray) -> np.ndarray:
    """[3,H,W] float in [0,1] (RGB, as load_and_preprocess_images returns) -> [H,W,3] uint8 BGR."""
    rgb = np.clip(np.transpose(img_chw, (1, 2, 0)), 0.0, 1.0)
    return cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _patch_grid_contour(label_hw: np.ndarray, h: int, w: int) -> np.ndarray:
    """Binary patch-grid label -> pixel-resolution contour mask, drawn at the patch blocks'
    own edges (nearest upsample, so the outline is honest about patch quantisation)."""
    up = cv2.resize(label_hw.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    eroded = cv2.erode(up, np.ones((3, 3), np.uint8), iterations=1)
    return (up - eroded).astype(bool)


def _fit_text(text: str, width: int, base: float, thick: int) -> float:
    """Largest scale <= base at which `text` still fits in `width`. Sintel frames are only
    518 px wide, so a fixed scale silently truncates the scores the banner exists to show."""
    scale = base
    while scale > 0.25:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        if tw <= width:
            break
        scale -= 0.05
    return scale


def _header_band(width: int, header: str, subheader: str) -> np.ndarray:
    """Score banner + an absolute 0..1 colour key, repeated on every GIF frame so the numbers
    are readable from any point in the animation (a GIF has no caption to scroll back to)."""
    pad, key_w = 10, 120
    band = np.full((96, width, 3), 24, np.uint8)

    s1 = _fit_text(header, width - 2 * pad, 0.85, 2)
    cv2.putText(band, header, (pad, 30), cv2.FONT_HERSHEY_SIMPLEX, s1, (255, 255, 255), 2, cv2.LINE_AA)
    s2 = _fit_text(subheader, width - 2 * pad, 0.45, 1)
    cv2.putText(band, subheader, (pad, 54), cv2.FONT_HERSHEY_SIMPLEX, s2, (170, 170, 170), 1, cv2.LINE_AA)

    # Legend row: colour key right-aligned, GT key on the left, on their own line so neither
    # can overrun the other however long the header above them is.
    key_x, key_y = width - pad - key_w, 68
    ramp = np.linspace(0, 1, key_w, dtype=np.float32)[None].repeat(14, 0)
    band[key_y:key_y + 14, key_x:key_x + key_w] = cv2.applyColorMap(
        (ramp * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.putText(band, "0", (key_x - 12, key_y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(band, "1", (key_x + key_w + 3, key_y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(band, "sigma(g)", (key_x - 78, key_y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(band, "green = GT dyn", (pad, key_y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (120, 255, 120), 1, cv2.LINE_AA)
    return band


def save_gate_overlay_gif(
    path: str,
    images: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    header: str,
    subheader: str = "",
    alpha: float = 0.45,
    duration_ms: int = 200,
) -> str:
    """One animated GIF per sequence: sigma(g) heat-overlaid on the RGB, GT outlined in green.

    The heatmap uses colorize_map's ABSOLUTE 0..1 scale, so frames, sequences and checkpoints
    stay directly comparable (a brighter red always means a more confident gate, never just
    "the max of this frame").

    Both `probs` and `labels` are on the PATCH grid -- exactly the arrays AUC/F1 scored, not a
    re-derived pixel mask. What you see is what was measured.

    Args:
        images: [S,3,H,W] float in [0,1], RGB.
        probs:  [S,ph,pw] sigma(g) in [0,1].
        labels: [S,ph,pw] binary GT patch label.
        header: banner line (sequence + scores), repeated on every frame.
        subheader: small provenance line (ckpt / thresholds).
        duration_ms: ms per frame.
    """
    from PIL import Image

    s, _, h, w = images.shape
    band = _header_band(w, header, subheader)
    frames = []
    for i in range(s):
        bgr = _to_bgr_u8(images[i])
        heat = colorize_map(probs[i].astype(np.float32), h, w)
        cell = cv2.addWeighted(bgr, 1.0 - alpha, heat, alpha, 0.0)
        cell[_patch_grid_contour(labels[i], h, w)] = (0, 255, 0)   # GT outline
        for colour, thick in ((0, 0, 0), 4), ((255, 255, 255), 1):
            cv2.putText(cell, f"f{i:02d}/{s - 1:02d}", (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, thick, cv2.LINE_AA)
        rgb = cv2.cvtColor(np.vstack([band, cell]), cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)
    return path
