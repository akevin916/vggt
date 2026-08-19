"""Shared gate-visualization primitive.

Trimmed on 2026-08-19: the PO/Sintel panel builders and their driver functions
(``save_po_panel`` / ``save_sintel_panel`` / ``run_po_vis`` / ``run_sintel_vis`` /
``print_diag_summary``) went out with ``diag/vis/gate.py``. ``diag/vis/gate_gif.py``
composes its own panel and needs only the colormap step, which is what remains.
"""

from __future__ import annotations

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
