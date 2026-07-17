"""Shared helpers for gate diagnostics and ablations."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


def po_common_conf(args) -> SimpleNamespace:
    return SimpleNamespace(
        img_size=args.img_size,
        patch_size=args.patch_size,
        augs=SimpleNamespace(scales=None),
        rescale=True,
        rescale_aug=False,
        landscape_check=False,
        debug=False,
        training=False,
        get_nearby=True,
        load_depth=True,
        inside_random=False,
        allow_duplicate_img=False,
    )


def pool_to_patch(mask_shw: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
    s, h, w = mask_shw.shape
    pooled = F.adaptive_avg_pool2d(mask_shw.reshape(s, 1, h, w).float(), (patch_h, patch_w))
    return pooled.reshape(s, patch_h, patch_w)


def oracle_logits_from_masks(
    masks: List[Optional[np.ndarray]], s: int, ph: int, pw: int, k: float
) -> torch.Tensor:
    """Build an eval-only ``gate_logits_override`` [1, s, ph*pw] from pixel masks.

    Area-resizes each mask to the patch grid, then maps [0,1] -> [-k, +k]. This is the
    numpy/cv2 eval-side counterpart of loss.oracle_gate_logits_from_mask (torch/pooling,
    training-side); keep them separate -- the two are not bit-identical.
    """
    import cv2

    patch_probs = np.zeros((s, ph, pw), dtype=np.float32)
    for i in range(min(s, len(masks))):
        if masks[i] is None:
            continue
        patch_probs[i] = cv2.resize(masks[i], (pw, ph), interpolation=cv2.INTER_AREA)
    logits = (torch.from_numpy(patch_probs) * 2.0 - 1.0) * k
    return logits.reshape(1, s, ph * pw)


def summarize_gate_diag(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Minimal gate diagnostic: class separation and threshold accuracy."""
    labels = labels.astype(np.int8)
    dyn = labels == 1
    stat = labels == 0
    if len(probs) == 0:
        return {
            "n_patches": 0,
            "dynamic_fraction": 0.0,
            "mean_prob_dynamic": float("nan"),
            "mean_prob_static": float("nan"),
            "separation_gap": float("nan"),
            "accuracy_at_0_5": float("nan"),
        }

    mean_dyn = float(probs[dyn].mean()) if dyn.any() else float("nan")
    mean_stat = float(probs[stat].mean()) if stat.any() else float("nan")
    gap = mean_dyn - mean_stat if np.isfinite(mean_dyn) and np.isfinite(mean_stat) else float("nan")

    return {
        "n_patches": int(len(probs)),
        "dynamic_fraction": float(labels.mean()),
        "mean_prob_dynamic": mean_dyn,
        "mean_prob_static": mean_stat,
        "separation_gap": gap,
        "accuracy_at_0_5": float(((probs >= 0.5) == labels).mean()),
    }
