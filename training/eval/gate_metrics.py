"""Lightweight patch-level gate diagnostics."""

from __future__ import annotations

from typing import Dict

import numpy as np


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
        "accuracy_at_0_5": float((probs >= 0.5) == labels).mean(),
    }
