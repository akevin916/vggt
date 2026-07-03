"""Patch-level gate evaluation metrics."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def best_f1_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, 19):
        pred = probs >= t
        tp = float((pred & (labels == 1)).sum())
        fp = float((pred & (labels == 0)).sum())
        fn = float((~pred & (labels == 1)).sum())
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_f1, best_t


def classification_at_threshold(probs: np.ndarray, labels: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    pred = probs >= thr
    tp = float((pred & (labels == 1)).sum())
    fp = float((pred & (labels == 0)).sum())
    fn = float((~pred & (labels == 1)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    iou = tp / (tp + fp + fn + 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1, "iou": iou}


def summarize_gate(probs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    labels = labels.astype(np.int8)
    dyn = labels == 1
    stat = labels == 0
    cls = classification_at_threshold(probs, labels, thr=0.5)
    best_f1, best_thr = best_f1_threshold(probs, labels)

    out: Dict[str, Any] = {
        "n_patches": int(len(probs)),
        "dynamic_fraction": float(labels.mean()),
        "mean_prob_dynamic": float(probs[dyn].mean()) if dyn.any() else 0.0,
        "mean_prob_static": float(probs[stat].mean()) if stat.any() else 0.0,
        "f1_at_0_5": cls["f1"],
        "iou_at_0_5": cls["iou"],
        "precision_at_0_5": cls["precision"],
        "recall_at_0_5": cls["recall"],
        "best_f1": best_f1,
        "best_f1_threshold": best_thr,
    }

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if dyn.any() and stat.any():
            out["roc_auc"] = float(roc_auc_score(labels, probs))
            out["ap"] = float(average_precision_score(labels, probs))
        else:
            out["roc_auc"] = float("nan")
            out["ap"] = float("nan")
    except Exception as exc:
        out["sklearn_error"] = str(exc)

    return out
