"""Resolve default eval / benchmark output directories from checkpoint paths."""

from __future__ import annotations

import os

TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(TRAINING_DIR)

# Subdirectory names under logs/<exp>/ (or logs/train/<ckpt_stem>/ for extracted weights).
EVAL_SINTEL = "eval_sintel"
VIS_GATE = "vis_gate"
GATE_BIAS_ABLATION = "gate_bias_ablation"
GATE_BIAS_ABLATION_PO = "gate_bias_ablation_po"


def _join_path_parts(parts: list[str]) -> str:
    """Join split path parts, preserving absolute paths on POSIX."""
    if parts and parts[0] == "":
        return os.path.join(os.sep, *parts[1:])
    return os.path.join(*parts)


def exp_root_from_ckpt(ckpt: str, training_dir: str | None = None) -> str:
    """Training run root: parent of ``ckpts/``, or ``logs/train/<ckpt_stem>`` for extracted weights."""
    training_dir = training_dir or TRAINING_DIR
    ckpt_abs = os.path.abspath(ckpt)
    parts = ckpt_abs.replace("\\", "/").split("/")

    if "ckpts" in parts:
        idx = parts.index("ckpts")
        if idx >= 1:
            return _join_path_parts(parts[:idx])

    stem = os.path.splitext(os.path.basename(ckpt))[0]
    return os.path.join(training_dir, "logs", "train", stem)


def default_eval_dir(ckpt: str, eval_name: str, training_dir: str | None = None) -> str:
    """Default output directory for an eval script, e.g. ``logs/<exp>/vis_gate``."""
    return os.path.abspath(os.path.join(exp_root_from_ckpt(ckpt, training_dir), eval_name))
