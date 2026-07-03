"""Resolve default eval output directories from checkpoint paths."""

from __future__ import annotations

import os


def _join_path_parts(parts: list[str]) -> str:
    """Join split path parts, preserving absolute paths on POSIX."""
    if parts and parts[0] == "":
        return os.path.join(os.sep, *parts[1:])
    return os.path.join(*parts)


def exp_root_from_ckpt(ckpt: str, training_dir: str | None = None) -> str:
    """Training run root: parent of ``ckpts/``, or ``logs/train/<ckpt_stem>`` for extracted weights."""
    training_dir = training_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_abs = os.path.abspath(ckpt)
    parts = ckpt_abs.replace("\\", "/").split("/")

    if "ckpts" in parts:
        idx = parts.index("ckpts")
        if idx >= 1:
            return _join_path_parts(parts[:idx])

    stem = os.path.splitext(os.path.basename(ckpt))[0]
    return os.path.join(training_dir, "logs", "train", stem)


def default_eval_dir(ckpt: str, eval_name: str, training_dir: str | None = None) -> str:
    """Default output directory for an eval script, e.g. ``logs/<exp>/eval_gate``."""
    return os.path.abspath(os.path.join(exp_root_from_ckpt(ckpt, training_dir), eval_name))
