"""Resolve default output directories for eval / benchmark / diag scripts.

Single rule: ``logs/`` holds training artifacts only (ckpts, tensorboard, log.txt, and
the trainer's own pose_eval). Everything produced *after* training -- benchmark numbers,
ablation json, diagnostics, plots -- goes to ``outputs/<exp>/<tool>/``.

``<exp>`` is derived from the checkpoint, so a plot and the json next to it share one key
and re-running with a different ckpt can no longer silently overwrite the previous one.
"""

from __future__ import annotations

import os

TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(TRAINING_DIR)
OUTPUTS_DIR = os.path.join(REPO_DIR, "outputs")

# Tool names -> the <tool> level under outputs/<exp>/.
EVAL_SINTEL = "eval_sintel"
GATE_BIAS_ABLATION = "gate_bias_ablation"
GATE_BIAS_ABLATION_PO = "gate_bias_ablation_po"
GATE_QUALITY = "gate_quality"
PNP_POSE = "pnp_pose"
VIS_GATE = "vis_gate"
ERROR_GROWTH = "error_growth"
TRAJECTORY = "trajectory"
GATE_TEMPORAL = "gate_temporal"
GATE_GIF = "gate_gif"
TRAIN_CURVES = "train_curves"


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


def exp_name_from_ckpt(ckpt: str, training_dir: str | None = None) -> str:
    """Experiment name used as the ``<exp>`` level under outputs/."""
    return os.path.basename(exp_root_from_ckpt(ckpt, training_dir))


def default_output_dir(ckpt: str, tool_name: str, training_dir: str | None = None) -> str:
    """Default output directory for any post-training script: ``outputs/<exp>/<tool>/``."""
    return os.path.abspath(os.path.join(OUTPUTS_DIR, exp_name_from_ckpt(ckpt, training_dir), tool_name))


def output_dir_for_exp(exp_name: str, tool_name: str) -> str:
    """Same as ``default_output_dir`` for callers keyed by an experiment name, not a ckpt."""
    return os.path.abspath(os.path.join(OUTPUTS_DIR, exp_name, tool_name))
