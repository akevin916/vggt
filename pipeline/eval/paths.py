"""Resolve default output directories for eval / benchmark / diag scripts.

``logs/`` holds training artifacts only (ckpts, tensorboard, log.txt, and the trainer's own
pose_eval). Everything produced *after* training goes to ``outputs/``, split by KIND first:

    outputs/metrics/<tool>/<key>/   json / csv / txt numbers -- small, keep, back up
    outputs/vis/<tool>/<key>/       png / gif / mp4 / ply -- for a human to look at
    outputs/cache/<tool>/<key>/     npz / npy and their sidecar meta.json -- large, regenerable
    outputs/runlogs/<tool>/<key>/   stdout, batch logs, _error_log.txt

``<tool>`` is the producing script's file name, one tool per script. ``<key>`` is
``<exp>`` or ``<exp>__<run>``: ``<exp>`` comes from the checkpoint (``A_vs_B`` when comparing),
``<run>`` names the settings, ``<split>_<protocol>[_<variant>]``. The same key appears under
every kind, so a figure and the numbers behind it always sit at mirrored paths.

An explicit ``--out_dir`` collapses all four kinds into that one directory (``override=``),
which keeps old CLI invocations working.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(PIPELINE_DIR)
LOGS_DIR = os.path.join(REPO_DIR, "logs")
OUTPUTS_DIR = os.path.join(REPO_DIR, "outputs")

KINDS = ("metrics", "vis", "cache", "runlogs")
RUN_SEP = "__"

def _join_path_parts(parts: list[str]) -> str:
    """Join split path parts, preserving absolute paths on POSIX."""
    if parts and parts[0] == "":
        return os.path.join(os.sep, *parts[1:])
    return os.path.join(*parts)


def exp_root_from_ckpt(ckpt: str, logs_dir: str | None = None) -> str:
    """Training run root: parent of ``ckpts/``, or ``logs/train/<ckpt_stem>`` for extracted weights."""
    logs_dir = logs_dir or LOGS_DIR
    ckpt_abs = os.path.abspath(ckpt)
    parts = ckpt_abs.replace("\\", "/").split("/")

    if "ckpts" in parts:
        idx = parts.index("ckpts")
        if idx >= 1:
            return _join_path_parts(parts[:idx])

    stem = os.path.splitext(os.path.basename(ckpt))[0]
    return os.path.join(logs_dir, "train", stem)


def exp_name_from_ckpt(ckpt: str, logs_dir: str | None = None) -> str:
    """Experiment name used as the ``<exp>`` part of an output key."""
    return os.path.basename(exp_root_from_ckpt(ckpt, logs_dir))


def output_key(exp: str, run: str | None = None) -> str:
    """``<exp>`` or ``<exp>__<run>``."""
    return f"{exp}{RUN_SEP}{run}" if run else exp


@dataclass(frozen=True)
class OutputDirs:
    """One output key resolved under every kind. Nothing is created until a ``*_file`` call."""

    metrics: str
    vis: str
    cache: str
    runlogs: str

    def _file(self, root: str, name: str) -> str:
        path = os.path.join(root, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def metrics_file(self, name: str) -> str:
        return self._file(self.metrics, name)

    def vis_file(self, name: str) -> str:
        return self._file(self.vis, name)

    def cache_file(self, name: str) -> str:
        return self._file(self.cache, name)

    def runlog_file(self, name: str) -> str:
        return self._file(self.runlogs, name)

    def sub(self, name: str) -> "OutputDirs":
        """Same layout one level deeper, e.g. one sub-dir per sequence."""
        return OutputDirs(*(os.path.join(getattr(self, k), name) for k in KINDS))


def output_dirs(tool: str, exp: str, run: str | None = None,
                override: str | None = None) -> OutputDirs:
    """``outputs/<kind>/<tool>/<exp>[__<run>]/`` for every kind, or ``override`` for all of them."""
    if override:
        d = os.path.abspath(override)
        return OutputDirs(d, d, d, d)
    key = output_key(exp, run)
    return OutputDirs(*(os.path.abspath(os.path.join(OUTPUTS_DIR, k, tool, key)) for k in KINDS))


def output_dirs_from_ckpt(ckpt: str, tool: str, run: str | None = None,
                          override: str | None = None, logs_dir: str | None = None) -> OutputDirs:
    """``output_dirs`` keyed by the experiment the checkpoint belongs to."""
    return output_dirs(tool, exp_name_from_ckpt(ckpt, logs_dir), run, override)
