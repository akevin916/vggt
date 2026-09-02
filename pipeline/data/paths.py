"""Resolve dataset roots relative to the repo, not to a mount point.

Counterpart to ``pipeline/eval/paths.py``: that one resolves where results are *written*,
this one resolves where data is *read* from.

The datasets live on a separate disk and are reached through the ``data`` symlink at the
repo root. Going through that symlink instead of hardcoding the mount point is what keeps
every entry point alive when the disk is remounted under a different name -- udisks names
an automount after the filesystem label and appends a digit on collision, so the mount
point is not a stable identifier. Only the symlink is.

Set ``VGGT_DATA_ROOT`` to override on a machine where the symlink is absent.
"""

from __future__ import annotations

import os

_DATA_PKG_DIR = os.path.dirname(os.path.abspath(__file__))  # pipeline/data (this code)
PIPELINE_DIR = os.path.dirname(_DATA_PKG_DIR)
REPO_DIR = os.path.dirname(PIPELINE_DIR)

DATA_ROOT = os.environ.get("VGGT_DATA_ROOT") or os.path.join(REPO_DIR, "data")


def data_path(*parts: str) -> str:
    """Absolute path to a dataset under the repo-root ``data`` symlink."""
    return os.path.join(DATA_ROOT, *parts)


# Datasets are bucketed by role: ``train/`` is the training mix, ``eval/`` holds the
# benchmarks. PointOdyssey stays under train/ even though diag scripts probe on it --
# the bucket records what the data IS, not who happens to read it.
PO_DIR = data_path("train", "point_odyssey")
TARTANAIR_DIR = data_path("train", "tartanair")
WAYMO_DIR = data_path("train", "waymo_processed")
SPRING_DIR = data_path("train", "spring")

# Sintel is flattened: <eval>/sintel/ holds final/ depth/ camdata_left/ directly (the
# upstream layout nests them under a training/ split dir, which reads as a contradiction
# inside eval/).
SINTEL_DIR = data_path("eval", "sintel")
