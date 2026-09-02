"""Minimal binary-PLY writer shared by the point-cloud tools (diag/vis + the export UI)."""

from __future__ import annotations

import numpy as np


def write_ply(path: str, points: np.ndarray, colors: np.ndarray) -> int:
    """Binary little-endian PLY with per-vertex RGB. ``colors`` in [0, 1]. Returns point count."""
    points = np.ascontiguousarray(points, dtype=np.float32)
    rgb = np.clip(np.asarray(colors) * 255.0, 0, 255).astype(np.uint8)
    n = points.shape[0]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    # Interleave xyz (3×float32=12B) + rgb (3×uint8=3B) into one 15-byte record per vertex.
    vertex = np.empty(n, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    vertex["xyz"] = points
    vertex["rgb"] = rgb

    with open(path, "wb") as f:
        f.write(header)
        f.write(vertex.tobytes())
    return n
