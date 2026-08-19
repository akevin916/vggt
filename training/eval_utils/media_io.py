"""Write an image sequence out as a video.

Lives here rather than inside any one tool because both the preprocessors and the
benchmarks need it: a point cloud on its own is unreadable without the footage it came
from, so every output directory that holds a ``.ply`` should hold the matching clip too.
"""

from __future__ import annotations

import os
from typing import Iterable

import cv2
import numpy as np


def write_video(path: str, frames: Iterable[np.ndarray], fps: float = 10.0) -> int:
    """Encode RGB uint8 [H,W,3] frames to an mp4. Returns the frame count.

    Frames must all share a shape. ``fps`` is a *playback* rate, unrelated to the source
    capture rate -- de-duplicated endoscopic sequences look best slowed right down.
    """
    frames = list(frames)
    if not frames:
        return 0
    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {path}")
    try:
        for f in frames:
            a = np.asarray(f)
            if a.dtype != np.uint8:
                a = np.clip(a * 255 if a.max() <= 1.0 else a, 0, 255).astype(np.uint8)
            writer.write(a[:, :, ::-1])          # cv2 wants BGR
    finally:
        writer.release()
    return len(frames)
