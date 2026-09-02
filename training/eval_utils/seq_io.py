"""Write the ``seq.npz`` a reconstruction needs to be re-rendered later.

``cloud.npz`` is already a rendering decision: the points are subsampled and the colours
baked, so the polished renderer in diag/vis/cloud_polish.py cannot rebuild anything from
it. That renderer wants the per-frame depth / intrinsic / extrinsic / RGB instead, which
is what this file stores -- one npz beside the cloud, same keys everywhere, so timeline.py
--blend can read a VGGT dump and a MonST3R dump without knowing which produced it.

benchmark/eval_{lesion,gastric}.py write this format from a VGGT prediction dict; the
MonST3R runners and diag/dump_recon.py write it from a list of per-frame arrays, which is
what ``dump_seq_npz`` takes.
"""

from __future__ import annotations

import numpy as np


def _stack(seq):
    """List of (torch tensor | ndarray) -> one contiguous float array."""
    out = []
    for x in seq:
        out.append(x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x))
    return np.stack(out)


def dump_seq_npz(path, depth, intrinsic, extrinsic, images, **extra):
    """``depth`` (N,H,W), ``intrinsic`` (N,3,3), ``extrinsic`` (N,3,4) world-to-cam,
    ``images`` (N,H,W,3) float in [0,1]. Lists of per-frame arrays or torch tensors are
    accepted for the first three; everything is stacked here.

    depth goes to float16 and images to uint8 -- these files sit beside every arm of every
    sequence and the full-precision copies were the bulk of the disk cost for no visible
    difference in the render.
    """
    payload = dict(
        depth=_stack(depth).astype(np.float16),
        intrinsic=_stack(intrinsic).astype(np.float32),
        extrinsic=_stack(extrinsic).astype(np.float32),
        images=(np.clip(np.asarray(images), 0, 1) * 255).astype(np.uint8),
    )
    payload.update(extra)
    np.savez_compressed(path, **payload)
    return path
