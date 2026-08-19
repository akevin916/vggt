"""View synthesis by inverse warping, and the masked PSNR computed on it.

Used by ``benchmark/eval_lesion.py`` to score a stereo pair: synthesise the right view
from the left image and compare against the real right image.

Convention notes, because getting these wrong is silent:

* VGGT's ``extrinsic`` is [3,4] **world-to-camera** and frame 0 is the identity, so a
  relative pose is ``E_src @ inv(E_dst)`` in homogeneous form -- it takes a point in the
  destination camera's frame into the source camera's frame.
* The warp is an **inverse** warp: for every pixel of the destination view we unproject
  with the destination's own depth, push it into the source camera, project, and sample
  the source image. This is the SfMLearner/AF-SfMLearner convention and it is hole-free
  by construction, unlike forward splatting.
* Depth and pose come out of the *same* forward pass, so they share one arbitrary scale
  and the warp is self-consistent. No ground-truth calibration or scale alignment is
  needed -- which is what makes this metric usable on data that ships no calibration at
  all. It measures self-consistency, not metric accuracy; say so when reporting it.

Everything here is pure geometry on tensors; no model, no I/O.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def to_homogeneous_extrinsic(E: np.ndarray) -> np.ndarray:
    """[3,4] world-to-camera -> [4,4]."""
    M = np.eye(4, dtype=np.float64)
    M[:3, :4] = E.astype(np.float64)
    return M


def relative_pose(E_src: np.ndarray, E_dst: np.ndarray) -> np.ndarray:
    """[4,4] taking a point from the destination camera frame into the source camera frame."""
    return to_homogeneous_extrinsic(E_src) @ np.linalg.inv(to_homogeneous_extrinsic(E_dst))


def unproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """[H,W] depth + [3,3] intrinsic -> [H,W,3] points in that camera's own frame."""
    H, W = depth.shape
    ys, xs = np.meshgrid(np.arange(H, dtype=np.float64),
                         np.arange(W, dtype=np.float64), indexing="ij")
    ones = np.ones_like(xs)
    pix = np.stack([xs + 0.5, ys + 0.5, ones], axis=-1)          # pixel centres
    rays = pix @ np.linalg.inv(K.astype(np.float64)).T
    return (rays * depth[..., None]).astype(np.float32)


def unproject_to_world(depth: np.ndarray, K: np.ndarray, E: np.ndarray) -> np.ndarray:
    """[H,W,3] points in world coordinates -- the input to a point-cloud/PLY export."""
    cam = unproject(depth, K).reshape(-1, 3).astype(np.float64)
    M = np.linalg.inv(to_homogeneous_extrinsic(E))               # camera-to-world
    world = cam @ M[:3, :3].T + M[:3, 3]
    return world.reshape(depth.shape + (3,)).astype(np.float32)


def warp_to_view(
    src_img: np.ndarray,
    dst_depth: np.ndarray,
    K_src: np.ndarray,
    K_dst: np.ndarray,
    T_dst2src: np.ndarray,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthesise the destination view by sampling the source image.

    Args:
        src_img:   [H,W,3] float in [0,1] -- the image being sampled from.
        dst_depth: [H,W] float -- depth of the view being synthesised.
        K_src/K_dst: [3,3] intrinsics at the same resolution as the images.
        T_dst2src: [4,4] from ``relative_pose(E_src, E_dst)``.

    Returns:
        (warped [H,W,3], valid [H,W] bool). ``valid`` is False where the reprojection
        lands outside the source image or behind its camera; those pixels must be
        excluded from PSNR or they score whatever ``grid_sample`` padded with.
    """
    H, W = dst_depth.shape
    pts_dst = unproject(dst_depth, K_dst).reshape(-1, 3).astype(np.float64)
    pts_src = pts_dst @ T_dst2src[:3, :3].T + T_dst2src[:3, 3]

    z = pts_src[:, 2]
    in_front = z > 1e-8
    proj = pts_src @ K_src.astype(np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        u = proj[:, 0] / proj[:, 2]
        v = proj[:, 1] / proj[:, 2]

    # Pixel centres live at 0.5 .. W-0.5, so that -- not [0, W-1] -- is the range bilinear
    # sampling can serve without reaching past the border. The slack matters too: the
    # unproject/reproject round-trip lands on 0.49999978 rather than 0.5, and a literal
    # comparison then drops the whole border ring (~2.4% of an identity warp), quietly
    # shrinking the PSNR denominator on every pair.
    eps = 1e-3
    in_bounds = ((u >= 0.5 - eps) & (u <= W - 0.5 + eps)
                 & (v >= 0.5 - eps) & (v <= H - 0.5 + eps))
    valid = (in_front & in_bounds & np.isfinite(u) & np.isfinite(v)).reshape(H, W)

    # grid_sample wants normalised coords in [-1,1] with align_corners=False semantics.
    gx = np.nan_to_num(u.reshape(H, W), nan=-2.0) / W * 2.0 - 1.0
    gy = np.nan_to_num(v.reshape(H, W), nan=-2.0) / H * 2.0 - 1.0
    grid = torch.from_numpy(np.stack([gx, gy], axis=-1)).float()[None].to(device)

    src = torch.from_numpy(src_img).float().permute(2, 0, 1)[None].to(device)
    out = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros",
                        align_corners=False)
    warped = out[0].permute(1, 2, 0).cpu().numpy()
    warped[~valid] = 0.0
    return warped, valid


def masked_psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                data_range: float = 1.0) -> float:
    """PSNR over ``mask`` only. Returns NaN when the mask is empty."""
    m = mask.astype(bool)
    if m.sum() == 0:
        return float("nan")
    diff = (pred[m] - gt[m]).astype(np.float64)
    mse = float((diff ** 2).mean())
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def specular_mask(img: np.ndarray, thresh: float = 250.0 / 255.0) -> np.ndarray:
    """Saturated (view-dependent) highlights. An L->R warp cannot reproduce these, so
    they inflate the residual; excluding them is defensible but must be reported."""
    return img.max(axis=-1) >= thresh


def score_pair(
    img_src: np.ndarray,
    img_dst: np.ndarray,
    depth_dst: np.ndarray,
    K_src: np.ndarray,
    K_dst: np.ndarray,
    E_src: np.ndarray,
    E_dst: np.ndarray,
    extra_mask: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> dict:
    """Warp ``img_src`` into the destination view and score it both ways: over every
    valid pixel, and with saturated highlights removed. Reporting only the second
    invites the charge of picking the favourable pixels, so both go in the json."""
    T = relative_pose(E_src, E_dst)
    warped, valid = warp_to_view(img_src, depth_dst, K_src, K_dst, T, device=device)

    mask = valid if extra_mask is None else (valid & extra_mask.astype(bool))
    spec = specular_mask(img_dst) | specular_mask(img_src)

    return dict(
        warped=warped,
        valid=valid,
        psnr=masked_psnr(warped, img_dst, mask),
        psnr_no_specular=masked_psnr(warped, img_dst, mask & ~spec),
        valid_frac=float(valid.mean()),
        specular_frac=float(spec.mean()),
    )
