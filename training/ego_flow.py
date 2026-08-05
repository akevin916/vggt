"""Differentiable camera-induced optical flow (ego-flow), disparity form.

Shared geometry for the L_ego_flow training loss (`loss.compute_ego_flow_loss`). The
formulation is MonST3R's `warp_by_disp` (`reference/monst3r/dust3r/utils/goem_opt.py:196`,
`use_depth=False` branch); what differs downstream is the *target* — see
`docs/monst3r_loss_diff.md`.

Why a module rather than an import from `diag/`: `loss.py` is training-layer code and must
not depend on the diagnostic scripts (CLAUDE.md layering). `diag/flow_loss_probe.py` and
`diag/flow_pose_headroom.py` keep their own private copies of this geometry ON PURPOSE —
the numbers they produced are published in docs/table.md and must stay reproducible from
the code that produced them. If this file is ever changed, cross-check against the probe
(verification step 2 of the port plan) instead of editing the probe to match.
"""

from functools import lru_cache
from typing import Tuple

import torch


@lru_cache(maxsize=8)
def _pixel_grid_cached(H: int, W: int, device_str: str, dtype: torch.dtype) -> torch.Tensor:
    device = torch.device(device_str)
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xx, yy, torch.ones_like(xx)], 0).reshape(1, 3, H * W)


def pixel_grid(H: int, W: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Homogeneous pixel grid (1, 3, H*W), cached per (H, W, device, dtype).

    Rebuilt every call it would be ~0.5 MB of meshgrid per frame pair per step; the grid is
    a constant, so it is cached. The returned tensor is shared — do NOT write into it.
    """
    return _pixel_grid_cached(H, W, str(device), dtype)


def relative_w2c(extri_src: torch.Tensor, extri_tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Relative transform between two world-to-cam extrinsics (VGGT/OpenCV convention).

    X_cam2 = R2 (R1^T (X_cam1 - t1)) + t2  =>  R_rel = R2 R1^T,  t_rel = t2 - R_rel t1.

    MonST3R's `get_relative_transform` takes cam-to-world instead; converting here rather
    than inverting the poses keeps the numerics in one place.

    Args:
        extri_src, extri_tgt: (P, 3, 4) world-to-cam.
    Returns:
        R_rel (P, 3, 3), t_rel (P, 3, 1)
    """
    R1, t1 = extri_src[:, :3, :3], extri_src[:, :3, 3:]
    R2, t2 = extri_tgt[:, :3, :3], extri_tgt[:, :3, 3:]
    R_rel = R2.matmul(R1.transpose(-1, -2))
    t_rel = t2 - R_rel.matmul(t1)
    return R_rel, t_rel


def ego_flow_from_disp(
    R_rel: torch.Tensor,
    t_rel: torch.Tensor,
    disp: torch.Tensor,
    K_tgt: torch.Tensor,
    K_src_inv: torch.Tensor,
    coord: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Camera-induced flow, disparity form (MonST3R goem_opt.py:196, use_depth=False).

        tgt = (K2 R_rel K1^-1) u  +  disp * (K2 t_rel)
        tgt = tgt / tgt[2]
        flow = tgt - u

    Rotation contributes independently of depth; translation scales with disparity. Written
    in disparity rather than depth so z -> inf degenerates to a pure rotational homography
    instead of dividing by a large z.

    SCALE: the translation term is `disp * t`, which is invariant to a global rescaling of
    the scene (scale by s -> disp becomes disp/s, t becomes s*t). So a flow computed
    entirely from predicted quantities and one computed entirely from GT quantities are
    directly comparable in pixels with NO scale alignment — as long as neither is mixed.
    Mixing (GT depth with predicted translation, say) is what needs a scale factor; see
    `diag/flow_loss_probe.py`'s mixed variants.

    Args:
        R_rel: (P, 3, 3) cam_src -> cam_tgt rotation
        t_rel: (P, 3, 1) cam_src -> cam_tgt translation
        disp:  (P, 1, H, W) inverse depth of the SOURCE frame
        K_tgt: (P, 3, 3);  K_src_inv: (P, 3, 3)
        coord: (1, 3, H*W) homogeneous pixel grid, from `pixel_grid`
    Returns:
        flow: (P, 2, H, W) in pixels
        z:    (P, H, W) projective depth of the warped point, BEFORE normalisation. Pixels
              with z <= 0 land behind the target camera: the perspective divide flips their
              sign and the resulting "flow" is garbage with a reversed gradient, so callers
              must mask on z. (The probe had no need for this — it never backpropagates.)
    """
    P, _, H, W = disp.shape
    H_mat = K_tgt.matmul(R_rel.matmul(K_src_inv))            # (P,3,3)
    flat_disp = disp.reshape(P, 1, H * W)
    tgt_coord = torch.matmul(H_mat, coord) + flat_disp * torch.matmul(K_tgt, t_rel)
    z = tgt_coord[:, 2]                                      # (P, H*W)
    tgt_coord = tgt_coord / (z[:, None] + 1e-6)
    flow = (tgt_coord - coord).reshape(P, 3, H, W)[:, :2]
    return flow, z.reshape(P, H, W)
