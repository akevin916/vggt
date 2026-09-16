"""RAFT optical flow and flow-derived occlusion masks for the self-supervised losses.

Ported from ColonAdapter (reference/ColonAdapter/layers.py: `get_occu_mask_backward` +
`get_corresponding_map`, which ColonAdapter in turn takes from AF-SfMLearner). The one
substitution: ColonAdapter trains its own flow network (`PositionDecoder`) in a separate
stage 1; here torchvision's pretrained `raft_large` stands in for it, so there is no flow
training stage. It is the same RAFT the repo already uses for the dynamic-mask
preprocessing (pipeline/data/preprocess/tartanair_raft_dynmask.py).

WHY A FLOW-BASED OCCLUSION MASK AT ALL. The Monodepth2 auto-mask the self-supervised run used
("drop a pixel when the warp does not beat the un-warped frame") assumes neighbouring frames
differ a lot, which holds on KITTI and fails on endoscopy: consecutive SCARED frames are
nearly identical, so warped and un-warped errors sit in the same range and the test becomes
a coin flip -- scared_selfsup dropped 37-42% of pixels that way. Coverage of a backward flow
is a geometric test instead of an error test: a target pixel that no source pixel maps onto
is not visible in the source, whatever the photometric error says.

Conventions, because getting them wrong is silent:
  * flow is (dx, dy) in PIXELS, defined on the grid of the FIRST image, pointing to where that
    pixel lands in the SECOND image.
  * RAFT runs with autocast disabled (fp32) and under no_grad -- flow is a fixed target, never
    a thing the model is trained through.
  * Everything is returned at the training image resolution; RAFT itself runs at `proc`x`proc`
    and the flow is rescaled on the way back up.
"""

from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F

_RAFT: Dict[str, torch.nn.Module] = {}


def _raft(device: torch.device) -> torch.nn.Module:
    """Lazily-built, frozen raft_large, one per device (weights come from the torch hub cache)."""
    key = str(device)
    if key not in _RAFT:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _RAFT[key] = model
    return _RAFT[key]


@torch.no_grad()
def raft_flow(img1: torch.Tensor, img2: torch.Tensor, proc: int = 256, iters: int = 12) -> torch.Tensor:
    """Optical flow img1 -> img2.

    Args:
        img1, img2: (N, 3, H, W) in [0, 1].
        proc: square side RAFT runs at (must be divisible by 8).
    Returns:
        (N, 2, H, W) flow in pixels at the input resolution, on img1's grid.
    """
    if proc < 128 or proc % 8:
        # RAFT's correlation pyramid needs >= 16x16 feature maps at 1/8 resolution.
        raise ValueError(f"proc must be >= 128 and divisible by 8, got {proc}")
    N, _, H, W = img1.shape
    with torch.autocast(device_type=img1.device.type, enabled=False):
        a = F.interpolate(img1.float(), (proc, proc), mode="bilinear", align_corners=False) * 2 - 1
        b = F.interpolate(img2.float(), (proc, proc), mode="bilinear", align_corners=False) * 2 - 1
        flow = _raft(img1.device)(a, b, num_flow_updates=iters)[-1]
        flow = F.interpolate(flow, (H, W), mode="bilinear", align_corners=False)
        flow[:, 0] *= W / proc
        flow[:, 1] *= H / proc
    return flow


def coverage_map(flow: torch.Tensor) -> torch.Tensor:
    """Forward-splat every pixel of `flow`'s grid to where it lands and accumulate its bilinear
    weight on the destination grid. Port of ColonAdapter's `get_corresponding_map`.

    Args:
        flow: (N, 2, H, W), on the SOURCE grid, pointing into the destination.
    Returns:
        (N, H, W) coverage on the DESTINATION grid (~1 = seen once, ~0 = nothing lands here).
    """
    N, _, H, W = flow.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=flow.device, dtype=flow.dtype),
                            torch.arange(W, device=flow.device, dtype=flow.dtype), indexing="ij")
    x = (xs[None] + flow[:, 0]).reshape(N, -1)
    y = (ys[None] + flow[:, 1]).reshape(N, -1)
    x0, y0 = torch.floor(x), torch.floor(y)
    out = torch.zeros(N, H * W, device=flow.device, dtype=flow.dtype)
    for dx in (0.0, 1.0):
        for dy in (0.0, 1.0):
            xi, yi = x0 + dx, y0 + dy
            w = (1 - (x - xi).abs()) * (1 - (y - yi).abs())
            inside = (xi >= 0) & (xi <= W - 1) & (yi >= 0) & (yi <= H - 1)
            idx = (yi.clamp(0, H - 1) * W + xi.clamp(0, W - 1)).long()
            out.scatter_add_(1, idx, torch.where(inside, w, torch.zeros_like(w)))
    return out.view(N, H, W)


def pair_flows(batch: dict, offsets: Iterable[int], proc: int = 256,
               occ_thresh: float = 0.95) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Flow and visibility for every (target, target + offset) pair in the batch window.

    Cached on the batch dict under "_flow_cache", so the photometric loss and the flow-geometry
    loss share one set of RAFT evaluations per step. Every ordered frame pair is computed once:
    for offsets (-1, +1) the flow t->t+1 serves as both the forward flow of offset +1 and the
    backward flow of offset -1.

    Returns:
        {offset: (tgt, flow, visible)} with
          tgt      (n,)            target frame indices, the same ordering the losses iterate in
          flow     (B, n, 2, H, W) target -> source flow, on the target grid
          visible  (B, n, H, W)    bool; coverage of the source -> target flow > occ_thresh,
                                   i.e. the target pixel is actually seen in the source frame
    """
    offsets = [int(o) for o in offsets if int(o) != 0]
    cache = batch.setdefault("_flow_cache", {})
    key = (tuple(offsets), int(proc), float(occ_thresh))
    if key in cache:
        return cache[key]

    images = batch["images"]
    B, S, _, H, W = images.shape
    need = set()
    for off in offsets:
        for t in range(max(0, -off), min(S, S - off)):
            need.add((t, t + off))
            need.add((t + off, t))
    need = sorted(need)
    out: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    if not need:
        cache[key] = out
        return out

    i1 = torch.tensor([p[0] for p in need], device=images.device)
    i2 = torch.tensor([p[1] for p in need], device=images.device)
    P = len(need)
    flows = raft_flow(images[:, i1].reshape(B * P, 3, H, W),
                      images[:, i2].reshape(B * P, 3, H, W), proc=proc).view(B, P, 2, H, W)
    lut = {p: k for k, p in enumerate(need)}

    for off in offsets:
        tgt = list(range(max(0, -off), min(S, S - off)))
        if not tgt:
            continue
        n = len(tgt)
        fwd = torch.stack([flows[:, lut[(t, t + off)]] for t in tgt], dim=1)   # tgt grid -> src
        bwd = torch.stack([flows[:, lut[(t + off, t)]] for t in tgt], dim=1)   # src grid -> tgt
        visible = coverage_map(bwd.reshape(B * n, 2, H, W)).view(B, n, H, W) > occ_thresh
        out[off] = (torch.tensor(tgt, device=images.device), fwd, visible)

    cache[key] = out
    return out
