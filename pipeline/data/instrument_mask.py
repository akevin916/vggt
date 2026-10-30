# Patch flags for 方向 3 (planning/方向3_StereoMIS器械遮罩訓練.md): which patches a key_mask hides.
#
# Shared by the trainer (flags from the augmented, batch-aligned pixel mask) and the eval
# (flags from mask_left on the inference crop), so both sides use one definition:
#   inst    a patch is flagged when instrument pixels cover >= 50% of it
#   random  per frame, the SAME number of patches as `inst`, at positions drawn from a generator
#           seeded by (sequence, frame row) -- a frame gets the same draw every time it is seen
#           on the same patch grid.
import zlib

import torch
import torch.nn.functional as F


def patch_flags(pixel_mask: torch.Tensor, patch_size: int = 14, thresh: float = 0.5) -> torch.Tensor:
    """pixel_mask [..., H, W] float in {0, 1} (1 = instrument) -> [..., (H/p)*(W/p)] bool."""
    lead, (H, W) = pixel_mask.shape[:-2], pixel_mask.shape[-2:]
    if H % patch_size or W % patch_size:
        raise ValueError(f"mask {H}x{W} is not a multiple of patch {patch_size}")
    m = pixel_mask.reshape(-1, 1, H, W).float()
    frac = F.avg_pool2d(m, patch_size)
    return (frac >= thresh).reshape(*lead, -1)


def random_flags_like(flags: torch.Tensor, keys) -> torch.Tensor:
    """flags [S, P] bool; keys: S strings (e.g. "P2_0:1234"). Same per-frame count, random places."""
    S, P = flags.shape
    out = torch.zeros_like(flags)
    counts = flags.sum(dim=1).tolist()
    for s in range(S):
        if counts[s] == 0:
            continue
        g = torch.Generator().manual_seed(zlib.crc32(str(keys[s]).encode()))
        out[s, torch.randperm(P, generator=g)[: int(counts[s])].to(flags.device)] = True
    return out
