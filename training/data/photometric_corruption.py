"""Photometric corruption: appearance-only perturbations for the influence loss.

WHAT THIS IS FOR. Endoscopic video fails in a specific way that VGGT has no mechanism to
absorb: a SINGLE frame goes bad (auto-white-balance re-locks and the field turns green, the
gain steps, a specular blob saturates) and the WHOLE sequence's reconstruction drifts, because
global attention lets every frame attend to that frame's patch tokens with no notion of
"this view is not trustworthy". These corruptions are the probe used to (a) train against that
failure and (b) measure it.

THE ONE HARD RULE: geometry is never touched. No warp, crop, resize, flip, or shift -- only a
per-pixel value mapping. compute_influence_loss compares the clean and corrupted forward passes
pixel-by-pixel, so the two must stay pixel-aligned or the loss is meaningless. That is also what
makes the finite difference it computes span ONLY the appearance directions: the corruption
carries no geometric component, so what the loss measures is exactly the appearance channel of
one frame's influence on the others.

TWO CLASSES, and they are not interchangeable:
  C1 (information-preserving) -- colour cast, exposure, gamma, smooth light field, contrast.
      The frame's content survives the mapping; geometry is still recoverable in principle.
      L_inf uses ONLY these, because "the other frames must be unaffected" is then a statement
      about nuisance sensitivity and nothing else.
  C2 (information-destroying) -- specular saturation (clipped, data gone), haze/smoke, fluid
      smear, motion blur, defocus, sensor noise. Here the frame genuinely stops carrying
      geometry, so demanding zero change in the others is slightly too strong for L_inf. They
      exist for the reliability-gate arm and for the held-out robustness evaluation: train on C1
      only, test on C2, and a gate that still fires has learned "unreliable" rather than "green".

MULTIPLICATIVE LIGHT ACTS IN LINEAR SPACE. Endoscopic frames are gamma-encoded; a light source
moving closer scales radiance, not the sRGB code value. Gains and the light field therefore
round-trip through linear light, while the tone-curve operators (gamma, contrast) act directly
on the encoded values, which is where they physically happen.

SEVERITY IS BIMODAL, NOT UNIFORM. The real failure is rare-but-severe. Uniform mild jitter
trains for a distribution that does not occur and leaves the tail -- the case that actually
breaks reconstruction -- unpractised.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch

# C1: the frame's content survives the mapping.
C1_TYPES = ("color_cast", "exposure", "gamma", "light_field", "contrast")
# C2: the frame genuinely loses information.
C2_TYPES = ("specular", "haze", "blur", "noise")


# --------------------------------------------------------------------------------------
# sRGB <-> linear light
# --------------------------------------------------------------------------------------
def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp(min=0) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(min=0.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055).clamp(0.0, 1.0)


def _u(shape, lo, hi, device, dtype=torch.float32) -> torch.Tensor:
    return torch.rand(shape, device=device, dtype=dtype) * (hi - lo) + lo


def _smooth_field(n: int, h: int, w: int, device, res: int = 5) -> torch.Tensor:
    """n independent smooth [0,1] fields of size (h, w), from a low-res random grid."""
    coarse = torch.rand(n, 1, res, res, device=device)
    return torch.nn.functional.interpolate(
        coarse, size=(h, w), mode="bilinear", align_corners=False
    )


# --------------------------------------------------------------------------------------
# which frames get hit
# --------------------------------------------------------------------------------------
def sample_frame_mask(
    B: int,
    S: int,
    device,
    p_sequence: float = 0.8,
    n_frames: Sequence[int] = (1, 2),
    p_burst: float = 0.3,
    p_first_frame: float = 0.2,
) -> torch.Tensor:
    """[B, S] bool mask -- True where the frame is to be corrupted.

    p_sequence < 1 leaves some sequences entirely clean on purpose: without them the model can
    learn to assume every clip contains a bad frame.
    p_first_frame deliberately over-samples frame 0, which defines the reference coordinate
    system and is therefore the worst case, not just one case among S.
    """
    mask = torch.zeros(B, S, dtype=torch.bool, device=device)
    lo, hi = int(n_frames[0]), int(n_frames[-1])
    for b in range(B):
        if torch.rand(1).item() > p_sequence:
            continue
        k = int(torch.randint(lo, hi + 1, (1,)).item())
        k = min(k, S)
        if torch.rand(1).item() < p_burst and S > k:
            # A real AWB / exposure event persists for several frames rather than one.
            start = int(torch.randint(0, S - k + 1, (1,)).item())
            idx = list(range(start, start + k))
        else:
            idx = torch.randperm(S)[:k].tolist()
        if torch.rand(1).item() < p_first_frame and 0 not in idx:
            idx[0] = 0
        mask[b, idx] = True
    return mask


# --------------------------------------------------------------------------------------
# the corruption itself
# --------------------------------------------------------------------------------------
def apply_corruption(
    images: torch.Tensor,
    frame_mask: torch.Tensor,
    types: Sequence[str] = C1_TYPES,
    p_type: float = 0.5,
    p_severe: float = 0.25,
    mild: Sequence[float] = (0.2, 0.5),
    severe: Sequence[float] = (0.7, 1.0),
) -> torch.Tensor:
    """Corrupt the frames selected by `frame_mask`; return a new [B, S, 3, H, W] in [0, 1].

    Frames outside the mask are restored bit-for-bit from the input. This matters: the sRGB
    round-trip is not exactly the identity, and a clean frame that differs from the teacher's
    input by even a rounding error would feed L_inf a signal that has nothing to do with the
    corruption it is supposed to be measuring.

    Every corrupted frame draws its own severity and its own subset of `types` (at least one),
    since real events are combinations -- a white-balance jump arrives with an exposure step.
    """
    B, S, C, H, W = images.shape
    device = images.device
    n = B * S
    flat = images.reshape(n, C, H, W).float()

    # Per-frame severity in [0, 1]: bimodal, so the tail is actually practised.
    is_severe = torch.rand(n, device=device) < p_severe
    sev = torch.where(is_severe, _u((n,), *severe, device), _u((n,), *mild, device))
    sev = sev.view(n, 1, 1, 1)

    # Per-frame, per-type on/off, with at least one type active.
    active: Dict[str, torch.Tensor] = {}
    any_on = torch.zeros(n, dtype=torch.bool, device=device)
    for t in types:
        on = torch.rand(n, device=device) < p_type
        active[t] = on
        any_on |= on
    if types:
        forced = torch.randint(0, len(types), (n,), device=device)
        for i, t in enumerate(types):
            active[t] = active[t] | ((~any_on) & (forced == i))

    def gate(t: str) -> torch.Tensor:
        return active[t].view(n, 1, 1, 1).float()

    out = flat

    # ---- linear-light operators (radiance actually scales here) ----
    needs_linear = any(t in types for t in ("color_cast", "exposure", "light_field"))
    if needs_linear:
        lin = srgb_to_linear(out)

        if "color_cast" in types:
            # Per-channel gain: AWB re-lock, blood/bile tint. This is the "green frame".
            a = 0.45 * sev
            g = torch.exp(_u((n, 3, 1, 1), -1.0, 1.0, device) * a)
            lin = lin * (1.0 + (g - 1.0) * gate("color_cast"))

        if "exposure" in types:
            # Global gain step: AGC reacting to a bright wall.
            b = 0.9 * sev
            g = torch.exp(_u((n, 1, 1, 1), -1.0, 1.0, device) * b)
            lin = lin * (1.0 + (g - 1.0) * gate("exposure"))

        if "light_field" in types:
            # Smooth low-frequency multiplicative field: falloff / vignetting as the scope moves.
            # Amplitude is kept deliberately modest -- real falloff is a function of DEPTH, and a
            # strong random field would be teaching the model to ignore shape-from-shading, which
            # is a genuine geometric cue in endoscopy rather than a nuisance.
            d = 0.5 * sev
            f = torch.exp((_smooth_field(n, H, W, device) * 2.0 - 1.0) * d)
            lin = lin * (1.0 + (f - 1.0) * gate("light_field"))

        out = linear_to_srgb(lin)

    # ---- tone-curve operators (these happen on the encoded signal) ----
    if "gamma" in types:
        c = 0.7 * sev
        gm = torch.exp(_u((n, 1, 1, 1), -1.0, 1.0, device) * c)
        gm = 1.0 + (gm - 1.0) * gate("gamma")
        out = out.clamp(1e-6, 1.0) ** gm

    if "contrast" in types:
        e = 0.5 * sev
        k = torch.exp(_u((n, 1, 1, 1), -1.0, 1.0, device) * e)
        k = 1.0 + (k - 1.0) * gate("contrast")
        mean = out.mean(dim=(1, 2, 3), keepdim=True)
        out = mean + (out - mean) * k

    # ---- C2: information-destroying ----
    if "specular" in types:
        # Additive Gaussian blobs clipped at 1.0 -- the clip is the point: saturated pixels have
        # genuinely lost their content, which is what a wet-mucosa highlight does.
        g = gate("specular")
        yy = torch.linspace(0, 1, H, device=device).view(1, H, 1)
        xx = torch.linspace(0, 1, W, device=device).view(1, 1, W)
        blob = torch.zeros(n, 1, H, W, device=device)
        for _ in range(3):
            cy = _u((n, 1, 1), 0.15, 0.85, device)
            cx = _u((n, 1, 1), 0.15, 0.85, device)
            r = _u((n, 1, 1), 0.03, 0.12, device)
            amp = _u((n, 1, 1), 0.3, 1.2, device) * sev.view(n, 1, 1)
            d2 = (yy - cy) ** 2 + (xx - cx) ** 2
            blob = blob + (amp * torch.exp(-d2 / (2 * r ** 2))).unsqueeze(1)
        out = out + blob * g

    if "haze" in types:
        # Standard veiling model: cautery smoke, lens fog.
        g = gate("haze")
        t = _smooth_field(n, H, W, device, res=3) * (0.6 * sev)
        out = (1 - t * g) * out + (t * g) * 0.9

    if "blur" in types:
        g = gate("blur")
        ks = 9
        coords = torch.arange(ks, device=device, dtype=torch.float32) - (ks - 1) / 2
        sigma = (0.5 + 2.5 * sev).view(n, 1)
        k1 = torch.exp(-(coords.view(1, ks) ** 2) / (2 * sigma ** 2))
        k1 = k1 / k1.sum(dim=1, keepdim=True)
        pad = ks // 2
        x = out.reshape(1, n * C, H, W)
        kh = k1.repeat_interleave(C, dim=0).view(n * C, 1, 1, ks)
        kv = k1.repeat_interleave(C, dim=0).view(n * C, 1, ks, 1)
        x = torch.nn.functional.conv2d(
            torch.nn.functional.pad(x, (pad, pad, 0, 0), mode="reflect"), kh, groups=n * C)
        x = torch.nn.functional.conv2d(
            torch.nn.functional.pad(x, (0, 0, pad, pad), mode="reflect"), kv, groups=n * C)
        blurred = x.reshape(n, C, H, W)
        out = out + (blurred - out) * g

    if "noise" in types:
        g = gate("noise")
        out = out + torch.randn_like(out) * (0.05 * sev) * g

    out = out.clamp(0.0, 1.0).reshape(B, S, C, H, W)
    # Restore uncorrupted frames exactly (see docstring).
    keep = (~frame_mask).view(B, S, 1, 1, 1)
    return torch.where(keep, images, out.to(images.dtype))


def corrupt_batch(
    images: torch.Tensor,
    cfg: Optional[Dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper: sample which frames to hit, then hit them.

    Args:
        images: [B, S, 3, H, W] in [0, 1].
        cfg: keys of sample_frame_mask and apply_corruption (all optional).
    Returns:
        (corrupted images, [B, S] bool mask of which frames were corrupted)
    """
    cfg = dict(cfg or {})
    cfg.pop("enabled", None)
    cfg.pop("warmup_steps", None)
    B, S = images.shape[0], images.shape[1]
    mask_kw = {k: cfg[k] for k in ("p_sequence", "n_frames", "p_burst", "p_first_frame") if k in cfg}
    apply_kw = {k: cfg[k] for k in ("types", "p_type", "p_severe", "mild", "severe") if k in cfg}
    frame_mask = sample_frame_mask(B, S, images.device, **mask_kw)
    if not bool(frame_mask.any()):
        return images, frame_mask
    return apply_corruption(images, frame_mask, **apply_kw), frame_mask
