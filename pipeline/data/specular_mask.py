"""Specular-highlight mask for endoscopic frames, and the two strata built on it.

ONE DETECTOR, ONE PLACE. Three different specular detectors had grown up in diag/ under two
names, so a number reported against "the specular region" did not identify which region. This
module is the only one, and it is the chromaticity test that was validated by eye over SCARED and
C3VD (outputs/specular_check/, 36 overlays, 2026-09-07): a specular pixel is BRIGHT and its colour
sits near the white point, because the reflection carries the illuminant's colour rather than the
mucosa's red.

NO ROI. Earlier versions masked out the endoscope's black letterbox first. The data in use has no
letterbox, and a border test that never fires is a border test nobody notices has stopped working.

DILATION IS PART OF THE MASK, deliberately. A highlight does not end where the saturation ends --
it fades through a penumbra that is already corrupted but no longer passes either threshold.
Scoring only the core would attribute that ring to `clean`, which is the one place it must not go.
"""

from typing import Tuple

import cv2
import numpy as np

# Stratum ids. Kept here rather than in a caller so every script labelling a pixel agrees.
CLEAN = 0
SPECULAR = 1
STRATA = {CLEAN: "clean", SPECULAR: "specular"}

# Validated defaults -- see the module docstring for what they were checked against.
BRIGHT_THR = 0.82     # mean of the three channels, on [0,1]
CHROMA_TOL = 0.07     # distance in the rg-chromaticity plane from the white point (1/3, 1/3)
DILATE = 7            # square structuring element, pixels


def specular_mask(bgr: np.ndarray, bright_thr: float = BRIGHT_THR,
                  chroma_tol: float = CHROMA_TOL, dilate: int = DILATE) -> np.ndarray:
    """[H,W] uint8 (0/1). Input is BGR, i.e. straight from cv2.imread.

    Channel order matters: the white-point distance is measured in the (r, g) plane, so feeding
    BGR to a formula written for RGB silently swaps r for b and scores a different colour.
    """
    f = bgr.astype(np.float32) / 255.0
    B, G, R = f[..., 0], f[..., 1], f[..., 2]

    total = R + G + B + 1e-6
    r, g = R / total, G / total
    brightness = total / 3.0
    dist_to_white = np.sqrt((r - 1.0 / 3.0) ** 2 + (g - 1.0 / 3.0) ** 2)

    m = ((brightness > bright_thr) & (dist_to_white < chroma_tol)).astype(np.uint8)
    if dilate and dilate > 1:
        m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
    return m


def stratify(bgr: np.ndarray, bright_thr: float = BRIGHT_THR,
             chroma_tol: float = CHROMA_TOL, dilate: int = DILATE) -> np.ndarray:
    """[H,W] int8 stratum map over the WHOLE frame: SPECULAR where the mask fires, CLEAN elsewhere."""
    lab = np.full(bgr.shape[:2], CLEAN, dtype=np.int8)
    lab[specular_mask(bgr, bright_thr, chroma_tol, dilate) == 1] = SPECULAR
    return lab
