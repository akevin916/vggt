# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


# Implementation of 2D Rotary Position Embeddings (RoPE).

# This module provides a clean implementation of 2D Rotary Position Embeddings,
# which extends the original RoPE concept to handle 2D spatial positions.

# Inspired by:
#         https://github.com/meta-llama/codellama/blob/main/llama/model.py
#         https://github.com/naver-ai/rope-vit


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid.

    This class efficiently manages the generation of spatial coordinates for patches
    in a 2D grid, caching results to avoid redundant computations.

    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.
    """

    def __init__(self):
        """Initializes the position generator with an empty cache."""
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generates spatial positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        Returns:
            Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
            for each position in the grid, repeated for each batch item.
        """
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding implementation.

    This module applies rotary position embeddings to input tokens based on their
    2D spatial positions. It handles the position-dependent rotation of features
    separately for vertical and horizontal dimensions.

    Args:
        frequency: Base frequency for the position embeddings. Default: 100.0
        scaling_factor: Scaling factor for frequency computation. Default: 1.0

    Attributes:
        base_frequency: Base frequency for computing position embeddings.
        scaling_factor: Factor to scale the computed frequencies.
        frequency_cache: Cache for storing precomputed frequency components.
    """

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        """Initializes the 2D RoPE module."""
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings.

        Args:
            dim: Feature dimension (must be even).
            seq_len: Maximum sequence length.
            device: Target device for computations.
            dtype: Data type for the computed tensors.

        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            # Compute frequency bands
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)

            # Generate position-dependent frequencies
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            # Compute and cache frequency components
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Performs feature rotation by splitting and recombining feature dimensions.

        Args:
            x: Input tensor to rotate.

        Returns:
            Rotated feature tensor.
        """
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
        """Applies 1D rotary position embeddings along one dimension.

        Args:
            tokens: Input token features.
            positions: Position indices.
            cos_comp: Cosine components for rotation.
            sin_comp: Sine components for rotation.

        Returns:
            Tokens with applied rotary position embeddings.
        """
        # Embed positions with frequency components
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]

        # Apply rotation
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies 2D rotary position embeddings to input tokens.

        Args:
            tokens: Input tensor of shape (batch_size, n_heads, n_tokens, dim).
                   The feature dimension (dim) must be divisible by 4.
            positions: Position tensor of shape (batch_size, n_tokens, 2) containing
                      the y and x coordinates for each token.

        Returns:
            Tensor of same shape as input with applied 2D rotary position embeddings.

        Raises:
            AssertionError: If input dimensions are invalid or positions are malformed.
        """
        # Validate inputs
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must have shape (batch_size, n_tokens, 2)"

        # Compute feature dimension for each spatial direction
        feature_dim = tokens.size(-1) // 2

        # Get frequency components
        max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(feature_dim, max_position, tokens.device, tokens.dtype)

        # Split features for vertical and horizontal processing
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)

        # Apply RoPE separately for each dimension
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)

        # Combine processed features
        return torch.cat((vertical_features, horizontal_features), dim=-1)


class RotaryPositionEmbedding1D(RotaryPositionEmbedding2D):
    # NEW: Dyn-VGGT 時間軸 RoPE。與 2D 版同構但只編「時間」一個軸（整個 head_dim 都用幀索引旋轉），
    #      只在 temporal attention 內作用，空間 RoPE 一個字不動，保留預訓練 warm-start（見 docs §3.1）。
    """1D Rotary Position Embedding for the temporal axis.

    Reuses the frequency / rotate-half machinery of the 2D version, but rotates the
    full feature dimension by a single (time) coordinate instead of splitting it into
    a vertical/horizontal pair.

    Args:
        tokens: Input tensor of shape (batch, n_heads, n_tokens, head_dim).
        positions: Integer position tensor of shape (batch, n_tokens) — frame indices.
                   Tokens given position 0 receive an identity rotation (no temporal RoPE),
                   which is how register tokens are excluded (decision B4).
    """

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 2, "Positions must have shape (batch, n_tokens)"

        feature_dim = tokens.size(-1)
        max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(feature_dim, max_position, tokens.device, tokens.dtype)

        return self._apply_1d_rope(tokens, positions, cos_comp, sin_comp)


class RotaryPositionEmbedding3D(RotaryPositionEmbedding2D):
    """3D RoPE over (y, x, t): the head_dim is split between two spatial axes and FRAME INDEX.

    Why this exists: it injects frame-order information into the SAME place the temporal blocks
    do (global attention) but with ZERO new parameters, which is the control that separates
    "temporal helps because of the information" from "temporal helps because of its 100.8M
    parameters" (docs: temporal 100.8M -> pose chunk5 -11.5%, weight-shared 16k -> -1.2%).

    ``dims`` are per-axis feature widths and must sum to head_dim (64 for VGGT-1B: 2D RoPE uses
    32/32, i.e. 16 frequency pairs per axis; the default 24/24/16 here is 12/12/8 pairs).
    ⚠️ This RE-ASSIGNS which channels carry which axis, so pretrained weights are NOT warm-started
    the way temporal (LayerScale 0) or the dual-stream bias (|p| ~ 0) are. Measure the damage of
    the re-split alone by running with ``t`` forced to 0 before reading anything into a trained run.

    Frame attention is unaffected for free: t is constant within a frame and RoPE only sees
    position differences, so the time rotation cancels there and only global attention feels it.
    """

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0,
                 dims: Tuple[int, int, int] = (24, 24, 16)):
        super().__init__(frequency=frequency, scaling_factor=scaling_factor)
        assert all(d % 2 == 0 for d in dims), f"every axis width must be even, got {dims}"
        self.dims = tuple(int(d) for d in dims)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """tokens (batch, n_heads, n_tokens, head_dim); positions (batch, n_tokens, 3) = y, x, t."""
        assert positions.ndim == 3 and positions.shape[-1] == 3, "positions must be (batch, n_tokens, 3)"
        assert tokens.size(-1) == sum(self.dims), f"head_dim {tokens.size(-1)} != sum{self.dims}"

        out = []
        for axis, (feat, dim) in enumerate(zip(torch.split(tokens, self.dims, dim=-1), self.dims)):
            coord = positions[..., axis]
            cos_comp, sin_comp = self._compute_frequency_components(
                dim, int(coord.max()) + 1, tokens.device, tokens.dtype
            )
            out.append(self._apply_1d_rope(feat, coord, cos_comp, sin_comp))
        return torch.cat(out, dim=-1)


class RotaryPositionEmbedding2DTime(nn.Module):
    """2D RoPE with a TIME phase added on top of the existing spatial one (no re-split).

    Per frequency pair k, for the channel half that owns axis u (y for the first half, x for the
    second), and with t the token's frame index:

        camera / register / illu token:  theta_k = alpha[h] * nu_k * t
        patch token:                     theta_k = omega_k * u_k + alpha[h] * nu_k * t

        omega_k = spatial_base ** (-2k/m)     m = head_dim // 2, spatial_base = 100 (UNCHANGED)
        nu_k    = time_base    ** (-2k/m)     time_base = 10 by default: periods ~6..54 frames,
                                              which is the range training (4-12) and eval (64) use

    Why this and not the 24/24/16 re-split (RotaryPositionEmbedding3D): re-splitting changes every
    spatial frequency, and the zero-training probe measured +112% chunk5 / +170% chunk64 on trained
    base weights before a single step. Here the spatial ladder is untouched and alpha starts at 0,
    so the module is EXACTLY the pretrained 2D RoPE at init and time fades in as alpha grows.

    alpha is per-head and one instance is built per block, so its shape in a checkpoint is
    [num_heads] per block -- the only new parameters (16 per block).

    Cost: specials have no spatial term, so camera/register carry time phase with no space-time
    aliasing; patch tokens share channels between space and time, which aliases "next frame" with
    a spatial shift of alpha*nu_k/omega_k. That trade-off is inherent to not re-splitting; the
    per-k ratio varies because time_base != spatial_base, which is what lets heads separate them.
    """

    def __init__(self, num_heads: int, frequency: float = 100.0, time_base: float = 10.0):
        super().__init__()
        self.base_frequency = float(frequency)
        self.time_base = float(time_base)
        self.alpha = nn.Parameter(torch.zeros(num_heads))   # identity at init -> exact warm start

    def _ladder(self, m: int, base: float, device, dtype) -> torch.Tensor:
        exponents = torch.arange(0, m, 2, device=device, dtype=torch.float32) / m
        return (1.0 / (base ** exponents)).to(dtype)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """tokens (B, H, N, head_dim); positions (B, N, 3) = (y, x, t), all integers."""
        assert positions.ndim == 3 and positions.shape[-1] == 3, "positions must be (B, N, 3)"
        B, H, N, D = tokens.shape
        m = D // 2
        f32 = torch.float32
        omega = self._ladder(m, self.base_frequency, tokens.device, f32)          # [m/2]
        nu = self._ladder(m, self.time_base, tokens.device, f32)                  # [m/2]
        y, x, t = (positions[..., i].to(f32) for i in range(3))                   # [B, N]

        # time phase, per head: [B, H, N, m/2]
        time_phase = self.alpha.to(f32).view(1, H, 1, 1) * t.view(B, 1, N, 1) * nu.view(1, 1, 1, -1)
        out = []
        for coord, feat in zip((y, x), torch.split(tokens, m, dim=-1)):
            theta = coord.view(B, 1, N, 1) * omega.view(1, 1, 1, -1) + time_phase
            theta = torch.cat((theta, theta), dim=-1).to(tokens.dtype)            # [B, H, N, m]
            cos, sin = theta.cos(), theta.sin()
            half = feat.shape[-1] // 2
            rotated = torch.cat((-feat[..., half:], feat[..., :half]), dim=-1)
            out.append(feat * cos + rotated * sin)
        return torch.cat(out, dim=-1)
