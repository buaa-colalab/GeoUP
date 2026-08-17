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
        self.position_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}

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
        device_key = str(device)
        cache_key = (height, width, device_key)
        if cache_key not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
            positions = torch.stack((grid_y, grid_x), dim=-1).reshape(height * width, 2)
            self.position_cache[cache_key] = positions

        cached_positions = self.position_cache[cache_key]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1)

try:
    from .curope import cuRoPE2D

    RotaryPositionEmbedding2D = cuRoPE2D
except ImportError:
    print(
        "Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead"
    )

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

        def __init__(self, freq: float = 100.0, F0: float = 1.0):
            """Initializes the 2D RoPE module."""
            super().__init__()
            self.base_frequency = freq
            self.scaling_factor = F0
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
                if dim % 2 != 0:
                    raise ValueError(f"RoPE axis dimension must be even, got {dim}")

                # Keep only the unique half-dimension frequency bands. The old
                # path duplicated them to full dim and then duplicated work
                # again in `_rotate_features`.
                exponents = torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
                inv_freq = 1.0 / (self.base_frequency**exponents)

                # Generate position-dependent frequencies
                positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
                angles = torch.einsum("i,j->ij", positions, inv_freq)

                # Compute and cache frequency components
                angles = angles.to(dtype)
                cos_components = angles.cos()
                sin_components = angles.sin()
                self.frequency_cache[cache_key] = (cos_components, sin_components)

            return self.frequency_cache[cache_key]

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
            half_dim = tokens.shape[-1] // 2
            first_half, second_half = tokens[..., :half_dim], tokens[..., half_dim:]

            # Embed positions with frequency components
            cos = F.embedding(positions, cos_comp)[:, None, :, :]
            sin = F.embedding(positions, sin_comp)[:, None, :, :]

            # Apply rotation without materializing a full rotated copy.
            return torch.cat(
                (first_half * cos - second_half * sin,
                 second_half * cos + first_half * sin),
                dim=-1,
            )

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
            assert tokens.size(-1) % 4 == 0, "Feature dimension must be divisible by 4"
            assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must have shape (batch_size, n_tokens, 2)"

            # Compute feature dimension for each spatial direction
            feature_dim = tokens.size(-1) // 2

            # Get frequency components
            # All call sites in this repo pass dense patch-grid coordinates
            # plus optional zero-valued special tokens, so token count is a
            # safe upper bound for coordinate values and avoids a CUDA sync
            # from `positions.max()`.
            max_position = max(positions.shape[-2], 1)
            cos_comp, sin_comp = self._compute_frequency_components(feature_dim, max_position, tokens.device, tokens.dtype)

            # Split features for vertical and horizontal processing
            vertical_features, horizontal_features = tokens.chunk(2, dim=-1)

            # Apply RoPE separately for each dimension
            vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
            horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)

            # Combine processed features
            return torch.cat((vertical_features, horizontal_features), dim=-1)
