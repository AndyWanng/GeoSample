# -*- coding: utf-8 -*-
"""Grid utilities for 3D symmetric sampling via grid_sample."""

from __future__ import annotations

import torch


def make_base_grid_normalized(
    depth: int, height: int, width: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create voxel-center base grid in normalized coordinates for align_corners=False.

    Returns:
        (gx, gy, gz) each shaped (1, D, H, W).
    """
    z = torch.arange(depth, device=device, dtype=dtype)
    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)

    gz = (2.0 * z + 1.0) / depth - 1.0
    gy = (2.0 * y + 1.0) / height - 1.0
    gx = (2.0 * x + 1.0) / width - 1.0

    gz = gz.view(1, depth, 1, 1).expand(1, depth, height, width)
    gy = gy.view(1, 1, height, 1).expand(1, depth, height, width)
    gx = gx.view(1, 1, 1, width).expand(1, depth, height, width)
    return gx, gy, gz
