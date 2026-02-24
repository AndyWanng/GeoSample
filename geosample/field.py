# -*- coding: utf-8 -*-
"""Geometry field prediction for GeoSample (SO(3) frame + step sizes)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from .rotation import normalize_quaternion, quaternion_to_matrix
from .utils import EPS


@dataclass
class GeoField:
    """Per-voxel geometric field used by GeoSample.

    Attributes:
        U: (N, K, 3, D, H, W) unit directions in image Cartesian axes.
        r: (N, K, 1, D, H, W) step sizes in voxel units.
        q: (N, 4, D, H, W) unit quaternion (for fusion in ConsensusField).
    """
    U: torch.Tensor
    r: torch.Tensor
    q: torch.Tensor


class GeoFieldHead3D(nn.Module):
    """Predict a local SO(3) frame and bounded per-direction step sizes."""

    def __init__(
        self,
        in_channels: int,
        K: int = 3,
        r_range: Tuple[float, float] = (0.5, 2.0),
    ) -> None:
        super().__init__()
        self.K = int(K)
        self.rmin, self.rmax = float(r_range[0]), float(r_range[1])

        out_channels = 4 + self.K
        self.pre = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.head = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)

    @torch.no_grad()
    def init_parameters(self, r0: float = 1.0) -> None:
        """Initialize to identity rotation and a stable step size prior."""
        if self.head.bias is None:
            return

        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.head.bias.data[0] = 1.0

        sigma = (float(r0) - self.rmin) / (self.rmax - self.rmin + EPS)
        sigma = min(max(float(sigma), 1e-4), 1.0 - 1e-4)
        rho_bias = math.log(sigma / (1.0 - sigma))
        self.head.bias.data[4 : 4 + self.K].fill_(rho_bias)

    def forward(self, x: torch.Tensor) -> GeoField:
        N, C, D, H, W = x.shape
        p = self.head(self.pre(x))

        q_raw = p[:, 0:4]                 # (N,4,D,H,W)
        rho = p[:, 4 : 4 + self.K]        # (N,K,D,H,W)

        q = q_raw.permute(0, 2, 3, 4, 1).contiguous()  # (N,D,H,W,4)
        q = normalize_quaternion(q)
        R = quaternion_to_matrix(q)                    # (N,D,H,W,3,3)

        U_list = []
        for i in range(self.K):
            ui = R[..., i]
            ui = ui / (ui.norm(dim=-1, keepdim=True) + EPS)
            U_list.append(ui)

        U = torch.stack(U_list, dim=1)                 # (N,K,D,H,W,3)
        U = U.permute(0, 1, 5, 2, 3, 4).contiguous()   # (N,K,3,D,H,W)

        r = self.rmin + torch.sigmoid(rho) * (self.rmax - self.rmin)
        r = r.unsqueeze(2)                             # (N,K,1,D,H,W)

        q_out = q.permute(0, 4, 1, 2, 3).contiguous()  # (N,4,D,H,W)
        return GeoField(U=U, r=r, q=q_out)
