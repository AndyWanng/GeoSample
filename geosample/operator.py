# -*- coding: utf-8 -*-
"""GeoSample: geometry-guided symmetric sampling operator for 3D feature volumes."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .consensus import ConsensusField3D
from .field import GeoField, GeoFieldHead3D
from .grid import make_base_grid_normalized
from .mixer import TokenGatedMixer3D
from .utils import EPS


class GeoSample3D(nn.Module):
    """A unified local operator for stride-1 refinement and stride>1 downsampling."""

    def __init__(
        self,
        channels: int,
        K: int = 3,
        r_range: Tuple[float, float] = (0.5, 2.0),
        use_consensus: bool = True,
        consensus_omega_init: float = 0.5,
        separate_down_mixer: bool = True,
        padding_mode: str = "reflection",
    ) -> None:
        super().__init__()
        self.C = int(channels)
        self.K = int(K)
        self.padding_mode = str(padding_mode)

        self.field_head = GeoFieldHead3D(in_channels=self.C, K=self.K, r_range=r_range)

        self.num_tokens = 1 + self.K + 3 + 1
        self.mixer = TokenGatedMixer3D(channels=self.C, num_tokens=self.num_tokens, separate_down=separate_down_mixer)

        self.consensus = ConsensusField3D(K=self.K, omega_init=consensus_omega_init) if use_consensus else None

    def init_parameters(self, r0: float = 1.0) -> None:
        self.field_head.init_parameters(r0=r0)

    def forward(
        self,
        x: torch.Tensor,
        stride: Tuple[int, int, int] = (1, 1, 1),
        reference_field: Optional[GeoField] = None,
        return_field: bool = False,
    ):
        N, C, D, H, W = x.shape
        if C != self.C:
            raise ValueError(f"Expected {self.C} channels, got {C}.")

        field = self.field_head(x)
        if (reference_field is not None) and (self.consensus is not None):
            field = self.consensus(field, reference_field)

        U = field.U           # (N,K,3,D,H,W)
        r = field.r           # (N,K,1,D,H,W)

        gx0, gy0, gz0 = make_base_grid_normalized(D, H, W, x.device, x.dtype)

        ux, uy, uz = U[:, :, 0], U[:, :, 1], U[:, :, 2]    # (N,K,D,H,W)
        r_s = r[:, :, 0]                                    # (N,K,D,H,W)

        dx = 2.0 * r_s * ux / (W + EPS)
        dy = 2.0 * r_s * uy / (H + EPS)
        dz = 2.0 * r_s * uz / (D + EPS)

        grid_plus = torch.stack([gx0 + dx, gy0 + dy, gz0 + dz], dim=-1).view(-1, D, H, W, 3).contiguous()
        grid_minus = torch.stack([gx0 - dx, gy0 - dy, gz0 - dz], dim=-1).view(-1, D, H, W, 3).contiguous()

        x_rep = x.repeat_interleave(self.K, dim=0)

        x_plus = F.grid_sample(
            x_rep,
            grid_plus,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=False,
        ).view(N, self.K, C, D, H, W)

        x_minus = F.grid_sample(
            x_rep,
            grid_minus,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=False,
        ).view(N, self.K, C, D, H, W)

        a = 0.5 * (x_plus + x_minus)
        o = 0.5 * (x_plus - x_minus)

        inv_r = 1.0 / (r_s.detach() + EPS)
        d1 = o * inv_r.unsqueeze(2)

        G = (U.unsqueeze(3) * d1.unsqueeze(2)).sum(dim=1)     # (N,3,C,D,H,W)
        G_flat = G.contiguous().view(N, 3 * C, D, H, W)

        inv_r2 = inv_r * inv_r
        x0 = x.unsqueeze(1)
        d2 = (x_plus + x_minus - 2.0 * x0) * inv_r2.unsqueeze(2)
        L = d2.mean(dim=1)

        a_flat = a.contiguous().view(N, self.K * C, D, H, W)
        tokens = torch.cat([x, a_flat, G_flat, L], dim=1)

        delta = self.mixer.refine(tokens)
        y = x + delta

        if stride == (1, 1, 1):
            if return_field:
                return y, field
            return y

        tokens_s2 = torch.cat([x, a_flat, G_flat.abs(), L], dim=1)
        tokens_coarse = F.avg_pool3d(tokens_s2, kernel_size=stride, stride=stride, ceil_mode=True)
        delta_down = self.mixer.downsample(tokens_coarse)

        y_pool = F.avg_pool3d(y, kernel_size=stride, stride=stride, ceil_mode=True)
        y_down = y_pool + delta_down

        if return_field:
            return y_down, field
        return y_down
