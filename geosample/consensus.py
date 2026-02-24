# -*- coding: utf-8 -*-
"""Rotation-consistent fusion of geometry fields (Consensus Field)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .field import GeoField
from .rotation import normalize_quaternion, quaternion_to_matrix, slerp
from .utils import EPS


class ConsensusField3D(nn.Module):
    """Fuse a current field with a reference field in a rotation-consistent manner."""

    def __init__(self, K: int = 3, omega_init: float = 0.5) -> None:
        super().__init__()
        self.K = int(K)
        self.gate = nn.Conv3d(4, 1, kernel_size=1, bias=True)
        self.gate._skip_global_init = True

        with torch.no_grad():
            self.gate.weight.zero_()
            w0 = min(max(float(omega_init), 1e-4), 1.0 - 1e-4)
            self.gate.bias.fill_(math.log(w0 / (1.0 - w0)))

    def forward(self, current: GeoField, reference: GeoField) -> GeoField:
        U, r, q = current.U, current.r, current.q
        U_ref, r_ref, q_ref = reference.U, reference.r, reference.q

        qv = q.permute(0, 2, 3, 4, 1).contiguous()         # (N,D,H,W,4)
        qv_ref = q_ref.permute(0, 2, 3, 4, 1).contiguous()

        qv = normalize_quaternion(qv)
        qv_ref = normalize_quaternion(qv_ref)

        c = torch.abs((qv * qv_ref).sum(dim=-1)).clamp(0.0, 1.0)  # (N,D,H,W)
        c = c.unsqueeze(1)

        r_mean = r.mean(dim=1)         # (N,1,D,H,W)
        r_mean_ref = r_ref.mean(dim=1)

        ones = torch.ones_like(r_mean)
        omega = torch.sigmoid(self.gate(torch.cat([r_mean, r_mean_ref, c, ones], dim=1)))  # (N,1,D,H,W)

        t = omega.permute(0, 2, 3, 4, 1).contiguous()      # (N,D,H,W,1)
        q_bar = slerp(qv, qv_ref, t)
        R_bar = quaternion_to_matrix(q_bar)                # (N,D,H,W,3,3)

        r_bar = omega.unsqueeze(1) * r + (1.0 - omega.unsqueeze(1)) * r_ref

        U_bar_list = []
        for i in range(self.K):
            ui = R_bar[..., i]
            ui = ui / (ui.norm(dim=-1, keepdim=True) + EPS)
            U_bar_list.append(ui)

        U_bar = torch.stack(U_bar_list, dim=1)             # (N,K,D,H,W,3)
        U_bar = U_bar.permute(0, 1, 5, 2, 3, 4).contiguous()

        q_out = q_bar.permute(0, 4, 1, 2, 3).contiguous()
        return GeoField(U=U_bar, r=r_bar, q=q_out)
