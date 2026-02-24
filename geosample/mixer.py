# -*- coding: utf-8 -*-
"""Token gating and 1x1x1 mixing used by GeoSample."""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenGatedMixer3D(nn.Module):
    """Token-wise gating followed by full 1x1x1 mixing.

    The input is a flattened token tensor of shape (N, C*M, D, H, W),
    where M is the number of token blocks and each block has C channels.
    """

    def __init__(self, channels: int, num_tokens: int, separate_down: bool = True) -> None:
        super().__init__()
        self.C = int(channels)
        self.M = int(num_tokens)

        self.gate = nn.Conv3d(self.C * self.M, self.M, kernel_size=1, groups=self.M, bias=True)
        self.gate._skip_global_init = True

        self.mix_refine = nn.Conv3d(self.C * self.M, self.C, kernel_size=1, bias=True)
        self.mix_down = nn.Conv3d(self.C * self.M, self.C, kernel_size=1, bias=True) if separate_down else self.mix_refine

        with torch.no_grad():
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def _apply_gate(self, feat_flat: torch.Tensor) -> torch.Tensor:
        N, CM, D, H, W = feat_flat.shape
        if CM != self.C * self.M:
            raise ValueError(f"Expected {self.C*self.M} channels, got {CM}.")

        logits = self.gate(feat_flat)              # (N,M,D,H,W)
        w = torch.sigmoid(logits)
        w = w * (self.M / (w.sum(dim=1, keepdim=True) + 1e-8))

        feat = feat_flat.view(N, self.M, self.C, D, H, W)
        feat = feat * w.unsqueeze(2)
        return feat.view(N, self.C * self.M, D, H, W)

    def refine(self, feat_flat: torch.Tensor) -> torch.Tensor:
        return self.mix_refine(self._apply_gate(feat_flat))

    def downsample(self, feat_flat: torch.Tensor) -> torch.Tensor:
        return self.mix_down(self._apply_gate(feat_flat))
