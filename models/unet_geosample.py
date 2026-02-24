# -*- coding: utf-8 -*-
"""Example 3D U-Net built from GeoSample and Consensus Field blocks."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from geosample import GeoField, GeoFieldHead3D, GeoSample3D


def init_conv_weights(module: nn.Module) -> None:
    """Kaiming init for Conv3d, excluding modules that opt out."""
    for m in module.modules():
        if isinstance(m, nn.Conv3d):
            if getattr(m, "_skip_global_init", False):
                continue
            nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class GeoSampleUnit3D(nn.Module):
    """GeoSample operator followed by a pointwise projection + norm + activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int, int] = (1, 1, 1),
        K: int = 3,
        r_range: Tuple[float, float] = (0.5, 2.0),
        consensus_omega_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.stride = tuple(stride)

        separate_down = self.stride != (1, 1, 1)
        self.op = GeoSample3D(
            channels=in_channels,
            K=K,
            r_range=r_range,
            use_consensus=True,
            consensus_omega_init=consensus_omega_init,
            separate_down_mixer=separate_down,
        )
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True, track_running_stats=False)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def init_parameters(self, r0: float = 1.0) -> None:
        self.op.init_parameters(r0=r0)

    def forward(self, x: torch.Tensor, reference_field: Optional[GeoField] = None) -> torch.Tensor:
        y = self.op(x, stride=self.stride, reference_field=reference_field, return_field=False)
        y = self.act(self.norm(self.proj(y)))
        return y


class DoubleGeoSample3D(nn.Module):
    """Two stride-1 GeoSample units (U-Net style double-conv)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 3,
        r_range: Tuple[float, float] = (0.5, 2.0),
        consensus_omega_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.block1 = GeoSampleUnit3D(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(1, 1, 1),
            K=K,
            r_range=r_range,
            consensus_omega_init=consensus_omega_init,
        )
        self.block2 = GeoSampleUnit3D(
            in_channels=out_channels,
            out_channels=out_channels,
            stride=(1, 1, 1),
            K=K,
            r_range=r_range,
            consensus_omega_init=consensus_omega_init,
        )

    def init_parameters(self, r0: float = 1.0) -> None:
        self.block1.init_parameters(r0=r0)
        self.block2.init_parameters(r0=r0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x


class GeoSampleDownsample3D(nn.Module):
    """Stride>1 GeoSample downsampling unit."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int, int] = (2, 2, 2),
        K: int = 3,
        r_range: Tuple[float, float] = (0.5, 2.0),
        consensus_omega_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.unit = GeoSampleUnit3D(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            K=K,
            r_range=r_range,
            consensus_omega_init=consensus_omega_init,
        )

    def init_parameters(self, r0: float = 1.0) -> None:
        self.unit.init_parameters(r0=r0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unit(x)


class GeoSampleUpBlock3D(nn.Module):
    """Decoder upsampling with Consensus Field alignment before skip fusion."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        K: int = 3,
        r_range: Tuple[float, float] = (0.5, 2.0),
        consensus_omega_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.align = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

        self.skip_field_head = GeoFieldHead3D(in_channels=skip_channels, K=K, r_range=r_range)
        self.pre_skip_align = GeoSampleUnit3D(
            in_channels=out_channels,
            out_channels=out_channels,
            stride=(1, 1, 1),
            K=K,
            r_range=r_range,
            consensus_omega_init=consensus_omega_init,
        )
        self.fuse = DoubleGeoSample3D(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            K=K,
            r_range=r_range,
            consensus_omega_init=consensus_omega_init,
        )

    def init_parameters(self, r0: float = 1.0) -> None:
        self.skip_field_head.init_parameters(r0=r0)
        self.pre_skip_align.init_parameters(r0=r0)
        self.fuse.init_parameters(r0=r0)

    @staticmethod
    def _match_size(x: torch.Tensor, ref_shape: Tuple[int, int, int]) -> torch.Tensor:
        if x.shape[2:] == ref_shape:
            return x
        return F.interpolate(x, size=ref_shape, mode="trilinear", align_corners=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.align(self.up(x))
        x = self._match_size(x, skip.shape[2:])

        reference_field = self.skip_field_head(skip.detach())
        x = self.pre_skip_align(x, reference_field=reference_field)

        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return x


class UNet3D_GeoSample(nn.Module):
    """A compact 3D U-Net where conv/pooling primitives are replaced by GeoSample."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        num_stages: int = 4,
        blocks_per_stage: int = 1,
        final_act: Optional[nn.Module] = None,
        K: int = 3,
        r_range: Tuple[float, float] = (0.5, 2.0),
        consensus_omega_init: float = 0.5,
    ) -> None:
        super().__init__()
        if num_stages < 2:
            raise ValueError("num_stages must be >= 2.")
        self.final_act = final_act

        chs = [base_channels * (2 ** i) for i in range(num_stages)]

        self.enc_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        for i in range(num_stages):
            stage_in = in_channels if i == 0 else chs[i]
            stage_out = chs[i]

            blocks = [DoubleGeoSample3D(stage_in, stage_out, K=K, r_range=r_range, consensus_omega_init=consensus_omega_init)]
            for _ in range(blocks_per_stage - 1):
                blocks.append(DoubleGeoSample3D(stage_out, stage_out, K=K, r_range=r_range, consensus_omega_init=consensus_omega_init))
            self.enc_blocks.append(nn.Sequential(*blocks))

            if i < num_stages - 1:
                self.down_blocks.append(
                    GeoSampleDownsample3D(stage_out, chs[i + 1], stride=(2, 2, 2), K=K, r_range=r_range, consensus_omega_init=consensus_omega_init)
                )

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(num_stages - 1)):
            self.up_blocks.append(
                GeoSampleUpBlock3D(
                    in_channels=chs[i + 1],
                    skip_channels=chs[i],
                    out_channels=chs[i],
                    K=K,
                    r_range=r_range,
                    consensus_omega_init=consensus_omega_init,
                )
            )

        self.head = nn.Conv3d(chs[0], out_channels, kernel_size=1)

        init_conv_weights(self)
        self.init_parameters(r0=1.0)

    def init_parameters(self, r0: float = 1.0) -> None:
        for m in self.modules():
            if m is self:
                continue
            if hasattr(m, "init_parameters") and callable(getattr(m, "init_parameters")):
                m.init_parameters(r0=r0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = x

        for i, enc in enumerate(self.enc_blocks):
            h = enc(h)
            if i < len(self.down_blocks):
                skips.append(h)
                h = self.down_blocks[i](h)

        for up, skip in zip(self.up_blocks, reversed(skips)):
            h = up(h, skip)

        logits = self.head(h)
        return self.final_act(logits) if self.final_act is not None else logits
