# -*- coding: utf-8 -*-
"""Quaternion helpers for rotation fields (SO(3))."""

from __future__ import annotations

import torch

from .utils import EPS


def normalize_quaternion(q: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Normalize quaternions along the last dimension."""
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions (...,4) to rotation matrices (...,3,3)."""
    q = normalize_quaternion(q)
    w, x, y, z = q.unbind(dim=-1)

    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    m00 = ww + xx - yy - zz
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)

    m10 = 2 * (xy + wz)
    m11 = ww - xx + yy - zz
    m12 = 2 * (yz - wx)

    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = ww - xx - yy + zz

    return torch.stack(
        [
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1),
        ],
        dim=-2,
    )


def slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Spherical linear interpolation between unit quaternions.

    Args:
        q0: (...,4) start quaternion.
        q1: (...,4) end quaternion.
        t:  (...,1) or broadcastable interpolation factor in [0,1].

    Returns:
        (...,4) interpolated quaternion.
    """
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1_adj = torch.where(dot < 0.0, -q1, q1)
    dot = torch.abs(dot)

    near = dot > (1.0 - 1e-6)
    q_lerp = normalize_quaternion((1.0 - t) * q0 + t * q1_adj, eps=eps)

    theta = torch.acos(dot.clamp(-1.0 + eps, 1.0 - eps))
    sin_theta = torch.sin(theta)
    w0 = torch.sin((1.0 - t) * theta) / (sin_theta + eps)
    w1 = torch.sin(t * theta) / (sin_theta + eps)
    q_slerp = w0 * q0 + w1 * q1_adj

    return torch.where(near, q_lerp, q_slerp)
