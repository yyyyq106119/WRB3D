"""Shared 3-D residual U-Net components for both frequency branches."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int, num_timesteps: int) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.num_timesteps = int(num_timesteps)

    def forward(self, t: Tensor) -> Tensor:
        half = self.dimension // 2
        if half == 0:
            return t.float().reshape(-1, 1)
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            -scale * torch.arange(half, device=t.device, dtype=torch.float32)
        )
        angles = (t.float() / max(self.num_timesteps, 1)).reshape(-1, 1) * frequencies.reshape(1, -1)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        if embedding.shape[1] < self.dimension:
            embedding = F.pad(embedding, (0, self.dimension - embedding.shape[1]))
        return embedding


def summarize_covariance(
    covariance: float | Tensor,
    *,
    batch_size: int,
    num_bands: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return a ``[B,num_bands]`` log-covariance conditioning tensor."""
    cov = torch.as_tensor(covariance, device=device, dtype=torch.float32)
    if cov.ndim == 0:
        summary = cov.reshape(1, 1).expand(batch_size, num_bands)
    elif cov.ndim == 1:
        if cov.numel() == 1:
            summary = cov.reshape(1, 1).expand(batch_size, num_bands)
        elif cov.numel() == num_bands:
            summary = cov.reshape(1, num_bands).expand(batch_size, num_bands)
        elif cov.numel() % num_bands == 0:
            summary = cov.reshape(num_bands, -1).mean(dim=1).reshape(1, num_bands)
            summary = summary.expand(batch_size, num_bands)
        else:
            raise ValueError("covariance vector cannot be summarized into requested bands")
    elif cov.ndim == 2:
        if cov.shape[0] not in {1, batch_size}:
            raise ValueError("case-wise covariance batch mismatch")
        if cov.shape[1] == num_bands:
            summary = cov
        elif cov.shape[1] % num_bands == 0:
            summary = cov.reshape(cov.shape[0], num_bands, -1).mean(dim=2)
        else:
            raise ValueError("case-wise covariance channel mismatch")
        summary = summary.expand(batch_size, num_bands)
    else:
        if cov.ndim == 4:
            cov = cov.unsqueeze(0)
        if cov.shape[0] not in {1, batch_size}:
            raise ValueError("spatial covariance batch mismatch")
        channel_mean = cov.reshape(cov.shape[0], cov.shape[1], -1).mean(dim=2)
        if channel_mean.shape[1] == num_bands:
            summary = channel_mean
        elif channel_mean.shape[1] % num_bands == 0:
            summary = channel_mean.reshape(channel_mean.shape[0], num_bands, -1).mean(dim=2)
        else:
            raise ValueError("spatial covariance channel mismatch")
        summary = summary.expand(batch_size, num_bands)
    if not torch.isfinite(summary).all() or torch.any(summary < 0):
        raise ValueError("covariance summary must be finite and non-negative")
    return torch.log(summary.clamp_min(1e-12)).to(dtype=dtype)


class ConditionEmbedding(nn.Module):
    def __init__(self, num_timesteps: int, covariance_bands: int, dimension: int = 128) -> None:
        super().__init__()
        self.covariance_bands = int(covariance_bands)
        self.time = SinusoidalTimeEmbedding(dimension, num_timesteps)
        self.time_mlp = nn.Sequential(
            nn.Linear(dimension, dimension), nn.SiLU(), nn.Linear(dimension, dimension)
        )
        self.covariance_mlp = nn.Sequential(
            nn.Linear(self.covariance_bands, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )

    def forward(self, t: Tensor, covariance: float | Tensor, dtype: torch.dtype) -> Tensor:
        cov = summarize_covariance(
            covariance,
            batch_size=t.shape[0],
            num_bands=self.covariance_bands,
            device=t.device,
            dtype=torch.float32,
        )
        return (self.time_mlp(self.time(t)) + self.covariance_mlp(cov)).to(dtype=dtype)


class ConditionedResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.condition = nn.Linear(condition_dim, out_channels)
        self.skip = (
            nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, 1)
        )

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.condition(condition).reshape(condition.shape[0], -1, 1, 1, 1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: Tensor, shape: Sequence[int]) -> Tensor:
        return self.conv(F.interpolate(x, size=tuple(shape), mode="trilinear", align_corners=False))


class BottleneckCrossFrequencyAttention(nn.Module):
    """Optional low-to-high attention used only in its explicit ablation modes."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        heads = 4 if channels % 4 == 0 else 1
        self.high_norm = nn.LayerNorm(channels)
        self.low_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.output = nn.Linear(channels, channels)

    def forward(self, high: Tensor, low: Tensor) -> Tensor:
        if low.shape[-3:] != high.shape[-3:]:
            low = F.interpolate(low, size=high.shape[-3:], mode="trilinear", align_corners=False)
        b, c, d, h, w = high.shape
        query = high.flatten(2).transpose(1, 2)
        key_value = low.flatten(2).transpose(1, 2)
        attended, _ = self.attention(
            self.high_norm(query), self.low_norm(key_value), self.low_norm(key_value), need_weights=False
        )
        return high + self.output(attended).transpose(1, 2).reshape(b, c, d, h, w)


class ConditionedUNet3D(nn.Module):
    """U-Net with optional explicitly gated low-frequency features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        condition_dim: int,
        *,
        auxiliary_channels: Sequence[int] | None = None,
        feature_gating: bool = False,
        cross_attention: bool = False,
        output_init_std: float = 1e-3,
    ) -> None:
        super().__init__()
        self.channels = tuple(int(v) for v in channels)
        if len(self.channels) < 2:
            raise ValueError("U-Net requires at least two channel levels")
        self.stem = nn.Conv3d(in_channels, self.channels[0], 3, padding=1)
        self.encoder = nn.ModuleList(
            [ConditionedResidualBlock3D(ch, ch, condition_dim) for ch in self.channels]
        )
        self.down = nn.ModuleList(
            [Downsample3D(a, b) for a, b in zip(self.channels[:-1], self.channels[1:])]
        )
        self.up = nn.ModuleList(
            [Upsample3D(self.channels[i + 1], self.channels[i]) for i in range(len(self.channels) - 1)]
        )
        self.decoder = nn.ModuleList(
            [
                ConditionedResidualBlock3D(2 * self.channels[i], self.channels[i], condition_dim)
                for i in range(len(self.channels) - 1)
            ]
        )
        self.feature_gating = bool(feature_gating)
        self.cross_attention_enabled = bool(cross_attention)
        if self.feature_gating:
            if auxiliary_channels is None or len(auxiliary_channels) != len(self.channels):
                raise ValueError("feature gating requires one auxiliary channel count per scale")
            self.auxiliary_projection = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.GroupNorm(_groups(int(aux)), int(aux)),
                        nn.Conv3d(int(aux), ch, 1),
                    )
                    for aux, ch in zip(auxiliary_channels, self.channels)
                ]
            )
            initial_logit = math.log(0.1 / 0.9)
            self.auxiliary_gate_logits = nn.Parameter(torch.full((len(self.channels),), initial_logit))
        else:
            self.auxiliary_projection = nn.ModuleList()
            self.register_parameter("auxiliary_gate_logits", None)
        self.cross_attention = (
            BottleneckCrossFrequencyAttention(self.channels[-1])
            if self.cross_attention_enabled
            else None
        )
        self.output = nn.Conv3d(self.channels[0], out_channels, 3, padding=1)
        nn.init.normal_(self.output.weight, mean=0.0, std=float(output_init_std))
        nn.init.zeros_(self.output.bias)

    @property
    def feature_channels(self) -> tuple[int, ...]:
        return self.channels

    def _inject(self, index: int, h: Tensor, auxiliary: Sequence[Tensor] | None) -> Tensor:
        if not self.feature_gating:
            return h
        if auxiliary is None or len(auxiliary) != len(self.channels):
            raise ValueError("feature-gated high branch requires all low multi-scale features")
        value = auxiliary[index]
        if value.shape[-3:] != h.shape[-3:]:
            value = F.interpolate(value, size=h.shape[-3:], mode="trilinear", align_corners=False)
        gate = torch.sigmoid(self.auxiliary_gate_logits[index]).to(dtype=h.dtype)
        return h + gate * self.auxiliary_projection[index](value)

    def forward(
        self,
        x: Tensor,
        condition: Tensor,
        *,
        auxiliary: Sequence[Tensor] | None = None,
        return_features: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        h = self.stem(x)
        features: list[Tensor] = []
        for index, block in enumerate(self.encoder):
            h = block(h, condition)
            h = self._inject(index, h, auxiliary)
            features.append(h)
            if index < len(self.down):
                h = self.down[index](h)
        if self.cross_attention is not None:
            if auxiliary is None:
                raise ValueError("cross-frequency attention requires low features")
            h = self.cross_attention(h, auxiliary[-1])
        for index in range(len(self.channels) - 2, -1, -1):
            h = self.up[index](h, features[index].shape[-3:])
            h = self.decoder[index](torch.cat((h, features[index]), dim=1), condition)
        output = self.output(h)
        return (output, features) if return_features else output

