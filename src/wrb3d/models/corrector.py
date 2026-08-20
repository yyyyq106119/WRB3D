"""Inference-safe case-adaptive correction of the seven high Haar bands."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class CaseAdaptiveWaveletCorrector(nn.Module):
    """Predict bounded per-case gains from inference-available features only."""

    def __init__(
        self,
        low_bottleneck_channels: int,
        high_bottleneck_channels: int,
        *,
        modality_channels: int = 1,
        hidden_dim: int = 128,
        gamma: float = 0.15,
        epsilon: float = 1e-8,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        if low_bottleneck_channels <= 0 or high_bottleneck_channels <= 0:
            raise ValueError("bottleneck channel counts must be positive")
        if modality_channels <= 0 or hidden_dim <= 0:
            raise ValueError("modality_channels and hidden_dim must be positive")
        if not math.isfinite(float(gamma)) or float(gamma) <= 0:
            raise ValueError("corrector gamma must be finite and positive")
        self.modality_channels = int(modality_channels)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        input_dim = int(low_bottleneck_channels) + int(high_bottleneck_channels) + 14
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 7),
        )
        if identity_init:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _gap(feature: Tensor) -> Tensor:
        if feature.ndim != 5:
            raise ValueError("corrector bottleneck features must be [B,C,D,H,W]")
        return feature.float().mean(dim=(-3, -2, -1))

    def _band_log_energy(self, value: Tensor) -> Tensor:
        if value.ndim != 5 or value.shape[1] != 7 * self.modality_channels:
            raise ValueError("corrector high tensor must contain seven bands")
        b, _, d, h, w = value.shape
        bands = value.float().reshape(b, 7, self.modality_channels, d, h, w)
        return torch.log(bands.square().mean(dim=(2, 3, 4, 5)) + self.epsilon)

    def forward(
        self,
        low_bottleneck: Tensor,
        high_bottleneck: Tensor,
        raw_high_residual: Tensor,
        mri_high: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if raw_high_residual.shape != mri_high.shape:
            raise ValueError("raw high residual and MRI high tensors must match")
        features = torch.cat(
            (
                self._gap(low_bottleneck),
                self._gap(high_bottleneck),
                self._band_log_energy(raw_high_residual),
                self._band_log_energy(mri_high),
            ),
            dim=1,
        )
        logits = self.mlp(features)
        gains = torch.exp(self.gamma * torch.tanh(logits))
        expanded = gains.repeat_interleave(self.modality_channels, dim=1)
        corrected = raw_high_residual * expanded.to(raw_high_residual.dtype).reshape(
            raw_high_residual.shape[0], -1, 1, 1, 1
        )
        return corrected, gains

    @property
    def gain_bounds(self) -> tuple[float, float]:
        return math.exp(-self.gamma), math.exp(self.gamma)
