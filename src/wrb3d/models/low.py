"""Low-frequency clean residual endpoint predictor."""

from __future__ import annotations

from typing import Sequence

from torch import Tensor, nn
import torch

from .common import ConditionEmbedding, ConditionedUNet3D


class LowResidualBridgeNet(nn.Module):
    """Predict ``r_B^L`` from current residual, MRI LLL, and all MRI high bands."""

    def __init__(
        self,
        input_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256),
        condition_dim: int = 128,
        num_timesteps: int = 1000,
        output_init_std: float = 1e-3,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.condition_embedding = ConditionEmbedding(num_timesteps, 1, condition_dim)
        self.unet = ConditionedUNet3D(
            9 * self.input_channels,
            self.input_channels,
            channels,
            condition_dim,
            output_init_std=output_init_std,
        )

    @property
    def feature_channels(self) -> tuple[int, ...]:
        return self.unet.feature_channels

    def forward(
        self,
        residual_state_low: Tensor,
        mri_low: Tensor,
        mri_high: Tensor,
        t: Tensor,
        covariance: float | Tensor,
        *,
        return_features: bool = True,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        if mri_high.shape[1] != 7 * self.input_channels:
            raise ValueError("MRI high tensor must contain seven subbands per modality channel")
        condition = self.condition_embedding(t, covariance, residual_state_low.dtype)
        x = torch.cat((residual_state_low, mri_low, mri_high), dim=1)
        return self.unet(x, condition, return_features=return_features)

