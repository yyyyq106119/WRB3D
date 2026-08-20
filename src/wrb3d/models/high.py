"""Seven-band high-frequency clean residual endpoint predictor."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from .common import ConditionEmbedding, ConditionedUNet3D

CONDITION_MODES = {"none", "feature_gating", "cross_frequency_attention", "both"}


class HighResidualBridgeNet(nn.Module):
    """Predict all seven ``r_B^H`` subbands in one synchronized bridge."""

    def __init__(
        self,
        input_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256),
        low_feature_channels: Sequence[int] = (32, 64, 128, 256),
        condition_dim: int = 128,
        num_timesteps: int = 1000,
        condition_mode: str = "feature_gating",
        include_mri_low: bool = True,
        output_init_std: float = 1e-3,
    ) -> None:
        super().__init__()
        if condition_mode not in CONDITION_MODES:
            raise ValueError(f"unsupported low-to-high condition mode={condition_mode!r}")
        self.input_channels = int(input_channels)
        self.condition_mode = condition_mode
        self.include_mri_low = bool(include_mri_low)
        self.uses_low_condition = condition_mode != "none"
        self.condition_embedding = ConditionEmbedding(num_timesteps, 7, condition_dim)
        # residual H (7C), MRI H (7C), residual-state L (C), predicted residual L (C),
        # and, in the formal residual model, MRI L (C): 17C total.
        if self.uses_low_condition:
            high_input_channels = (17 if self.include_mri_low else 16) * self.input_channels
        else:
            high_input_channels = 14 * self.input_channels
        use_features = condition_mode in {"feature_gating", "both"}
        use_attention = condition_mode in {"cross_frequency_attention", "both"}
        self.unet = ConditionedUNet3D(
            high_input_channels,
            7 * self.input_channels,
            channels,
            condition_dim,
            auxiliary_channels=low_feature_channels,
            feature_gating=use_features,
            cross_attention=use_attention,
            output_init_std=output_init_std,
        )

    def forward(
        self,
        residual_state_high: Tensor,
        mri_high: Tensor,
        mri_low: Tensor,
        residual_state_low: Tensor,
        predicted_low_residual: Tensor,
        low_features: Sequence[Tensor],
        t: Tensor,
        covariance: float | Tensor,
        *,
        return_features: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        expected = 7 * self.input_channels
        if residual_state_high.shape[1] != expected or mri_high.shape[1] != expected:
            raise ValueError("high state and MRI condition must contain seven subbands")
        inputs = [residual_state_high, mri_high]
        if self.uses_low_condition:
            if self.include_mri_low:
                inputs.append(mri_low)
            inputs.extend((residual_state_low, predicted_low_residual))
        condition = self.condition_embedding(t, covariance, residual_state_high.dtype)
        auxiliary = low_features if self.uses_low_condition else None
        return self.unet(
            torch.cat(inputs, dim=1),
            condition,
            auxiliary=auxiliary,
            return_features=return_features,
        )
