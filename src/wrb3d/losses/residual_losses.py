"""Minimal formal stage-A objective on raw residual and image outputs."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def charbonnier(error: Tensor, epsilon: float = 1e-3) -> Tensor:
    return torch.sqrt(error.float().square() + float(epsilon) ** 2).mean().to(error.dtype)


class ResidualBridgeLoss(nn.Module):
    """Low residual + normalized high residual + raw PET + raw range loss."""

    def __init__(
        self,
        *,
        lambda_low: float = 1.0,
        lambda_high: float = 1.0,
        lambda_image: float = 2.0,
        lambda_range: float = 0.05,
        endpoint_loss: str = "charbonnier",
        high_band_stds: Tensor | None = None,
        max_high_weight: float = 10.0,
        epsilon: float = 1e-3,
    ) -> None:
        super().__init__()
        if endpoint_loss not in {"charbonnier", "l1"}:
            raise ValueError("endpoint_loss must be charbonnier or l1")
        self.lambda_low = float(lambda_low)
        self.lambda_high = float(lambda_high)
        self.lambda_image = float(lambda_image)
        self.lambda_range = float(lambda_range)
        self.endpoint_loss = endpoint_loss
        self.max_high_weight = float(max_high_weight)
        self.epsilon = float(epsilon)
        if high_band_stds is None:
            high_band_stds = torch.ones(7, dtype=torch.float32)
        stds = torch.as_tensor(high_band_stds, dtype=torch.float32).flatten()
        if stds.numel() != 7 or not torch.isfinite(stds).all() or torch.any(stds <= 0):
            raise ValueError("high_band_stds must contain seven positive train-derived values")
        self.register_buffer("high_band_stds", stds)

    def _endpoint(self, error: Tensor) -> Tensor:
        return charbonnier(error, self.epsilon) if self.endpoint_loss == "charbonnier" else error.abs().mean()

    def forward(
        self,
        predicted_low_residual: Tensor,
        target_low_residual: Tensor,
        predicted_high_residual: Tensor,
        target_high_residual: Tensor,
        raw_pet: Tensor,
        target_pet: Tensor,
        *,
        image_for_loss: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        components = self.components(
            predicted_low_residual,
            target_low_residual,
            predicted_high_residual,
            target_high_residual,
            raw_pet,
            target_pet,
            image_for_loss=image_for_loss,
        )
        low = components["low"]
        high = components["high"]
        raw_image = components["raw_image"]
        image = components["image"]
        value_range = components["range"]
        per_band = components["per_band"]
        total = components["base_total"]
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite formal residual bridge loss")
        logs: dict[str, Tensor] = {
            "loss_total": total.detach(),
            "loss_base": total.detach(),
            "loss_low_residual": low.detach(),
            "loss_high_residual": high.detach(),
            "loss_raw_image": raw_image.detach(),
            "loss_image_supervised": image.detach(),
            "image_supervision_is_projected": torch.tensor(
                image_for_loss is not None, device=raw_pet.device
            ),
            "loss_range": value_range.detach(),
            "raw_out_of_range_ratio": ((raw_pet < 0) | (raw_pet > 1)).float().mean().detach(),
        }
        for index, value in enumerate(per_band):
            logs[f"high_residual_band_{index}_loss"] = value.detach()
        return total, logs

    def components(
        self,
        predicted_low_residual: Tensor,
        target_low_residual: Tensor,
        predicted_high_residual: Tensor,
        target_high_residual: Tensor,
        raw_pet: Tensor,
        target_pet: Tensor,
        *,
        image_for_loss: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return graph-connected base components for gradient calibration."""
        if predicted_high_residual.shape[1] % 7 != 0:
            raise ValueError("high residual channels must be divisible by seven")
        low = self._endpoint(predicted_low_residual - target_low_residual)
        b, channels, d, h, w = predicted_high_residual.shape
        modality_channels = channels // 7
        high_error = (predicted_high_residual - target_high_residual).reshape(
            b, 7, modality_channels, d, h, w
        )
        weights = (1.0 / self.high_band_stds.clamp_min(1e-12)).clamp_max(
            self.max_high_weight
        ).to(device=high_error.device, dtype=high_error.dtype)
        if self.endpoint_loss == "charbonnier":
            per_band = torch.sqrt(high_error.float().square() + self.epsilon**2).mean(
                dim=(0, 2, 3, 4, 5)
            ).to(high_error.dtype)
        else:
            per_band = high_error.abs().mean(dim=(0, 2, 3, 4, 5))
        high = (per_band * weights).mean()
        raw_image = charbonnier(raw_pet - target_pet, self.epsilon)
        supervised_image = raw_pet if image_for_loss is None else image_for_loss
        image = raw_image if image_for_loss is None else charbonnier(
            supervised_image - target_pet, self.epsilon
        )
        value_range = F.relu(-raw_pet).mean() + F.relu(raw_pet - 1.0).mean()
        base_total = (
            self.lambda_low * low
            + self.lambda_high * high
            + self.lambda_image * image
            + self.lambda_range * value_range
        )
        return {
            "low": low,
            "high": high,
            "raw_image": raw_image,
            "image": image,
            "range": value_range,
            "per_band": per_band,
            "base_total": base_total,
            "weighted_image": self.lambda_image * image,
            "weighted_high": self.lambda_high * high,
        }
