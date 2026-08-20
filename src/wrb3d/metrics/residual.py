"""Residual endpoint metrics with safe batch-size-one reductions."""

from __future__ import annotations

import torch
from torch import Tensor


def _pearson(prediction: Tensor, target: Tensor, epsilon: float = 1e-8) -> Tensor:
    x = prediction.float().reshape(prediction.shape[0], -1)
    y = target.float().reshape(target.shape[0], -1)
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    value = (x * y).sum(dim=1) / (
        torch.sqrt(x.square().sum(dim=1) * y.square().sum(dim=1)).clamp_min(epsilon)
    )
    return value.mean()


def residual_metrics(prediction: Tensor, target: Tensor, epsilon: float = 1e-8) -> dict[str, Tensor]:
    pred_energy = prediction.float().square().mean()
    target_energy = target.float().square().mean()
    nonzero_threshold = 1e-6
    return {
        "mae": (prediction - target).abs().mean(),
        "mse": (prediction - target).square().mean(),
        "pearson": _pearson(prediction, target, epsilon),
        "energy_ratio": pred_energy / target_energy.clamp_min(epsilon),
        "sign_agreement": (torch.sign(prediction) == torch.sign(target)).float().mean(),
        "nonzero_ratio": (prediction.abs() > nonzero_threshold).float().mean(),
        "predicted_energy": pred_energy,
        "gt_energy": target_energy,
    }


def high_band_metrics(prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
    if prediction.shape != target.shape or prediction.shape[1] % 7 != 0:
        raise ValueError("high tensors must match and contain seven bands")
    b, channels, d, h, w = prediction.shape
    c = channels // 7
    pred = prediction.reshape(b, 7, c, d, h, w)
    truth = target.reshape(b, 7, c, d, h, w)
    output: dict[str, Tensor] = {}
    for band in range(7):
        for name, value in residual_metrics(pred[:, band], truth[:, band]).items():
            output[f"high_band_{band}_{name}"] = value
    return output

