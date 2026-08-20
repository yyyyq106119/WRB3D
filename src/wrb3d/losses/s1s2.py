"""Hotspot and directional high-band losses for the S1/S2 experiment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class AuxiliaryWeights:
    hotspot: float = 0.0
    underestimation: float = 0.0
    aligned_amplitude: float = 0.0
    orthogonal_error: float = 0.0


def _band_view(value: Tensor) -> Tensor:
    if value.ndim != 5 or value.shape[1] % 7 != 0:
        raise ValueError("high residual must be [B,7*C,D,H,W]")
    b, channels, d, h, w = value.shape
    return value.reshape(b, 7, channels // 7, d, h, w)


def build_hotspot_mask(
    target_pet: Tensor,
    *,
    roi: Tensor | None = None,
    foreground_threshold: float = 1e-6,
    quantile: float = 0.90,
    temperature_scale: float = 0.05,
    minimum_foreground_fraction: float = 0.01,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Build detached per-case soft Q90 hotspot masks."""
    if target_pet.ndim != 5:
        raise ValueError("target PET must be [B,C,D,H,W]")
    if not 0.0 < quantile < 1.0:
        raise ValueError("hotspot quantile must be within (0,1)")
    detached = target_pet.detach().float()
    masks: list[Tensor] = []
    valid_rows: list[bool] = []
    thresholds: list[Tensor] = []
    temperatures: list[Tensor] = []
    fractions: list[Tensor] = []
    for index in range(detached.shape[0]):
        case = detached[index]
        if roi is not None:
            candidate = roi[index].detach().to(device=case.device).bool()
            foreground = candidate if bool(candidate.any()) else case > foreground_threshold
        else:
            foreground = case > foreground_threshold
        fraction = foreground.float().mean()
        valid = bool(fraction >= minimum_foreground_fraction) and bool(foreground.any())
        if valid:
            values = case[foreground]
            tau = torch.quantile(values, quantile)
            q95 = torch.quantile(values, 0.95)
            q50 = torch.quantile(values, 0.50)
            delta = torch.maximum(
                temperature_scale * (q95 - q50),
                torch.tensor(1e-4, device=case.device),
            )
            mask = foreground.float() * torch.sigmoid((case - tau) / delta)
        else:
            tau = torch.tensor(float("nan"), device=case.device)
            delta = torch.tensor(float("nan"), device=case.device)
            mask = torch.zeros_like(case)
        masks.append(mask)
        valid_rows.append(valid)
        thresholds.append(tau)
        temperatures.append(delta)
        fractions.append(fraction)
    mask_tensor = torch.stack(masks, dim=0).detach()
    valid_tensor = torch.tensor(valid_rows, device=target_pet.device, dtype=torch.bool)
    return mask_tensor, valid_tensor, {
        "hotspot_threshold": torch.stack(thresholds),
        "hotspot_temperature": torch.stack(temperatures),
        "foreground_fraction": torch.stack(fractions),
    }


def hotspot_losses(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    valid_cases: Tensor,
    *,
    charbonnier_epsilon: float = 1e-3,
) -> dict[str, Tensor]:
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("prediction, target, and hotspot mask must match")
    flat_mask = mask.float().flatten(1)
    denominator = flat_mask.sum(dim=1).clamp_min(1e-8)
    error = prediction.float() - target.float()
    hot = (
        flat_mask
        * torch.sqrt(error.square() + float(charbonnier_epsilon) ** 2).flatten(1)
    ).sum(dim=1) / denominator
    under = (flat_mask * F.relu(-error).flatten(1)).sum(dim=1) / denominator
    over = (flat_mask * F.relu(error).flatten(1)).sum(dim=1) / denominator
    valid = valid_cases.float()
    normalizer = valid.sum().clamp_min(1.0)
    return {
        "hotspot": (hot * valid).sum() / normalizer,
        "underestimation": (under * valid).sum() / normalizer,
        "overestimation": (over * valid).sum() / normalizer,
        "valid_case_fraction": valid.mean(),
    }


def aligned_high_losses(
    prediction: Tensor,
    target: Tensor,
    band_epsilon: Tensor,
    *,
    smooth_l1_beta: float = 0.1,
) -> dict[str, Tensor]:
    pred = _band_view(prediction).float()
    truth = _band_view(target).float()
    if pred.shape != truth.shape:
        raise ValueError("predicted and target high residuals must match")
    pred_centered = pred - pred.mean(dim=(-3, -2, -1), keepdim=True)
    truth_centered = truth - truth.mean(dim=(-3, -2, -1), keepdim=True)
    spatial_n = pred.shape[-3] * pred.shape[-2] * pred.shape[-1]
    denominator_energy = truth_centered.square().sum(dim=(-3, -2, -1))
    eps = torch.as_tensor(band_epsilon, device=pred.device, dtype=torch.float32).flatten()
    if eps.numel() != 7 or not torch.isfinite(eps).all() or torch.any(eps <= 0):
        raise ValueError("band_epsilon must contain seven finite positive values")
    eps_bc = eps.reshape(1, 7, 1)
    valid = denominator_energy >= float(spatial_n) * eps_bc
    numerator = (pred_centered * truth_centered).sum(dim=(-3, -2, -1))
    beta = numerator / (denominator_energy.detach() + float(spatial_n) * eps_bc)
    beta_loss = F.smooth_l1_loss(
        beta,
        torch.ones_like(beta),
        reduction="none",
        beta=float(smooth_l1_beta),
    )
    orthogonal = pred_centered - beta.detach().reshape(
        *beta.shape, 1, 1, 1
    ) * truth_centered
    eta = orthogonal.abs().mean(dim=(-3, -2, -1)) / (
        truth_centered.abs().mean(dim=(-3, -2, -1)).detach() + eps_bc.sqrt()
    )
    valid_float = valid.float()
    normalizer = valid_float.sum().clamp_min(1.0)
    return {
        "aligned_amplitude": (beta_loss * valid_float).sum() / normalizer,
        "orthogonal_error": (eta * valid_float).sum() / normalizer,
        "beta": beta,
        "orthogonal_ratio": eta,
        "valid": valid,
        "valid_band_fraction": valid_float.mean(),
    }


def gain_regularization(gains: Tensor | None) -> Tensor:
    if gains is None:
        raise ValueError("gain regularization requires explicit S1 gains")
    return torch.log(gains.float()).square().mean()
