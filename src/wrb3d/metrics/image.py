"""Patient-safe 3-D image metrics with an explicit intensity data range."""

from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F


def _window_size(shape: tuple[int, int, int], requested: int) -> int:
    value = min(int(requested), *shape)
    if value % 2 == 0:
        value -= 1
    return max(value, 1)


def structural_similarity_3d(
    prediction: Tensor,
    target: Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
) -> Tensor:
    """Return mean local 3-D SSIM over a `[B,C,D,H,W]` batch."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("SSIM inputs must be matching [B,C,D,H,W] tensors")
    if not math.isfinite(float(data_range)) or float(data_range) <= 0:
        raise ValueError("SSIM data_range must be finite and positive")
    x = prediction.float()
    y = target.float()
    window = _window_size(tuple(int(value) for value in x.shape[-3:]), window_size)
    padding = window // 2
    pool = lambda value: F.avg_pool3d(value, window, stride=1, padding=padding)
    mu_x = pool(x)
    mu_y = pool(y)
    sigma_x = pool(x.square()) - mu_x.square()
    sigma_y = pool(y.square()) - mu_y.square()
    sigma_xy = pool(x * y) - mu_x * mu_y
    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(torch.finfo(torch.float32).eps)).mean()


def image_metrics(
    prediction: Tensor, target: Tensor, *, data_range: float = 1.0
) -> dict[str, Tensor]:
    """Return raw-volume MAE/MSE/PSNR/3-D SSIM averaged over patients."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("image metric inputs must be matching [B,C,D,H,W] tensors")
    error = prediction.float() - target.float()
    per_case_mse = error.square().flatten(1).mean(dim=1)
    psnr = torch.where(
        per_case_mse == 0,
        torch.full_like(per_case_mse, float("inf")),
        10.0
        * torch.log10(
            torch.tensor(float(data_range) ** 2, device=error.device) / per_case_mse
        ),
    )
    return {
        "mae": error.abs().flatten(1).mean(dim=1).mean(),
        "mse": per_case_mse.mean(),
        "psnr": psnr.mean(),
        "ssim3d": structural_similarity_3d(
            prediction, target, data_range=float(data_range)
        ),
    }
