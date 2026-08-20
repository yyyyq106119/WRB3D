"""Formal validation metrics for S1/S2; no metric participates in training."""

from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F

from ..losses import aligned_high_losses, build_hotspot_mask
from .image import image_metrics


def _ssim_and_contrast(
    prediction: Tensor, target: Tensor, data_range: float, window_size: int = 11
) -> tuple[Tensor, Tensor]:
    window = min(window_size, *prediction.shape[-3:])
    if window % 2 == 0:
        window -= 1
    window = max(window, 1)
    padding = window // 2
    x = prediction.float()
    y = target.float()
    mu_x = F.avg_pool3d(x, window, stride=1, padding=padding)
    mu_y = F.avg_pool3d(y, window, stride=1, padding=padding)
    sigma_x = F.avg_pool3d(x.square(), window, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool3d(y.square(), window, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool3d(x * y, window, 1, padding) - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    contrast = (2.0 * sigma_xy + c2) / (sigma_x + sigma_y + c2).clamp_min(1e-12)
    luminance = (2.0 * mu_x * mu_y + c1) / (
        mu_x.square() + mu_y.square() + c1
    ).clamp_min(1e-12)
    return (luminance * contrast).mean(), contrast.mean()


def multi_scale_structural_similarity_3d(
    prediction: Tensor, target: Tensor, *, data_range: float = 1.0
) -> Tensor:
    """Legacy-compatible five-weight 3-D MS-SSIM on the raw [0,1] volumes."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("MS-SSIM inputs must be matching [B,C,D,H,W] tensors")
    canonical = torch.tensor(
        [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
        device=prediction.device,
        dtype=torch.float32,
    )
    x = prediction.float()
    y = target.float()
    values: list[Tensor] = []
    contrasts: list[Tensor] = []
    for level in range(5):
        ssim, contrast = _ssim_and_contrast(x, y, float(data_range))
        values.append(ssim.clamp_min(1e-8))
        contrasts.append(contrast.clamp_min(1e-8))
        if level == 4 or min(x.shape[-3:]) < 2:
            break
        x = F.avg_pool3d(x, kernel_size=2, stride=2, ceil_mode=True)
        y = F.avg_pool3d(y, kernel_size=2, stride=2, ceil_mode=True)
    weights = canonical[: len(values)]
    weights = weights / weights.sum()
    if len(values) == 1:
        return values[0]
    terms = [
        contrasts[index].pow(weights[index])
        for index in range(len(values) - 1)
    ]
    terms.append(values[-1].pow(weights[-1]))
    return torch.stack(terms).prod()


def _safe_ratio(numerator: Tensor, denominator: Tensor, epsilon: float = 1e-8) -> Tensor:
    return numerator / denominator.clamp_min(epsilon)


def _centered_band_metrics(prediction: Tensor, target: Tensor, prefix: str) -> dict[str, Tensor]:
    if prediction.ndim != 5 or prediction.shape != target.shape:
        raise ValueError("residual metric tensors must match [B,C,D,H,W]")
    pred = prediction.float().flatten(1)
    truth = target.float().flatten(1)
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    truth_c = truth - truth.mean(dim=1, keepdim=True)
    denominator = truth_c.square().sum(dim=1)
    covariance = (pred_c * truth_c).sum(dim=1)
    slope = covariance / denominator.clamp_min(1e-8)
    pearson = covariance / torch.sqrt(
        pred_c.square().sum(dim=1) * denominator
    ).clamp_min(1e-8)
    energy_ratio = _safe_ratio(pred.square().mean(dim=1), truth.square().mean(dim=1))
    return {
        f"{prefix}_mae": (pred - truth).abs().mean(dim=1).mean(),
        f"{prefix}_pearson": pearson.mean(),
        f"{prefix}_slope": slope.mean(),
        f"{prefix}_energy_ratio": energy_ratio.mean(),
    }


def _hotspot_metrics(prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
    soft, valid, metadata = build_hotspot_mask(target)
    error = prediction.float() - target.float()
    mask = soft.float()
    denominator = mask.flatten(1).sum(dim=1).clamp_min(1e-8)
    mae = (mask * error.abs()).flatten(1).sum(dim=1) / denominator
    rmse = torch.sqrt((mask * error.square()).flatten(1).sum(dim=1) / denominator)
    under = (mask * F.relu(-error)).flatten(1).sum(dim=1) / denominator
    over = (mask * F.relu(error)).flatten(1).sum(dim=1) / denominator
    mean_bias = (mask * error).flatten(1).sum(dim=1) / denominator
    rows: dict[str, list[Tensor]] = {
        "hotspot_peak_intensity_bias": [],
        "hotspot_top10_pred_gt_mean_ratio": [],
        "hotspot_top5_pred_gt_mean_ratio": [],
        "hotspot_top1_pred_gt_mean_ratio": [],
        "hotspot_false_positive_energy": [],
        "hotspot_gt_recall": [],
    }
    for index in range(target.shape[0]):
        truth = target[index].float()
        pred = prediction[index].float()
        foreground = truth > 1e-6
        if not bool(valid[index]):
            for values in rows.values():
                values.append(torch.tensor(0.0, device=target.device))
            continue
        truth_values = truth[foreground]
        pred_values = pred[foreground]
        rows["hotspot_peak_intensity_bias"].append(
            pred_values.max() - truth_values.max()
        )
        for fraction, key in (
            (0.10, "hotspot_top10_pred_gt_mean_ratio"),
            (0.05, "hotspot_top5_pred_gt_mean_ratio"),
            (0.01, "hotspot_top1_pred_gt_mean_ratio"),
        ):
            threshold = torch.quantile(truth_values, 1.0 - fraction)
            selected = foreground & (truth >= threshold)
            rows[key].append(_safe_ratio(pred[selected].mean(), truth[selected].mean()))
        gt_threshold = torch.quantile(truth_values, 0.90)
        pred_threshold = torch.quantile(pred_values, 0.90)
        gt_hot = foreground & (truth >= gt_threshold)
        pred_hot = foreground & (pred >= pred_threshold)
        rows["hotspot_gt_recall"].append(
            _safe_ratio((gt_hot & pred_hot).float().sum(), gt_hot.float().sum())
        )
        rows["hotspot_false_positive_energy"].append(
            _safe_ratio(
                pred[pred_hot & ~gt_hot].square().sum(),
                truth[gt_hot].square().sum(),
            )
        )
    valid_float = valid.float()
    normalizer = valid_float.sum().clamp_min(1.0)
    output = {
        "hotspot_mae": (mae * valid_float).sum() / normalizer,
        "hotspot_rmse": (rmse * valid_float).sum() / normalizer,
        "hotspot_underestimation": (under * valid_float).sum() / normalizer,
        "hotspot_overestimation": (over * valid_float).sum() / normalizer,
        "hotspot_mean_intensity_bias": (mean_bias * valid_float).sum() / normalizer,
        "hotspot_valid_case_fraction": valid_float.mean(),
        "hotspot_foreground_fraction": metadata["foreground_fraction"].mean(),
    }
    for key, values in rows.items():
        stacked = torch.stack(values)
        output[key] = (stacked * valid_float).sum() / normalizer
    return output


def _intensity_calibration(prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
    boundaries = (0.0, 0.50, 0.80, 0.90, 0.95, 0.99, 1.0)
    output: dict[str, list[Tensor]] = {}
    for index in range(target.shape[0]):
        truth = target[index].float()
        pred = prediction[index].float()
        foreground = truth > 1e-6
        if not bool(foreground.any()):
            continue
        values = truth[foreground]
        quantiles = [torch.quantile(values, value) for value in boundaries]
        for bin_index, (lower_q, upper_q) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            lower = quantiles[bin_index]
            upper = quantiles[bin_index + 1]
            selected = foreground & (truth >= lower)
            if upper_q < 1.0:
                selected &= truth < upper
            else:
                selected &= truth <= upper
            label = f"intensity_q{int(lower_q*100):02d}_{int(upper_q*100):02d}"
            if bool(selected.any()):
                output.setdefault(f"{label}_pred_mean", []).append(pred[selected].mean())
                output.setdefault(f"{label}_gt_mean", []).append(truth[selected].mean())
                output.setdefault(f"{label}_bias", []).append((pred[selected] - truth[selected]).mean())
    return {key: torch.stack(values).mean() for key, values in output.items()}


def formal_case_metrics(
    model: torch.nn.Module,
    mri: Tensor,
    pet: Tensor,
    inference: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Compute the complete raw-PET and wavelet audit for a validation case."""
    prediction = inference["B_raw"]
    output = image_metrics(prediction, pet, data_range=1.0)
    output["msssim3d"] = multi_scale_structural_similarity_3d(
        prediction, pet, data_range=1.0
    )
    output["raw_out_of_range_ratio"] = (
        (prediction < 0) | (prediction > 1)
    ).float().mean()
    output.update(_hotspot_metrics(prediction, pet))
    output.update(_intensity_calibration(prediction, pet))
    mri_low, mri_high, _ = model.dwt(mri)
    pred_low, pred_high, _ = model.dwt(prediction)
    gt_low, gt_high, _ = model.dwt(pet)
    pred_low_residual = pred_low - mri_low
    pred_high_residual = pred_high - mri_high
    gt_low_residual = gt_low - mri_low
    gt_high_residual = gt_high - mri_high
    output.update(
        _centered_band_metrics(pred_low_residual, gt_low_residual, "low_residual")
    )
    b, channels, d, h, w = pred_high_residual.shape
    modality_channels = channels // 7
    pred_bands = pred_high_residual.reshape(b, 7, modality_channels, d, h, w)
    gt_bands = gt_high_residual.reshape(b, 7, modality_channels, d, h, w)
    aligned = aligned_high_losses(
        pred_high_residual,
        gt_high_residual,
        model.aligned_band_epsilon,
        smooth_l1_beta=model.aligned_smooth_l1_beta,
    )
    for band in range(7):
        output.update(
            _centered_band_metrics(
                pred_bands[:, band], gt_bands[:, band], f"high_band_{band}"
            )
        )
        output[f"high_band_{band}_aligned_amplitude_beta"] = aligned["beta"][
            :, band
        ].mean()
        output[f"high_band_{band}_orthogonal_error_ratio"] = aligned[
            "orthogonal_ratio"
        ][:, band].mean()
        output[f"high_band_{band}_valid_fraction"] = aligned["valid"][
            :, band
        ].float().mean()
        output[f"high_band_{band}_gt_energy"] = gt_bands[:, band].square().mean()
    gains = inference.get("corrector_gains")
    if gains is not None:
        raw = inference["high_pred_raw"]
        corrected = inference["high_pred_corrected"]
        raw_bands = raw.reshape(b, 7, modality_channels, d, h, w)
        corrected_bands = corrected.reshape(b, 7, modality_channels, d, h, w)
        for band in range(7):
            output[f"corrector_gain_band_{band}"] = gains[:, band].mean()
            output[f"corrector_gain2_band_{band}"] = gains[:, band].square().mean()
            output.update(
                _centered_band_metrics(
                    raw_bands[:, band], gt_bands[:, band], f"high_raw_band_{band}"
                )
            )
            output.update(
                _centered_band_metrics(
                    corrected_bands[:, band],
                    gt_bands[:, band],
                    f"high_corrected_band_{band}",
                )
            )
    for key, value in output.items():
        if not torch.isfinite(value):
            raise FloatingPointError(f"non-finite formal metric: {key}")
    return output
