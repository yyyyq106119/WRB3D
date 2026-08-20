"""Audited controls for the post-S2 PSNR refinement experiment."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..losses import AuxiliaryWeights
from .checkpoint import load_checkpoint
from .ema import EMA
from .s1s2 import sha256_file


_AUXILIARY_NAMES = (
    "hotspot",
    "underestimation",
    "aligned_amplitude",
    "orthogonal_error",
)


def validated_auxiliary_weights(values: dict[str, Any] | None) -> dict[str, float]:
    """Validate the frozen S2 auxiliary weights inherited by S3."""
    if not isinstance(values, dict) or set(values) != set(_AUXILIARY_NAMES):
        raise RuntimeError(
            "S3 requires all four calibrated S2 auxiliary weights in its source checkpoint"
        )
    output = {name: float(values[name]) for name in _AUXILIARY_NAMES}
    if any(not math.isfinite(value) or value <= 0.0 for value in output.values()):
        raise RuntimeError("S3 source auxiliary weights must be finite and positive")
    return output


def load_ema_as_base_initialization(
    path: str | Path,
    model: nn.Module,
    *,
    config: dict[str, Any],
    expected_epoch: int,
    expected_experiment_id: str = "S2",
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Copy a verified embedded EMA into Base without restoring training state."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"S3 source checkpoint does not exist: {path}")
    source_ema = EMA(model)
    restored = load_checkpoint(
        path,
        model,
        ema=source_ema,
        config=config,
        map_location=map_location,
    )
    if not restored["ema_loaded"]:
        raise RuntimeError("S3 source checkpoint has no embedded EMA state")
    if int(restored["epoch"]) != int(expected_epoch):
        raise RuntimeError(
            f"S3 source epoch mismatch: {restored['epoch']} != {int(expected_epoch)}"
        )
    source_config = restored.get("checkpoint_config")
    if not isinstance(source_config, dict):
        raise RuntimeError("S3 source checkpoint has no resolved source configuration")
    source_experiment = source_config.get("experiment", {})
    if str(source_experiment.get("id")) != str(expected_experiment_id):
        raise RuntimeError(
            "S3 must start from the predeclared S2 experiment checkpoint; "
            f"observed experiment id {source_experiment.get('id')!r}"
        )
    corrector = source_config.get("model", {}).get(
        "case_adaptive_wavelet_corrector", {}
    )
    if bool(corrector.get("enabled", False)):
        raise RuntimeError("S3 PSNR refinement source must be the no-corrector S2 model")
    model.load_state_dict(source_ema.module.state_dict(), strict=True)
    calibrated = validated_auxiliary_weights(
        restored["experiment_state"].get("calibrated_auxiliary_weights")
    )
    return {
        "kind": "s2_embedded_ema_to_new_base",
        "path": str(path.resolve()),
        "checkpoint_sha256": sha256_file(path),
        "source_epoch": int(restored["epoch"]),
        "source_global_step": int(restored["global_step"]),
        "source_experiment_id": str(source_experiment.get("id")),
        "source_experiment_name": str(source_experiment.get("name")),
        "ema_decay": float(source_ema.decay),
        "ema_num_updates": int(source_ema.num_updates),
        "calibrated_auxiliary_weights": calibrated,
        "optimizer_restored": False,
        "scheduler_restored": False,
        "scaler_restored": False,
        "global_step_reset": True,
        "new_ema_initialized_from_new_base": True,
    }


def _gradient_norm(
    loss: Tensor, parameters: Sequence[nn.Parameter], *, retain_graph: bool
) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    total = torch.zeros((), device=loss.device, dtype=torch.float64)
    for gradient in gradients:
        if gradient is not None:
            total += gradient.detach().double().square().sum()
    return float(total.sqrt().cpu())


def collect_mse_gradient_ratios(
    model: nn.Module,
    batches: Iterable[dict[str, Any]],
    covariance_low: Tensor,
    covariance_high: Tensor,
    *,
    device: torch.device,
    maximum_batches: int,
    t_sampling: str,
    endpoint_probability: float,
) -> dict[str, Any]:
    """Measure raw-MSE/reference gradient ratios without an optimizer step."""
    model.train()
    if any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules()):
        raise RuntimeError("S3 gradient calibration refuses mutable BatchNorm statistics")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    ratios: list[float] = []
    reference_norms: list[float] = []
    mse_norms: list[float] = []
    case_ids: list[str] = []
    for batch_index, batch in enumerate(batches):
        if batch_index >= maximum_batches:
            break
        mri = batch["mri"].to(device, non_blocking=True)
        pet = batch["pet"].to(device, non_blocking=True)
        roi = batch.get("roi")
        if roi is not None:
            roi = roi.to(device, non_blocking=True)
        result = model.forward_train(
            mri,
            pet,
            covariance_low,
            covariance_high,
            t_sampling=t_sampling,
            endpoint_probability=endpoint_probability,
            roi=roi,
            auxiliary_weights=AuxiliaryWeights(),
        )
        reference = result["loss_components"]["weighted_image"]
        mse = F.mse_loss(result["B_raw"].float(), pet.float())
        reference_norm = _gradient_norm(reference, parameters, retain_graph=True)
        mse_norm = _gradient_norm(mse, parameters, retain_graph=False)
        ratio = mse_norm / (reference_norm + 1e-12)
        if not all(math.isfinite(value) and value > 0.0 for value in (reference_norm, mse_norm, ratio)):
            raise RuntimeError("S3 encountered an invalid MSE gradient calibration sample")
        reference_norms.append(reference_norm)
        mse_norms.append(mse_norm)
        ratios.append(ratio)
        case_ids.extend(str(value) for value in batch.get("case_id", []))
    if not ratios:
        raise RuntimeError("S3 MSE gradient calibration received no batches")
    return {
        "ratios": ratios,
        "reference_gradient_norms": reference_norms,
        "mse_gradient_norms": mse_norms,
        "case_ids": case_ids,
    }


def solve_mse_weight(
    ratios: Sequence[float],
    *,
    target_weighted_ratio: float = 0.10,
    required_batches: int = 32,
) -> tuple[float, dict[str, Any]]:
    values = [float(value) for value in ratios][:required_batches]
    if len(values) != int(required_batches):
        raise RuntimeError(
            f"S3 MSE calibration requires exactly {required_batches} batches; "
            f"observed {len(values)}"
        )
    if not 0.0 < float(target_weighted_ratio) <= 1.0:
        raise ValueError("S3 MSE target gradient ratio must be within (0,1]")
    middle = median(values)
    if not math.isfinite(middle) or middle <= 0.0:
        raise RuntimeError(f"invalid S3 median MSE gradient ratio: {middle}")
    weight = float(target_weighted_ratio) / middle
    if not 1e-6 <= weight <= 10.0:
        raise RuntimeError(
            f"calibrated S3 MSE weight {weight:.9g} is outside the safety range [1e-6,10]"
        )
    return weight, {
        "required_batches": int(required_batches),
        "target_weighted_ratio": float(target_weighted_ratio),
        "unweighted_ratios": values,
        "median_unweighted_ratio": middle,
        "calibrated_weight": weight,
    }


def mse_weight_for_epoch(epoch: int, target: float, warmup_epochs: int) -> float:
    if int(warmup_epochs) <= 0:
        raise ValueError("S3 MSE warmup_epochs must be positive")
    fraction = min(max((int(epoch) + 1) / float(warmup_epochs), 0.0), 1.0)
    return float(target) * fraction


def refinement_constraints(
    candidate: dict[str, float],
    baseline: dict[str, float],
    selection: dict[str, Any],
) -> dict[str, bool]:
    return {
        "msssim_guard": float(candidate["msssim3d"])
        >= float(baseline["msssim3d"])
        - float(selection.get("msssim_max_drop", 0.001)),
        "hotspot_mae_guard": float(candidate["hotspot_mae"])
        <= float(baseline["hotspot_mae"])
        * float(selection.get("hotspot_mae_max_ratio", 1.02)),
        "raw_out_of_range_guard": float(candidate["raw_out_of_range_ratio"])
        <= float(baseline["raw_out_of_range_ratio"])
        + float(selection.get("raw_out_of_range_max_increase", 0.002)),
    }


def update_refinement_best(
    best: dict[str, Any],
    candidate: dict[str, float],
    baseline: dict[str, float],
    selection: dict[str, Any],
    epoch: int,
) -> tuple[list[str], dict[str, bool]]:
    improved: list[str] = []
    ordinary = {
        "psnr": ("psnr", True),
        "whole_mae": ("mae", False),
        "hotspot_mae": ("hotspot_mae", False),
        "msssim": ("msssim3d", True),
    }
    for name, (metric, higher) in ordinary.items():
        value = float(candidate[metric])
        previous = best.get(name)
        if previous is None or (value > previous["value"] if higher else value < previous["value"]):
            best[name] = {"epoch": int(epoch), "metric": metric, "value": value}
            improved.append(name)
    checks = refinement_constraints(candidate, baseline, selection)
    if all(checks.values()):
        value = float(candidate["psnr"])
        previous = best.get("psnr_guarded")
        if previous is None or value > previous["value"]:
            best["psnr_guarded"] = {
                "epoch": int(epoch),
                "metric": "psnr",
                "value": value,
                "constraints": checks,
            }
            improved.append("psnr_guarded")
        target = float(selection.get("target_psnr", 24.0))
        if value >= target:
            previous = best.get("psnr_target_guarded")
            if previous is None or value > previous["value"]:
                best["psnr_target_guarded"] = {
                    "epoch": int(epoch),
                    "metric": "psnr",
                    "value": value,
                    "target": target,
                    "constraints": checks,
                }
                improved.append("psnr_target_guarded")
    return improved, checks
