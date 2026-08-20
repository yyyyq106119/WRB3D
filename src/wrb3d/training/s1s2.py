"""Deterministic experiment controls shared by formal S1 and S2 training."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor, nn

from ..losses import AuxiliaryWeights


def canonical_config_fingerprint(config: dict[str, Any]) -> str:
    payload = copy.deepcopy(config)
    payload.pop("_config_path", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class WarmupCosineStepScheduler:
    """Per-optimizer-update 20-epoch warmup and no-restart cosine decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        steps_per_epoch: int,
        total_epochs: int = 800,
        warmup_epochs: int = 20,
        peak_lr: float = 1e-4,
        minimum_lr: float = 1e-6,
        step_index: int = 0,
    ) -> None:
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        if total_epochs <= warmup_epochs:
            raise ValueError("total_epochs must exceed warmup_epochs")
        self.optimizer = optimizer
        self.steps_per_epoch = int(steps_per_epoch)
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.peak_lr = float(peak_lr)
        self.minimum_lr = float(minimum_lr)
        self.step_index = int(step_index)
        self._set_lr(self.lr_for_progress(self.step_index / self.steps_per_epoch))

    def lr_for_epoch(self, epoch: float) -> float:
        if epoch < self.warmup_epochs:
            return self.peak_lr * min((float(epoch) + 1.0) / self.warmup_epochs, 1.0)
        progress = (float(epoch) - self.warmup_epochs) / (
            self.total_epochs - self.warmup_epochs
        )
        progress = min(max(progress, 0.0), 1.0)
        return self.minimum_lr + 0.5 * (self.peak_lr - self.minimum_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    def lr_for_progress(self, progress_epoch: float) -> float:
        return self.lr_for_epoch(progress_epoch)

    def _set_lr(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(value)

    def prepare_step(self) -> float:
        value = self.lr_for_progress(self.step_index / self.steps_per_epoch)
        self._set_lr(value)
        return value

    def step(self) -> None:
        self.step_index += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "steps_per_epoch": self.steps_per_epoch,
            "total_epochs": self.total_epochs,
            "warmup_epochs": self.warmup_epochs,
            "peak_lr": self.peak_lr,
            "minimum_lr": self.minimum_lr,
            "step_index": self.step_index,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key in (
            "steps_per_epoch",
            "total_epochs",
            "warmup_epochs",
            "peak_lr",
            "minimum_lr",
        ):
            expected = getattr(self, key)
            if state.get(key) != expected:
                raise RuntimeError(
                    f"scheduler resume mismatch for {key}: {state.get(key)!r} != {expected!r}"
                )
        self.step_index = int(state["step_index"])
        self._set_lr(self.lr_for_progress(self.step_index / self.steps_per_epoch))


def auxiliary_weights_for_epoch(
    epoch: int, targets: dict[str, float] | None
) -> AuxiliaryWeights:
    if not targets:
        return AuxiliaryWeights()

    def ramp(target: float, start: int, end: int) -> float:
        fraction = min(max((int(epoch) - start) / float(end - start), 0.0), 1.0)
        return float(target) * fraction

    return AuxiliaryWeights(
        hotspot=ramp(targets["hotspot"], 20, 70),
        underestimation=ramp(targets["underestimation"], 40, 100),
        aligned_amplitude=ramp(targets["aligned_amplitude"], 20, 70),
        orthogonal_error=ramp(targets["orthogonal_error"], 20, 70),
    )


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


def collect_gradient_ratios(
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
    """Measure unweighted auxiliary/reference gradient ratios without stepping."""
    model.train()
    if any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules()):
        raise RuntimeError("gradient calibration refuses mutable BatchNorm statistics")
    all_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "case_adaptive_wavelet_corrector" not in name
    ]
    high_parameters = [
        parameter for parameter in model.high_model.parameters() if parameter.requires_grad
    ]
    ratios = {name: [] for name in ("hotspot", "underestimation", "aligned_amplitude", "orthogonal_error")}
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
        components = result["loss_components"]
        image_norm = _gradient_norm(
            components["weighted_image"], all_parameters, retain_graph=True
        )
        hotspot_norm = _gradient_norm(
            components["hotspot"], all_parameters, retain_graph=True
        )
        under_norm = _gradient_norm(
            components["underestimation"], all_parameters, retain_graph=True
        )
        high_norm = _gradient_norm(
            components["weighted_high"], high_parameters, retain_graph=True
        )
        aligned_norm = _gradient_norm(
            components["aligned_amplitude"], high_parameters, retain_graph=True
        )
        orthogonal_norm = _gradient_norm(
            components["orthogonal_error"], high_parameters, retain_graph=False
        )
        epsilon = 1e-12
        ratios["hotspot"].append(hotspot_norm / (image_norm + epsilon))
        ratios["underestimation"].append(under_norm / (image_norm + epsilon))
        ratios["aligned_amplitude"].append(aligned_norm / (high_norm + epsilon))
        ratios["orthogonal_error"].append(orthogonal_norm / (high_norm + epsilon))
        ids = batch.get("case_id", [])
        case_ids.extend(str(value) for value in ids)
    if not ratios["hotspot"]:
        raise RuntimeError("gradient calibration received no batches")
    return {"ratios": ratios, "case_ids": case_ids}


def solve_calibrated_weights(
    ratio_records: dict[str, list[float]],
    *,
    required_batches: int = 32,
) -> tuple[dict[str, float], dict[str, Any]]:
    targets = {
        "hotspot": 0.08,
        "underestimation": 0.04,
        "aligned_amplitude": 0.05,
        "orthogonal_error": 0.05,
    }
    weights: dict[str, float] = {}
    audit: dict[str, Any] = {"required_batches": required_batches, "losses": {}}
    for name, target in targets.items():
        values = [float(value) for value in ratio_records[name]][:required_batches]
        if len(values) != required_batches:
            raise RuntimeError(
                f"gradient calibration requires exactly {required_batches} batches; "
                f"{name} has {len(values)}"
            )
        middle = median(values)
        if not math.isfinite(middle) or middle <= 0:
            raise RuntimeError(f"invalid median gradient ratio for {name}: {middle}")
        weight = target / middle
        if not 1e-5 <= weight <= 1.0:
            raise RuntimeError(
                f"calibrated {name} weight {weight:.9g} is outside [1e-5,1.0]"
            )
        weights[name] = weight
        audit["losses"][name] = {
            "target_weighted_ratio": target,
            "unweighted_ratios": values,
            "median_unweighted_ratio": middle,
            "calibrated_weight": weight,
        }
    return weights, audit


def load_shared_backbone_initialization(
    model: nn.Module, path: str | Path
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") != "shared_random_backbone":
        raise RuntimeError("shared initialization artifact has the wrong schema")
    state = payload.get("backbone")
    if not isinstance(state, dict):
        raise RuntimeError("shared initialization has no backbone state")
    target = model.state_dict()
    expected = {
        key
        for key in target
        if key.startswith("low_model.") or key.startswith("high_model.")
    }
    if set(state) != expected:
        missing = sorted(expected - set(state))
        extra = sorted(set(state) - expected)
        raise RuntimeError(
            f"shared backbone key mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )
    for key in sorted(expected):
        if tuple(state[key].shape) != tuple(target[key].shape):
            raise RuntimeError(f"shared initialization shape mismatch for {key}")
    model.load_state_dict(state, strict=False)
    return {
        "seed": int(payload["seed"]),
        "parameter_tensor_count": len(state),
        "artifact_sha256": sha256_file(path),
    }
