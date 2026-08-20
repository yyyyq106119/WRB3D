"""Checkpoint schema with architecture and output-semantics locks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import warnings

import torch
from torch import nn

from ..utils.config import resolve_project_path
from .ema import EMA


_SEMANTIC_SCHEMA_VERSION = 2


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(config: dict[str, Any], value: str | Path) -> Path:
    return resolve_project_path(config, value)


def checkpoint_semantics(config: dict[str, Any] | None, model: nn.Module) -> dict[str, Any]:
    """Return the canonical inference/training semantics bound to a checkpoint."""
    base = _model(model)
    config = config or {}
    wavelet = config.get("wavelet", {})
    bridge = config.get("bridge", {})
    covariance = config.get("covariance", {})
    data = config.get("data", {})
    statistics_path = covariance.get("statistics_path")
    statistics_hash = None
    if statistics_path:
        statistics_hash = _sha256_file(_project_path(config, statistics_path))
    manifest_hashes: dict[str, str] = {}
    manifest_dir = data.get("manifest_dir")
    if manifest_dir:
        directory = _project_path(config, manifest_dir)
        for name in ("train_patients.json", "val_patients.json", "test_patients.json", "split_audit.json"):
            digest = _sha256_file(directory / name)
            if digest is not None:
                manifest_hashes[name] = digest
    return {
        "schema_version": _SEMANTIC_SCHEMA_VERSION,
        "architecture_key": getattr(base, "architecture_key", None),
        "prediction_target": getattr(base, "prediction_target", None),
        "wavelet": {
            "type": wavelet.get("type", "fixed_orthonormal_haar3d"),
            "levels": int(wavelet.get("levels", 1)),
            "low_scaling": wavelet.get("low_scaling", "sum_div_sqrt8"),
            "band_order": list(
                wavelet.get(
                    "band_order",
                    ["LLL", "HLL", "LHL", "HHL", "LLH", "HLH", "LHH", "HHH"],
                )
            ),
            "pad_mode": wavelet.get("pad_mode", "replicate"),
            "channel_layout": "band_major_v2",
        },
        "bridge": {
            "type": bridge.get("type", "residual_brownian_bridge"),
            "num_timesteps": int(
                bridge.get("num_timesteps", getattr(getattr(base, "bridge", None), "num_timesteps", 1000))
            ),
            "schedule": bridge.get("schedule", "linear"),
            "endpoint_at_t0": bridge.get("endpoint_at_t0", "clean_mri_to_pet_residual"),
            "endpoint_at_tT": bridge.get("endpoint_at_tT", "zero_residual"),
        },
        "covariance": {
            "q_low": covariance.get("q_low"),
            "q_high": covariance.get("q_high"),
            "direct_low": covariance.get("low"),
            "direct_high": covariance.get("high"),
            "statistics_sha256": statistics_hash,
        },
        "normalization": data.get("normalization"),
        "dataset_root": str(data.get("root")) if data.get("root") is not None else None,
        "manifest_sha256": manifest_hashes,
    }


def _semantic_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _model(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    ema: EMA | None = None,
    epoch: int = 0,
    global_step: int = 0,
    config: dict | None = None,
    scheduler: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    base = _model(model)
    semantics = checkpoint_semantics(config, base)
    payload = {
        "schema_version": 2,
        "architecture_key": getattr(base, "architecture_key", None),
        "prediction_target": getattr(base, "prediction_target", None),
        "semantic_config": semantics,
        "semantic_fingerprint": _semantic_fingerprint(semantics),
        "model": base.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if ema is not None:
        payload["ema"] = ema.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if extra is not None:
        payload["experiment_state"] = extra
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _state_dict(payload: dict) -> dict[str, torch.Tensor]:
    for key in ("model", "model_state_dict", "state_dict"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        return payload
    raise KeyError("checkpoint has no recognized model state dictionary")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    ema: EMA | None = None,
    warm_start: bool = False,
    config: dict[str, Any] | None = None,
    allow_semantic_mismatch: bool = False,
    map_location: str | torch.device = "cpu",
    scheduler: Any | None = None,
    expected_config_fingerprint: str | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint root must be a mapping")
    experiment_state = payload.get("experiment_state", {})
    if expected_config_fingerprint is not None:
        observed = experiment_state.get("config_fingerprint")
        if observed != expected_config_fingerprint:
            raise RuntimeError(
                "checkpoint experiment config fingerprint mismatch: "
                f"{observed!r} != {expected_config_fingerprint!r}"
            )
    base = _model(model)
    expected_architecture = getattr(base, "architecture_key", None)
    expected_semantics = getattr(base, "prediction_target", None)
    source_architecture = payload.get("architecture_key")
    source_semantics = payload.get("prediction_target", "endpoint_x0_legacy")
    semantic_mismatch = source_semantics != expected_semantics
    if semantic_mismatch:
        warnings.warn(
            "Checkpoint output semantics are incompatible: "
            f"source={source_semantics!r}, target={expected_semantics!r}. "
            "Weights are not evidence that the residual endpoint model is trained.",
            RuntimeWarning,
            stacklevel=2,
        )
        if config is not None and not allow_semantic_mismatch:
            raise RuntimeError(
                "checkpoint prediction semantics are incompatible; use an explicit "
                "legacy/mismatch override only for a controlled conversion experiment"
            )
    architecture_mismatch = (
        source_architecture is not None
        and expected_architecture is not None
        and source_architecture != expected_architecture
    )
    source_semantic_config = payload.get("semantic_config")
    source_fingerprint = payload.get("semantic_fingerprint")
    if isinstance(source_semantic_config, dict):
        embedded_fingerprint = _semantic_fingerprint(source_semantic_config)
        if source_fingerprint != embedded_fingerprint:
            raise RuntimeError("checkpoint semantic fingerprint is missing or corrupt")
    receiving_semantic_config = checkpoint_semantics(config, base) if config is not None else None
    receiving_fingerprint = (
        _semantic_fingerprint(receiving_semantic_config)
        if receiving_semantic_config is not None
        else None
    )
    config_compatible = bool(
        source_fingerprint is not None
        and receiving_fingerprint is not None
        and source_fingerprint == receiving_fingerprint
    )
    if source_fingerprint is None or receiving_fingerprint is None:
        message = (
            "Checkpoint semantic configuration could not be fully verified. Formal training/inference "
            "must pass the resolved receiving config and use a schema-v2 checkpoint."
        )
        if config is not None and not allow_semantic_mismatch:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    elif not config_compatible:
        message = (
            "checkpoint covariance/wavelet/normalization/dataset semantics do not match the "
            "receiving configuration"
        )
        if not allow_semantic_mismatch:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    state = _state_dict(payload)
    loaded_keys: list[str]
    skipped_keys: list[str]
    if warm_start:
        target = base.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in target and tuple(value.shape) == tuple(target[key].shape)
        }
        base.load_state_dict(compatible, strict=False)
        loaded_keys = sorted(compatible)
        skipped_keys = sorted(set(state) - set(compatible))
        if not loaded_keys:
            raise RuntimeError("warm start found no shape-compatible parameters")
        if ema is not None:
            ema.module.load_state_dict(base.state_dict(), strict=True)
            ema.num_updates = 0
    else:
        if architecture_mismatch:
            raise RuntimeError(
                f"checkpoint architecture mismatch: {source_architecture!r} != {expected_architecture!r}"
            )
        base.load_state_dict(state, strict=True)
        loaded_keys = sorted(state)
        skipped_keys = []
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if scaler is not None and "scaler" in payload:
            scaler.load_state_dict(payload["scaler"])
        if ema is not None and "ema" in payload:
            ema.load_state_dict(payload["ema"])
        if scheduler is not None:
            if "scheduler" not in payload:
                raise RuntimeError("formal resume checkpoint has no scheduler state")
            scheduler.load_state_dict(payload["scheduler"])
    return {
        "epoch": int(payload.get("epoch", 0)),
        "global_step": int(payload.get("global_step", 0)),
        "semantic_mismatch": semantic_mismatch,
        "architecture_mismatch": architecture_mismatch,
        "config_compatible": config_compatible,
        "source_semantic_fingerprint": source_fingerprint,
        "receiving_semantic_fingerprint": receiving_fingerprint,
        "loaded_keys": loaded_keys,
        "skipped_keys": skipped_keys,
        "warm_start": bool(warm_start),
        "ema_loaded": bool(ema is not None and "ema" in payload and not warm_start),
        "experiment_state": experiment_state,
        "checkpoint_config": payload.get("config"),
    }
