"""Strict YAML loading and construction for the independent package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from ..models import WaveletResidualBridgeModel


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"configuration {path} must contain a mapping")
    parent = data.pop("extends", None)
    if parent is not None:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        base = load_config(parent_path)
        base.pop("_config_path", None)

        def merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
            output = dict(left)
            for key, value in right.items():
                if isinstance(value, dict) and isinstance(output.get(key), dict):
                    output[key] = merge(output[key], value)
                else:
                    output[key] = value
            return output

        data = merge(base, data)
    data["_config_path"] = str(path.resolve())
    return data


def build_model(config: dict[str, Any]) -> WaveletResidualBridgeModel:
    model = config["model"]
    bridge = config["bridge"]
    loss = config.get("loss", {})
    projection = config.get("projection", {})
    corrector = model.get("case_adaptive_wavelet_corrector", {})
    high_band_stds = loss.get("high_band_stds")
    if high_band_stds is None:
        statistics_path = config.get("covariance", {}).get("statistics_path")
        if not statistics_path:
            raise RuntimeError(
                "loss.high_band_stds must be explicit for smoke runs or derived from "
                "covariance.statistics_path for formal training"
            )
        resolved = resolve_project_path(config, statistics_path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"train-derived high-band normalization statistics are required: {resolved}"
            )
        statistics = _verified_statistics(config, resolved)
        high_band_stds = statistics["std_high_residual"]
    aligned = dict(loss.get("aligned_amplitude", {}))
    if bool(aligned.get("enabled", False)):
        band_epsilon = aligned.get("band_epsilon")
        aligned_statistics_path = aligned.get("statistics_path")
        if band_epsilon is None:
            if not aligned_statistics_path:
                raise RuntimeError(
                    "formal aligned-amplitude training requires a train-derived statistics_path"
                )
            aligned_path = resolve_project_path(config, aligned_statistics_path)
            if not aligned_path.exists():
                raise FileNotFoundError(
                    f"train-derived aligned-band statistics are required: {aligned_path}"
                )
            aligned_statistics = json.loads(aligned_path.read_text(encoding="utf-8"))
            provenance = aligned_statistics.get("provenance", {})
            if provenance.get("split") != "train":
                raise RuntimeError("aligned-band statistics must have train-split provenance")
            if _canonical_path(provenance.get("dataset_root", "")) != _canonical_path(
                config["data"]["root"]
            ):
                raise RuntimeError("aligned-band statistics dataset root does not match config")
            band_epsilon = aligned_statistics.get("band_epsilon")
        aligned["band_epsilon"] = band_epsilon
    return WaveletResidualBridgeModel(
        input_channels=int(model.get("input_channels", 1)),
        channels=tuple(int(value) for value in model.get("channels", [32, 64, 128, 256])),
        condition_dim=int(model.get("condition_dim", 128)),
        num_timesteps=int(bridge.get("num_timesteps", 1000)),
        prediction_target=str(model.get("prediction_target", "residual_x0")),
        low_to_high_condition=str(model.get("low_to_high_condition", "feature_gating")),
        projection_mode=str(projection.get("mode", "none")),
        projection_beta=float(projection.get("beta", 10.0)),
        corrector_kwargs={
            "enabled": bool(corrector.get("enabled", False)),
            "hidden_dim": int(corrector.get("hidden_dim", 128)),
            "gamma": float(corrector.get("gamma", 0.15)),
            "identity_init": bool(corrector.get("identity_init", True)),
        },
        auxiliary_loss_kwargs={
            "hotspot": dict(loss.get("hotspot", {})),
            "aligned_amplitude": aligned,
            "gain_regularization": dict(loss.get("gain_regularization", {})),
        },
        loss_kwargs={
            "lambda_low": float(loss.get("lambda_low", 1.0)),
            "lambda_high": float(loss.get("lambda_high", 1.0)),
            "lambda_image": float(loss.get("lambda_image", 2.0)),
            "lambda_range": float(loss.get("lambda_range", 0.05)),
            "endpoint_loss": str(loss.get("endpoint_loss", "charbonnier")),
            "high_band_stds": torch.as_tensor(high_band_stds),
            "max_high_weight": float(loss.get("max_high_weight", 10.0)),
            "image_domain": str(loss.get("image_domain", "raw")),
        },
    )


def project_root(config: dict[str, Any]) -> Path:
    """Locate the package root independently of config nesting depth."""
    source = config.get("_config_path")
    if not source:
        return Path.cwd().resolve()
    config_directory = Path(source).expanduser().resolve().parent
    for candidate in (config_directory, *config_directory.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "wrb3d").is_dir():
            return candidate
    return config_directory


def resolve_project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root(config) / path
    return path.resolve()


def _canonical_path(value: str | Path) -> str:
    return str(Path(value).expanduser().resolve())


def _verified_statistics(config: dict[str, Any], path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    provenance = data.get("provenance", {})
    if provenance.get("split") != "train":
        raise RuntimeError("residual statistics must have train-split provenance")
    expected_root = config.get("data", {}).get("root")
    recorded_root = provenance.get("dataset_root")
    if expected_root is None or recorded_root is None:
        raise RuntimeError("statistics provenance must bind a dataset root")
    if _canonical_path(expected_root) != _canonical_path(recorded_root):
        raise RuntimeError(
            "statistics dataset provenance does not match the configured dataset root; "
            "the statistics are stale or belong to another dataset"
        )
    if bool(config.get("data", {}).get("require_patient_manifests", False)):
        manifest_dir = config["data"].get("manifest_dir", "splits")
        manifest_path = resolve_project_path(config, manifest_dir) / "split_audit.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"dataset identity manifest is required before loading statistics: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_identity = manifest.get("dataset_identity_sha256")
        observed_identity = provenance.get("dataset_identity_sha256")
        if not expected_identity or observed_identity != expected_identity:
            raise RuntimeError(
                "statistics provenance is stale: dataset/split content identity does not match"
            )
    return data


def load_covariances(config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    covariance = config["covariance"]
    path = covariance.get("statistics_path")
    if path:
        path = resolve_project_path(config, path)
        if not path.exists():
            raise FileNotFoundError(
                f"residual statistics are required before formal training: {path}. "
                "Run wrb3d-estimate-stats on the training split."
            )
        data = _verified_statistics(config, path)
        q_low = float(covariance.get("q_low", 0.5))
        q_high = float(covariance.get("q_high", 0.5))
        low_std = float(data["std_low_residual"])
        high_stds = torch.tensor(data["std_high_residual"], dtype=torch.float32)
        low = torch.tensor(2.0 * q_low**2 * low_std**2, dtype=torch.float32)
        high = 2.0 * q_high**2 * high_stds.square()
        return low, high, data
    if not bool(covariance.get("allow_unverified_for_smoke", False)):
        raise RuntimeError("direct covariance is allowed only in an explicitly marked smoke configuration")
    low = torch.tensor(float(covariance["low"]), dtype=torch.float32)
    high = torch.tensor(covariance["high"], dtype=torch.float32)
    if high.numel() != 7:
        raise ValueError("smoke high covariance must contain seven values")
    return low, high, {"provenance": {"split": "synthetic_smoke"}}
