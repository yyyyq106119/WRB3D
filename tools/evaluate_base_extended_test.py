"""One-shot, Base-only evaluation on the immutable paired test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from wrb3d.datasets import PairedVolumeDataset, verify_split_manifests
from wrb3d.metrics import image_metrics
from wrb3d.training import load_checkpoint
from wrb3d.training.checkpoint import checkpoint_semantics
from wrb3d.training.manifests import collect_split_records, resolve_manifest_dir
from wrb3d.utils import build_model, load_config, load_covariances
from wrb3d.wavelets import compute_residuals


DEFAULT_STEPS = (1, 5, 15, 50)
CORE_SEMANTIC_KEYS = (
    "schema_version",
    "architecture_key",
    "prediction_target",
    "wavelet",
    "bridge",
    "covariance",
    "residual_normalization",
    "loss",
    "normalization",
    "dataset_root",
    "dataset_identity_sha256",
    "manifest_sha256",
    "split_identity_sha256",
)
DIAGNOSTIC_ONLY_SEMANTIC_KEYS = (
    "ema",
    "experiment",
    "initialization",
    "evaluation",
    "run_contract",
    "runtime_approval",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", nargs="+", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--expected-completed-epochs", type=int, default=800)
    parser.add_argument(
        "--diagnostic-direct-base-load",
        action="store_true",
        help=(
            "load checkpoint['model'] directly after exact architecture, prediction-"
            "target and semantic-fingerprint verification; this is diagnostic and "
            "does not confer formal E2 eligibility"
        ),
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _core_semantics(payload: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in CORE_SEMANTIC_KEYS if key not in payload]
    if missing:
        raise RuntimeError(
            f"checkpoint/config core semantic fields are missing: {missing}"
        )
    return {key: payload[key] for key in CORE_SEMANTIC_KEYS}


def _semantic_differences(
    left: Any, right: Any, path: str = ""
) -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        rows: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left:
                rows.append(
                    {"path": child, "checkpoint": "<missing>", "receiving": right[key]}
                )
            elif key not in right:
                rows.append(
                    {"path": child, "checkpoint": left[key], "receiving": "<missing>"}
                )
            else:
                rows.extend(_semantic_differences(left[key], right[key], child))
        return rows
    if left != right:
        return [{"path": path, "checkpoint": left, "receiving": right}]
    return []


def _pairwise_numeric_equivalence(
    left: Any, right: Any, path: str = ""
) -> tuple[Any, Any, list[str]]:
    if isinstance(left, dict) and isinstance(right, dict):
        normalized_left: dict[str, Any] = {}
        normalized_right: dict[str, Any] = {}
        normalized_paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left:
                normalized_right[key] = right[key]
                continue
            if key not in right:
                normalized_left[key] = left[key]
                continue
            child_left, child_right, child_paths = _pairwise_numeric_equivalence(
                left[key], right[key], child
            )
            normalized_left[key] = child_left
            normalized_right[key] = child_right
            normalized_paths.extend(child_paths)
        return normalized_left, normalized_right, normalized_paths
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        normalized_left_list: list[Any] = []
        normalized_right_list: list[Any] = []
        normalized_paths: list[str] = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            child = f"{path}[{index}]"
            child_left, child_right, child_paths = _pairwise_numeric_equivalence(
                left_item, right_item, child
            )
            normalized_left_list.append(child_left)
            normalized_right_list.append(child_right)
            normalized_paths.extend(child_paths)
        return normalized_left_list, normalized_right_list, normalized_paths
    left_is_number = isinstance(left, (int, float)) and not isinstance(left, bool)
    right_is_number = isinstance(right, (int, float)) and not isinstance(right, bool)
    if isinstance(left, str) and right_is_number:
        try:
            parsed = float(left)
        except ValueError:
            return left, right, []
        if math.isfinite(parsed) and parsed == float(right):
            return float(right), float(right), [path]
    if left_is_number and isinstance(right, str):
        try:
            parsed = float(right)
        except ValueError:
            return left, right, []
        if math.isfinite(parsed) and float(left) == parsed:
            return float(left), float(left), [path]
    return left, right, []


def _tensor_endpoint_metrics(
    prediction: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8
) -> dict[str, float]:
    x = prediction.detach().float().reshape(prediction.shape[0], -1)
    y = target.detach().float().reshape(target.shape[0], -1)
    x_centered = x - x.mean(dim=1, keepdim=True)
    y_centered = y - y.mean(dim=1, keepdim=True)
    covariance = (x_centered * y_centered).sum(dim=1)
    target_variance = y_centered.square().sum(dim=1)
    pearson = covariance / torch.sqrt(
        x_centered.square().sum(dim=1) * target_variance
    ).clamp_min(eps)
    slope = covariance / target_variance.clamp_min(eps)
    energy_ratio = x.square().mean(dim=1) / y.square().mean(dim=1).clamp_min(eps)
    return {
        "mae": float((x - y).abs().mean().cpu()),
        "pearson": float(pearson.mean().cpu()),
        "slope": float(slope.mean().cpu()),
        "energy_ratio": float(energy_ratio.mean().cpu()),
    }


def _residual_endpoint_metrics(
    model: torch.nn.Module,
    mri: torch.Tensor,
    pet: torch.Tensor,
    predicted_low: torch.Tensor,
    predicted_high: torch.Tensor,
    band_order: list[str],
) -> dict[str, float]:
    mri_low, mri_high, _ = model.dwt(mri)
    pet_low, pet_high, _ = model.dwt(pet)
    target = compute_residuals(mri_low, mri_high, pet_low, pet_high)
    output = {
        f"low_residual_{name}": value
        for name, value in _tensor_endpoint_metrics(
            predicted_low, target.low
        ).items()
    }
    if predicted_high.shape != target.high.shape:
        raise RuntimeError("predicted and target high residual shapes differ")
    if predicted_high.shape[1] % 7 != 0:
        raise RuntimeError("high residual tensor does not contain seven bands")
    channels = predicted_high.shape[1] // 7
    names = list(band_order[1:])
    if len(names) != 7:
        raise RuntimeError("wavelet band order must contain LLL plus seven high bands")
    predicted_bands = predicted_high.reshape(
        predicted_high.shape[0],
        7,
        channels,
        *predicted_high.shape[-3:],
    )
    target_bands = target.high.reshape(
        target.high.shape[0],
        7,
        channels,
        *target.high.shape[-3:],
    )
    for index, band in enumerate(names):
        for metric, value in _tensor_endpoint_metrics(
            predicted_bands[:, index], target_bands[:, index]
        ).items():
            output[f"band_{band}_{metric}"] = value
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    preferred = ["case_id", "patient_id", "sampling_steps"]
    fields = preferred + [name for name in fields if name not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _patient_safe_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for steps in sorted({int(row["sampling_steps"]) for row in rows}):
        step_rows = [row for row in rows if int(row["sampling_steps"]) == steps]
        metric_names = sorted(
            key
            for key, value in step_rows[0].items()
            if key not in {"case_id", "patient_id", "sampling_steps"}
            and isinstance(value, (int, float))
        )
        by_patient: dict[str, list[dict[str, Any]]] = {}
        for row in step_rows:
            by_patient.setdefault(str(row["patient_id"]), []).append(row)
        patient_rows: list[dict[str, float]] = []
        for patient_values in by_patient.values():
            patient_rows.append(
                {
                    name: sum(float(row[name]) for row in patient_values)
                    / len(patient_values)
                    for name in metric_names
                }
            )
        result[str(steps)] = {
            "case_count": len(step_rows),
            "patient_count": len(patient_rows),
            "metrics": {
                name: sum(row[name] for row in patient_rows) / len(patient_rows)
                for name in metric_names
            },
        }
    return result


def _finite_rows(rows: list[dict[str, Any]]) -> bool:
    return all(
        math.isfinite(float(value))
        for row in rows
        for key, value in row.items()
        if key not in {"case_id", "patient_id", "sampling_steps"}
        and isinstance(value, (int, float))
    )


@torch.inference_mode()
def main() -> None:
    args = _arguments()
    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {checkpoint}")
    if output.exists():
        raise FileExistsError(
            f"test output already exists and will not be overwritten: {output}"
        )
    if len(set(args.steps)) != len(args.steps) or any(value <= 0 for value in args.steps):
        raise ValueError("--steps must contain unique positive integers")

    raw_checkpoint = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(raw_checkpoint, dict):
        raise TypeError("checkpoint root must be a mapping")
    if not isinstance(raw_checkpoint.get("model"), dict):
        raise RuntimeError(
            "formal Base evaluation requires checkpoint['model']; EMA-only or "
            "ambiguous checkpoint loading is forbidden"
        )
    source_epoch = int(raw_checkpoint.get("epoch", -1))
    completed_epochs = source_epoch + 1
    if completed_epochs != int(args.expected_completed_epochs):
        raise RuntimeError(
            f"checkpoint completed_epochs={completed_epochs}, expected "
            f"{args.expected_completed_epochs}"
        )
    source_global_step = int(raw_checkpoint.get("global_step", -1))
    if source_global_step <= 0:
        raise RuntimeError("checkpoint global_step is missing or invalid")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda" and not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be set before deterministic CUDA evaluation"
        )
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    config = load_config(config_path)
    split_audit = verify_split_manifests(
        collect_split_records(config), resolve_manifest_dir(config)
    )
    if any(split_audit.get("overlaps", {}).values()):
        raise RuntimeError("patient leakage is present in the audited split manifests")
    model = build_model(config).to(device)
    if args.diagnostic_direct_base_load:
        source_semantics = raw_checkpoint.get("semantic_config")
        source_fingerprint = raw_checkpoint.get("semantic_fingerprint")
        if not isinstance(source_semantics, dict):
            raise RuntimeError(
                "diagnostic direct Base loading requires embedded semantic_config"
            )
        if source_fingerprint != _semantic_fingerprint(source_semantics):
            raise RuntimeError(
                "checkpoint semantic fingerprint is missing or corrupt"
            )
        expected_architecture = getattr(model, "architecture_key", None)
        if raw_checkpoint.get("architecture_key") != expected_architecture:
            raise RuntimeError(
                "checkpoint architecture does not exactly match the evaluation model"
            )
        expected_target = getattr(model, "prediction_target", None)
        if raw_checkpoint.get("prediction_target") != expected_target:
            raise RuntimeError(
                "checkpoint prediction target does not exactly match the evaluation model"
            )
        source_coordinate_system = (
            source_semantics.get("experiment", {}).get("coordinate_system")
            if isinstance(source_semantics.get("experiment"), dict)
            else None
        )
        if (
            source_coordinate_system != "physical_residual"
            or getattr(model, "coordinate_system", None) != "physical_residual"
        ):
            raise RuntimeError(
                "v4 diagnostic Base evaluation requires matching physical-residual "
                "checkpoint and model coordinate systems"
            )
        receiving_semantics = checkpoint_semantics(config, model)
        receiving_fingerprint = _semantic_fingerprint(receiving_semantics)
        source_core_semantics = _core_semantics(source_semantics)
        receiving_core_semantics = _core_semantics(receiving_semantics)
        (
            comparable_source_core,
            comparable_receiving_core,
            numeric_equivalent_paths,
        ) = _pairwise_numeric_equivalence(
            source_core_semantics, receiving_core_semantics
        )
        source_core_fingerprint = _semantic_fingerprint(comparable_source_core)
        receiving_core_fingerprint = _semantic_fingerprint(
            comparable_receiving_core
        )
        core_differences = _semantic_differences(
            comparable_source_core, comparable_receiving_core
        )
        if source_core_fingerprint != receiving_core_fingerprint:
            print(
                json.dumps(
                    {
                        "status": "BLOCKED",
                        "reason": "core_inference_semantics_mismatch",
                        "differences": core_differences,
                    },
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                flush=True,
            )
            raise RuntimeError(
                "diagnostic direct Base loading requires exact core inference "
                "semantics; see the printed field-level differences"
            )
        model.load_state_dict(raw_checkpoint["model"], strict=True)
        checkpoint_info = {
            "config_compatible": True,
            "ema_loaded": False,
            "source_semantic_fingerprint": source_fingerprint,
            "receiving_semantic_fingerprint": receiving_fingerprint,
            "source_core_semantic_fingerprint": source_core_fingerprint,
            "receiving_core_semantic_fingerprint": receiving_core_fingerprint,
            "numeric_equivalent_semantic_paths": numeric_equivalent_paths,
        }
        checkpoint_load_mode = "diagnostic_direct_base_state"
    else:
        checkpoint_info = load_checkpoint(
            checkpoint,
            model,
            config=config,
            map_location=device,
        )
        checkpoint_load_mode = "formal_checkpoint_loader"
    if not checkpoint_info["config_compatible"]:
        raise RuntimeError("checkpoint and evaluation config are not semantically compatible")
    if checkpoint_info["ema_loaded"]:
        raise RuntimeError("EMA was unexpectedly loaded during Base-only evaluation")
    model.eval()
    covariance_low, covariance_high, statistics = load_covariances(config)
    covariance_low = covariance_low.to(device)
    covariance_high = covariance_high.to(device)
    data = config["data"]
    dataset = PairedVolumeDataset(
        data["root"],
        "test",
        tuple(data.get("patch_size", [128, 128, 64])),
        data.get("patient_id_regex"),
    )
    expected_test_patients = int(
        split_audit.get("patients_per_split", {}).get("test", -1)
    )
    observed_test_patients = len(
        {str(dataset[index]["patient_id"]) for index in range(len(dataset))}
    )
    if observed_test_patients != expected_test_patients:
        raise RuntimeError(
            f"test patient count mismatch: {observed_test_patients} != "
            f"{expected_test_patients}"
        )

    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    progress_path = output / "test_per_case_metrics.jsonl"
    band_order = list(
        config.get("wavelet", {}).get(
            "band_order",
            ["LLL", "HLL", "LHL", "HHL", "LLH", "HLH", "LHH", "HHH"],
        )
    )
    for index in range(len(dataset)):
        sample = dataset[index]
        mri = sample["mri"].unsqueeze(0).to(device)
        pet = sample["pet"].unsqueeze(0).to(device)
        for steps in args.steps:
            result = model.infer(
                mri,
                covariance_low,
                covariance_high,
                num_steps=int(steps),
                stochastic=False,
            )
            metrics = {
                name: float(value.detach().cpu())
                for name, value in image_metrics(
                    result["B_raw"], pet, data_range=1.0
                ).items()
            }
            metrics["raw_out_of_range_ratio"] = float(
                (
                    (result["B_raw"] < 0.0) | (result["B_raw"] > 1.0)
                ).float().mean().cpu()
            )
            metrics.update(
                _residual_endpoint_metrics(
                    model,
                    mri,
                    pet,
                    result["predicted_low_residual"],
                    result["predicted_high_residual"],
                    band_order,
                )
            )
            row: dict[str, Any] = {
                "case_id": str(sample["case_id"]),
                "patient_id": str(sample["patient_id"]),
                "sampling_steps": int(steps),
                **metrics,
            }
            rows.append(row)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
        print(
            f"Base test {index + 1}/{len(dataset)}: {sample['case_id']}",
            flush=True,
        )
    if not rows or not _finite_rows(rows):
        raise RuntimeError("test metrics are empty or non-finite")

    per_case_csv = output / "test_per_case_metrics.csv"
    _write_csv(per_case_csv, rows)
    summary = {
        "schema_version": 1,
        "kind": "base_extended_500_to_800_test_evaluation",
        "status": "COMPLETE",
        "evaluated_weights": "base",
        "selection_policy": (
            "checkpoint fixed before test evaluation; test metrics must not be "
            "used to select another checkpoint"
        ),
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "checkpoint_epoch": source_epoch,
        "completed_epochs": completed_epochs,
        "checkpoint_global_step": source_global_step,
        "checkpoint_model_key": "model",
        "checkpoint_load_mode": checkpoint_load_mode,
        "formal_e2_provenance_enforced": not args.diagnostic_direct_base_load,
        "formal_e2_eligible": not args.diagnostic_direct_base_load,
        "diagnostic_limitation": (
            "formal E2 provenance was not asserted; result is a Base-only "
            "diagnostic test evaluation"
            if args.diagnostic_direct_base_load
            else None
        ),
        "checkpoint_contains_ema_but_ignored": isinstance(
            raw_checkpoint.get("ema"), dict
        ),
        "checkpoint_config_compatible": checkpoint_info["config_compatible"],
        "source_core_semantic_fingerprint": checkpoint_info.get(
            "source_core_semantic_fingerprint"
        ),
        "receiving_core_semantic_fingerprint": checkpoint_info.get(
            "receiving_core_semantic_fingerprint"
        ),
        "diagnostic_only_semantic_fields_not_required_to_match": (
            list(DIAGNOSTIC_ONLY_SEMANTIC_KEYS)
            if args.diagnostic_direct_base_load
            else []
        ),
        "numeric_string_equivalence_normalized": bool(
            checkpoint_info.get("numeric_equivalent_semantic_paths")
        ),
        "numeric_equivalent_semantic_paths": checkpoint_info.get(
            "numeric_equivalent_semantic_paths", []
        ),
        "test_case_count": len(dataset),
        "test_patient_count": observed_test_patients,
        "sampling_steps": list(args.steps),
        "seed": int(args.seed),
        "stochastic": False,
        "deterministic": True,
        "postprocessing": "none_raw_pet",
        "data_range": 1.0,
        "dataset_identity_sha256": split_audit["dataset_identity_sha256"],
        "split_patient_counts": split_audit["patients_per_split"],
        "split_overlaps": split_audit["overlaps"],
        "statistics_split": statistics.get("provenance", {}).get("split"),
        "patient_safe_summary": _patient_safe_summary(rows),
        "artifacts": {
            "per_case_csv": str(per_case_csv),
            "per_case_jsonl": str(progress_path),
        },
    }
    summary_path = output / "test_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "evaluated_weights": summary["evaluated_weights"],
                "completed_epochs": completed_epochs,
                "test_case_count": len(dataset),
                "test_patient_count": observed_test_patients,
                "output_dir": str(output),
                "summary": str(summary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
