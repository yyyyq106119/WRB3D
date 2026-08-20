"""Audited AMP/DDP entry point for the formal S2 -> S1 800-epoch matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from ..datasets import PairedVolumeDataset
from ..metrics import formal_case_metrics
from ..utils import build_model, load_config, load_covariances
from .checkpoint import load_checkpoint, save_checkpoint
from .distributed import (
    DistributedEvalSampler,
    gather_metric_records,
    initialize_distributed,
    reduce_metrics,
)
from .ema import EMA
from .manifests import verify_train_val_without_test_access
from .s1s2 import (
    WarmupCosineStepScheduler,
    auxiliary_weights_for_epoch,
    canonical_config_fingerprint,
    collect_gradient_ratios,
    load_shared_backbone_initialization,
    sha256_file,
    solve_calibrated_weights,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal S1/S2 800-epoch trainer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--shared-backbone-init", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def _worker_init():
    def initialize(worker_id: int) -> None:
        del worker_id
        value = int(torch.initial_seed() % (2**32))
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)

    return initialize


def _json_line(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _environment_snapshot(
    config: dict[str, Any],
    config_path: Path,
    shared_path: Path,
    world_size: int,
) -> dict[str, Any]:
    gpu_rows = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu_rows.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "capability": list(properties.major_minor)
                    if hasattr(properties, "major_minor")
                    else [properties.major, properties.minor],
                }
            )
    return {
        "schema_version": 1,
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "world_size": world_size,
        "gpus": gpu_rows,
        "source_git_sha": None,
        "source_git_status": "NOT_A_GIT_REPOSITORY",
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "config_fingerprint": canonical_config_fingerprint(config),
        "shared_initialization_path": str(shared_path.resolve()),
        "shared_initialization_sha256": sha256_file(shared_path),
    }


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    loader: DataLoader,
    covariance_low: torch.Tensor,
    covariance_high: torch.Tensor,
    world_size: int,
    *,
    maximum_cases: int | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    records: list[dict[str, Any]] = []
    seen = 0
    for batch in loader:
        mri_batch = batch["mri"].to(covariance_low.device, non_blocking=True)
        pet_batch = batch["pet"].to(covariance_low.device, non_blocking=True)
        for item in range(mri_batch.shape[0]):
            if maximum_cases is not None and seen >= maximum_cases:
                break
            mri = mri_batch[item : item + 1]
            pet = pet_batch[item : item + 1]
            inference = model.infer(
                mri,
                covariance_low,
                covariance_high,
                num_steps=1,
                stochastic=False,
            )
            metrics = formal_case_metrics(model, mri, pet, inference)
            records.append(
                {
                    "case_id": str(batch["case_id"][item]),
                    "patient_id": str(batch["patient_id"][item]),
                    "metrics": {
                        name: float(value.detach().float().cpu())
                        for name, value in metrics.items()
                    },
                }
            )
            seen += 1
        if maximum_cases is not None and seen >= maximum_cases:
            break
    records = gather_metric_records(records, world_size)
    if not records:
        raise RuntimeError("validation loader produced no cases")
    by_patient: dict[str, list[dict[str, float]]] = {}
    for row in records:
        by_patient.setdefault(row["patient_id"], []).append(row["metrics"])
    patients: list[dict[str, Any]] = []
    for patient_id, values in sorted(by_patient.items()):
        patients.append(
            {
                "patient_id": patient_id,
                "metrics": {
                    name: sum(item[name] for item in values) / len(values)
                    for name in values[0]
                },
            }
        )
    aggregate = {
        name: sum(item["metrics"][name] for item in patients) / len(patients)
        for name in patients[0]["metrics"]
    }
    aggregate["patient_count"] = len(patients)
    aggregate["case_count"] = len(records)
    aggregate["high_mean_orthogonal_error"] = sum(
        aggregate[f"high_band_{band}_orthogonal_error_ratio"] for band in range(7)
    ) / 7.0
    aggregate["high_mean_pearson"] = sum(
        aggregate[f"high_band_{band}_pearson"] for band in range(7)
    ) / 7.0
    return aggregate, patients


def _aggregate_calibration(
    local: dict[str, Any], world_size: int, rank: int
) -> tuple[dict[str, float], dict[str, Any]]:
    gathered: list[Any]
    if world_size > 1:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    combined = {
        name: [
            value
            for rank_payload in gathered
            for value in rank_payload["ratios"][name]
        ]
        for name in local["ratios"]
    }
    all_case_ids = [
        value for rank_payload in gathered for value in rank_payload["case_ids"]
    ]
    payload: list[Any] = [None]
    if rank == 0:
        weights, audit = solve_calibrated_weights(combined, required_batches=32)
        audit["case_ids"] = all_case_ids[:32]
        audit["world_size"] = world_size
        audit["status"] = "PASS"
        payload[0] = (weights, audit)
    if world_size > 1:
        dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _best_rules() -> dict[str, tuple[str, bool]]:
    return {
        "whole_mae": ("mae", False),
        "hotspot_mae": ("hotspot_mae", False),
        "peak_bias": ("hotspot_peak_intensity_bias_abs", False),
        "msssim": ("msssim3d", True),
        "orthogonal_error": ("high_mean_orthogonal_error", False),
    }


def _update_best(
    best: dict[str, Any], validation: dict[str, float], epoch: int
) -> list[str]:
    values = dict(validation)
    values["hotspot_peak_intensity_bias_abs"] = abs(
        validation["hotspot_peak_intensity_bias"]
    )
    improved: list[str] = []
    for name, (metric, higher_is_better) in _best_rules().items():
        candidate = float(values[metric])
        previous = best.get(name)
        if previous is None or (
            candidate > previous["value"] if higher_is_better else candidate < previous["value"]
        ):
            best[name] = {"epoch": int(epoch), "value": candidate, "metric": metric}
            improved.append(name)
    return improved


def _checkpoint_extra(
    *,
    config_fingerprint: str,
    shared_initialization_sha256: str,
    calibrated_weights: dict[str, float] | None,
    best: dict[str, Any],
    overflow_events: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config_fingerprint": config_fingerprint,
        "shared_initialization_sha256": shared_initialization_sha256,
        "calibrated_auxiliary_weights": calibrated_weights,
        "best_checkpoints": best,
        "amp_overflow_events": overflow_events,
    }


def main() -> None:
    args = _arguments()
    config_path = Path(args.config)
    shared_path = Path(args.shared_backbone_init)
    config = load_config(config_path)
    train_cfg = config["train"]
    experiment = config.get("experiment", {})
    seed = int(experiment.get("seed", train_cfg.get("seed", 1234)))
    total_epochs = int(args.max_epochs or experiment.get("epochs", train_cfg.get("epochs", 800)))
    if args.max_epochs is None and total_epochs != 800:
        raise RuntimeError("formal S1/S2 runs must use exactly 800 epochs")
    distributed, rank, local_rank, world_size = initialize_distributed()
    expected_gpus = int(train_cfg.get("expected_gpus", 4))
    if args.max_steps is None and world_size != expected_gpus:
        raise RuntimeError(
            f"formal run requires DDP world_size={expected_gpus}, observed {world_size}"
        )
    process_seed = seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    data_cfg = config["data"]
    if bool(data_cfg.get("require_patient_manifests", True)):
        verify_train_val_without_test_access(config)
    dataset = PairedVolumeDataset(
        data_cfg["root"],
        "train",
        tuple(data_cfg.get("patch_size", [128, 128, 64])),
        data_cfg.get("patient_id_regex"),
    )
    sampler = (
        DistributedSampler(dataset, shuffle=True, seed=seed, drop_last=False)
        if distributed
        else None
    )
    generator = torch.Generator()
    generator.manual_seed(seed + rank)
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size_per_gpu", 1)),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        worker_init_fn=_worker_init(),
        generator=generator,
        drop_last=False,
    )
    validation_dataset = PairedVolumeDataset(
        data_cfg["root"],
        "val",
        tuple(data_cfg.get("patch_size", [128, 128, 64])),
        data_cfg.get("patient_id_regex"),
    )
    validation_sampler = DistributedEvalSampler(validation_dataset) if distributed else None
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        sampler=validation_sampler,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        worker_init_fn=_worker_init(),
    )
    model = build_model(config).to(device)
    shared_audit = load_shared_backbone_initialization(model, shared_path)
    if shared_audit["seed"] != seed:
        raise RuntimeError("shared initialization seed does not match experiment seed")
    covariance_low, covariance_high, covariance_statistics = load_covariances(config)
    covariance_low = covariance_low.to(device)
    covariance_high = covariance_high.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["optimizer"].get("lr_peak", 1e-4)),
        betas=tuple(float(value) for value in config["optimizer"].get("betas", [0.9, 0.999])),
        weight_decay=float(config["optimizer"].get("weight_decay", 0.0)),
    )
    trainable_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    optimizer_ids = [
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    ]
    if len(optimizer_ids) != len(set(optimizer_ids)) or set(optimizer_ids) != set(trainable_ids):
        raise RuntimeError("each trainable parameter must occur in exactly one optimizer group")
    scheduler = WarmupCosineStepScheduler(
        optimizer,
        steps_per_epoch=len(loader),
        total_epochs=total_epochs,
        warmup_epochs=int(config["optimizer"].get("warmup_epochs", 20)),
        peak_lr=float(config["optimizer"].get("lr_peak", 1e-4)),
        minimum_lr=float(config["optimizer"].get("lr_min", 1e-6)),
    )
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, float(train_cfg.get("ema_decay", 0.9999)))
    config_fingerprint = canonical_config_fingerprint(config)
    shared_sha256 = sha256_file(shared_path)
    output = Path(args.output_dir or train_cfg["output_dir"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if not args.resume and (output / "latest.pt").exists():
            raise RuntimeError("output already has latest.pt; use an exact same-run --resume")
        (output / "resolved_config.json").write_text(
            json.dumps(config, indent=2, default=str), encoding="utf-8"
        )
        (output / "environment.json").write_text(
            json.dumps(
                _environment_snapshot(config, config_path, shared_path, world_size),
                indent=2,
            ),
            encoding="utf-8",
        )
        (output / "covariance_provenance.json").write_text(
            json.dumps(covariance_statistics, indent=2), encoding="utf-8"
        )
    if distributed:
        dist.barrier()
    start_epoch = 0
    global_step = 0
    calibrated_weights: dict[str, float] | None = None
    best: dict[str, Any] = {}
    overflow_events = 0
    if args.resume:
        restored = load_checkpoint(
            args.resume,
            model,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            config=config,
            map_location=device,
            scheduler=scheduler,
            expected_config_fingerprint=config_fingerprint,
        )
        state = restored["experiment_state"]
        if state.get("shared_initialization_sha256") != shared_sha256:
            raise RuntimeError("resume checkpoint uses another shared initialization")
        start_epoch = restored["epoch"] + 1
        global_step = restored["global_step"]
        calibrated_weights = state.get("calibrated_auxiliary_weights")
        best = dict(state.get("best_checkpoints", {}))
        overflow_events = int(state.get("amp_overflow_events", 0))
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=True,
        )
    maximum_steps = args.max_steps
    stop = False
    consecutive_overflows = 0
    for epoch in range(start_epoch, total_epochs):
        if maximum_steps is not None and global_step >= maximum_steps:
            break
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_seed = process_seed + epoch * 100_003
        generator.manual_seed(epoch_seed)
        random.seed(epoch_seed)
        np.random.seed(epoch_seed % (2**32))
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(epoch_seed)
        base = model.module if hasattr(model, "module") else model
        if epoch == 20 and calibrated_weights is None:
            local_batches = math.ceil(32 / world_size)
            local_calibration = collect_gradient_ratios(
                base,
                iter(loader),
                covariance_low,
                covariance_high,
                device=device,
                maximum_batches=local_batches,
                t_sampling=str(config["bridge"].get("t_sampling", "endpoint_mixture")),
                endpoint_probability=float(config["bridge"].get("endpoint_probability", 0.15)),
            )
            calibrated_weights, calibration_audit = _aggregate_calibration(
                local_calibration, world_size, rank
            )
            if rank == 0:
                calibration_audit.update(
                    {
                        "experiment": experiment.get("name"),
                        "epoch": 20,
                        "config_fingerprint": config_fingerprint,
                    }
                )
                (output / f"gradient_calibration_{str(experiment.get('id')).lower()}.json").write_text(
                    json.dumps(calibration_audit, indent=2), encoding="utf-8"
                )
        weights = auxiliary_weights_for_epoch(epoch, calibrated_weights)
        model.train()
        for batch in loader:
            if maximum_steps is not None and global_step >= maximum_steps:
                stop = True
                break
            mri = batch["mri"].to(device, non_blocking=True)
            pet = batch["pet"].to(device, non_blocking=True)
            roi = batch.get("roi")
            if roi is not None:
                roi = roi.to(device, non_blocking=True)
            learning_rate = scheduler.prepare_step()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                result = model(
                    mri,
                    pet,
                    covariance_low,
                    covariance_high,
                    t_sampling=str(config["bridge"].get("t_sampling", "endpoint_mixture")),
                    endpoint_probability=float(config["bridge"].get("endpoint_probability", 0.15)),
                    roi=roi,
                    auxiliary_weights=weights,
                )
                loss = result["loss"]
            scale_before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["optimizer"].get("grad_clip_norm", 1.0))
            )
            nonfinite = torch.tensor(
                0 if bool(torch.isfinite(gradient_norm).item()) else 1,
                device=device,
                dtype=torch.int32,
            )
            if distributed:
                dist.all_reduce(nonfinite, op=dist.ReduceOp.MAX)
            if bool(nonfinite.item()):
                if not use_amp:
                    raise FloatingPointError("non-finite gradient norm without AMP")
                overflow_events += 1
                consecutive_overflows += 1
                scale_after = scale_before * float(scaler.get_backoff_factor())
                scaler.update(new_scale=scale_after)
                optimizer.zero_grad(set_to_none=True)
                if rank == 0:
                    _json_line(
                        output / "amp_overflow_events.jsonl",
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "event": "synchronized_optimizer_step_skipped",
                            "scale_before": scale_before,
                            "scale_after": scale_after,
                            "consecutive_overflows": consecutive_overflows,
                            "world_size": world_size,
                        },
                    )
                if consecutive_overflows >= 5:
                    raise FloatingPointError(
                        "persistent AMP overflow: five consecutive synchronized skipped steps"
                    )
                continue
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            consecutive_overflows = 0
            scheduler.step()
            ema.update(base)
            global_step += 1
            reduced = reduce_metrics(result["logs"], world_size)
            if rank == 0:
                _json_line(
                    output / "lr_steps.jsonl",
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "lr": learning_rate,
                    },
                )
                if global_step % int(train_cfg.get("log_every", 10)) == 0:
                    row = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "lr": learning_rate,
                        "gradient_norm_pre_clip": float(gradient_norm.detach().cpu()),
                        "amp_scale": scale_after,
                    }
                    row.update(
                        {
                            name: float(value.detach().cpu())
                            for name, value in reduced.items()
                        }
                    )
                    _json_line(output / "train_metrics.jsonl", row)
        validation_base: dict[str, float] | None = None
        improved: list[str] = []
        validation_due = (epoch + 1) % int(train_cfg.get("validate_every", 10)) == 0
        if validation_due:
            validation_base, base_patients = _validate(
                base,
                validation_loader,
                covariance_low,
                covariance_high,
                world_size,
            )
            validation_ema, ema_patients = _validate(
                ema.module,
                validation_loader,
                covariance_low,
                covariance_high,
                world_size,
            )
            expected_patients = int(train_cfg.get("expected_validation_patients", 43))
            if int(validation_base["patient_count"]) != expected_patients:
                raise RuntimeError(
                    f"formal validation requires {expected_patients} patients; "
                    f"observed {validation_base['patient_count']}"
                )
            if rank == 0:
                improved = _update_best(best, validation_base, epoch)
                for model_name, aggregate, patients in (
                    ("base", validation_base, base_patients),
                    ("ema", validation_ema, ema_patients),
                ):
                    row = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model": model_name,
                        **aggregate,
                    }
                    _json_line(output / "validation_metrics.jsonl", row)
                    per_case = output / (
                        f"validation_patients_epoch_{epoch:04d}_{model_name}.jsonl"
                    )
                    per_case.write_text(
                        "".join(json.dumps(value, sort_keys=True) + "\n" for value in patients),
                        encoding="utf-8",
                    )
        if rank == 0:
            extra = _checkpoint_extra(
                config_fingerprint=config_fingerprint,
                shared_initialization_sha256=shared_sha256,
                calibrated_weights=calibrated_weights,
                best=best,
                overflow_events=overflow_events,
            )
            latest = output / "latest.pt"
            save_checkpoint(
                latest,
                model,
                optimizer=optimizer,
                scaler=scaler,
                ema=ema,
                epoch=epoch,
                global_step=global_step,
                config=config,
                scheduler=scheduler,
                extra=extra,
            )
            if validation_due:
                versioned = output / f"epoch_{epoch:04d}.pt"
                shutil.copy2(latest, versioned)
                for name in improved:
                    shutil.copy2(latest, output / f"best_val_{name}.pt")
                _json_line(
                    output / "checkpoint_manifest.jsonl",
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "latest_sha256": sha256_file(latest),
                        "versioned": versioned.name,
                        "versioned_sha256": sha256_file(versioned),
                        "best_updated": improved,
                        "validation_base": validation_base,
                    },
                )
        if distributed:
            dist.barrier()
        if stop:
            break
    if rank == 0 and maximum_steps is None and total_epochs == 800:
        final_checkpoint = output / "epoch_0799.pt"
        if not final_checkpoint.is_file():
            raise RuntimeError("formal run ended without epoch_0799.pt")
        completion = {
            "status": "COMPLETE",
            "experiment": experiment.get("name"),
            "epochs": 800,
            "last_epoch": 799,
            "global_step": global_step,
            "config_fingerprint": config_fingerprint,
            "shared_initialization_sha256": shared_sha256,
            "epoch_0799_sha256": sha256_file(final_checkpoint),
            "test_executed": False,
        }
        (output / "FORMAL_TRAINING_COMPLETE.json").write_text(
            json.dumps(completion, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
