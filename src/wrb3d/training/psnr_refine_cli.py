"""Independent S3 refinement: verified S2 EMA to Base plus weak raw-PET MSE."""

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
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from ..datasets import PairedVolumeDataset
from ..losses import AuxiliaryWeights
from ..utils import build_model, load_config, load_covariances
from ..utils.config import resolve_project_path
from .checkpoint import load_checkpoint, save_checkpoint
from .cli import _json_line, _validate, _worker_init
from .distributed import DistributedEvalSampler, initialize_distributed, reduce_metrics
from .ema import EMA
from .manifests import verify_train_val_without_test_access
from .psnr_refine import (
    collect_mse_gradient_ratios,
    load_ema_as_base_initialization,
    mse_weight_for_epoch,
    solve_mse_weight,
    update_refinement_best,
    validated_auxiliary_weights,
)
from .s1s2 import WarmupCosineStepScheduler, canonical_config_fingerprint, sha256_file


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audited S3 PSNR refinement trainer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def _environment_snapshot(
    config: dict[str, Any],
    config_path: Path,
    source_path: Path,
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
                    "capability": [properties.major, properties.minor],
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
        "world_size": int(world_size),
        "gpus": gpu_rows,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "config_fingerprint": canonical_config_fingerprint(config),
        "source_checkpoint_path": str(source_path.resolve()),
        "source_checkpoint_sha256": sha256_file(source_path),
        "test_accessed": False,
    }


def _aggregate_mse_calibration(
    local: dict[str, Any],
    *,
    world_size: int,
    rank: int,
    required_batches: int,
    target_ratio: float,
) -> tuple[float, dict[str, Any]]:
    if world_size > 1:
        gathered: list[Any] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    ratios = [value for payload in gathered for value in payload["ratios"]]
    case_ids = [value for payload in gathered for value in payload["case_ids"]]
    reference_norms = [
        value for payload in gathered for value in payload["reference_gradient_norms"]
    ]
    mse_norms = [
        value for payload in gathered for value in payload["mse_gradient_norms"]
    ]
    broadcast: list[Any] = [None]
    if rank == 0:
        weight, audit = solve_mse_weight(
            ratios,
            target_weighted_ratio=target_ratio,
            required_batches=required_batches,
        )
        audit.update(
            {
                "schema_version": 1,
                "status": "PASS",
                "kind": "raw_pet_mse_gradient_calibration",
                "reference_loss": "existing_weighted_raw_image_loss",
                "case_ids": case_ids[:required_batches],
                "reference_gradient_norms": reference_norms[:required_batches],
                "mse_gradient_norms": mse_norms[:required_batches],
                "world_size": int(world_size),
                "train_split_only": True,
            }
        )
        broadcast[0] = (weight, audit)
    if world_size > 1:
        dist.broadcast_object_list(broadcast, src=0)
    return broadcast[0]


def _checkpoint_extra(
    *,
    config_fingerprint: str,
    source: dict[str, Any],
    source_auxiliary_weights: dict[str, float],
    mse_calibration: dict[str, Any],
    baseline: dict[str, float],
    best: dict[str, Any],
    overflow_events: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "s3_psnr_refinement",
        "config_fingerprint": config_fingerprint,
        "source_initialization": source,
        "frozen_source_auxiliary_weights": source_auxiliary_weights,
        "mse_gradient_calibration": mse_calibration,
        "selection_baseline": baseline,
        "best_checkpoints": best,
        "amp_overflow_events": int(overflow_events),
        "test_accessed": False,
    }


def main() -> None:
    args = _arguments()
    config_path = Path(args.config)
    config = load_config(config_path)
    experiment = config.get("experiment", {})
    refine_cfg = config.get("psnr_refinement", {})
    train_cfg = config["train"]
    if str(experiment.get("kind")) != "s3_psnr_refinement":
        raise RuntimeError("this entry point requires experiment.kind=s3_psnr_refinement")
    configured_epochs = int(experiment.get("epochs", train_cfg.get("epochs", 80)))
    total_epochs = int(args.max_epochs or configured_epochs)
    if args.max_steps is None and total_epochs != configured_epochs:
        raise RuntimeError("formal S3 must use its predeclared epoch budget")
    source_value = args.source_checkpoint or refine_cfg.get("source_checkpoint")
    if not source_value:
        raise RuntimeError("S3 requires a predeclared source_checkpoint")
    source_path = resolve_project_path(config, source_value)
    expected_source_sha256 = refine_cfg.get("source_checkpoint_sha256")
    if expected_source_sha256 and sha256_file(source_path) != str(expected_source_sha256):
        raise RuntimeError("S3 source checkpoint SHA256 does not match the frozen config")
    distributed, rank, local_rank, world_size = initialize_distributed()
    expected_gpus = int(train_cfg.get("expected_gpus", 4))
    if args.max_steps is None and world_size != expected_gpus:
        raise RuntimeError(
            f"formal S3 requires DDP world_size={expected_gpus}, observed {world_size}"
        )
    seed = int(experiment.get("seed", train_cfg.get("seed", 1234)))
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
    generator = torch.Generator().manual_seed(process_seed)
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
    source_provenance: dict[str, Any] = {}
    if not args.resume:
        source_provenance = load_ema_as_base_initialization(
            source_path,
            model,
            config=config,
            expected_epoch=int(refine_cfg.get("source_epoch", 589)),
            expected_experiment_id=str(refine_cfg.get("source_experiment_id", "S2")),
            map_location=device,
        )
    covariance_low, covariance_high, covariance_statistics = load_covariances(config)
    covariance_low = covariance_low.to(device)
    covariance_high = covariance_high.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["optimizer"].get("lr_peak", 5e-6)),
        betas=tuple(float(value) for value in config["optimizer"].get("betas", [0.9, 0.999])),
        weight_decay=float(config["optimizer"].get("weight_decay", 0.0)),
    )
    scheduler = WarmupCosineStepScheduler(
        optimizer,
        steps_per_epoch=len(loader),
        total_epochs=total_epochs,
        warmup_epochs=int(config["optimizer"].get("warmup_epochs", 5)),
        peak_lr=float(config["optimizer"].get("lr_peak", 5e-6)),
        minimum_lr=float(config["optimizer"].get("lr_min", 5e-7)),
    )
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, float(train_cfg.get("ema_decay", 0.9999)))
    config_fingerprint = canonical_config_fingerprint(config)
    output = Path(args.output_dir or train_cfg["output_dir"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if not args.resume and (output / "latest.pt").exists():
            raise RuntimeError("S3 output already has latest.pt; use exact-run --resume")
        (output / "resolved_config.json").write_text(
            json.dumps(config, indent=2, default=str), encoding="utf-8"
        )
        (output / "environment.json").write_text(
            json.dumps(
                _environment_snapshot(config, config_path, source_path, world_size),
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
    source_auxiliary_weights: dict[str, float] = {}
    mse_calibration: dict[str, Any] = {}
    baseline: dict[str, float] = {}
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
        if state.get("kind") != "s3_psnr_refinement":
            raise RuntimeError("resume checkpoint is not an S3 PSNR refinement run")
        source_provenance = dict(state["source_initialization"])
        if source_provenance.get("checkpoint_sha256") != sha256_file(source_path):
            raise RuntimeError("S3 resume source checkpoint provenance changed")
        source_auxiliary_weights = validated_auxiliary_weights(
            state.get("frozen_source_auxiliary_weights")
        )
        mse_calibration = dict(state["mse_gradient_calibration"])
        baseline = dict(state["selection_baseline"])
        best = dict(state.get("best_checkpoints", {}))
        overflow_events = int(state.get("amp_overflow_events", 0))
        start_epoch = int(restored["epoch"]) + 1
        global_step = int(restored["global_step"])
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=True,
        )
    base = model.module if hasattr(model, "module") else model
    if not args.resume:
        baseline, baseline_patients = _validate(
            base,
            validation_loader,
            covariance_low,
            covariance_high,
            world_size,
        )
        expected_patients = int(train_cfg.get("expected_validation_patients", 43))
        if int(baseline["patient_count"]) != expected_patients:
            raise RuntimeError(
                f"S3 baseline requires {expected_patients} validation patients; "
                f"observed {baseline['patient_count']}"
            )
        source_auxiliary_weights = validated_auxiliary_weights(
            source_provenance["calibrated_auxiliary_weights"]
        )
        required_batches = int(refine_cfg.get("gradient_calibration_batches", 32))
        local_batches = math.ceil(required_batches / world_size)
        if sampler is not None:
            sampler.set_epoch(0)
        calibration_seed = int(refine_cfg.get("gradient_calibration_seed", seed + 30_003))
        random.seed(calibration_seed + rank)
        np.random.seed((calibration_seed + rank) % (2**32))
        torch.manual_seed(calibration_seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(calibration_seed + rank)
        local_calibration = collect_mse_gradient_ratios(
            base,
            iter(loader),
            covariance_low,
            covariance_high,
            device=device,
            maximum_batches=local_batches,
            t_sampling=str(config["bridge"].get("t_sampling", "endpoint_mixture")),
            endpoint_probability=float(config["bridge"].get("endpoint_probability", 0.15)),
        )
        mse_weight, mse_calibration = _aggregate_mse_calibration(
            local_calibration,
            world_size=world_size,
            rank=rank,
            required_batches=required_batches,
            target_ratio=float(refine_cfg.get("target_mse_gradient_ratio", 0.10)),
        )
        mse_calibration.update(
            {
                "experiment": experiment.get("name"),
                "config_fingerprint": config_fingerprint,
                "source_checkpoint_sha256": source_provenance["checkpoint_sha256"],
            }
        )
        if rank == 0:
            (output / "source_initialization.json").write_text(
                json.dumps(source_provenance, indent=2), encoding="utf-8"
            )
            (output / "selection_baseline.json").write_text(
                json.dumps(baseline, indent=2), encoding="utf-8"
            )
            (output / "mse_gradient_calibration.json").write_text(
                json.dumps(mse_calibration, indent=2), encoding="utf-8"
            )
            _json_line(
                output / "validation_metrics.jsonl",
                {"epoch": -1, "global_step": 0, "model": "source_s2_ema_as_base", **baseline},
            )
            (output / "validation_patients_source_s2_ema_as_base.jsonl").write_text(
                "".join(json.dumps(value, sort_keys=True) + "\n" for value in baseline_patients),
                encoding="utf-8",
            )
            baseline_guards = {
                "msssim_guard": True,
                "hotspot_mae_guard": True,
                "raw_out_of_range_guard": True,
            }
            for name, metric, value in (
                ("psnr", "psnr", baseline["psnr"]),
                ("whole_mae", "mae", baseline["mae"]),
                ("hotspot_mae", "hotspot_mae", baseline["hotspot_mae"]),
                ("msssim", "msssim3d", baseline["msssim3d"]),
            ):
                best[name] = {
                    "epoch": -1,
                    "metric": metric,
                    "value": float(value),
                    "source_baseline": True,
                }
            best["psnr_guarded"] = {
                "epoch": -1,
                "metric": "psnr",
                "value": float(baseline["psnr"]),
                "constraints": baseline_guards,
                "source_baseline": True,
            }
            if float(baseline["psnr"]) >= float(
                refine_cfg.get("selection", {}).get("target_psnr", 24.0)
            ):
                best["psnr_target_guarded"] = dict(best["psnr_guarded"])
            initial_extra = _checkpoint_extra(
                config_fingerprint=config_fingerprint,
                source=source_provenance,
                source_auxiliary_weights=source_auxiliary_weights,
                mse_calibration=mse_calibration,
                baseline=baseline,
                best=best,
                overflow_events=0,
            )
            initial_checkpoint = output / "source_s2_ema_as_base.pt"
            save_checkpoint(
                initial_checkpoint,
                model,
                optimizer=optimizer,
                scaler=scaler,
                ema=ema,
                epoch=-1,
                global_step=0,
                config=config,
                scheduler=scheduler,
                extra=initial_extra,
            )
            for name in best:
                shutil.copy2(initial_checkpoint, output / f"best_val_{name}.pt")
    else:
        mse_weight = float(mse_calibration["calibrated_weight"])
    fixed_auxiliary = AuxiliaryWeights(**source_auxiliary_weights)
    maximum_steps = args.max_steps
    stop = False
    consecutive_overflows = 0
    warmup_epochs = int(refine_cfg.get("mse_warmup_epochs", 10))
    selection = dict(refine_cfg.get("selection", {}))
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
        actual_mse_weight = mse_weight_for_epoch(epoch, mse_weight, warmup_epochs)
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
                    auxiliary_weights=fixed_auxiliary,
                )
                original_loss = result["loss"]
                raw_mse = F.mse_loss(result["B_raw"].float(), pet.float())
                loss = original_loss + actual_mse_weight * raw_mse
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite S3 total loss")
                result["logs"]["loss_pre_psnr_refinement"] = original_loss.detach()
                result["logs"]["loss_raw_pet_mse"] = raw_mse.detach()
                result["logs"]["lambda_raw_pet_mse_actual"] = torch.tensor(
                    actual_mse_weight, device=device
                )
                result["logs"]["loss_total"] = loss.detach()
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
                    raise FloatingPointError("non-finite S3 gradient norm without AMP")
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
                        "persistent S3 AMP overflow: five consecutive synchronized skipped steps"
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
                    {"epoch": epoch, "global_step": global_step, "lr": learning_rate},
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
                        {name: float(value.detach().cpu()) for name, value in reduced.items()}
                    )
                    _json_line(output / "train_metrics.jsonl", row)
        validation_due = (epoch + 1) % int(train_cfg.get("validate_every", 5)) == 0
        validation_base: dict[str, float] | None = None
        improved: list[str] = []
        guard_checks: dict[str, bool] = {}
        if validation_due:
            validation_base, base_patients = _validate(
                base, validation_loader, covariance_low, covariance_high, world_size
            )
            validation_ema, ema_patients = _validate(
                ema.module, validation_loader, covariance_low, covariance_high, world_size
            )
            expected_patients = int(train_cfg.get("expected_validation_patients", 43))
            if int(validation_base["patient_count"]) != expected_patients:
                raise RuntimeError(
                    f"formal S3 validation requires {expected_patients} patients; "
                    f"observed {validation_base['patient_count']}"
                )
            if rank == 0:
                improved, guard_checks = update_refinement_best(
                    best, validation_base, baseline, selection, epoch
                )
                for model_name, aggregate, patients in (
                    ("base", validation_base, base_patients),
                    ("ema", validation_ema, ema_patients),
                ):
                    _json_line(
                        output / "validation_metrics.jsonl",
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "model": model_name,
                            **aggregate,
                        },
                    )
                    (output / f"validation_patients_epoch_{epoch:04d}_{model_name}.jsonl").write_text(
                        "".join(json.dumps(value, sort_keys=True) + "\n" for value in patients),
                        encoding="utf-8",
                    )
        if rank == 0:
            extra = _checkpoint_extra(
                config_fingerprint=config_fingerprint,
                source=source_provenance,
                source_auxiliary_weights=source_auxiliary_weights,
                mse_calibration=mse_calibration,
                baseline=baseline,
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
                        "guard_checks": guard_checks,
                        "validation_base": validation_base,
                    },
                )
        if distributed:
            dist.barrier()
        if stop:
            break
    if rank == 0 and maximum_steps is None and total_epochs == configured_epochs:
        final_checkpoint = output / f"epoch_{configured_epochs - 1:04d}.pt"
        if not final_checkpoint.is_file():
            raise RuntimeError(f"formal S3 ended without {final_checkpoint.name}")
        guarded = best.get("psnr_guarded")
        target = float(selection.get("target_psnr", 24.0))
        completion = {
            "status": "COMPLETE",
            "experiment": experiment.get("name"),
            "epochs": configured_epochs,
            "last_epoch": configured_epochs - 1,
            "global_step": global_step,
            "config_fingerprint": config_fingerprint,
            "source_initialization": source_provenance,
            "mse_gradient_calibration": mse_calibration,
            "selection_baseline": baseline,
            "best_checkpoints": best,
            "target_psnr": target,
            "target_reached_under_all_guards": bool(
                guarded is not None and float(guarded["value"]) >= target
            ),
            "final_checkpoint_sha256": sha256_file(final_checkpoint),
            "test_executed": False,
        }
        (output / "S3_REFINEMENT_COMPLETE.json").write_text(
            json.dumps(completion, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
