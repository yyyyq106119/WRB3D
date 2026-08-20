"""One-batch/one-step/one-validation-case/checkpoint roundtrip for S1 and S2."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile

import torch

from wrb3d.metrics import formal_case_metrics
from wrb3d.training import EMA, load_checkpoint, save_checkpoint
from wrb3d.training.s1s2 import (
    WarmupCosineStepScheduler,
    canonical_config_fingerprint,
    load_shared_backbone_initialization,
    sha256_file,
)
from wrb3d.utils import build_model, load_config


def _config(path: str) -> dict:
    config = load_config(path)
    config["loss"]["high_band_stds"] = [1.0] * 7
    config["loss"]["aligned_amplitude"]["band_epsilon"] = [1e-10] * 7
    return config


def _run(identifier: str, config_path: str, shared_path: str, root: Path) -> dict:
    config = _config(config_path)
    torch.manual_seed(1234)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    load_shared_backbone_initialization(model, shared_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = WarmupCosineStepScheduler(
        optimizer, steps_per_epoch=1, total_epochs=800
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = EMA(model, 0.9999)
    mri = torch.rand(1, 1, 8, 8, 8, device=device)
    pet = torch.rand(1, 1, 8, 8, 8, device=device)
    optimizer.zero_grad(set_to_none=True)
    lr = scheduler.prepare_step()
    with torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        result = model.forward_train(
            mri,
            pet,
            torch.tensor(0.1, device=device),
            torch.full((7,), 0.1, device=device),
            t=torch.tensor([5], device=device),
            noise_low=torch.zeros(1, 1, 4, 4, 4, device=device),
            noise_high=torch.zeros(1, 7, 4, 4, 4, device=device),
            auxiliary_weights={
                "hotspot": 1e-3,
                "underestimation": 1e-3,
                "aligned_amplitude": 1e-3,
                "orthogonal_error": 1e-3,
            },
        )
    scaler.scale(result["loss"]).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not bool(torch.isfinite(norm).item()):
        raise FloatingPointError(f"{identifier} smoke produced a non-finite gradient norm")
    before_step = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    scaler.step(optimizer)
    scaler.update()
    optimizer_step_changed_parameters = any(
        not torch.equal(before_step[name], parameter.detach())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if not optimizer_step_changed_parameters:
        raise RuntimeError(f"{identifier} smoke did not complete an optimizer update")
    scheduler.step()
    ema.update(model)
    model.eval()
    with torch.no_grad():
        inference = model.infer(
            mri,
            torch.tensor(0.1, device=device),
            torch.full((7,), 0.1, device=device),
            num_steps=1,
            stochastic=False,
        )
        validation = formal_case_metrics(model, mri, pet, inference)
    directory = root / identifier
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "roundtrip.pt"
    fingerprint = canonical_config_fingerprint(config)
    shared_sha = sha256_file(shared_path)
    save_checkpoint(
        checkpoint,
        model,
        optimizer=optimizer,
        scaler=scaler,
        ema=ema,
        epoch=0,
        global_step=1,
        config=config,
        scheduler=scheduler,
        extra={
            "config_fingerprint": fingerprint,
            "shared_initialization_sha256": shared_sha,
        },
    )
    restored_model = build_model(copy.deepcopy(config)).to(device)
    load_shared_backbone_initialization(restored_model, shared_path)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-4)
    restored_scheduler = WarmupCosineStepScheduler(
        restored_optimizer, steps_per_epoch=1, total_epochs=800
    )
    restored_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    restored_ema = EMA(restored_model, 0.9999)
    restored = load_checkpoint(
        checkpoint,
        restored_model,
        optimizer=restored_optimizer,
        scaler=restored_scaler,
        ema=restored_ema,
        config=config,
        scheduler=restored_scheduler,
        expected_config_fingerprint=fingerprint,
        map_location=device,
    )
    tensors_equal = all(
        torch.equal(value, restored_model.state_dict()[key])
        for key, value in model.state_dict().items()
    )
    if not tensors_equal or restored_scheduler.state_dict() != scheduler.state_dict():
        raise RuntimeError(f"{identifier} checkpoint roundtrip mismatch")
    return {
        "status": "PASS",
        "experiment": identifier,
        "device": str(device),
        "one_train_batch": True,
        "one_backward": True,
        "one_optimizer_step": optimizer_step_changed_parameters,
        "one_validation_case": True,
        "checkpoint_roundtrip": True,
        "loss": float(result["loss"].detach().cpu()),
        "gradient_norm": float(norm.detach().cpu()),
        "first_step_lr": lr,
        "validation_metric_count": len(validation),
        "checkpoint_sha256": sha256_file(checkpoint),
        "restored_epoch": restored["epoch"],
        "restored_global_step": restored["global_step"],
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="wrb3d_s1s2_smoke_") as directory:
        root = Path(directory)
        rows = [
            _run(
                "S2",
                "configs/s2_no_corrector_hotspot_aligned_800e.yaml",
                "artifacts/shared_backbone_init_seed1234.pt",
                root,
            ),
            _run(
                "S1",
                "configs/s1_case_adaptive_corrector_hotspot_aligned_800e.yaml",
                "artifacts/shared_backbone_init_seed1234.pt",
                root,
            ),
        ]
    payload = {"status": "PASS", "experiments": rows}
    Path("artifacts/engineering_smoke_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
