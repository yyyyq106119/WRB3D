from __future__ import annotations

from pathlib import Path

import torch

from tools.audit_base_extended_500_to_800 import (
    config_differences,
    inspect_checkpoint,
    sha256_file,
    state_dict_sha256,
)


def _config() -> dict:
    return {
        "model": {"name": "base"},
        "loss": {"lambda_low": 1.0},
        "train": {
            "epochs": 500,
            "output_dir": "outputs/stage_a/from_scratch",
            "optimizer": "Adam",
        },
    }


def _write_checkpoint(
    path: Path,
    *,
    include_scheduler: bool = True,
    config: dict | None = None,
) -> tuple[str, str]:
    model = {"weight": torch.arange(4, dtype=torch.float32)}
    payload = {
        "model": model,
        "optimizer": {"param_groups": [{"lr": 1e-4}], "state": {}},
        "scaler": {"scale": 65536.0},
        "epoch": 499,
        "global_step": 39000,
        "config": config or _config(),
    }
    if include_scheduler:
        payload["scheduler"] = {"last_epoch": 499, "T_max": 800}
    torch.save(payload, path)
    return sha256_file(path), state_dict_sha256(model)


def test_config_diff_allows_only_declared_continuation_fields():
    original = _config()
    checkpoint = _config()
    checkpoint["train"] = {
        **checkpoint["train"],
        "epochs": 800,
        "output_dir": "experiments/BASE_EXTENDED_500_TO_800",
    }
    rows, blockers = config_differences(checkpoint, original)
    assert rows
    assert not blockers


def test_config_diff_blocks_loss_change():
    original = _config()
    checkpoint = _config()
    checkpoint["loss"] = {"lambda_low": 2.0}
    _, blockers = config_differences(checkpoint, original)
    assert [row["path"] for row in blockers] == ["loss.lambda_low"]


def test_static_audit_accepts_exact_original_base_state(tmp_path):
    checkpoint = tmp_path / "original_base.pt"
    file_hash, model_hash = _write_checkpoint(checkpoint)
    result = inspect_checkpoint(
        checkpoint,
        expected_sha256=file_hash,
        expected_model_sha256=model_hash,
        expected_epoch=499,
        expected_global_step=39000,
        original_config=_config(),
    )
    assert result["identity_verified"] is True
    assert result["allowed_use"] is True
    assert result["model_base_key"] == "model"
    assert result["ema_key"] is None


def test_static_audit_fails_closed_without_scheduler(tmp_path):
    checkpoint = tmp_path / "original_base.pt"
    file_hash, model_hash = _write_checkpoint(
        checkpoint, include_scheduler=False
    )
    result = inspect_checkpoint(
        checkpoint,
        expected_sha256=file_hash,
        expected_model_sha256=model_hash,
        expected_epoch=499,
        expected_global_step=39000,
        original_config=_config(),
    )
    assert result["identity_verified"] is True
    assert result["allowed_use"] is False
    assert "scheduler_state_present" in result["blockers"]


def test_static_audit_rejects_banned_checkpoint_provenance(tmp_path):
    directory = tmp_path / "e2"
    directory.mkdir()
    checkpoint = directory / "original_base.pt"
    file_hash, model_hash = _write_checkpoint(checkpoint)
    result = inspect_checkpoint(
        checkpoint,
        expected_sha256=file_hash,
        expected_model_sha256=model_hash,
        expected_epoch=499,
        expected_global_step=39000,
        original_config=_config(),
    )
    assert result["identity_verified"] is False
    assert result["allowed_use"] is False
    assert "e2" in result["banned_provenance_keywords"]
