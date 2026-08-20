"""Fail-closed CPU preflight for the independent S3 refinement run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wrb3d.training.manifests import verify_train_val_without_test_access
from wrb3d.training.psnr_refine import load_ema_as_base_initialization
from wrb3d.utils import build_model, load_config
from wrb3d.utils.config import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/s3_s2_ema589_psnr_refine_80e.yaml"
    )
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--output", default="metrics/s3_psnr_refine_preflight.json")
    args = parser.parse_args()
    config = load_config(args.config)
    refine = config["psnr_refinement"]
    source = resolve_project_path(
        config, args.source_checkpoint or refine["source_checkpoint"]
    )
    split_audit = verify_train_val_without_test_access(config)
    model = build_model(config)
    provenance = load_ema_as_base_initialization(
        source,
        model,
        config=config,
        expected_epoch=int(refine["source_epoch"]),
        expected_experiment_id=str(refine["source_experiment_id"]),
        map_location="cpu",
    )
    selection = refine["selection"]
    checks = {
        "experiment_kind": config["experiment"].get("kind")
        == "s3_psnr_refinement",
        "source_is_s2": provenance["source_experiment_id"] == "S2",
        "source_epoch_589": provenance["source_epoch"] == 589,
        "embedded_ema_loaded": provenance["kind"]
        == "s2_embedded_ema_to_new_base",
        "optimizer_reset": provenance["optimizer_restored"] is False,
        "scheduler_reset": provenance["scheduler_restored"] is False,
        "scaler_reset": provenance["scaler_restored"] is False,
        "four_source_auxiliary_weights": len(
            provenance["calibrated_auxiliary_weights"]
        )
        == 4,
        "corrector_disabled": config["model"]["case_adaptive_wavelet_corrector"][
            "enabled"
        ]
        is False,
        "raw_mse_no_clamp": refine["mse_domain"] == "raw_pet_no_clamp",
        "thirty_two_train_calibration_batches": int(
            refine["gradient_calibration_batches"]
        )
        == 32,
        "target_gradient_ratio_0p10": float(
            refine["target_mse_gradient_ratio"]
        )
        == 0.10,
        "target_psnr_24": float(selection["target_psnr"]) == 24.0,
        "msssim_guard": float(selection["msssim_max_drop"]) == 0.001,
        "hotspot_guard": float(selection["hotspot_mae_max_ratio"]) == 1.02,
        "no_test_access": split_audit is not None,
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source_initialization": provenance,
        "selection": selection,
        "test_executed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not all(checks.values()):
        raise RuntimeError("S3 preflight failed")


if __name__ == "__main__":
    main()
