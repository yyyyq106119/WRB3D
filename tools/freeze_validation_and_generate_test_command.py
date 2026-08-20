"""Freeze one validation-selected checkpoint and generate, but never run, test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wrb3d.training.s1s2 import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-experiment", choices=["S1", "S2"], required=True)
    parser.add_argument(
        "--selected-view",
        choices=[
            "best_val_whole_mae",
            "best_val_hotspot_mae",
            "best_val_peak_bias",
            "epoch_0799",
        ],
        required=True,
    )
    parser.add_argument(
        "--comparison",
        default="reports/validation_comparison/paired_comparison.json",
    )
    parser.add_argument("--output", default="artifacts/frozen_test_decision.json")
    args = parser.parse_args()
    comparison_path = Path(args.comparison)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison.get("test_used") is not False:
        raise RuntimeError("comparison is not a validation-only decision artifact")
    run_name = (
        "S1_case_adaptive_corrector_hotspot_aligned_800e"
        if args.selected_experiment == "S1"
        else "S2_no_corrector_hotspot_aligned_800e"
    )
    checkpoint_name = (
        f"{args.selected_view}.pt"
        if args.selected_view != "epoch_0799"
        else "epoch_0799.pt"
    )
    run = Path("outputs") / run_name
    checkpoint = run / checkpoint_name
    config = Path("configs") / (
        "s1_case_adaptive_corrector_hotspot_aligned_800e.yaml"
        if args.selected_experiment == "S1"
        else "s2_no_corrector_hotspot_aligned_800e.yaml"
    )
    if not checkpoint.is_file() or not (run / "FORMAL_TRAINING_COMPLETE.json").is_file():
        raise RuntimeError("selected formal checkpoint/run is incomplete")
    payload = {
        "schema_version": 1,
        "status": "FROZEN_VALIDATION_DECISION",
        "selected_experiment": args.selected_experiment,
        "selected_view": args.selected_view,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(config),
        "config_sha256": sha256_file(config),
        "comparison_sha256": sha256_file(comparison_path),
        "selection_source": "fixed_43_patient_validation_only",
        "test_used_for_selection": False,
        "test_executed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    command = (
        "CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "
        f"python tools/evaluate_frozen_test.py --decision {output.as_posix()} "
        "--output-dir outputs/final_test_frozen"
    )
    Path("artifacts/FINAL_TEST_COMMAND.txt").write_text(command + "\n", encoding="utf-8")
    print(command)


if __name__ == "__main__":
    main()
