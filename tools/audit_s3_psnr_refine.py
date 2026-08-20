"""Audit a completed S3 run without accessing the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wrb3d.training.s1s2 import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", default="outputs/S3_S2_ema589_psnr_refine_80e"
    )
    args = parser.parse_args()
    run = Path(args.run)
    completion_path = run / "S3_REFINEMENT_COMPLETE.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    best = completion["best_checkpoints"]
    guarded = best.get("psnr_guarded")
    target_guarded = best.get("psnr_target_guarded")
    final = run / f"epoch_{int(completion['last_epoch']):04d}.pt"
    guarded_path = run / "best_val_psnr_guarded.pt"
    checks = {
        "complete": completion.get("status") == "COMPLETE",
        "source_s2_epoch589_ema": completion["source_initialization"].get(
            "source_epoch"
        )
        == 589
        and completion["source_initialization"].get("source_experiment_id")
        == "S2",
        "optimizer_not_restored": completion["source_initialization"].get(
            "optimizer_restored"
        )
        is False,
        "mse_calibration_pass": completion["mse_gradient_calibration"].get(
            "status"
        )
        == "PASS",
        "guarded_candidate_exists": guarded is not None and guarded_path.is_file(),
        "final_checkpoint_hash": final.is_file()
        and sha256_file(final) == completion.get("final_checkpoint_sha256"),
        "no_test_access": completion.get("test_executed") is False,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "baseline": completion["selection_baseline"],
        "best_psnr_guarded": guarded,
        "best_psnr_target_guarded": target_guarded,
        "target_psnr": completion["target_psnr"],
        "target_reached_under_all_guards": completion[
            "target_reached_under_all_guards"
        ],
        "required_action": (
            "FREEZE_BEST_VAL_PSNR_TARGET_GUARDED"
            if target_guarded is not None
            else (
                "KEEP_SOURCE_S2_EMA589"
                if guarded is not None and int(guarded["epoch"]) == -1
                else "FREEZE_BEST_VAL_PSNR_GUARDED_AS_BEST_EFFORT"
            )
        ),
    }
    print(json.dumps(result, indent=2))
    if not all(checks.values()):
        raise RuntimeError("S3 completion audit failed")


if __name__ == "__main__":
    main()
