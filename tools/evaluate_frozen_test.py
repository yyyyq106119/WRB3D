"""Run the single frozen test evaluation; this script is never called automatically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wrb3d.datasets import PairedVolumeDataset
from wrb3d.metrics import formal_case_metrics
from wrb3d.training import load_checkpoint
from wrb3d.training.s1s2 import canonical_config_fingerprint, sha256_file
from wrb3d.utils import build_model, load_config, load_covariances


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    decision_path = Path(args.decision)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("status") != "FROZEN_VALIDATION_DECISION":
        raise RuntimeError("test requires a frozen validation decision")
    checkpoint = Path(decision["checkpoint"])
    if sha256_file(checkpoint) != decision["checkpoint_sha256"]:
        raise RuntimeError("frozen checkpoint hash mismatch")
    config = load_config(decision["config"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    load_checkpoint(
        checkpoint,
        model,
        config=config,
        map_location=device,
        expected_config_fingerprint=canonical_config_fingerprint(config),
    )
    covariance_low, covariance_high, _ = load_covariances(config)
    covariance_low = covariance_low.to(device)
    covariance_high = covariance_high.to(device)
    dataset = PairedVolumeDataset(
        config["data"]["root"],
        "test",
        tuple(config["data"].get("patch_size", [128, 128, 64])),
        config["data"].get("patient_id_regex"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    rows = []
    model.eval()
    for batch in loader:
        mri = batch["mri"].to(device)
        pet = batch["pet"].to(device)
        inference = model.infer(
            mri, covariance_low, covariance_high, num_steps=1, stochastic=False
        )
        metrics = formal_case_metrics(model, mri, pet, inference)
        rows.append(
            {
                "case_id": str(batch["case_id"][0]),
                "patient_id": str(batch["patient_id"][0]),
                "metrics": {key: float(value.cpu()) for key, value in metrics.items()},
            }
        )
    patients = {row["patient_id"] for row in rows}
    if len(patients) != 88:
        raise RuntimeError(f"frozen test requires 88 patients, observed {len(patients)}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "test_per_case.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        key: sum(row["metrics"][key] for row in rows) / len(rows)
        for key in rows[0]["metrics"]
    }
    payload = {
        "status": "COMPLETE",
        "decision_sha256": sha256_file(decision_path),
        "checkpoint_sha256": decision["checkpoint_sha256"],
        "patient_count": 88,
        "case_count": len(rows),
        "summary": summary,
    }
    (output / "test_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
