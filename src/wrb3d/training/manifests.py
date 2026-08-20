"""Build immutable patient split manifests before formal training."""

from __future__ import annotations

import argparse
from pathlib import Path

import json

from ..datasets import PairedVolumeDataset, write_split_manifests
from ..utils import load_config, resolve_project_path


def collect_split_records(
    config: dict, splits: tuple[str, ...] = ("train", "val", "test")
) -> dict[str, list[dict]]:
    data = config["data"]
    return {
        split: PairedVolumeDataset(
            data["root"],
            split,
            patch_size=None,
            patient_id_regex=data.get("patient_id_regex"),
        ).manifest_records()
        for split in splits
    }


def verify_train_val_without_test_access(config: dict) -> dict:
    """Verify fixed train/val membership while never discovering test files."""
    records = collect_split_records(config, ("train", "val"))
    directory = resolve_manifest_dir(config)
    patient_sets: dict[str, set[str]] = {}
    for split in ("train", "val"):
        path = directory / f"{split}_patients.json"
        if not path.is_file():
            raise FileNotFoundError(f"required patient manifest is missing: {path}")
        recorded = {str(value) for value in json.loads(path.read_text(encoding="utf-8"))}
        observed = {str(row["patient_id"]) for row in records[split]}
        if observed != recorded:
            raise RuntimeError(f"stale {split} patient manifest")
        patient_sets[split] = observed
    shared = patient_sets["train"] & patient_sets["val"]
    if shared:
        raise RuntimeError(f"patient leakage across train/val: {sorted(shared)[:10]}")
    audit_path = directory / "split_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"required immutable split audit is missing: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("dataset_identity_sha256"):
        raise RuntimeError("split audit has no dataset identity")
    counts = audit.get("patients_per_split", {})
    if counts.get("train") != len(patient_sets["train"]) or counts.get("val") != len(
        patient_sets["val"]
    ):
        raise RuntimeError("split audit train/val counts are stale")
    train_cfg = config.get("train", {})
    expected_train = int(train_cfg.get("expected_train_patients", len(patient_sets["train"])))
    expected_val = int(train_cfg.get("expected_validation_patients", len(patient_sets["val"])))
    if len(patient_sets["train"]) != expected_train or len(patient_sets["val"]) != expected_val:
        raise RuntimeError(
            "formal split cardinality mismatch: "
            f"train={len(patient_sets['train'])}/{expected_train}, "
            f"val={len(patient_sets['val'])}/{expected_val}"
        )
    if any(audit.get("overlaps", {}).values()):
        raise RuntimeError("immutable split audit records patient leakage")
    return audit


def resolve_manifest_dir(config: dict) -> Path:
    return resolve_project_path(config, config["data"].get("manifest_dir", "splits"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and audit fixed patient-level splits")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(args.output_dir) if args.output_dir else resolve_manifest_dir(config)
    audit = write_split_manifests(collect_split_records(config), output)
    print(f"wrote audited patient manifests to {output.resolve()}")
    print(audit)


if __name__ == "__main__":
    main()
