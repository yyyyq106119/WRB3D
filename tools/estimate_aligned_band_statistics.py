"""Estimate train-only centered GT high-band energy medians for S1/S2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wrb3d.datasets import PairedVolumeDataset
from wrb3d.training.manifests import verify_train_val_without_test_access
from wrb3d.utils import load_config
from wrb3d.wavelets import DWT3D


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    manifest = verify_train_val_without_test_access(config)
    dataset = PairedVolumeDataset(
        data["root"],
        "train",
        patch_size=None,
        patient_id_regex=data.get("patient_id_regex"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    dwt = DWT3D()
    rows: list[torch.Tensor] = []
    patients: set[str] = set()
    case_ids: list[str] = []
    for batch in loader:
        mri = batch["mri"].float()
        pet = batch["pet"].float()
        _, mri_high, _ = dwt(mri)
        _, pet_high, _ = dwt(pet)
        residual = (pet_high - mri_high).reshape(1, 7, 1, *mri_high.shape[-3:])
        centered = residual - residual.mean(dim=(-3, -2, -1), keepdim=True)
        rows.append(centered.square().mean(dim=(2, 3, 4, 5)).squeeze(0))
        patients.update(str(value) for value in batch["patient_id"])
        case_ids.extend(str(value) for value in batch["case_id"])
    energies = torch.stack(rows)
    medians = energies.median(dim=0).values
    epsilon = (1e-3 * medians).clamp_min(torch.finfo(torch.float32).tiny)
    payload = {
        "schema_version": 1,
        "kind": "train_centered_gt_high_band_energy",
        "band_order": ["HLL", "LHL", "HHL", "LLH", "HLH", "LHH", "HHH"],
        "median_mean_centered_gt_energy": [float(value) for value in medians],
        "epsilon_scale": 0.001,
        "band_epsilon": [float(value) for value in epsilon],
        "provenance": {
            "split": "train",
            "dataset_root": str(Path(data["root"]).expanduser().resolve()),
            "dataset_identity_sha256": manifest["dataset_identity_sha256"],
            "case_count": len(case_ids),
            "patient_count": len(patients),
            "full_volumes": True,
            "case_id_sha256": hashlib.sha256(
                "\n".join(sorted(case_ids)).encode("utf-8")
            ).hexdigest(),
        },
    }
    expected = int(config["train"].get("expected_train_patients", len(patients)))
    if len(patients) != expected:
        raise RuntimeError(
            f"expected {expected} training patients, observed {len(patients)}"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
