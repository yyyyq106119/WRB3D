"""Estimate train-only wavelet residual standard deviations and bridge covariance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..datasets import PairedVolumeDataset
from ..utils import load_config
from ..wavelets import DWT3D, compute_residuals
from .manifests import verify_train_val_without_test_access


class _Moments:
    def __init__(self, bands: int) -> None:
        self.count = torch.zeros(bands, dtype=torch.float64)
        self.total = torch.zeros(bands, dtype=torch.float64)
        self.square = torch.zeros(bands, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        bands = self.count.numel()
        reshaped = values.detach().double().reshape(values.shape[0], bands, -1)
        self.count += reshaped.shape[0] * reshaped.shape[2]
        self.total += reshaped.sum(dim=(0, 2)).cpu()
        self.square += reshaped.square().sum(dim=(0, 2)).cpu()

    def std(self) -> torch.Tensor:
        mean = self.total / self.count.clamp_min(1)
        variance = self.square / self.count.clamp_min(1) - mean.square()
        return variance.clamp_min(0).sqrt().float()


def estimate(config: dict, max_cases: int | None = None) -> dict:
    data = config["data"]
    if bool(data.get("require_patient_manifests", True)):
        manifest_audit = verify_train_val_without_test_access(config)
    else:
        raise RuntimeError("formal S1/S2 statistics require immutable patient manifests")
    dataset = PairedVolumeDataset(
        data["root"], "train", patch_size=None, patient_id_regex=data.get("patient_id_regex")
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    transform = DWT3D()
    low = _Moments(1)
    high = _Moments(7)
    patients = set()
    cases = 0
    for batch in loader:
        mri = batch["mri"].float()
        pet = batch["pet"].float()
        mri_low, mri_high, _ = transform(mri)
        pet_low, pet_high, _ = transform(pet)
        residual = compute_residuals(mri_low, mri_high, pet_low, pet_high)
        low.update(residual.low)
        high.update(residual.high.reshape(residual.high.shape[0], 7, -1))
        patients.update(batch["patient_id"])
        cases += 1
        if max_cases is not None and cases >= max_cases:
            break
    low_std = low.std()[0]
    high_std = high.std()
    q_low = float(config["covariance"].get("q_low", 0.5))
    q_high = float(config["covariance"].get("q_high", 0.5))
    return {
        "std_low_residual": float(low_std),
        "std_high_residual": [float(value) for value in high_std],
        "omega_low": float(2.0 * q_low**2 * low_std**2),
        "omega_high": [float(value) for value in 2.0 * q_high**2 * high_std.square()],
        "midpoint_noise_std_low": float((0.5 * 2.0 * q_low**2 * low_std**2).sqrt()),
        "midpoint_noise_std_high": [
            float(value) for value in (0.5 * 2.0 * q_high**2 * high_std.square()).sqrt()
        ],
        "noise_to_residual_std_ratio_low": q_low,
        "noise_to_residual_std_ratio_high": q_high,
        "provenance": {
            "split": "train",
            "dataset_root": str(Path(data["root"]).expanduser().resolve()),
            "dataset_identity_sha256": manifest_audit["dataset_identity_sha256"],
            "case_count": cases,
            "patient_count": len(patients),
            "full_volumes": True,
            "max_cases": max_cases,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    result = estimate(load_config(args.config), args.max_cases)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
