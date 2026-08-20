"""Fail unless every same-name S1/S2 backbone tensor is exactly identical."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from wrb3d.training.s1s2 import load_shared_backbone_initialization, sha256_file
from wrb3d.utils import build_model, load_config


def _construction_config(path: str) -> dict:
    config = load_config(path)
    config["loss"]["high_band_stds"] = [1.0] * 7
    config["loss"]["aligned_amplitude"]["band_epsilon"] = [1e-8] * 7
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--s1-config", default="configs/s1_case_adaptive_corrector_hotspot_aligned_800e.yaml"
    )
    parser.add_argument(
        "--s2-config", default="configs/s2_no_corrector_hotspot_aligned_800e.yaml"
    )
    parser.add_argument(
        "--initialization", default="artifacts/shared_backbone_init_seed1234.pt"
    )
    parser.add_argument(
        "--output", default="artifacts/shared_initialization_audit.json"
    )
    args = parser.parse_args()
    s1 = build_model(_construction_config(args.s1_config))
    s2 = build_model(_construction_config(args.s2_config))
    load_shared_backbone_initialization(s1, args.initialization)
    load_shared_backbone_initialization(s2, args.initialization)
    s1_state = s1.state_dict()
    s2_state = s2.state_dict()
    shared_keys = sorted(
        key
        for key in s1_state
        if key.startswith("low_model.") or key.startswith("high_model.")
    )
    mismatches = [
        key for key in shared_keys if not torch.equal(s1_state[key], s2_state[key])
    ]
    s2_corrector_keys = [
        key for key in s2_state if "case_adaptive_wavelet_corrector" in key
    ]
    audit = {
        "status": "PASS" if not mismatches and not s2_corrector_keys else "FAIL",
        "initialization_sha256": sha256_file(args.initialization),
        "shared_tensor_count": len(shared_keys),
        "elementwise_identical": not mismatches,
        "mismatches": mismatches,
        "s1_corrector_identity_gain_verified_separately": True,
        "s2_corrector_parameter_count": len(s2_corrector_keys),
    }
    Path(args.output).write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "PASS":
        raise RuntimeError("shared initialization audit failed")


if __name__ == "__main__":
    main()
