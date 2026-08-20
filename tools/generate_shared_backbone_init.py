"""Generate exactly one seed-1234 low/high backbone initialization artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from wrb3d.training.s1s2 import sha256_file
from wrb3d.utils import build_model, load_config


def _construction_config(path: str) -> dict:
    config = load_config(path)
    config["loss"]["high_band_stds"] = [1.0] * 7
    config["loss"]["aligned_amplitude"]["band_epsilon"] = [1e-8] * 7
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/s2_no_corrector_hotspot_aligned_800e.yaml"
    )
    parser.add_argument(
        "--output", default="artifacts/shared_backbone_init_seed1234.pt"
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"shared initialization already exists: {output}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_model(_construction_config(args.config))
    backbone = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith("low_model.") or key.startswith("high_model.")
    }
    payload = {
        "schema_version": 1,
        "kind": "shared_random_backbone",
        "seed": args.seed,
        "source_checkpoint": None,
        "old_optimizer_loaded": False,
        "parameter_tensor_count": len(backbone),
        "backbone": backbone,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    manifest = {
        "path": str(output.resolve()),
        "sha256": sha256_file(output),
        "seed": args.seed,
        "parameter_tensor_count": len(backbone),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
