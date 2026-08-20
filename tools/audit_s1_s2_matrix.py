"""Audit that S1/S2 differ only by identity, output path, corrector, and gain reg."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wrb3d.utils import load_config


def _differences(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        output = []
        for key in sorted(set(left) | set(right)):
            if key == "_config_path":
                continue
            child = f"{path}.{key}" if path else key
            if key not in left or key not in right:
                output.append({"path": child, "s1": left.get(key), "s2": right.get(key)})
            else:
                output.extend(_differences(left[key], right[key], child))
        return output
    if left != right:
        return [{"path": path, "s1": left, "s2": right}]
    return []


def main() -> None:
    s1 = load_config("configs/s1_case_adaptive_corrector_hotspot_aligned_800e.yaml")
    s2 = load_config("configs/s2_no_corrector_hotspot_aligned_800e.yaml")
    differences = _differences(s1, s2)
    allowed = {
        "experiment.id",
        "experiment.name",
        "model.case_adaptive_wavelet_corrector.enabled",
        "loss.gain_regularization.weight",
        "train.output_dir",
    }
    observed = {row["path"] for row in differences}
    payload = {
        "status": "PASS" if observed == allowed else "FAIL",
        "allowed_difference_paths": sorted(allowed),
        "observed_difference_paths": sorted(observed),
        "differences": differences,
        "s0_present": any(Path("configs").glob("*s0*")),
    }
    Path("artifacts/s1_s2_matrix_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS" or payload["s0_present"]:
        raise RuntimeError("S1/S2 matrix audit failed")


if __name__ == "__main__":
    main()
