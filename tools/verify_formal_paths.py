#!/usr/bin/env python3
"""Fail fast when a formal config resolves package-relative files incorrectly."""

from __future__ import annotations

import json
from pathlib import Path

from wrb3d.utils import build_model, load_config, load_covariances, project_root, resolve_project_path


def main() -> None:
    repository = Path.cwd().resolve()
    results: list[dict[str, object]] = []
    for name in (
        "configs/s2_no_corrector_hotspot_aligned_800e.yaml",
        "configs/s1_case_adaptive_corrector_hotspot_aligned_800e.yaml",
    ):
        config = load_config(name)
        covariance = resolve_project_path(config, config["covariance"]["statistics_path"])
        aligned = resolve_project_path(
            config, config["loss"]["aligned_amplitude"]["statistics_path"]
        )
        manifest = resolve_project_path(config, config["data"]["manifest_dir"])
        if project_root(config) != repository:
            raise RuntimeError(
                f"{name} resolved repository root to {project_root(config)}, expected {repository}"
            )
        if covariance.parent != repository / "artifacts":
            raise RuntimeError(f"{name} resolved covariance statistics outside project: {covariance}")
        if aligned.parent != repository / "artifacts":
            raise RuntimeError(f"{name} resolved aligned statistics outside project: {aligned}")
        if not covariance.is_file() or not aligned.is_file():
            raise FileNotFoundError(
                f"{name} requires generated statistics: covariance={covariance}, aligned={aligned}"
            )
        if not manifest.is_dir():
            raise FileNotFoundError(f"{name} manifest directory is missing: {manifest}")
        model = build_model(config)
        _, _, statistics = load_covariances(config)
        results.append(
            {
                "config": name,
                "project_root": str(project_root(config)),
                "covariance_statistics": str(covariance),
                "aligned_statistics": str(aligned),
                "manifest_dir": str(manifest),
                "model_architecture_key": model.architecture_key,
                "statistics_split": statistics.get("provenance", {}).get("split"),
                "status": "PASS",
            }
        )
    payload = {"status": "PASS", "checks": results}
    output = repository / "artifacts" / "formal_path_preflight.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
