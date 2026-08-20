#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
[[ -f artifacts/shared_backbone_init_seed1234.pt ]] || python tools/generate_shared_backbone_init.py
python tools/verify_shared_initialization.py
[[ -f artifacts/residual_statistics_train.json ]] || python -m wrb3d.training.statistics --config configs/s2_no_corrector_hotspot_aligned_800e.yaml --output artifacts/residual_statistics_train.json
[[ -f artifacts/aligned_band_statistics_train.json ]] || python tools/estimate_aligned_band_statistics.py --config configs/s2_no_corrector_hotspot_aligned_800e.yaml --output artifacts/aligned_band_statistics_train.json
python tools/verify_formal_paths.py
python -m pytest -q
python tools/run_engineering_smoke.py
