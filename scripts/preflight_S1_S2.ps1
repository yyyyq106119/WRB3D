Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) { $PSNativeCommandUseErrorActionPreference = $true }
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = (Join-Path $Root "src")
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
if (-not (Test-Path "artifacts/shared_backbone_init_seed1234.pt")) { & python tools/generate_shared_backbone_init.py }
& python tools/verify_shared_initialization.py
if (-not (Test-Path "artifacts/residual_statistics_train.json")) { & python -m wrb3d.training.statistics --config configs/s2_no_corrector_hotspot_aligned_800e.yaml --output artifacts/residual_statistics_train.json }
if (-not (Test-Path "artifacts/aligned_band_statistics_train.json")) { & python tools/estimate_aligned_band_statistics.py --config configs/s2_no_corrector_hotspot_aligned_800e.yaml --output artifacts/aligned_band_statistics_train.json }
& python tools/verify_formal_paths.py
& python -m pytest -q
& python tools/run_engineering_smoke.py
