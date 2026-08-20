Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) { $PSNativeCommandUseErrorActionPreference = $true }
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = (Join-Path $Root "src")
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:CUDA_VISIBLE_DEVICES = "0,1,2,3"
$Output = Join-Path $Root "outputs/S2_no_corrector_hotspot_aligned_800e"
$Complete = Join-Path $Output "FORMAL_TRAINING_COMPLETE.json"
if (Test-Path $Complete) { Write-Host "S2 already complete; safely skipping."; exit 0 }
if (-not (Test-Path "artifacts/residual_statistics_train.json") -or -not (Test-Path "artifacts/aligned_band_statistics_train.json")) { & (Join-Path $PSScriptRoot "preflight_S1_S2.ps1") }
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$Arguments = @("--standalone","--nnodes=1","--nproc_per_node=4","-m","wrb3d.training.cli","--config","configs/s2_no_corrector_hotspot_aligned_800e.yaml","--shared-backbone-init","artifacts/shared_backbone_init_seed1234.pt","--output-dir",$Output)
$Latest = Join-Path $Output "latest.pt"
if (Test-Path $Latest) { $Arguments += @("--resume",$Latest) }
("torchrun " + ($Arguments -join " ")) | Set-Content -Encoding utf8 (Join-Path $Output "launch_command.txt")
& torchrun @Arguments 2>&1 | Tee-Object -FilePath (Join-Path $Output "train_console.log") -Append
if ($LASTEXITCODE -ne 0) { throw "S2 torchrun failed with exit code $LASTEXITCODE" }
if (-not (Test-Path $Complete)) { throw "S2 exited without formal completion marker" }
