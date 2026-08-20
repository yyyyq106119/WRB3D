Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Test-Path Variable:PSNativeCommandUseErrorActionPreference) { $PSNativeCommandUseErrorActionPreference = $true }
& (Join-Path $PSScriptRoot "preflight_S1_S2.ps1")
& (Join-Path $PSScriptRoot "run_S2.ps1")
& (Join-Path $PSScriptRoot "run_S1.ps1")
