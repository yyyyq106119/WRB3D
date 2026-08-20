#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0,1,2,3
OUTPUT="$ROOT/outputs/S2_no_corrector_hotspot_aligned_800e"
COMPLETE="$OUTPUT/FORMAL_TRAINING_COMPLETE.json"
[[ -f "$COMPLETE" ]] && { echo "S2 already complete; safely skipping."; exit 0; }
[[ -f artifacts/residual_statistics_train.json && -f artifacts/aligned_band_statistics_train.json ]] || bash scripts/preflight_S1_S2.sh
mkdir -p "$OUTPUT"
ARGS=(--standalone --nnodes=1 --nproc_per_node=4 -m wrb3d.training.cli --config configs/s2_no_corrector_hotspot_aligned_800e.yaml --shared-backbone-init artifacts/shared_backbone_init_seed1234.pt --output-dir "$OUTPUT")
[[ -f "$OUTPUT/latest.pt" ]] && ARGS+=(--resume "$OUTPUT/latest.pt")
printf 'torchrun %q ' "${ARGS[@]}" > "$OUTPUT/launch_command.txt"
printf '\n' >> "$OUTPUT/launch_command.txt"
torchrun "${ARGS[@]}" 2>&1 | tee -a "$OUTPUT/train_console.log"
[[ -f "$COMPLETE" ]] || { echo "S2 exited without formal completion marker" >&2; exit 1; }
