#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUTPUT="$ROOT/outputs/S3_S2_ema589_psnr_refine_80e"
SOURCE="$ROOT/outputs/S2_no_corrector_hotspot_aligned_800e/epoch_0589.pt"
[[ -f "$ROOT/outputs/S2_no_corrector_hotspot_aligned_800e/FORMAL_TRAINING_COMPLETE.json" ]] || { echo "S3 requires completed S2" >&2; exit 1; }
[[ -f "$SOURCE" ]] || { echo "S3 source checkpoint is missing: $SOURCE" >&2; exit 1; }
mkdir -p "$OUTPUT"
ARGS=(--standalone --nnodes=1 --nproc_per_node=4 -m wrb3d.training.psnr_refine_cli --config configs/s3_s2_ema589_psnr_refine_80e.yaml --source-checkpoint "$SOURCE" --output-dir "$OUTPUT")
[[ -f "$OUTPUT/latest.pt" ]] && ARGS+=(--resume "$OUTPUT/latest.pt")
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" PYTHONUNBUFFERED=1 torchrun "${ARGS[@]}" 2>&1 | tee -a "$OUTPUT/train_console.log"
