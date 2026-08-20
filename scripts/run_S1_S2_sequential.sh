#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
bash scripts/preflight_S1_S2.sh
bash scripts/run_S2.sh
bash scripts/run_S1.sh
