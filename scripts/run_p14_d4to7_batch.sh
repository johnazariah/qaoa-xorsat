#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

for D in 4 5 6 7; do
  echo "===== $(date '+%F %T') START D=${D} ====="
  LOG_D="logs/p14-d${D}-memfix-$(date +%Y%m%d-%H%M%S).log"
  JULIA_NUM_THREADS=12 julia --project=. scripts/run_p14_d_warm.jl "$D" | tee "$LOG_D"
  echo "===== $(date '+%F %T') END D=${D} ====="
  echo
done
