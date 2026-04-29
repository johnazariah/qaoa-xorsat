#!/usr/bin/env bash
# Canonical local startup script for MaxCut sweeps.
#
# Usage:
#   bash scripts/start-maxcut-local.sh [D|all] [P_MAX|auto] [SEED]
#
# Examples:
#   bash scripts/start-maxcut-local.sh all auto 42
#   bash scripts/start-maxcut-local.sh 8 12 42

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

LOG_DIR="$REPO_DIR/logs/maxcut-local"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/maxcut-local-$(date -u +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

D_SELECTION="${1:-all}"
P_MAX_ARG="${2:-auto}"
SEED="${3:-42}"
THREADS="${QAOA_THREADS:-$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 12)}"

echo "=== Local MaxCut startup ==="
echo "Repo:    $REPO_DIR"
echo "Host:    $(hostname)"
echo "Threads: $THREADS"
echo "Seed:    $SEED"
echo "Log:     $LOG_FILE"
echo "Date:    $(date -u)"
echo ""

if ! command -v julia >/dev/null 2>&1; then
    echo "Julia not found; installing with juliaup..."
    curl -fsSL https://install.julialang.org | sh -s -- --yes
    export PATH="$HOME/.juliaup/bin:$PATH"
fi

echo "Julia: $(julia --version)"
echo "Instantiating project..."
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
echo ""

if [ "$P_MAX_ARG" = "auto" ]; then
    TOTAL_MEM_GB=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1/1024/1024/1024}' || echo 64)
    if [ "$TOTAL_MEM_GB" -ge 140 ]; then
        P_MAX=14
    elif [ "$TOTAL_MEM_GB" -ge 40 ]; then
        P_MAX=13
    else
        P_MAX=12
    fi
    echo "Memory ${TOTAL_MEM_GB}GB: targeting p=$P_MAX"
else
    P_MAX="$P_MAX_ARG"
fi

if [ "$D_SELECTION" = "all" ]; then
    D_VALUES=(3 4 5 6 7 8)
else
    D_VALUES=("$D_SELECTION")
fi

for D in "${D_VALUES[@]}"; do
    echo ""
    echo "============================================"
    echo "=== MaxCut D=$D through p=$P_MAX ==="
    echo "============================================"
    julia --project=. -t "$THREADS" scripts/maxcut_sweep.jl "$D" "$P_MAX" "$SEED"
done

echo ""
echo "=== Local MaxCut complete ==="
echo "Results: results/maxcut-k2-d*-sweep.csv"
echo "Log:     $LOG_FILE"
