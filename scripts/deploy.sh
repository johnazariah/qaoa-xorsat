#!/usr/bin/env bash
#
# Deploy and run the QAOA-XORSAT computation on a cloud VM.
#
# Prerequisites: Julia 1.12+ installed (or use the Docker container)
#
# Usage:
#   # Option 1: Docker (recommended for reproducibility)
#   docker build -t qaoa-xorsat .
#   docker run --rm -v $(pwd)/results:/workspace/results qaoa-xorsat \
#     experiments/full-table.toml
#
#   # Option 2: Native Julia
#   ./scripts/deploy.sh [TOML_CONFIG] [THREADS]
#
# Examples:
#   ./scripts/deploy.sh experiments/full-table.toml 16
#   ./scripts/deploy.sh experiments/resume-p13-14.toml 32

set -euo pipefail

CONFIG="${1:-experiments/full-table.toml}"
THREADS="${2:-$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 12)}"

echo "=== QAOA-XORSAT Deployment ==="
echo "Config:  $CONFIG"
echo "Threads: $THREADS"
echo "Julia:   $(julia --version 2>/dev/null || echo 'not found')"
echo "Memory:  $(free -h 2>/dev/null | awk '/^Mem:/{print $2}' || sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f GB\n", $1/1024/1024/1024}')"
echo ""

# Install dependencies
echo "Installing dependencies..."
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Prevent sleep on macOS
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Starting caffeinate..."
    caffeinate -dims &
    CAFFEINATE_PID=$!
    trap "kill $CAFFEINATE_PID 2>/dev/null" EXIT
fi

# Determine script based on config
if [[ "$CONFIG" == *"full-table"* ]]; then
    SCRIPT="scripts/run_full_table.jl"
    # Extract p_max from filename or default to 11
    P_MAX="${3:-11}"
    ARGS="$P_MAX"
else
    SCRIPT="scripts/optimize_qaoa.jl"
    ARGS="$CONFIG"
fi

# Create timestamped log
TIMESTAMP=$(date -u +%Y%m%dT%H%M%S)
LOGFILE="results/logs/deploy-${TIMESTAMP}.log"
mkdir -p results/logs

echo "Running: julia --project=. -t $THREADS $SCRIPT $ARGS"
echo "Log:     $LOGFILE"
echo ""

# Run with logging
julia --project=. -t "$THREADS" "$SCRIPT" $ARGS 2>&1 | tee "$LOGFILE"

echo ""
echo "=== Complete ==="
echo "Results in: results/"
echo "Log:        $LOGFILE"
