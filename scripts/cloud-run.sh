#!/usr/bin/env bash
#
# One-command setup and run for a fresh Linux VM.
# Copy-paste this into an SSH session.
#
# Tested on: Ubuntu 22.04, Debian 12
# Hardware: needs 512GB+ RAM for p=14, 1.5TB+ for p=15
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/johnazariah/qaoa-xorsat/main/scripts/cloud-run.sh | bash -s -- [P_MAX] [THREADS]
#
# Or download and run:
#   wget https://raw.githubusercontent.com/johnazariah/qaoa-xorsat/main/scripts/cloud-run.sh
#   bash cloud-run.sh 14 28

set -euo pipefail

P_MAX="${1:-14}"
THREADS="${2:-$(nproc)}"

echo "=== QAOA-XORSAT Cloud Setup ==="
echo "P_MAX:   $P_MAX"
echo "Threads: $THREADS"
echo "RAM:     $(free -h | awk '/^Mem:/{print $2}')"
echo "CPU:     $(lscpu | grep 'Model name' | sed 's/.*: *//')"
echo ""

# Install Julia if not present
if ! command -v julia &> /dev/null; then
    echo "Installing Julia..."
    curl -fsSL https://install.julialang.org | sh -s -- --yes
    export PATH="$HOME/.juliaup/bin:$PATH"
fi
echo "Julia: $(julia --version)"

# Clone repo
if [ ! -d "qaoa-xorsat" ]; then
    echo "Cloning repository..."
    git clone https://github.com/johnazariah/qaoa-xorsat.git
fi
cd qaoa-xorsat

# Install dependencies
echo "Installing Julia dependencies..."
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Quick smoke test
echo "Smoke test..."
julia --project=. -e 'using QaoaXorsat; println("QaoaXorsat loaded OK")'

# Create results directory
mkdir -p results/logs

# Determine the script and log name
TIMESTAMP=$(date -u +%Y%m%dT%H%M%S)
LOGFILE="results/logs/cloud-${TIMESTAMP}-p${P_MAX}.log"

echo ""
echo "=== Starting sweep: all 15 pairs through p=$P_MAX ==="
echo "Log: $LOGFILE"
echo "Monitor: tail -f $LOGFILE"
echo ""

# Run the full table
nohup julia --project=. -t "$THREADS" scripts/run_full_table.jl "$P_MAX" > "$LOGFILE" 2>&1 &
PID=$!
echo "Started PID=$PID"
echo "To check: tail -20 $LOGFILE"
echo "To stop:  kill $PID"

# Wait a bit and show initial output
sleep 30
echo ""
echo "=== Initial output ==="
head -20 "$LOGFILE" 2>/dev/null || echo "(waiting for output...)"
