#!/bin/bash
# Setup and run MaxCut sweep on a fresh machine (Linux or WSL on Windows).
#
# Usage:
#   1. Copy this repo to the machine
#   2. Run: bash scripts/setup-and-run-maxcut.sh
#
# Prerequisites: curl, tar (standard on Linux/WSL)

set -euo pipefail

echo "=== MaxCut Sweep Setup ==="
echo "Host: $(hostname)"
echo "CPUs: $(nproc)"
echo "Memory: $(free -h 2>/dev/null | grep Mem | awk '{print $2}' || echo 'unknown')"
echo "Date: $(date -u)"
echo ""

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# ── Install Julia if not present ──────────────────────────────────
if command -v julia &>/dev/null; then
    echo "Julia found: $(julia --version)"
else
    echo "Installing Julia 1.12..."
    # Detect platform
    case "$(uname -s)" in
        Linux*)
            ARCH="$(uname -m)"
            if [ "$ARCH" = "x86_64" ]; then
                JULIA_URL="https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-1.12.5-linux-x86_64.tar.gz"
            else
                echo "ERROR: Unsupported architecture $ARCH"
                exit 1
            fi
            ;;
        MINGW*|MSYS*|CYGWIN*)
            JULIA_URL="https://julialang-s3.julialang.org/bin/winnt/x64/1.12/julia-1.12.5-win64.zip"
            ;;
        *)
            echo "ERROR: Unsupported OS $(uname -s). Install Julia manually."
            exit 1
            ;;
    esac

    echo "Downloading from $JULIA_URL ..."
    mkdir -p "$HOME/.local"
    cd "$HOME/.local"

    if [[ "$JULIA_URL" == *.tar.gz ]]; then
        curl -fSL "$JULIA_URL" | tar xz
        JULIA_DIR=$(ls -d julia-* | head -1)
    else
        curl -fSL "$JULIA_URL" -o julia.zip
        unzip -q julia.zip
        rm julia.zip
        JULIA_DIR=$(ls -d julia-* | head -1)
    fi

    export PATH="$HOME/.local/$JULIA_DIR/bin:$PATH"
    cd "$REPO_DIR"

    echo "Installed: $(julia --version)"
    echo ""
    echo "Add to your PATH permanently:"
    echo "  export PATH=\"$HOME/.local/$JULIA_DIR/bin:\$PATH\""
    echo ""
fi

# ── Install Julia packages ────────────────────────────────────────
echo "Installing Julia packages..."
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
echo "Packages ready."
echo ""

# ── Determine thread count and which D values to run ──────────────
THREADS=$(nproc)
# Use all threads for one D at a time (better memory bandwidth)
echo "Running with $THREADS threads per sweep"
echo ""

# ── Run MaxCut sweeps ─────────────────────────────────────────────
# Run all D values sequentially with full thread count.
# Each writes to results/maxcut-k2-dD-sweep.csv with resume logic.
# Adjust P_MAX based on available memory:
#   p=13 needs ~28 GB, p=14 needs ~120 GB

TOTAL_MEM_GB=$(free -g 2>/dev/null | grep Mem | awk '{print $2}' || echo 64)
if [ "$TOTAL_MEM_GB" -ge 140 ]; then
    P_MAX=14
    echo "Memory ${TOTAL_MEM_GB}GB: targeting p=$P_MAX"
elif [ "$TOTAL_MEM_GB" -ge 40 ]; then
    P_MAX=13
    echo "Memory ${TOTAL_MEM_GB}GB: targeting p=$P_MAX"
else
    P_MAX=12
    echo "Memory ${TOTAL_MEM_GB}GB: targeting p=$P_MAX"
fi
echo ""

for D in 3 4 5 6 7 8; do
    echo "============================================"
    echo "=== MaxCut D=$D, p=1..$P_MAX ==="
    echo "============================================"
    julia --project=. -t $THREADS scripts/maxcut_sweep.jl $D $P_MAX 42
    echo ""
done

echo "=== All sweeps complete ==="
echo "Results in results/maxcut-k2-d*-sweep.csv"
