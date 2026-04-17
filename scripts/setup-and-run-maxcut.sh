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
TOTAL_MEM_GB=$(free -g 2>/dev/null | grep Mem | awk '{print $2}' || echo 64)

# Memory budget per p level (approximate):
#   p=12 ~ 7 GB, p=13 ~ 28 GB, p=14 ~ 120 GB, p=15 ~ 480 GB
D_VALUES=(3 4 5 6 7 8)
N_JOBS=${#D_VALUES[@]}
THREADS_PER_JOB=$((THREADS / N_JOBS))
# Ensure at least 1 thread per job
[ "$THREADS_PER_JOB" -lt 1 ] && THREADS_PER_JOB=1

echo "Threads: $THREADS total, $THREADS_PER_JOB per parallel job"
echo "Memory:  ${TOTAL_MEM_GB}GB"
echo "D values: ${D_VALUES[*]}"
echo ""

# ── Run MaxCut sweeps ─────────────────────────────────────────────
# Phase 1: Run ALL D values in parallel up to p=13.
#   Peak memory: 6 × ~28 GB = 168 GB (fits in 180 GB).
#   Each gets THREADS/6 threads for restart-level parallelism.
# Phase 2: Push to p=14 one D at a time with ALL threads.
#   Needs ~120 GB per run — must be sequential.
# Resume logic in maxcut_sweep.jl skips already-completed p levels.

if [ "$TOTAL_MEM_GB" -ge 140 ]; then
    P_PARALLEL=13
    P_MAX=14
elif [ "$TOTAL_MEM_GB" -ge 40 ]; then
    P_PARALLEL=13
    P_MAX=13
else
    P_PARALLEL=12
    P_MAX=12
fi

echo "Strategy: parallel phase p=1..$P_PARALLEL, sequential phase p=$((P_PARALLEL+1))..$P_MAX"
echo ""

# ── Phase 1: parallel sweep up to P_PARALLEL ─────────────────────
LOG_DIR="$REPO_DIR/results"
mkdir -p "$LOG_DIR"
PIDS=()
FAILED=0

echo "============================================"
echo "=== Phase 1: D={${D_VALUES[*]}}, p=1..$P_PARALLEL  (${N_JOBS} parallel, ${THREADS_PER_JOB} threads each) ==="
echo "============================================"
echo ""

for D in "${D_VALUES[@]}"; do
    LOG_FILE="$LOG_DIR/maxcut-k2-d${D}-phase1.log"
    echo "Starting D=$D → $LOG_FILE"
    julia --project=. -t "$THREADS_PER_JOB" scripts/maxcut_sweep.jl "$D" "$P_PARALLEL" 42 \
        > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Waiting for ${#PIDS[@]} parallel jobs..."
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  D=${D_VALUES[$i]} finished (pid ${PIDS[$i]})"
    else
        echo "  D=${D_VALUES[$i]} FAILED (pid ${PIDS[$i]})"
        FAILED=$((FAILED + 1))
    fi
done
echo ""

if [ "$FAILED" -gt 0 ]; then
    echo "WARNING: $FAILED parallel jobs failed — check logs in $LOG_DIR"
fi

# ── Phase 2: sequential sweep for remaining high-p levels ─────────
if [ "$P_MAX" -gt "$P_PARALLEL" ]; then
    echo "============================================"
    echo "=== Phase 2: D={${D_VALUES[*]}}, p=$((P_PARALLEL+1))..$P_MAX  (sequential, $THREADS threads) ==="
    echo "============================================"
    echo ""

    for D in "${D_VALUES[@]}"; do
        echo "--- MaxCut D=$D, pushing to p=$P_MAX ($THREADS threads) ---"
        julia --project=. -t "$THREADS" scripts/maxcut_sweep.jl "$D" "$P_MAX" 42
        echo ""
    done
fi

echo "=== All sweeps complete ==="
echo "Results in results/maxcut-k2-d*-sweep.csv"
echo "Phase 1 logs in results/maxcut-k2-d*-phase1.log"
