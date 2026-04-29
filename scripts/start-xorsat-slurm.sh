#!/usr/bin/env bash
# Canonical SLURM startup script for Stephen's Max-k-XORSAT cluster run.
#
# Usage on the cluster login node:
#   cd ~/qaoa-xorsat
#   git pull --ff-only origin main
#   bash scripts/start-xorsat-slurm.sh [--dry-run]
#
# This submits scripts/qaoa_cluster_p16.sh, which owns the SLURM array,
# Double64 arithmetic, CPU checkpointing, progress logs, and auto-push.

set -euo pipefail

DRY_RUN="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${QAOA_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SUBMIT_SCRIPT="scripts/qaoa_cluster_p16.sh"

cd "$REPO"
export PATH="$HOME/.juliaup/bin:$PATH"

echo "=== Stephen SLURM XORSAT startup ==="
echo "Repo:      $REPO"
echo "Git HEAD:  $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Branch:    $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
echo "Submit:    sbatch $SUBMIT_SCRIPT"
echo "Date:      $(date -u)"
echo ""

if [ ! -f "$SUBMIT_SCRIPT" ]; then
    echo "ERROR: missing $SUBMIT_SCRIPT"
    exit 2
fi

if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "Dry run only; not submitting."
    echo "Command: sbatch $SUBMIT_SCRIPT"
    exit 0
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch not found. Run this on the SLURM login node."
    exit 2
fi

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "WARNING: tracked local changes are present. The submitted job uses this working tree."
    git status --short
    echo ""
fi

echo "Checking Julia environment..."
julia --project=. -e '
using Pkg
Pkg.instantiate()
# Auto-install CUDA if a NVIDIA GPU is detected (no-op on non-CUDA hosts).
try
    if success(`nvidia-smi -L`) && !haskey(Pkg.project().dependencies, "CUDA")
        @info "NVIDIA GPU detected — installing CUDA.jl"
        Pkg.add("CUDA")
    end
catch
    # nvidia-smi not present; CPU-only environment is fine.
end
using DoubleFloats, QaoaXorsat
println("Environment ready.")
'
echo ""

JOBID=$(sbatch --parsable "$SUBMIT_SCRIPT")
echo "Submitted job: $JOBID"
echo "Monitor: squeue -j $JOBID"
echo "Logs:    logs/cluster-p16/ and qaoa-p16_${JOBID}-*.out/.err"