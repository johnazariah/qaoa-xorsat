#!/bin/bash
#
# SLURM job array for QAOA-XORSAT: all 15 (k,D) pairs from Jordan et al.
#
# Each array task runs one (k,D) pair from p=1 through P_MAX with warm-starting.
# runner.py maps SLURM_ARRAY_TASK_ID to (k,D) and launches Julia.
#
# Memory guide (per node, Complex{Float64} adjoint cache):
#   p=12:  ~19 GB    p=13:  ~84 GB    p=14: ~394 GB    p=15: ~1.6 TB
#
# Usage:
#   sbatch scripts/qaoa_sweep.sh                # default: p=15, all 15 pairs
#   sbatch --export=QAOA_P_MAX=12 scripts/qaoa_sweep.sh   # override p_max
#   sbatch --array=1-5 scripts/qaoa_sweep.sh    # only k=3 family
#
# Task ID → (k,D) mapping:
#    1:(3,4)   2:(3,5)   3:(3,6)   4:(3,7)   5:(3,8)
#    6:(4,5)   7:(4,6)   8:(4,7)   9:(4,8)
#   10:(5,6)  11:(5,7)  12:(5,8)
#   13:(6,7)  14:(6,8)
#   15:(7,8)

#SBATCH --job-name=qaoa
#SBATCH --array=1-15
#SBATCH --partition=c3d
#SBATCH --time=999:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=2700G
#SBATCH --output=qaoa_%A-%a.out
#SBATCH --error=qaoa_%A-%a.err

set -euo pipefail

# ── Configuration (override via --export or environment) ──────────────────────
QAOA_P_MAX="${QAOA_P_MAX:-15}"
QAOA_THREADS="${QAOA_THREADS:-${SLURM_CPUS_PER_TASK:-28}}"

echo "=== QAOA-XORSAT SLURM Task ==="
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Array ID:  ${SLURM_ARRAY_TASK_ID}"
echo "Node:      $(hostname)"
echo "CPUs:      ${QAOA_THREADS}"
echo "RAM:       $(free -h 2>/dev/null | awk '/^Mem:/{print $2}' || echo 'unknown')"
echo "P_MAX:     ${QAOA_P_MAX}"
echo "Started:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd ~/qaoa-xorsat

# ── Ensure Julia dependencies are ready ───────────────────────────────────────
if [ ! -f "Manifest.toml" ] || ! julia --project=. -e 'using QaoaXorsat' 2>/dev/null; then
    echo "Installing Julia dependencies..."
    julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
fi

# ── Export for runner.py and Julia ────────────────────────────────────────────
export QAOA_P_MAX QAOA_THREADS
export QAOA_RUNNER_LABEL="slurm-${SLURM_JOB_ID:-0}-${SLURM_ARRAY_TASK_ID}"
export QAOA_RUN_KIND="cluster"

# ── Run ───────────────────────────────────────────────────────────────────────
srun python3 scripts/runner.py "$SLURM_ARRAY_TASK_ID" \
    --p-max "$QAOA_P_MAX" \
    --threads "$QAOA_THREADS"

echo ""
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
