#!/bin/bash
#
# SLURM Double64 swarm sweep — ALL 15 pairs from p=1.
# Uses Double64 arithmetic to avoid Float64 precision loss at high (k,D).
# With 55 nodes, all 15 pairs run simultaneously.
#
# ── CLEAN START INSTRUCTIONS ──────────────────────────────────────
#
# On the login node, run these steps IN ORDER before sbatch:
#
#   1. Kill all running jobs:
#        scancel -u $USER
#
#   2. Wait for CG-state jobs to clear (check with squeue -u $USER).
#      If stuck, ask admin to clear them.
#
#   3. Pull latest code:
#        cd ~/qaoa-xorsat
#        git pull origin main
#
#   4. Remove ALL old result files (critical — resume logic reads these):
#        rm -f results/swarm-d64-k*.csv
#
#   5. Precompile ONCE on the login node (avoids 15-way race):
#        julia --project=. -e 'using DoubleFloats, QaoaXorsat; println("Ready")'
#
#   6. Submit:
#        sbatch scripts/qaoa_d64_sweep.sh
#
#   7. Push results periodically:
#        git add -A results/
#        git commit -m "Stephen: pure D64 swarm results"
#        git push origin HEAD:stephen-d64-results
#
# ──────────────────────────────────────────────────────────────────
#
#SBATCH --job-name=qaoa-d64
#SBATCH --array=1-15
#SBATCH --partition=c3d
#SBATCH --time=999:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=2700G
#SBATCH --output=qaoa-d64_%A-%a.out
#SBATCH --error=qaoa-d64_%A-%a.err

set -euo pipefail

PAIRS=(
    "3 4"   #  1
    "3 5"   #  2
    "3 6"   #  3
    "3 7"   #  4
    "3 8"   #  5
    "4 5"   #  6
    "4 6"   #  7
    "4 7"   #  8
    "4 8"   #  9
    "5 6"   # 10
    "5 7"   # 11
    "5 8"   # 12
    "6 7"   # 13
    "6 8"   # 14
    "7 8"   # 15
)

TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
PAIR="${PAIRS[$TASK_INDEX]}"
K=$(echo $PAIR | cut -d' ' -f1)
D=$(echo $PAIR | cut -d' ' -f2)

echo "=== QAOA Double64 Swarm ==="
echo "Task:  $SLURM_ARRAY_TASK_ID / 15"
echo "Pair:  k=$K, D=$D"
echo "Node:  $(hostname)"
echo "CPUs:  ${SLURM_CPUS_PER_TASK:-28}"
echo "Start: $(date -u)"
echo ""

cd ~/qaoa-xorsat
export PATH="$HOME/.juliaup/bin:$PATH"

# Clean start: remove any old result file for THIS pair so resume
# logic doesn't pick up garbage from a previous run.
RESULTS_FILE="results/swarm-d64-k${K}d${D}.csv"
if [ -f "$RESULTS_FILE" ]; then
    echo "Removing stale $RESULTS_FILE"
    rm -f "$RESULTS_FILE"
fi

# Task 1 precompiles; all others wait for it via a lockfile on shared fs.
LOCKFILE="$HOME/qaoa-xorsat/.precompile-done-${SLURM_ARRAY_JOB_ID}"
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    echo "Task 1: precompiling..."
    julia --project=. -e 'using DoubleFloats, QaoaXorsat; println("Precompile done")' 2>&1
    touch "$LOCKFILE"
else
    echo "Waiting for task 1 to finish precompilation..."
    while [ ! -f "$LOCKFILE" ]; do sleep 5; done
    echo "Precompilation ready."
fi

julia --project=. -t ${SLURM_CPUS_PER_TASK:-28} scripts/swarm_chain_d64.jl $K $D 15 100 10 20 42

echo ""
echo "Done: $(date -u)"
