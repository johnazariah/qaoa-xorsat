#!/bin/bash
#
# SLURM Double64 swarm sweep — ALL 15 pairs from p=1.
# Uses Double64 arithmetic to avoid Float64 precision loss at high (k,D).
# With 55 nodes, all 15 pairs run simultaneously.
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

# Stagger startup to avoid 15 tasks racing to precompile simultaneously.
# Each task waits (task_id * 10) seconds before starting.
# Precompilation should be done on the login node BEFORE sbatch,
# but this provides a safety net.
DELAY=$(( (SLURM_ARRAY_TASK_ID - 1) * 10 ))
echo "Stagger delay: ${DELAY}s (task $SLURM_ARRAY_TASK_ID)"
sleep $DELAY

julia --project=. -e 'using DoubleFloats, QaoaXorsat; println("Ready")' 2>&1

julia --project=. -t ${SLURM_CPUS_PER_TASK:-28} scripts/swarm_chain_d64.jl $K $D 15 100 10 20 42

echo ""
echo "Done: $(date -u)"
