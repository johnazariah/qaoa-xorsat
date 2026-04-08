#!/bin/bash
#
# SLURM swarm sweep for stuck (k,D) pairs.
# Each task runs the swarm optimizer from p=1 through p=15.
# Results go to results/swarm-k{K}d{D}.csv (written immediately per depth).
#
#SBATCH --job-name=qaoa-swarm
#SBATCH --array=1-10
#SBATCH --partition=c3d
#SBATCH --time=999:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=2700G
#SBATCH --output=qaoa-swarm_%A-%a.out
#SBATCH --error=qaoa-swarm_%A-%a.err

set -euo pipefail

# Pairs that need swarm (stuck at 0.500 with warm-start)
PAIRS=(
    "3 8"   # 1
    "4 6"   # 2
    "4 7"   # 3
    "4 8"   # 4
    "5 6"   # 5
    "5 7"   # 6
    "5 8"   # 7
    "6 7"   # 8
    "6 8"   # 9
    "7 8"   # 10
)

TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
PAIR="${PAIRS[$TASK_INDEX]}"
K=$(echo $PAIR | cut -d' ' -f1)
D=$(echo $PAIR | cut -d' ' -f2)

echo "=== QAOA Swarm ==="
echo "Task:  $SLURM_ARRAY_TASK_ID / 10"
echo "Pair:  k=$K, D=$D"
echo "Node:  $(hostname)"
echo "CPUs:  ${SLURM_CPUS_PER_TASK:-28}"
echo "Start: $(date -u)"
echo ""

cd ~/qaoa-xorsat
export PATH="$HOME/.juliaup/bin:$PATH"
julia --project=. -t ${SLURM_CPUS_PER_TASK:-28} scripts/swarm_chain.jl $K $D 15 100 10 20 42

echo ""
echo "Done: $(date -u)"
