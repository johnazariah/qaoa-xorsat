#!/bin/bash
#
# SLURM warm-start sweep — generated 2026-04-08
# Each task runs one (k,D) pair from its warm-start depth through p=15.
#
#SBATCH --job-name=qaoa-ws
#SBATCH --array=1-15
#SBATCH --partition=c3d
#SBATCH --time=999:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=2700G
#SBATCH --output=qaoa-ws_%A-%a.out
#SBATCH --error=qaoa-ws_%A-%a.err

set -euo pipefail

CONFIGS=(
    "experiments/warmstart/k3-d4.toml"  # 1: k=3 D=4 from p=14
    "experiments/warmstart/k3-d5.toml"  # 2: k=3 D=5 from p=14
    "experiments/warmstart/k3-d6.toml"  # 3: k=3 D=6 from p=15
    "experiments/warmstart/k3-d7.toml"  # 4: k=3 D=7 from p=14
    "experiments/warmstart/k3-d8.toml"  # 5: k=3 D=8 from p=13
    "experiments/warmstart/k4-d5.toml"  # 6: k=4 D=5 from p=14
    "experiments/warmstart/k4-d6.toml"  # 7: k=4 D=6 from p=12
    "experiments/warmstart/k4-d7.toml"  # 8: k=4 D=7 from p=12
    "experiments/warmstart/k4-d8.toml"  # 9: k=4 D=8 from p=12
    "experiments/warmstart/k5-d6.toml"  # 10: k=5 D=6 from p=12
    "experiments/warmstart/k5-d7.toml"  # 11: k=5 D=7 from p=11
    "experiments/warmstart/k5-d8.toml"  # 12: k=5 D=8 from p=10
    "experiments/warmstart/k6-d7.toml"  # 13: k=6 D=7 from p=11
    "experiments/warmstart/k6-d8.toml"  # 14: k=6 D=8 from p=10
    "experiments/warmstart/k7-d8.toml"  # 15: k=7 D=8 from p=10
)

TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
CONFIG="${CONFIGS[$TASK_INDEX]}"

echo "=== QAOA Warm-Start Sweep ==="
echo "Task:   $SLURM_ARRAY_TASK_ID / 15"
echo "Config: $CONFIG"
echo "Node:   $(hostname)"
echo "CPUs:   ${SLURM_CPUS_PER_TASK:-28}"
echo "Start:  $(date -u)"
echo ""

cd ~/qaoa-xorsat
export PATH="/root/.juliaup/bin:$HOME/.juliaup/bin:$PATH"
julia --project=. -t ${SLURM_CPUS_PER_TASK:-28} scripts/optimize_qaoa.jl "$CONFIG"

echo ""
echo "Done: $(date -u)"

