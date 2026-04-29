#!/bin/bash
#
# SLURM Double64 + CPU-checkpointed sweep for pushing XORSAT past p=13.
# One array task per target (k,D) pair. Each task writes a fresh
# results/cluster-p16-k{K}d{D}.csv and uses existing swarm/warm-start CSVs
# only as read-only seeds.
#
# Submit from the cluster repo checkout via the canonical startup wrapper:
#   bash scripts/start-xorsat-slurm.sh
#
# Optional overrides:
#   QAOA_REPO=$HOME/qaoa-xorsat
#   QAOA_PUSH_BRANCH=cluster-p16-results
#   QAOA_POPULATION=100 QAOA_GENERATIONS=10 QAOA_BURST=20
#   QAOA_MAX_RAM_CHECKPOINTS=4 QAOA_SWARM_CONCURRENCY=1
#
#SBATCH --job-name=qaoa-p16
#SBATCH --array=1-9
#SBATCH --partition=c3dssd
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=1400G
#SBATCH --requeue
#SBATCH --comment="maintain_node"
#SBATCH --output=qaoa-p16_%A-%a.out
#SBATCH --error=qaoa-p16_%A-%a.err

set -uo pipefail

REPO="${QAOA_REPO:-$HOME/qaoa-xorsat}"
PUSH_BRANCH="${QAOA_PUSH_BRANCH:-cluster-p16-results}"
LOGDIR="$REPO/logs/cluster-p16"
mkdir -p "$LOGDIR"

# Format: K D P_TARGET
TARGETS=(
    "3 4 16"  # 1: main k=3, reaches p=16
    "3 5 16"  # 2: main k=3, reaches p=16
    "3 6 15"  # 3: precision wall from p=12, target p=15
    "3 7 14"  # 4
    "3 8 14"  # 5
    "4 5 14"  # 6: priority 2
    "4 6 13"  # 7
    "4 7 13"  # 8
    "4 8 13"  # 9
)

TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
TARGET="${TARGETS[$TASK_INDEX]}"
K=$(echo "$TARGET" | cut -d' ' -f1)
D=$(echo "$TARGET" | cut -d' ' -f2)
P_TARGET=$(echo "$TARGET" | cut -d' ' -f3)

LOGFILE="$LOGDIR/task-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}-k${K}d${D}.log"
RESULTS_FILE="$REPO/results/cluster-p16-k${K}d${D}.csv"
PROGRESS_FILE="$LOGDIR/progress-k${K}d${D}.log"
CHECKPOINT_DIR="${QAOA_CHECKPOINT_DIR:-${TMPDIR:-/tmp}/qaoa-checkpoints-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}}"

exec > >(tee -a "$LOGFILE") 2>&1

echo "=== QAOA p16 cluster sweep ==="
echo "Task:        ${SLURM_ARRAY_TASK_ID} / 9"
echo "Pair:        k=$K D=$D"
echo "Target p:    $P_TARGET"
echo "Node:        $(hostname)"
echo "CPUs:        ${SLURM_CPUS_PER_TASK:-32}"
echo "Memory:      1400G requested"
echo "Start:       $(date -u)"
echo "Job:         ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Repo:        $REPO"
echo "Results:     $RESULTS_FILE"
echo "Progress:    $PROGRESS_FILE"
echo "Checkpoints: $CHECKPOINT_DIR"
echo ""

cd "$REPO" || exit 2
export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export JULIA_HEAP_SIZE_HINT="1200G"
export QAOA_RESULTS_FILE="$RESULTS_FILE"
export QAOA_PROGRESS_FILE="$PROGRESS_FILE"
export QAOA_CHECKPOINT_DIR="$CHECKPOINT_DIR"
export QAOA_MAX_RAM_CHECKPOINTS="${QAOA_MAX_RAM_CHECKPOINTS:-4}"
export QAOA_SWARM_CONCURRENCY="${QAOA_SWARM_CONCURRENCY:-1}"

mkdir -p "$(dirname "$RESULTS_FILE")" "$(dirname "$PROGRESS_FILE")" "$CHECKPOINT_DIR"

echo "--- Node diagnostics ---"
echo "Hostname: $(hostname)"
echo "Memory:   $(free -h 2>/dev/null | head -2 || echo 'N/A')"
echo "Disk:     $(df -h "$CHECKPOINT_DIR" 2>/dev/null | tail -1 || echo 'N/A')"
echo "Julia:    $(which julia 2>/dev/null || echo 'NOT FOUND')"
echo "Git HEAD: $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
echo "Threads:  $JULIA_NUM_THREADS"
echo ""

echo "Removing stale per-task outputs"
rm -f "$RESULTS_FILE" "$RESULTS_FILE.checkpoint" "$PROGRESS_FILE"
rm -rf "$CHECKPOINT_DIR"/* 2>/dev/null || true

echo "Checking Julia environment with precompile lock..."
PRECOMPILE_LOCK="$REPO/.julia-precompile-d64.lock"
PRECOMPILE_MARKER="$REPO/.julia-precompile-d64.done"
while [ ! -f "$PRECOMPILE_MARKER" ]; do
    if mkdir "$PRECOMPILE_LOCK" 2>/dev/null; then
        echo "Acquired precompile lock"
        julia --project=. -e 'using DoubleFloats, QaoaXorsat; println("Environment ready.")'
        RC=$?
        if [ $RC -ne 0 ]; then
            rmdir "$PRECOMPILE_LOCK" 2>/dev/null || true
            echo "FATAL: precompile failed with exit code $RC"
            exit $RC
        fi
        date -u > "$PRECOMPILE_MARKER"
        rmdir "$PRECOMPILE_LOCK" 2>/dev/null || true
    else
        echo "Waiting for another task to finish precompilation..."
        sleep 30
    fi
done

echo "Starting cluster_p16_chain.jl at $(date -u)"
julia --project=. -t "${SLURM_CPUS_PER_TASK:-32}" scripts/cluster_p16_chain.jl \
    "$K" "$D" "$P_TARGET" \
    "${QAOA_POPULATION:-100}" "${QAOA_GENERATIONS:-10}" "${QAOA_BURST:-20}" "${QAOA_SEED:-42}"
RC=$?

echo ""
echo "Exit code: $RC"
echo "Done: $(date -u)"

echo "Staging results/logs for push"
git add -f "$RESULTS_FILE" "$PROGRESS_FILE" "$LOGFILE" 2>/dev/null || true
if ! git diff --cached --quiet; then
    git commit -m "data: cluster p16 k=${K} D=${D} task ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" 2>/dev/null || true
    git push origin "HEAD:$PUSH_BRANCH" 2>/dev/null || true
else
    echo "No result/log changes to commit"
fi

if [ $RC -ne 0 ]; then
    echo "FAILED with exit code $RC"
fi

exit $RC