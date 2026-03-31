#!/usr/bin/env bash
set -eo pipefail
export HOME=${HOME:-/root}
export PATH="$HOME/.juliaup/bin:$PATH"
cd /home/azureuser/qaoa-xorsat
P_MAX=__PMAX__
THREADS=$(nproc)

# Find max completed p for a given (k,D) across all run CSVs
max_completed_p() {
    local k=$1 d=$2
    local max_p=0
    for csv in .project/results/optimization/runs/*/results.csv; do
        [ -f "$csv" ] || continue
        local p=$(awk -F, -v k="$k" -v d="$d" 'NR>1 && $8==k && $9==d {if($10+0 > max) max=$10+0} END {print max+0}' "$csv")
        [ "$p" -gt "$max_p" ] && max_p=$p
    done
    echo "$max_p"
}

for PAIR in __PAIRS__; do
    K=$(echo $PAIR | cut -d, -f1)
    D=$(echo $PAIR | cut -d, -f2)
    LOG="results/logs/pair-k${K}-d${D}.log"

    # Resume: find where we left off
    DONE_P=$(max_completed_p $K $D)
    P_MIN=$((DONE_P + 1))

    if [ "$P_MIN" -gt "$P_MAX" ]; then
        echo "$(date -u): (k=$K, D=$D) already complete through p=$DONE_P, skipping"
        continue
    fi

    if [ "$DONE_P" -gt 0 ]; then
        echo "$(date -u): (k=$K, D=$D) resuming from p=$P_MIN (completed through p=$DONE_P)"
    else
        echo "$(date -u): (k=$K, D=$D) starting fresh p=1..$P_MAX"
        P_MIN=1
    fi

    julia --project=. -t $THREADS scripts/optimize_qaoa.jl $K $D $P_MIN $P_MAX 2 320 1234 true adjoint >> "$LOG" 2>&1
    echo "$(date -u): Completed (k=$K, D=$D)"
done

echo "=== All pairs complete at $(date -u) ==="
