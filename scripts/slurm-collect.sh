#!/bin/bash
#
# Extract current progress from QAOA-XORSAT cluster runs into a single CSV.
#
# Aggregates completed results AND in-progress checkpoints from all run
# directories, producing a summary you can scp off the cluster.
#
# Output: results/cluster-progress-<timestamp>.csv
#
# Usage:
#   bash scripts/slurm-collect.sh                  # write to results/
#   bash scripts/slurm-collect.sh --stdout          # print to stdout
#   bash scripts/slurm-collect.sh --best            # only best value per (k,D)
#   bash scripts/slurm-collect.sh --checkpoints     # include in-progress angles

set -euo pipefail

cd "${0%/*}/.."

STDOUT_MODE=false
BEST_ONLY=false
INCLUDE_CHECKPOINTS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stdout) STDOUT_MODE=true; shift ;;
        --best) BEST_ONLY=true; shift ;;
        --checkpoints) INCLUDE_CHECKPOINTS=true; shift ;;
        *) echo "Usage: $0 [--stdout] [--best] [--checkpoints]" >&2; exit 1 ;;
    esac
done

RUNS_DIR=".project/results/optimization/runs"

if [[ ! -d "$RUNS_DIR" ]]; then
    echo "No results directory at $RUNS_DIR" >&2
    exit 1
fi

TIMESTAMP=$(date -u +%Y%m%dT%H%M%S)

# ── Collect completed results ─────────────────────────────────────────────────
HEADER="source,run_id,k,D,p,value,wall_time_seconds,converged,g_abstol,gamma,beta"

collect_completed() {
    for dir in $(ls -d "$RUNS_DIR"/*/ 2>/dev/null | sort); do
        results_csv="$dir/results.csv"
        [[ -f "$results_csv" ]] || continue
        run_id=$(basename "$dir")

        # Read header to find column positions
        head_line=$(head -1 "$results_csv")

        awk -F',' -v run_id="$run_id" '
        NR == 1 { next }
        {
            # Columns: 8=k, 9=D, 10=p, 15=value, 16=wall_time, 21=converged, 24=g_abstol, 25=gamma, 26=beta
            printf "completed,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
                run_id, $8, $9, $10, $15, $16, $21, $24, $25, $26
        }' "$results_csv"
    done
}

# ── Collect in-progress checkpoints ──────────────────────────────────────────
collect_checkpoints() {
    for dir in $(ls -d "$RUNS_DIR"/*/ 2>/dev/null | sort); do
        run_id=$(basename "$dir")
        for cp in "$dir"/checkpoint-p*.csv; do
            [[ -f "$cp" ]] || continue
            p_label=$(basename "$cp" | sed 's/checkpoint-p\([0-9]*\)\.csv/\1/')

            # Skip if we already have a completed result for this p
            results_csv="$dir/results.csv"
            if [[ -f "$results_csv" ]]; then
                k_from_manifest=$(awk -F',' 'NR==2{print $8}' "$results_csv" 2>/dev/null)
                d_from_manifest=$(awk -F',' 'NR==2{print $9}' "$results_csv" 2>/dev/null)
                has_completed=$(awk -F',' -v p="$p_label" 'NR>1 && $10==p {found=1} END{print found+0}' "$results_csv")
                [[ "$has_completed" == "1" ]] && continue
            fi

            # Parse checkpoint
            tail -1 "$cp" | awk -F',' -v run_id="$run_id" -v p="$p_label" -v k="${k_from_manifest:-?}" -v d="${d_from_manifest:-?}" '{
                printf "checkpoint,%s,%s,%s,%s,in-progress,0,false,,%s,%s\n",
                    run_id, k, d, p, $3, $4
            }'
        done
    done
}

# ── Filter to best per (k,D) ─────────────────────────────────────────────────
filter_best() {
    # Keep only the row with highest value for each (k,D) pair
    sort -t',' -k3,3n -k4,4n -k6,6rg | awk -F',' '
    {
        key = $3 "," $4
        if (!(key in best) || $6 > best_val[key]) {
            best[key] = $0
            best_val[key] = $6
        }
    }
    END {
        for (key in best) print best[key]
    }' | sort -t',' -k3,3n -k4,4n
}

# ── Assemble output ──────────────────────────────────────────────────────────
output() {
    echo "$HEADER"
    {
        collect_completed
        $INCLUDE_CHECKPOINTS && collect_checkpoints
    } | if $BEST_ONLY; then
        filter_best
    else
        sort -t',' -k3,3n -k4,4n -k5,5n
    fi
}

if $STDOUT_MODE; then
    output
else
    mkdir -p results
    OUTFILE="results/cluster-progress-${TIMESTAMP}.csv"
    output > "$OUTFILE"
    echo "Wrote $(wc -l < "$OUTFILE" | tr -d ' ') lines to $OUTFILE" >&2

    # Also create a symlink to the latest
    ln -sf "$(basename "$OUTFILE")" results/cluster-progress-latest.csv
    echo "Symlinked results/cluster-progress-latest.csv" >&2
fi
