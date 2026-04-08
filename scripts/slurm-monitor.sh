#!/bin/bash
#
# SLURM monitoring dashboard for QAOA-XORSAT cluster runs.
#
# Shows job status, current depth, best value, and wall time for all 15 tasks.
# Run from the login node or any node with access to the working directory.
#
# Usage:
#   bash scripts/slurm-monitor.sh                 # one-shot dashboard
#   watch -n 60 bash scripts/slurm-monitor.sh     # refresh every 60s
#   bash scripts/slurm-monitor.sh -j 12345        # specific job ID

set -euo pipefail

cd "${0%/*}/.."

# ── Parse args ────────────────────────────────────────────────────────────────
JOB_ID=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--job-id) JOB_ID="$2"; shift 2 ;;
        *) echo "Usage: $0 [-j JOB_ID]" >&2; exit 1 ;;
    esac
done

# Task ID → (k,D) mapping
declare -a PAIRS=(
    "3,4" "3,5" "3,6" "3,7" "3,8"
    "4,5" "4,6" "4,7" "4,8"
    "5,6" "5,7" "5,8"
    "6,7" "6,8"
    "7,8"
)

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                    QAOA-XORSAT Cluster Dashboard                       ║"
echo "╠══════════════════════════════════════════════════════════════════════════╣"
printf "║  Time: %-30s  Node: %-20s ║\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname)"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# ── SLURM job status (if squeue is available) ─────────────────────────────────
if command -v squeue &>/dev/null; then
    echo "── SLURM Job Status ──────────────────────────────────────────────────────"
    if [[ -n "$JOB_ID" ]]; then
        squeue -j "$JOB_ID" -o "%.8i %.4t %.10M %.6D %R" 2>/dev/null || echo "  (no active jobs for ID $JOB_ID)"
    else
        squeue -u "$USER" -n qaoa -o "%.10i %.4a %.4t %.10M %.6D %R" 2>/dev/null || echo "  (no active QAOA jobs)"
    fi
    echo ""
fi

# ── Scan results directories ─────────────────────────────────────────────────
RUNS_DIR=".project/results/optimization/runs"

if [[ ! -d "$RUNS_DIR" ]]; then
    echo "No results directory found at $RUNS_DIR"
    exit 0
fi

# Find the latest run directory (by timestamp prefix)
LATEST_RUN=""
for dir in $(ls -d "$RUNS_DIR"/*/ 2>/dev/null | sort -r); do
    if [[ -f "$dir/results.csv" ]]; then
        LATEST_RUN="$dir"
        break
    fi
done

# Also find any SLURM-labelled runs
SLURM_RUNS=()
for dir in $(ls -d "$RUNS_DIR"/*/ 2>/dev/null | sort); do
    manifest="$dir/manifest.json"
    if [[ -f "$manifest" ]] && grep -q '"slurm-' "$manifest" 2>/dev/null; then
        SLURM_RUNS+=("$dir")
    fi
done

echo "── Per-Task Progress ─────────────────────────────────────────────────────"
printf "%-6s  %-6s  %-12s  %-14s  %-10s  %-8s  %s\n" \
    "Task" "(k,D)" "Best p" "Best c̃" "Wall (s)" "Status" "Run ID"
printf "%-6s  %-6s  %-12s  %-14s  %-10s  %-8s  %s\n" \
    "----" "-----" "------" "----------" "--------" "------" "------"

for task_id in $(seq 1 15); do
    pair="${PAIRS[$((task_id-1))]}"
    k="${pair%,*}"
    D="${pair#*,}"

    best_p="-"
    best_value="-"
    best_wall="-"
    status="no data"
    run_id="-"

    # Search all run directories for results matching this (k,D)
    for dir in $(ls -d "$RUNS_DIR"/*/ 2>/dev/null | sort -r); do
        # Skip warm-start seed directories (not real results)
        [[ "$(basename "$dir")" == ws-* ]] && continue
        results_csv="$dir/results.csv"
        [[ -f "$results_csv" ]] || continue

        # Check if this run has results for our (k,D)
        match=$(awk -F',' -v k="$k" -v d="$D" \
            'NR>1 && $8==k && $9==d { print $10, $15, $16, $21 }' \
            "$results_csv" 2>/dev/null | tail -1)

        if [[ -n "$match" ]]; then
            read -r p val wall conv <<< "$match"
            best_p="$p"
            best_value="$val"
            best_wall="$wall"
            run_id=$(basename "$dir")

            if [[ "$conv" == "true" ]]; then
                status="done"
            else
                status="optim"
            fi

            # Check for in-progress checkpoint at p+1
            next_p=$((p + 1))
            checkpoint="$dir/checkpoint-p${next_p}.csv"
            if [[ -f "$checkpoint" ]]; then
                status="p=$next_p…"
            fi

            break
        fi
    done

    printf "%-6s  (%s,%s)  %-12s  %-14s  %-10s  %-8s  %s\n" \
        "$task_id" "$k" "$D" "$best_p" "$best_value" "$best_wall" "$status" "$run_id"
done

echo ""

# ── Active checkpoints (in-progress optimizations) ───────────────────────────
echo "── Active Checkpoints ────────────────────────────────────────────────────"
found_checkpoint=false
for dir in $(ls -d "$RUNS_DIR"/*/ 2>/dev/null | sort -r); do
    for cp in "$dir"/checkpoint-p*.csv; do
        [[ -f "$cp" ]] || continue
        found_checkpoint=true
        p_label=$(basename "$cp" | sed 's/checkpoint-p\([0-9]*\)\.csv/\1/')
        run=$(basename "$dir")
        age_seconds=$(( $(date +%s) - $(stat -f%m "$cp" 2>/dev/null || stat -c%Y "$cp" 2>/dev/null) ))
        age_min=$((age_seconds / 60))
        printf "  %s  p=%s  (updated %d min ago)\n" "$run" "$p_label" "$age_min"
    done
done
$found_checkpoint || echo "  (none)"

echo ""

# ── Partial traces (shows optimizer is still making progress) ─────────────────
echo "── Partial Traces (optimizer activity) ──────────────────────────────────"
found_trace=false
for dir in $(ls -d "$RUNS_DIR"/*/ 2>/dev/null | sort -r); do
    for trace in "$dir"/trace-p*-partial.csv; do
        [[ -f "$trace" ]] || continue
        found_trace=true
        p_label=$(basename "$trace" | sed 's/trace-p\([0-9]*\)-partial\.csv/\1/')
        run=$(basename "$dir")
        lines=$(wc -l < "$trace" | tr -d ' ')
        last_line=$(tail -1 "$trace")
        last_val=$(echo "$last_line" | awk -F',' '{print $3}')
        last_gnorm=$(echo "$last_line" | awk -F',' '{print $4}')
        age_seconds=$(( $(date +%s) - $(stat -f%m "$trace" 2>/dev/null || stat -c%Y "$trace" 2>/dev/null) ))
        age_min=$((age_seconds / 60))
        printf "  %s  p=%s  iters=%s  val=%s  g_norm=%s  (%d min ago)\n" \
            "$run" "$p_label" "$((lines-1))" "$last_val" "$last_gnorm" "$age_min"
    done
done
$found_trace || echo "  (none)"

echo ""

# ── Log file tails (if SLURM output files exist) ─────────────────────────────
if ls qaoa_*-*.out &>/dev/null 2>&1; then
    echo "── Latest Log Output ─────────────────────────────────────────────────────"
    for logfile in $(ls -t qaoa_*-*.out 2>/dev/null | head -5); do
        task=$(echo "$logfile" | sed 's/qaoa_[0-9]*-\([0-9]*\)\.out/\1/')
        pair="${PAIRS[$((task-1))]}"
        echo "  [$logfile] (k=${pair%,*}, D=${pair#*,}):"
        tail -3 "$logfile" 2>/dev/null | sed 's/^/    /'
        echo ""
    done
fi
