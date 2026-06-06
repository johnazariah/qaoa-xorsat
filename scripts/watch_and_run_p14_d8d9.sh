#!/usr/bin/env bash
# Watcher: poll sweep CSVs for D=8 and D=9 p=13 rows, and launch
# run_p14_d_warm.jl <D> for each as soon as the p=13 seed exists.
# Runs the two D values sequentially (D=8 then D=9).

set -u
cd "$(dirname "$0")/.."

POLL_SEC=${POLL_SEC:-60}
TS="$(date +%Y%m%d-%H%M%S)"
LOG="logs/p14-d8d9-watch-${TS}.log"
mkdir -p logs results

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

wait_for_p13_row() {
    local D=$1
    local csv="results/maxcut-k2-d${D}-sweep.csv"
    log "waiting for p=13 row in ${csv}"
    while true; do
        if [[ -f "$csv" ]] && grep -q "^2,${D},13," "$csv"; then
            log "p=13 row for D=${D} present"
            return 0
        fi
        sleep "$POLL_SEC"
    done
}

run_d() {
    local D=$1
    local rlog="logs/p14-d${D}-memfix-${TS}.log"
    log "launching run_p14_d_warm.jl ${D} -> ${rlog}"
    JULIA_NUM_THREADS=12 julia --project=. scripts/run_p14_d_warm.jl "$D" \
        >> "$rlog" 2>&1
    local rc=$?
    log "D=${D} exited rc=${rc} (log: ${rlog})"
    # Success guard: a new row for (k=2, D, p=14) in timing CSV
    if grep -q "^2,${D},14," results/maxcut-k2-p14-timing.csv 2>/dev/null; then
        log "timing row for D=${D} p=14 confirmed"
        return 0
    fi
    log "WARN: no timing row for D=${D} p=14; halting chain"
    return 1
}

log "===== watcher start ====="
log "PID=$$"

for D in 8 9; do
    wait_for_p13_row "$D" || { log "FATAL: wait failed for D=${D}"; exit 2; }
    run_d "$D" || { log "FATAL: run failed for D=${D}"; exit 3; }
done

log "===== watcher complete ====="
