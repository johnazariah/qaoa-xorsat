#!/bin/bash
#
# QAOA Double64 sweep — submit, monitor, and push results.
#
# Usage (on the cluster login node):
#   cd ~/qaoa-xorsat
#   git pull origin main
#   bash scripts/run-d64-sweep.sh 2>&1 | tee /tmp/qaoa-d64-run.log
#
set -uo pipefail

REPO=~/qaoa-xorsat
cd "$REPO"
export PATH="$HOME/.juliaup/bin:$PATH"

echo "============================================================"
echo "QAOA Double64 Sweep — $(date -u)"
echo "============================================================"
echo ""

# ── Pre-flight diagnostics ────────────────────────────────────────
echo "--- Git state ---"
git log --oneline -3
echo ""

echo "--- Julia version ---"
julia --version 2>&1 || { echo "ERROR: julia not found"; exit 1; }
echo ""

echo "--- Installing dependencies ---"
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' 2>&1
echo ""

echo "--- Smoke test ---"
julia --project=. -e '
using QaoaXorsat, DoubleFloats
v = Float64(basso_expectation_normalized(
    TreeParams(3, 4, 2),
    QAOAAngles(Double64.([0.3, 0.5]), Double64.([0.2, 0.4]));
    clause_sign=1))
println("Smoke test: ", v > 0 && v < 1 ? "PASS ($v)" : "FAIL ($v)")
' 2>&1
echo ""

# ── Cancel any existing jobs ──────────────────────────────────────
echo "--- Cancelling existing jobs ---"
scancel -u $USER 2>/dev/null && echo "Cancelled" || echo "No jobs to cancel"
sleep 2
echo ""

# ── Clear old D64 results (they were F64-optimized, not pure D64) ──
echo "--- Clearing old D64 result files (were F64-optimized garbage) ---"
rm -f results/swarm-d64-k*.csv 2>/dev/null && echo "Cleared" || echo "Nothing to clear"
echo ""

# ── Submit ────────────────────────────────────────────────────────
echo "--- Submitting D64 sweep ---"
JOBID=$(sbatch --parsable scripts/qaoa_d64_sweep.sh 2>&1)
echo "Submitted job: $JOBID"
echo ""

if [[ ! "$JOBID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: sbatch failed. Output was: $JOBID"
    echo ""
    echo "--- sbatch error details ---"
    sbatch scripts/qaoa_d64_sweep.sh 2>&1
    exit 1
fi

# ── Monitor loop ──────────────────────────────────────────────────
echo "--- Monitoring (Ctrl-C to stop monitoring; jobs keep running) ---"
echo ""

PUSH_INTERVAL=600  # push to git every 10 minutes
LAST_PUSH=$(date +%s)
ITERATION=0

while true; do
    ITERATION=$((ITERATION + 1))
    NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    # Check job status
    RUNNING=$(squeue -j $JOBID --noheader 2>/dev/null | wc -l | tr -d '[:space:]')
    COMPLETED=$(sacct -j $JOBID --format=State --noheader 2>/dev/null | grep -c "COMPLETED" || true)
    FAILED=$(sacct -j $JOBID --format=State --noheader 2>/dev/null | grep -cE "FAILED|CANCELLED|TIMEOUT" || true)

    # Default to 0 if empty
    RUNNING=${RUNNING:-0}
    COMPLETED=${COMPLETED:-0}
    FAILED=${FAILED:-0}

    echo "[$NOW] Running: $RUNNING  Completed: $COMPLETED  Failed: $FAILED"

    # Show latest results
    for f in results/swarm-d64-k*.csv; do
        [ -f "$f" ] || continue
        LAST=$(grep "^[0-9]" "$f" 2>/dev/null | tail -1)
        [ -z "$LAST" ] && continue
        K=$(echo "$LAST" | cut -d',' -f1)
        D=$(echo "$LAST" | cut -d',' -f2)
        P=$(echo "$LAST" | cut -d',' -f3)
        V=$(echo "$LAST" | cut -d',' -f4)
        echo "  ($K,$D) p=$P c̃=$V"
    done

    # Show any errors
    if [ "$FAILED" -gt 0 ]; then
        echo ""
        echo "  !!! $FAILED tasks failed. Error logs:"
        for f in $(ls -t qaoa-d64_${JOBID}-*.err 2>/dev/null | head -3); do
            echo "  === $(basename $f) ==="
            tail -10 "$f" 2>/dev/null
        done
    fi

    echo ""

    # Push to git periodically
    CURRENT=$(date +%s)
    ELAPSED=$((CURRENT - LAST_PUSH))
    if [ $ELAPSED -ge $PUSH_INTERVAL ]; then
        cd "$REPO"
        git add -f results/swarm-d64-*.csv 2>/dev/null
        git commit -m "d64: progress update $NOW" --allow-empty 2>/dev/null
        git push origin HEAD:stephen-d64-results 2>/dev/null && echo "  [pushed to stephen-d64-results]" || echo "  [push failed]"
        LAST_PUSH=$CURRENT
    fi

    # Exit if all done
    if [ "$RUNNING" -eq 0 ] && [ $ITERATION -gt 2 ]; then
        echo "All tasks finished."
        echo ""

        # Final push
        cd "$REPO"
        git add -f results/swarm-d64-*.csv 2>/dev/null
        git commit -m "d64: final results $NOW" 2>/dev/null
        git push origin HEAD:stephen-d64-results 2>/dev/null && echo "Pushed final results to stephen-d64-results" || echo "Push failed"

        echo ""
        echo "--- Final results ---"
        for f in results/swarm-d64-k*.csv; do
            [ -f "$f" ] || continue
            echo "=== $(basename $f) ==="
            grep "^[0-9]" "$f" 2>/dev/null | tail -3
            echo ""
        done

        echo "--- Job accounting ---"
        sacct -j $JOBID --format=JobID,JobName%20,State%12,Elapsed,ExitCode,MaxRSS -n 2>/dev/null

        break
    fi

    sleep 300  # check every 5 minutes
done

echo ""
echo "============================================================"
echo "Done. Results in results/swarm-d64-*.csv"
echo "Pushed to branch: stephen-d64-results"
echo "============================================================"
