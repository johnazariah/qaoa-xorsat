#!/bin/bash
#
# QAOA cluster diagnostic — run this and send John the output.
#
# Usage:
#   cd ~/qaoa-xorsat
#   bash scripts/cluster-diagnostic.sh 2>&1 | tee /tmp/qaoa-diagnostic.txt
#
# Then send John the file /tmp/qaoa-diagnostic.txt
#
set -uo pipefail

echo "============================================================"
echo "QAOA Cluster Diagnostic — $(date -u)"
echo "============================================================"
echo ""

echo "--- Git state ---"
cd ~/qaoa-xorsat 2>/dev/null || { echo "ERROR: ~/qaoa-xorsat not found"; exit 1; }
git log --oneline -5
echo ""
git status --short | head -20
echo ""
git diff --stat HEAD | head -10
echo ""

echo "--- Julia version ---"
export PATH="$HOME/.juliaup/bin:$PATH"
julia --version 2>&1 || echo "ERROR: julia not found"
echo ""

echo "--- Julia project deps ---"
julia --project=. -e 'using Pkg; Pkg.status()' 2>&1 | head -20
echo ""

echo "--- DoubleFloats installed? ---"
julia --project=. -e 'using DoubleFloats; println("DoubleFloats OK: ", Double64(pi))' 2>&1
echo ""

echo "--- Quick smoke test ---"
julia --project=. -e '
using QaoaXorsat, DoubleFloats
params = TreeParams(3, 4, 2)
angles = QAOAAngles([0.3, 0.5], [0.2, 0.4])
v64 = basso_expectation_normalized(params, angles; clause_sign=1)
vd64 = Float64(basso_expectation_normalized(params, QAOAAngles(Double64.([0.3, 0.5]), Double64.([0.2, 0.4])); clause_sign=1))
println("Float64:  ", v64)
println("Double64: ", vd64)
println("Match:    ", isapprox(v64, vd64, atol=1e-10))
' 2>&1
echo ""

echo "--- SLURM queue ---"
squeue -u $USER -o "%.10i %.4t %.10M %.6D %R" 2>/dev/null || echo "squeue not available"
echo ""

echo "--- Recent SLURM jobs (last 24h) ---"
sacct -u $USER --starttime=$(date -u -d "24 hours ago" +%Y-%m-%dT%H:%M 2>/dev/null || date -u -v-24H +%Y-%m-%dT%H:%M 2>/dev/null || echo "2026-04-09") --format=JobID,JobName%20,State%12,Elapsed,ExitCode,NodeList -n 2>/dev/null | head -30
echo ""

echo "--- Error logs (last 5 files, last 20 lines each) ---"
for f in $(ls -t qaoa-d64_*.err qaoa-swarm_*.err qaoa-ws_*.err 2>/dev/null | head -5); do
    echo "=== $f ==="
    tail -20 "$f"
    echo ""
done

echo "--- Output logs (last 5 files, last 10 lines each) ---"
for f in $(ls -t qaoa-d64_*.out qaoa-swarm_*.out qaoa-ws_*.out 2>/dev/null | head -5); do
    echo "=== $f ==="
    tail -10 "$f"
    echo ""
done

echo "--- Double64 swarm results ---"
for f in results/swarm-d64-k*.csv; do
    if [ -f "$f" ]; then
        echo "=== $(basename $f) ==="
        wc -l "$f" | awk '{print $1, "lines"}'
        grep "^[0-9]" "$f" | tail -3
        echo ""
    fi
done

echo "--- Float64 swarm results ---"
for f in results/swarm-k*.csv; do
    if [ -f "$f" ]; then
        echo "=== $(basename $f) ==="
        grep "^[0-9]" "$f" | tail -1
        echo ""
    fi
done

echo "--- Optimization run directories (last 10) ---"
ls -dt .project/results/optimization/runs/*/ 2>/dev/null | head -10
echo ""

echo "--- Disk usage ---"
du -sh .project/results/ results/ 2>/dev/null
echo ""

echo "--- Node info ---"
hostname
nproc 2>/dev/null || echo "nproc not available"
free -h 2>/dev/null | head -2 || echo "free not available"
echo ""

echo "============================================================"
echo "Diagnostic complete. Send /tmp/qaoa-diagnostic.txt to John."
echo "============================================================"
