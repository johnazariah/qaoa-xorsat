#!/bin/bash
#SBATCH --job-name=xorsat-k3-p16
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=1400G
#SBATCH --time=504:00:00
#SBATCH --output=logs/xorsat-k3-%a-%j.out
#SBATCH --error=logs/xorsat-k3-%a-%j.err
#SBATCH --requeue
#
# XORSAT k=3 clean sweep to p=16 with all optimizations
# Array index maps to D: 0→D=4, 1→D=5, 2→D=6, 3→D=7, 4→D=8
#
# Optimizations enabled:
#   - CPU gradient checkpointing (√p memory, p≥13)
#   - Disk spillover for checkpoints (p≥16, ~1.37TB in D64)
#   - Double64 precision past Float64 wall
#   - Gradient plateau detection (20-iter window)
#   - Warm-start chain from p-1
#   - Verbose progress logging
#
# Usage: sbatch scripts/slurm_xorsat_k3_p16.sh

set -euo pipefail

# ── Map array index to D ──────────────────────────────────────────────
D_VALUES=(4 5 6 7 8)
D=${D_VALUES[$SLURM_ARRAY_TASK_ID]}
K=3
P_MAX=16
CLAUSE_SIGN=1

# Float64 precision wall per D (beyond this, must use Double64)
declare -A F64_WALL
F64_WALL[4]=13
F64_WALL[5]=13
F64_WALL[6]=11
F64_WALL[7]=11
F64_WALL[8]=11

# Checkpointing threshold
CHECKPOINT_FROM=13

# Disk spillover directory (node-local NVMe)
DISK_DIR="/tmp/qaoa-checkpoints-k${K}-d${D}"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  XORSAT k=$K D=$D sweep p=1..$P_MAX                    ║"
echo "║  Node: $(hostname)                                      ║"
echo "║  CPUs: $SLURM_CPUS_PER_TASK                             ║"
echo "║  RAM:  $(free -g | awk '/^Mem:/{print $2}')G             ║"
echo "║  F64 wall: p=${F64_WALL[$D]}                             ║"
echo "║  Started: $(date)                                       ║"
echo "╚══════════════════════════════════════════════════════════╝"

# ── Setup Julia ───────────────────────────────────────────────────────
cd $HOME/qaoa-xorsat

# Precompile lockfile to avoid race with other array tasks
LOCKFILE="/tmp/qaoa-precompile.lock"
(
    flock -w 600 200 || { echo "Precompile lock timeout"; exit 1; }
    julia --project=. -e 'using QaoaXorsat; println("Precompiled OK")'
) 200>$LOCKFILE

# ── Run sweep ─────────────────────────────────────────────────────────
julia --project=. -t $SLURM_CPUS_PER_TASK -e "
using QaoaXorsat, Printf, Dates, Random
using DoubleFloats

K = $K
D = $D
P_MAX = $P_MAX
clause_sign = $CLAUSE_SIGN
F64_WALL = $(F64_WALL[$D])
CHECKPOINT_FROM = $CHECKPOINT_FROM
DISK_DIR = \"$DISK_DIR\"

fmt_time(s) = s < 60 ? @sprintf(\"%.1fs\", s) :
              s < 3600 ? @sprintf(\"%.1fmin\", s/60) :
              @sprintf(\"%.1fh\", s/3600)

results_file = joinpath(@__DIR__, \"results\", \"slurm-xorsat-k\$(K)-d\$(D)-p16.csv\")
mkpath(dirname(results_file))

# Read existing results for warm-start
function get_best_angles(file, k, D, target_p)
    best_val = -Inf
    best_gamma = Float64[]
    best_beta = Float64[]
    isfile(file) || return (best_val, best_gamma, best_beta)
    for line in eachline(file)
        startswith(line, '#') && continue
        startswith(line, 'k') && continue
        fields = split(line, ',')
        length(fields) >= 7 || continue
        lk = tryparse(Int, fields[1]); lk === nothing && continue
        lD = tryparse(Int, fields[2]); lD === nothing && continue
        lp = tryparse(Int, fields[3]); lp === nothing && continue
        lv = tryparse(Float64, fields[4]); lv === nothing && continue
        (lk == k && lD == D && lp == target_p) || continue
        if lv > best_val
            best_val = lv
            best_gamma = parse.(Float64, split(fields[6], ';'))
            best_beta = parse.(Float64, split(fields[7], ';'))
        end
    end
    return (best_val, best_gamma, best_beta)
end

# Also check composite-best for warm-start from previous runs
function get_warm_start(k, D, p)
    # Check our new results first
    v, g, b = get_best_angles(results_file, k, D, p)
    v > -Inf && return (v, g, b)
    # Fall back to composite-best
    composite = joinpath(@__DIR__, \"results\", \"composite-best.csv\")
    return get_best_angles(composite, k, D, p)
end

# Create CSV header if new
if !isfile(results_file)
    open(results_file, \"w\") do io
        println(io, \"# XORSAT k=\$K D=\$D sweep to p=\$P_MAX — \$(now())\")
        println(io, \"# Node: \$(gethostname()) CPUs: \$(Threads.nthreads())\")
        println(io, \"k,D,p,ctilde,wall_seconds,gamma,beta\")
    end
end

println(\"Threads: \$(Threads.nthreads())\")
grand_start = time()

for p in 1:P_MAX
    # Skip if already computed
    existing, _, _ = get_best_angles(results_file, K, D, p)
    if existing > -Inf
        @printf(\"  ⏭ p=%d already done (c̃=%.10f)\\n\", p, existing)
        continue
    end

    # Determine precision and checkpointing
    use_d64 = p > F64_WALL
    use_ckpt = p >= CHECKPOINT_FROM
    use_disk = p >= 16
    eltype = use_d64 ? Double64 : Float64

    # Get warm-start from p-1
    warm_starts = QAOAAngles[]
    if p > 1
        prev_v, prev_g, prev_b = get_warm_start(K, D, p - 1)
        if prev_v > -Inf
            warm_starts = [extend_angles(QAOAAngles(prev_g, prev_b), p)]
            @printf(\"  warm-start from p=%d: c̃=%.10f\\n\", p-1, prev_v)
        end
    end

    params = TreeParams(K, D, p)

    @printf(\"  ▶ p=%d D64=%s ckpt=%s disk=%s at %s\\n\",
            p, use_d64, use_ckpt, use_disk, Dates.format(now(), \"HH:MM:SS\"))
    flush(stdout)

    t0 = time()
    try
        result = optimize_angles(params;
            clause_sign,
            initial_guesses = warm_starts,
            restarts = p <= 8 ? 4 : 0,
            g_abstol = 1e-6,
            eval_eltype = eltype,
            checkpointed = use_ckpt,
            on_evaluation = (chunk, evals, elapsed, val, gnorm) -> begin
                @printf(\"    [k=%d D=%d p=%d] eval %d: c̃=%.10f |∇|=%.2e %s\\n\",
                        K, D, p, evals, val, gnorm, fmt_time(elapsed))
                flush(stdout)
            end
        )
        dt = time() - t0

        # Save
        gamma_str = join(string.(result.angles.γ), ';')
        beta_str = join(string.(result.angles.β), ';')
        open(results_file, \"a\") do io
            @printf(io, \"%d,%d,%d,%.12f,%.1f,%s,%s\\n\",
                K, D, p, result.value, dt, gamma_str, beta_str)
        end
        flush(stdout)

        @printf(\"  ✓ p=%d: c̃=%.12f in %s\\n\", p, result.value, fmt_time(dt))
    catch e
        dt = time() - t0
        @printf(\"  ✗ p=%d FAILED after %s: %s\\n\", p, fmt_time(dt), sprint(showerror, e))
        # Don't break — try next p with random start
    end
    flush(stdout)
end

# Cleanup disk spillover
rm(DISK_DIR; force=true, recursive=true)

grand_total = time() - grand_start
@printf(\"\\nDone! k=%d D=%d total: %s\\n\", K, D, fmt_time(grand_total))
"

echo ""
echo "Job finished at $(date)"
