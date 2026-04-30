#!/bin/bash
#SBATCH --job-name=xorsat-all-p16
#SBATCH --array=0-14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=1400G
#SBATCH --time=504:00:00
#SBATCH --output=logs/xorsat-%a-%j.out
#SBATCH --error=logs/xorsat-%a-%j.err
#SBATCH --requeue
#
# Full XORSAT sweep: all 15 (k,D) pairs to max feasible depth
# 15 nodes, one per pair, all running in parallel
#
# All optimizations enabled:
#   - CPU gradient checkpointing (√p memory, p≥13)
#   - Disk spillover (p≥16 in D64)
#   - Double64 past Float64 precision wall
#   - Gradient plateau detection
#   - Warm-start chain
#   - Verbose progress logging
#
# Usage: sbatch scripts/slurm_xorsat_all.sh

set -euo pipefail

# ── (k,D) pairs indexed by SLURM_ARRAY_TASK_ID ───────────────────────
K_VALUES=(3 3 3 3 3  4 4 4 4  5 5 5  6 6  7)
D_VALUES=(4 5 6 7 8  5 6 7 8  6 7 8  7 8  8)
# Max target depth per pair (memory-limited at 1.5TB with D64 checkpointing)
P_MAX_VALUES=(16 16 16 15 15  14 14 14 14  13 13 13  12 12  12)
# Float64 precision wall
F64_WALL_VALUES=(13 13 11 11 11  11 10 10 9  9 9 9  8 8  7)

IDX=$SLURM_ARRAY_TASK_ID
K=${K_VALUES[$IDX]}
D=${D_VALUES[$IDX]}
P_MAX=${P_MAX_VALUES[$IDX]}
F64_WALL=${F64_WALL_VALUES[$IDX]}
CLAUSE_SIGN=1
CHECKPOINT_FROM=13
DISK_DIR="/tmp/qaoa-checkpoints-k${K}-d${D}"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  XORSAT k=$K D=$D sweep p=1..$P_MAX                    ║"
echo "║  Node: $(hostname)                                      ║"
echo "║  CPUs: $SLURM_CPUS_PER_TASK                             ║"
echo "║  RAM:  $(free -g | awk '/^Mem:/{print $2}')G             ║"
echo "║  F64 wall: p=$F64_WALL                                  ║"
echo "║  Started: $(date)                                       ║"
echo "╚══════════════════════════════════════════════════════════╝"

cd $HOME/qaoa-xorsat
mkdir -p logs

# Precompile with lockfile
LOCKFILE="/tmp/qaoa-precompile.lock"
(
    flock -w 600 200 || { echo "Precompile lock timeout"; exit 1; }
    julia --project=. -e 'using QaoaXorsat; println("Precompiled OK")'
) 200>$LOCKFILE

julia --project=. -t $SLURM_CPUS_PER_TASK -e "
using QaoaXorsat, Printf, Dates, Random
using DoubleFloats

K = $K; D = $D; P_MAX = $P_MAX; clause_sign = $CLAUSE_SIGN
F64_WALL = $F64_WALL; CHECKPOINT_FROM = $CHECKPOINT_FROM
DISK_DIR = \"$DISK_DIR\"

fmt_time(s) = s < 60 ? @sprintf(\"%.1fs\", s) :
              s < 3600 ? @sprintf(\"%.1fmin\", s/60) :
              @sprintf(\"%.1fh\", s/3600)

results_file = joinpath(@__DIR__, \"results\", \"slurm-xorsat-k\$(K)-d\$(D).csv\")
mkpath(dirname(results_file))

function get_best_angles(file, k, D, target_p)
    best_val = -Inf; best_gamma = Float64[]; best_beta = Float64[]
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

function get_warm_start(k, D, p)
    v, g, b = get_best_angles(results_file, k, D, p)
    v > -Inf && return (v, g, b)
    for f in readdir(joinpath(@__DIR__, \"results\"); join=true)
        endswith(f, \".csv\") || continue
        v2, g2, b2 = get_best_angles(f, k, D, p)
        v2 > v && ((v, g, b) = (v2, g2, b2))
    end
    return (v, g, b)
end

if !isfile(results_file)
    open(results_file, \"w\") do io
        println(io, \"# XORSAT k=\$K D=\$D sweep — \$(now())\")
        println(io, \"k,D,p,ctilde,wall_seconds,gamma,beta\")
    end
end

println(\"Threads: \$(Threads.nthreads())\")
grand_start = time()

for p in 1:P_MAX
    existing, _, _ = get_best_angles(results_file, K, D, p)
    if existing > -Inf
        @printf(\"  ⏭ p=%d (c̃=%.10f)\\n\", p, existing)
        continue
    end

    use_d64 = p > F64_WALL
    use_ckpt = p >= CHECKPOINT_FROM
    eltype = use_d64 ? Double64 : Float64

    # Swarm at p≤12 where evals are cheap; warm-start only at p≥13
    use_swarm = p <= 12

    warm_starts = QAOAAngles[]
    if p > 1
        pv, pg, pb = get_warm_start(K, D, p - 1)
        if pv > -Inf
            warm_starts = [extend_angles(QAOAAngles(pg, pb), p)]
            @printf(\"  warm from p=%d: c̃=%.10f\\n\", p-1, pv)
        end
    end

    params = TreeParams(K, D, p)
    @printf(\"  ▶ k=%d D=%d p=%d D64=%s ckpt=%s swarm=%s at %s\\n\",
            K, D, p, use_d64, use_ckpt, use_swarm, Dates.format(now(), \"HH:MM:SS\"))
    flush(stdout)

    t0 = time()
    try
        if use_swarm
            # Swarm: population search to find the right basin
            pop = p <= 6 ? 50 : (p <= 9 ? 30 : 15)
            result = swarm_optimize(params;
                clause_sign,
                population = pop,
                generations = 5,
                burst_iters = 30,
                warm_starts = warm_starts,
                rng = Random.MersenneTwister(42 + p),
                g_abstol = 1e-6,
                eval_eltype = eltype,
                on_generation = (gen, best, npop, _angles) -> begin
                    elapsed = time() - t0
                    @printf(\"    [k=%d D=%d p=%d] gen %d: best=%.10f pop=%d %s\\n\",
                            K, D, p, gen, best, npop, fmt_time(elapsed))
                    flush(stdout)
                end,
            )
        else
            # High-p: single L-BFGS warm-start with checkpointing
            result = optimize_angles(params;
                clause_sign,
                initial_guesses = warm_starts,
                restarts = 0,
                g_abstol = 1e-6,
                eval_eltype = eltype,
                checkpointed = use_ckpt,
                on_evaluation = (chunk, evals, elapsed, val, gnorm) -> begin
                    @printf(\"    [k=%d D=%d p=%d] eval %d: c̃=%.10f |∇|=%.2e %s\\n\",
                            K, D, p, evals, val, gnorm, fmt_time(elapsed))
                    flush(stdout)
                end
            )
        end
        dt = time() - t0
        gs = join(string.(result.angles.γ), ';')
        bs = join(string.(result.angles.β), ';')
        open(results_file, \"a\") do io
            @printf(io, \"%d,%d,%d,%.12f,%.1f,%s,%s\\n\", K, D, p, result.value, dt, gs, bs)
        end
        @printf(\"  ✓ p=%d: c̃=%.12f in %s\\n\", p, result.value, fmt_time(dt))
    catch e
        dt = time() - t0
        @printf(\"  ✗ p=%d FAILED after %s: %s\\n\", p, fmt_time(dt), sprint(showerror, e))
    end
    flush(stdout)
end

rm(DISK_DIR; force=true, recursive=true)
@printf(\"\\nDone! k=%d D=%d total: %s\\n\", K, D, fmt_time(time() - grand_start))
"

echo "Job finished at $(date)"
