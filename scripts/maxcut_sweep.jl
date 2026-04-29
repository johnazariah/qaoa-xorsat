#!/usr/bin/env julia
# MaxCut (k=2) sweep across multiple D values.
#
# Usage: julia --project=. -t 16 scripts/maxcut_sweep.jl D P_MAX [SEED]
#
# Results written to results/maxcut-k2-dD-sweep.csv
# Supports resume: reads existing CSV and continues from last completed p.
#
# Optimizations enabled unconditionally on all platforms:
#   - GPU evaluator auto-detected: CUDA (NVIDIA) > Metal (Mac) > CPU
#   - CPU gradient checkpointing (√p memory) with disk spillover
#   - Double64 evaluation precision for k ≥ 6
#   - Memetic swarm + L-BFGS polish via swarm_optimize

using QaoaXorsat
using DoubleFloats
using Printf
using Random
using Dates

include(joinpath(@__DIR__, "..", "src", "gpu_backend.jl"))

D = parse(Int, get(ARGS, 1, "3"))
p_max = parse(Int, get(ARGS, 2, "13"))
seed = parse(Int, get(ARGS, 3, "42"))

k = 2
clause_sign = -1  # MaxCut

results_file = joinpath(@__DIR__, "..", "results", "maxcut-k2-d$(D)-sweep.csv")
mkpath(dirname(results_file))

# Resume logic
function resume_from_csv(results_file, k, D)
    max_p = 0
    for line in eachline(results_file)
        startswith(line, '#') && continue
        startswith(line, "k,") && continue
        fields = split(line, ',')
        length(fields) >= 7 || continue
        lk = tryparse(Int, fields[1]); lk === nothing && continue
        lD = tryparse(Int, fields[2]); lD === nothing && continue
        lp = tryparse(Int, fields[3]); lp === nothing && continue
        lv = tryparse(Float64, fields[4]); lv === nothing && continue
        (lk == k && lD == D) || continue
        QaoaXorsat.is_valid_qaoa_value(lv) || continue
        max_p = max(max_p, lp)
    end

    max_p == 0 && return (1, QAOAAngles[])

    best_value = -Inf
    best_warm = QAOAAngles[]
    for line in eachline(results_file)
        startswith(line, '#') && continue
        startswith(line, "k,") && continue
        fields = split(line, ',')
        length(fields) >= 7 || continue
        lk = tryparse(Int, fields[1]); lk === nothing && continue
        lD = tryparse(Int, fields[2]); lD === nothing && continue
        lp = tryparse(Int, fields[3]); lp === nothing && continue
        lv = tryparse(Float64, fields[4]); lv === nothing && continue
        (lk == k && lD == D && lp == max_p) || continue
        if lv > best_value
            best_value = lv
            gamma = parse.(Float64, split(fields[6], ';'))
            beta = parse.(Float64, split(fields[7], ';'))
            best_warm = [QAOAAngles(gamma, beta)]
        end
    end
    @printf("Resuming from p=%d (warm-start from p=%d, c̃=%.10f)\n", max_p + 1, max_p, best_value)
    return (max_p + 1, best_warm)
end

# ── Auto-detect GPU evaluator (CUDA > Metal > CPU) ──────────────────────
gpu_evaluator = make_gpu_evaluator()
gpu_status = GPU_BACKEND.label

# ── Disk spillover for high-p checkpoint storage ────────────────────────
tmp_root = joinpath(@__DIR__, "..", "tmp")
mkpath(tmp_root)
checkpoint_dir = mktempdir(tmp_root; prefix="qaoa-d$(D)-")
atexit(() -> try rm(checkpoint_dir; force=true, recursive=true) catch end)

# ── Evaluation precision: Double64 for k ≥ 6, Float64 otherwise ──────────
eval_eltype = k ≥ 6 ? Double64 : Float64

p_start = 1
warm = QAOAAngles[]

if isfile(results_file)
    p_start, warm = resume_from_csv(results_file, k, D)
else
    open(results_file, "w") do io
        println(io, "# MaxCut k=2 D=$D sweep — $(now())")
        println(io, "k,D,p,ctilde,wall_seconds,gamma,beta")
    end
end

println("=== MaxCut (k=$k, D=$D) sweep p=$p_start..$p_max ===")
println("Threads:        $(Threads.nthreads())")
println("GPU:            $gpu_status")
println("eval_eltype:    $eval_eltype")
println("Checkpoint dir: $checkpoint_dir")
println("Start:          $(now())")
println()
flush(stdout)

for p in p_start:p_max
    params = TreeParams(k, D, p)
    ws = isempty(warm) ? QAOAAngles[] : [extend_angles(warm[1], p)]

    # Swarm: small population at high p (memory-bounded), with warm-start.
    # MaxCut is single-basin so swarm exits early after 1-2 gens, then polishes.
    pop = p ≤ 8 ? 50 : (p ≤ 11 ? 20 : 10)

    t0 = time()
    local result
    try
        result = swarm_optimize(
            params;
            clause_sign,
            population = pop,
            generations = 5,
            burst_iters = 20,
            warm_starts = ws,
            rng = MersenneTwister(seed + p),
            g_abstol = 1e-8,
            gpu_evaluator,
            checkpointed = true,
            checkpoint_disk_dir = checkpoint_dir,
            checkpoint_max_ram_checkpoints = 4,
            eval_eltype,
            on_generation = (gen, best, npop, _angles) -> begin
                elapsed = time() - t0
                @printf("  p=%d gen %d: best=%.10f pop=%d elapsed=%.0fs\n",
                        p, gen, best, npop, elapsed)
                flush(stdout)
            end,
        )
    catch e
        dt = time() - t0
        @printf("p=%2d  FAILED after %.1fs: %s\n", p, dt, sprint(showerror, e))
        println("Stopping sweep — out of resources.")
        flush(stdout)
        break
    end
    dt = time() - t0

    gamma_str = join(string.(result.angles.γ), ';')
    beta_str = join(string.(result.angles.β), ';')

    line = @sprintf("%d,%d,%d,%.12f,%.1f,%s,%s",
        k, D, p, result.value, dt, gamma_str, beta_str)

    open(results_file, "a") do io
        println(io, line)
    end

    @printf("p=%2d  c̃=%.10f  time=%.1fs  converged=%s\n",
        p, result.value, dt, result.converged)
    flush(stdout)

    global warm = [result.angles]
end

println("\nDone: $(now())")
println("Results: $results_file")
