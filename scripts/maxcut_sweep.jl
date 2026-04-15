#!/usr/bin/env julia
# MaxCut (k=2) sweep across multiple D values.
#
# Usage: julia --project=. -t 16 scripts/maxcut_sweep.jl D P_MAX [SEED]
#
# Results written to results/maxcut-k2-dD-sweep.csv
# Supports resume: reads existing CSV and continues from last completed p.

using QaoaXorsat
using Printf
using Random
using Dates

D = parse(Int, get(ARGS, 1, "3"))
p_max = parse(Int, get(ARGS, 2, "13"))
seed = parse(Int, get(ARGS, 3, "42"))

k = 2
clause_sign = -1  # MaxCut

results_file = joinpath(@__DIR__, "..", "results", "maxcut-k2-d$(D)-sweep.csv")
mkpath(dirname(results_file))

# Resume logic
p_start = 1
warm = QAOAAngles[]

if isfile(results_file)
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
        gamma = parse.(Float64, split(fields[6], ';'))
        beta = parse.(Float64, split(fields[7], ';'))
        warm = [QAOAAngles(gamma, beta)]
        p_start = lp + 1
    end
    if p_start > 1
        @printf("Resuming from p=%d\n", p_start)
    end
else
    open(results_file, "w") do io
        println(io, "# MaxCut k=2 D=$D sweep — $(now())")
        println(io, "k,D,p,ctilde,wall_seconds,gamma,beta")
    end
end

println("=== MaxCut (k=$k, D=$D) sweep p=$p_start..$p_max ===")
println("Threads: $(Threads.nthreads())")
println("Start: $(now())")
println()
flush(stdout)

for p in p_start:p_max
    params = TreeParams(k, D, p)
    ws = isempty(warm) ? QAOAAngles[] : [extend_angles(warm[1], p)]

    t0 = time()
    result = optimize_angles(
        params;
        clause_sign,
        restarts = max(8, 2*p),
        maxiters = 1280,
        initial_guesses = ws,
        rng = MersenneTwister(seed + p),
        g_abstol = 1e-8,
    )
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
