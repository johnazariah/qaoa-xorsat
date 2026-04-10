#!/usr/bin/env julia
# Pure Double64 swarm chain for high (k,D) pairs where Float64 precision fails
#
# Usage: julia --project=. scripts/swarm_chain_d64.jl K D P_MAX [POP] [GENS] [BURST] [SEED]
#
# All evaluation and gradient computation runs in Double64 (~31 decimal digits).
# Optim.jl's L-BFGS parameter vector stays Float64; the evaluator promotes
# to Double64 internally and converts results back to Float64.

using QaoaXorsat
using Printf
using Random
using Dates
using DoubleFloats

k = parse(Int, get(ARGS, 1, "7"))
D = parse(Int, get(ARGS, 2, "8"))
p_max = parse(Int, get(ARGS, 3, "10"))
pop = parse(Int, get(ARGS, 4, "100"))
gens = parse(Int, get(ARGS, 5, "10"))
burst = parse(Int, get(ARGS, 6, "20"))
seed = parse(Int, get(ARGS, 7, "42"))

clause_sign = k == 2 ? -1 : 1

results_file = get(ENV, "QAOA_RESULTS_FILE",
    joinpath(@__DIR__, "..", "results", "swarm-d64-k$(k)d$(D).csv"))
mkpath(dirname(results_file))

function emit(msg)
    printstyled(msg, "\n")
    flush(stdout)
    open(results_file, "a") do io
        println(io, msg)
        flush(io)
    end
end

mutable struct State
    warm::Vector{QAOAAngles}
end

# Resume logic
p_start = 1
state = State(QAOAAngles[])

if isfile(results_file)
    @printf("Checking %s for resume point...\n", results_file)
    flush(stdout)
    for line in eachline(results_file)
        startswith(line, '#') && continue
        startswith(line, "k,") && continue
        fields = split(line, ',')
        length(fields) >= 8 || continue
        lk = tryparse(Int, fields[1]); lk === nothing && continue
        lD = tryparse(Int, fields[2]); lD === nothing && continue
        lp = tryparse(Int, fields[3]); lp === nothing && continue
        lv = tryparse(Float64, fields[4]); lv === nothing && continue
        (lk == k && lD == D) || continue
        QaoaXorsat.is_valid_qaoa_value(lv) && lv > 0.501 || continue
        gamma_strs = split(fields[7], ';')
        beta_strs = split(fields[8], ';')
        gamma = parse.(Float64, gamma_strs)
        beta = parse.(Float64, beta_strs)
        state.warm = [QAOAAngles(gamma, beta)]
        global p_start = lp + 1
        @printf("  Resuming from p=%d (c̃=%.10f)\n", lp, lv)
        flush(stdout)
    end
end

if p_start > 1
    @printf("Skipping p=1-%d, starting at p=%d\n", p_start - 1, p_start)
else
    emit("# swarm-d64 chain: k=$k, D=$D, p=1-$p_max, pop=$pop, gens=$gens, burst=$burst, seed=$seed")
    emit("k,D,p,ctilde,evals,wall_seconds,gamma,beta")
end
flush(stdout)

for p in p_start:p_max
    params = TreeParams(k, D, p)
    ws = isempty(state.warm) ? QAOAAngles[] : [extend_angles(state.warm[1], p)]

    t0 = time()

    # Pure D64: evaluation and gradient run in Double64 throughout the optimizer.
    # Optim.jl's L-BFGS parameter vector is still Float64; eval_eltype=Double64
    # promotes angles internally before calling the evaluator.
    result = swarm_optimize(
        params;
        clause_sign,
        population=pop,
        generations=gens,
        burst_iters=burst,
        rng=MersenneTwister(seed + p),
        warm_starts=ws,
        eval_eltype=Double64,
        on_generation=(gen, best, npop) -> begin
            emit(@sprintf("# p=%d gen %2d: best_d64=%.10f  pop=%d", p, gen, best, npop))
        end,
    )

    dt = time() - t0

    gamma_str = join(string.(result.angles.γ), ';')
    beta_str = join(string.(result.angles.β), ';')
    emit(@sprintf("%d,%d,%d,%.12f,%d,%.1f,%s,%s",
        k, D, p, result.value, result.evaluations, dt,
        gamma_str, beta_str))

    if QaoaXorsat.is_valid_qaoa_value(result.value) && result.value > 0.501
        state.warm = [result.angles]
    else
        emit(@sprintf("# chain broken at p=%d (d64 val=%.6f) — continuing with random starts", p, result.value))
        state.warm = QAOAAngles[]
    end
end

emit("# DONE")
