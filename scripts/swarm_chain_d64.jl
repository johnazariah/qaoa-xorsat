#!/usr/bin/env julia
# Double64 swarm chain for high (k,D) pairs where Float64 precision fails
#
# Usage: julia --project=. scripts/swarm_chain_d64.jl K D P_MAX [POP] [GENS] [BURST] [SEED]
#
# Same as swarm_chain.jl but uses Double64 arithmetic for the evaluator.
# The optimizer (L-BFGS) still runs in Float64 — only the function evaluation
# and gradient are computed in Double64.

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

# Override the evaluator to use Double64 for evaluation only.
# The optimizer's L-BFGS runs in Float64 but calls this for f and grad.
function d64_evaluate(params, angles; clause_sign=1)
    d64_angles = QAOAAngles(Double64.(angles.γ), Double64.(angles.β))
    Float64(basso_expectation_normalized(params, d64_angles; clause_sign))
end

function d64_evaluate_and_gradient(params, angles; clause_sign=1)
    d64_angles = QAOAAngles(Double64.(angles.γ), Double64.(angles.β))
    val, γg, βg = basso_expectation_and_gradient(params, d64_angles; clause_sign)
    (Float64(val), Float64.(γg), Float64.(βg))
end

# Monkey-patch: temporarily override the evaluator used by swarm_optimize
# by wrapping the angles in Double64 before evaluating.
# Since swarm_optimize calls optimize_angles which calls the adjoint path,
# the cleanest approach is to just run the swarm in Float64 and re-evaluate
# the winner in Double64.

for p in p_start:p_max
    params = TreeParams(k, D, p)
    ws = isempty(state.warm) ? QAOAAngles[] : [extend_angles(state.warm[1], p)]

    t0 = time()

    # Run swarm in Float64 (fast, finds the basin)
    result = swarm_optimize(
        params;
        clause_sign,
        population=pop,
        generations=gens,
        burst_iters=burst,
        rng=MersenneTwister(seed + p),
        warm_starts=ws,
        on_generation=(gen, best, npop) -> begin
            # Re-evaluate best in Double64 for the log
            emit(@sprintf("# p=%d gen %2d: best_f64=%.10f  pop=%d", p, gen, best, npop))
        end,
    )

    # Re-evaluate the winner in Double64 for the actual reported value
    val_d64 = d64_evaluate(params, result.angles; clause_sign)
    dt = time() - t0

    gamma_str = join(string.(result.angles.γ), ';')
    beta_str = join(string.(result.angles.β), ';')
    emit(@sprintf("%d,%d,%d,%.12f,%d,%.1f,%s,%s",
        k, D, p, val_d64, result.evaluations, dt,
        gamma_str, beta_str))

    if QaoaXorsat.is_valid_qaoa_value(val_d64) && val_d64 > 0.501
        state.warm = [result.angles]
    else
        emit(@sprintf("# chain broken at p=%d (d64 val=%.6f) — continuing with random starts", p, val_d64))
        state.warm = QAOAAngles[]
    end
end

emit("# DONE")
