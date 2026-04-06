#!/usr/bin/env julia
# Swarm-chain optimizer for high (k,D) pairs
using QaoaXorsat
using Printf
using Random

k = parse(Int, get(ARGS, 1, "7"))
D = parse(Int, get(ARGS, 2, "8"))
p_max = parse(Int, get(ARGS, 3, "10"))
pop = parse(Int, get(ARGS, 4, "100"))
gens = parse(Int, get(ARGS, 5, "10"))
burst = parse(Int, get(ARGS, 6, "20"))
seed = parse(Int, get(ARGS, 7, "42"))

clause_sign = k == 2 ? -1 : 1

# Write results to a dedicated file (bypasses stdout buffering entirely)
results_file = get(ENV, "QAOA_RESULTS_FILE",
    joinpath(@__DIR__, "..", "results", "swarm-k$(k)d$(D).csv"))
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

# ── Resume: check for existing results file and pick up where we left off ─────
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

        # Parse angles for warm start
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
    emit("# swarm chain: k=$k, D=$D, p=1-$p_max, pop=$pop, gens=$gens, burst=$burst, seed=$seed")
    emit("k,D,p,ctilde,evals,wall_seconds,gamma,beta")
end
flush(stdout)

for p in p_start:p_max
    params = TreeParams(k, D, p)
    ws = isempty(state.warm) ? QAOAAngles[] : [extend_angles(state.warm[1], p)]

    result = swarm_optimize(
        params;
        clause_sign,
        population=pop,
        generations=gens,
        burst_iters=burst,
        rng=MersenneTwister(seed + p),
        warm_starts=ws,
        on_generation=(gen, best, npop) -> begin
            if gen == 1 || gen == gens
                emit(@sprintf("# gen %2d: best=%.10f  pop=%d", gen, best, npop))
            end
        end,
    )

    gamma_str = join(string.(result.angles.γ), ';')
    beta_str = join(string.(result.angles.β), ';')
    emit(@sprintf("%d,%d,%d,%.12f,%d,%.1f,%s,%s",
        k, D, p, result.value, result.evaluations, result.wall_time_seconds,
        gamma_str, beta_str))

    if QaoaXorsat.is_valid_qaoa_value(result.value) && result.value > 0.501
        state.warm = [result.angles]
    else
        emit(@sprintf("# chain broken at p=%d — continuing with random starts", p))
        state.warm = QAOAAngles[]
    end
end

emit("# DONE")
