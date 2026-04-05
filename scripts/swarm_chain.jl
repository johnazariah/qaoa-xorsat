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

@printf("Swarm chain: k=%d, D=%d, p=1–%d, pop=%d, gens=%d, burst=%d, seed=%d\n",
    k, D, p_max, pop, gens, burst, seed)

mutable struct State
    warm::Vector{QAOAAngles}
end

state = State(QAOAAngles[])

for p in 1:p_max
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
                @printf("  gen %2d: best=%.10f  pop=%d\n", gen, best, npop)
            end
        end,
    )

    @printf("(%d,%d) p=%2d: c̃=%.10f  evals=%d  wall=%.1fs\n",
        k, D, p, result.value, result.evaluations, result.wall_time_seconds)
    flush(stdout)

    if QaoaXorsat.is_valid_qaoa_value(result.value) && result.value > 0.501
        state.warm = [result.angles]
    else
        @printf("  *** chain broken at p=%d — continuing with random starts\n", p)
        state.warm = QAOAAngles[]
    end
end
