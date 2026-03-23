#!/usr/bin/env julia

using Printf
using Random
using QaoaXorsat

function parse_int(name::String, value::String)
    try
        parse(Int, value)
    catch
        error("invalid $(name): $(value)")
    end
end

function usage()
    println(stderr, "Usage: julia --project=. scripts/optimize_qaoa.jl K D P_MIN P_MAX [RESTARTS] [MAXITERS] [SEED]")
end

length(ARGS) ≥ 4 || (usage(); exit(1))

k = parse_int("K", ARGS[1])
D = parse_int("D", ARGS[2])
p_min = parse_int("P_MIN", ARGS[3])
p_max = parse_int("P_MAX", ARGS[4])
restarts = length(ARGS) ≥ 5 ? parse_int("RESTARTS", ARGS[5]) : 8
maxiters = length(ARGS) ≥ 6 ? parse_int("MAXITERS", ARGS[6]) : 200
seed = length(ARGS) ≥ 7 ? parse_int("SEED", ARGS[7]) : 1234

p_max ≥ p_min || error("P_MAX must be ≥ P_MIN")

rng = MersenneTwister(seed)
clause_sign = k == 2 ? -1 : 1
results = optimize_depth_sequence(k, D, collect(p_min:p_max); clause_sign, restarts, maxiters, rng)

println("p,value,evaluations,starts,iterations,converged,gamma,beta")
for result in results
    @printf(
        "%d,%.12f,%d,%d,%d,%s,%s,%s\n",
        depth(result.angles),
        result.value,
        result.evaluations,
        result.starts,
        result.iterations,
        string(result.converged),
        join(string.(result.angles.γ), ';'),
        join(string.(result.angles.β), ';'),
    )
end
