#!/usr/bin/env julia
#
# Compute exact QAOA Max-k-XORSAT satisfaction fractions
# for D-regular k-uniform hypergraphs.
# Sweeps p = 1, 2, ..., P_MAX with warm-start angle seeding.
#
# Usage:
#   julia --project=. -t 8  scripts/run_xorsat.jl K D P_MAX
#   julia --project=. -t 8  scripts/run_xorsat.jl 3 4 12   # k=3, D=4, p=1..12
#   julia --project=. -t 16 scripts/run_xorsat.jl 4 5 10   # k=4, D=5, p=1..10
#
# Output: results/xorsat-k<K>-d<D>-sweep.csv

using QaoaXorsat
using DoubleFloats
using Printf
using Random
using Dates

if length(ARGS) < 3
    println(stderr, "Usage: julia --project=. -t THREADS scripts/run_xorsat.jl K D P_MAX [SEED]")
    exit(1)
end

k     = parse(Int, ARGS[1])
D     = parse(Int, ARGS[2])
p_max = parse(Int, ARGS[3])
seed  = length(ARGS) ≥ 4 ? parse(Int, ARGS[4]) : 42

results_file = joinpath(@__DIR__, "..", "results", "xorsat-k$(k)-d$(D)-sweep.csv")
mkpath(dirname(results_file))

# Write header if new file
if !isfile(results_file)
    open(results_file, "w") do io
        println(io, "# Max-k-XORSAT k=$k D=$D sweep — $(now())")
        println(io, "k,D,p,ctilde,wall_seconds,gamma,beta")
    end
end

# Use Double64 for large branching factors (k ≥ 6) to handle precision
eval_eltype = k ≥ 6 ? Double64 : Float64

@printf(stderr, "=== Max-%d-XORSAT (k=%d, D=%d) sweep p=1..%d, %d threads, %s ===\n",
        k, k, D, p_max, Threads.nthreads(), eval_eltype)

rng = MersenneTwister(seed)

results = optimize_depth_sequence(
    k, D, 1:p_max;
    clause_sign=1,
    restarts=8,
    maxiters=200,
    autodiff=:adjoint,
    rng,
    eval_eltype,
    on_result=result -> begin
        p = depth(result.angles)
        @printf(stderr, "  p=%d: c̃=%.10f (%.1fs, %s)\n",
                p, result.best_value, result.wall_time, result.converged ? "converged" : "not converged")

        open(results_file, "a") do io
            gamma_str = join([@sprintf("%.15f", g) for g in result.angles.γ], ";")
            beta_str  = join([@sprintf("%.15f", b) for b in result.angles.β], ";")
            @printf(io, "%d,%d,%d,%.15f,%.3f,%s,%s\n",
                    k, D, p, result.best_value, result.wall_time, gamma_str, beta_str)
        end
    end,
)

println(stderr, "\nDone. Results written to $results_file")
