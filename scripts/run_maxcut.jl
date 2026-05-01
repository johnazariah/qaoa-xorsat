#!/usr/bin/env julia
#
# Compute exact QAOA MaxCut satisfaction fractions for D-regular graphs.
# Sweeps p = 1, 2, ..., P_MAX with warm-start angle seeding.
#
# Usage:
#   julia --project=. -t 8  scripts/run_maxcut.jl D P_MAX
#   julia --project=. -t 16 scripts/run_maxcut.jl 3 12    # D=3, p=1..12
#   julia --project=. -t 16 scripts/run_maxcut.jl 4 10    # D=4, p=1..10
#
# Output: results/maxcut-k2-d<D>-sweep.csv

using QaoaXorsat
using Printf
using Random
using Dates

if length(ARGS) < 2
    println(stderr, "Usage: julia --project=. -t THREADS scripts/run_maxcut.jl D P_MAX [SEED]")
    exit(1)
end

D     = parse(Int, ARGS[1])
p_max = parse(Int, ARGS[2])
seed  = length(ARGS) ≥ 3 ? parse(Int, ARGS[3]) : 42
k     = 2

results_file = joinpath(@__DIR__, "..", "results", "maxcut-k2-d$(D)-sweep.csv")
mkpath(dirname(results_file))

# Write header if new file
if !isfile(results_file)
    open(results_file, "w") do io
        println(io, "# MaxCut k=2 D=$D sweep — $(now())")
        println(io, "k,D,p,ctilde,wall_seconds,gamma,beta")
    end
end

@printf(stderr, "=== MaxCut (k=%d, D=%d) sweep p=1..%d, %d threads ===\n", k, D, p_max, Threads.nthreads())

rng = MersenneTwister(seed)

results = optimize_depth_sequence(
    k, D, 1:p_max;
    clause_sign=-1,
    restarts=8,
    maxiters=200,
    autodiff=:adjoint,
    rng,
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
