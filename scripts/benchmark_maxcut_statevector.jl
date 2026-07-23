#!/usr/bin/env julia
#
# Time the exact finite-N MaxCut statevector forward and adjoint evaluations.
#
# Usage:
#   julia --project=. scripts/benchmark_maxcut_statevector.jl [N] [p] [repeats]

using QaoaXorsat
using Printf
using Random

N = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 16
p = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 4
repeats = length(ARGS) ≥ 3 ? parse(Int, ARGS[3]) : 5

N ≥ 2 || throw(ArgumentError("N must be at least 2"))
p ≥ 1 || throw(ArgumentError("p must be at least 1"))
repeats ≥ 1 || throw(ArgumentError("repeats must be at least 1"))

rng = MersenneTwister(20260723)
edges = [(vertex, vertex == N ? 1 : vertex + 1) for vertex in 1:N]
ev = prepare_maxcut_eval(N, edges)
angles = QAOAAngles(randn(rng, p), randn(rng, p))
gradient = zeros(2p)

maxcut_expectation(ev, angles)
maxcut_expectation_and_gradient!(gradient, ev, angles)

forward_seconds = @elapsed for _ in 1:repeats
    maxcut_expectation(ev, angles)
end
gradient_seconds = @elapsed for _ in 1:repeats
    maxcut_expectation_and_gradient!(gradient, ev, angles)
end

@printf(
    "N=%d p=%d states=%d repeats=%d forward=%.6f s gradient=%.6f s\n",
    N,
    p,
    1 << N,
    repeats,
    forward_seconds / repeats,
    gradient_seconds / repeats,
)
