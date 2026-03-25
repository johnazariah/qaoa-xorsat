#!/usr/bin/env julia
#
# Benchmark: SIMD WHT + In-place Adjoint vs Baseline
#
# Compares the optimized branch (this worktree) against the baseline by
# measuring per-evaluation time of basso_expectation_and_gradient at p=6..10.
#
# Run from the worktree:
#   julia --project=.. benchmark.jl
#
# Run from main for baseline:
#   cd /Users/johnaz/PhD/qaoa-xorsat && julia --project=. .worktree/wht-optimization/spike/benchmark.jl

using QaoaXorsat
using Random
using Statistics

println("=" ^ 72)
println("WHT Optimization Benchmark")
println("=" ^ 72)
println()
println("Threads: ", Threads.nthreads())
println()

k, D = 3, 4

# Use fixed angles from the production results for reproducibility
rng = MersenneTwister(42)

println("─" ^ 72)
println("Per-evaluation timing: basso_expectation_and_gradient")
println("  (includes forward + backward pass, measures wall-clock per call)")
println("─" ^ 72)
println()

for p in [6, 7, 8, 9, 10]
    params = TreeParams(k, D, p)
    N = 2^(2p + 1)
    angles = random_angles(p; rng=MersenneTwister(42))

    # Warmup
    basso_expectation_and_gradient(params, angles)

    # Measure: run enough times to get stable median
    n_trials = p ≤ 7 ? 20 : (p ≤ 8 ? 10 : (p ≤ 9 ? 5 : 3))
    times = Float64[]
    for _ in 1:n_trials
        t = @elapsed basso_expectation_and_gradient(params, angles)
        push!(times, t)
    end

    med = median(times)
    mn = minimum(times)
    println("  p=$p (N=$(N)):  median=$(round(med, digits=4))s  min=$(round(mn, digits=4))s  trials=$n_trials")
end

println()

# Also benchmark just the WHT itself
println("─" ^ 72)
println("Standalone WHT timing")
println("─" ^ 72)
println()

for log2N in [17, 19, 21, 23]
    N = 1 << log2N
    v = randn(ComplexF64, N)

    # Warmup
    QaoaXorsat.wht!(copy(v))

    n_trials = log2N ≤ 19 ? 50 : (log2N ≤ 21 ? 20 : 10)
    times = Float64[]
    for _ in 1:n_trials
        vc = copy(v)
        t = @elapsed QaoaXorsat.wht!(vc)
        push!(times, t)
    end

    med = median(times)
    mn = minimum(times)
    mb = N * sizeof(ComplexF64) / 1e6
    println("  N=2^$log2N ($(round(mb, digits=1)) MB):  median=$(round(med*1000, digits=2))ms  min=$(round(mn*1000, digits=2))ms")
end

println()
println("Done.")
