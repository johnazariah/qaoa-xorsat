#!/usr/bin/env julia
#
# Compute Basso large-D asymptotic reference for all 15 (k,D) pairs.
#
# Runs the exact evaluator at D_ref=100 (approximating D→∞) for each k value.
# The O(1/D) correction at D=100 is ~1%, making these tight reference values.
#
# Output: CSV with columns k, D_ref, p, ctilde_asymptotic
# Plus per-k TOML configs for the full finite-D runs.
#
# Usage:
#   julia --project=. -t 12 scripts/asymptotic_reference.jl [P_MAX]

using Printf
using Random
using QaoaXorsat

p_max = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 13
D_ref = 100  # large enough that O(1/D) ≈ 1%
seed = 1234
restarts = 2
maxiters = 320

# Stephen's table: all k values that appear in the 15 pairs
k_values = [3, 4, 5, 6, 7]

# The 15 (k,D) pairs from Jordan et al.
pairs = [
    (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
    (4, 5), (4, 6), (4, 7), (4, 8),
    (5, 6), (5, 7), (5, 8),
    (6, 7), (6, 8),
    (7, 8),
]

# JIT warmup
@info "JIT warmup..."
optimize_depth_sequence(3, 100, [1]; clause_sign=1, restarts=0, maxiters=5,
    autodiff=:adjoint, rng=MersenneTwister(0))
@info "JIT warmup complete"

# Output CSV
output_path = joinpath(@__DIR__, "..", ".project", "results", "asymptotic-reference.csv")
mkpath(dirname(output_path))
open(output_path, "w") do io
    println(io, "k,D_ref,p,ctilde_asymptotic,wall_time_seconds,converged,g_abstol")

    for k in k_values
        clause_sign = k == 2 ? -1 : 1
        rng = MersenneTwister(seed)

        @printf(stderr, "\n=== k=%d, D_ref=%d, p=1..%d ===\n", k, D_ref, p_max)
        flush(stderr)

        results = optimize_depth_sequence(
            k, D_ref, 1:p_max;
            clause_sign, restarts, maxiters,
            autodiff=:adjoint, rng,
            on_result=result -> begin
                p = depth(result.angles)
                @printf(stderr, "  k=%d p=%d: c̃=%.12f (%.3fs, %s)\n",
                    k, p, result.value, result.wall_time_seconds,
                    result.converged ? "converged" : "NOT CONVERGED")
                flush(stderr)
                @printf(io, "%d,%d,%d,%.12f,%.6f,%s,%.1e\n",
                    k, D_ref, p, result.value, result.wall_time_seconds,
                    string(result.converged), result.g_abstol)
                flush(io)
            end,
        )
    end
end

@printf(stderr, "\nSaved: %s\n", output_path)

# Print summary comparison table for (k=3, D=4) reference
println(stderr, "\n=== Quick comparison: k=3 asymptotic vs finite-D=4 ===")
println(stderr, "  p | D=100 (≈∞) | D=4 (exact)  | Δ (O(1/D))")
println(stderr, "  --|------------|--------------|----------")

# Known finite-D=4 values from our runs
finite_d4 = Dict(
    1 => 0.676056660061,
    2 => 0.739122784672,
    3 => 0.777144062579,
    4 => 0.802204018074,
    5 => 0.820480966923,
    6 => 0.834391132449,
    7 => 0.845313272946,
    8 => 0.854102062227,
    9 => 0.861321028427,
    10 => 0.867359599246,
    11 => 0.872489761801,
)

# Re-read the CSV we just wrote
for line in eachline(output_path)
    startswith(line, "3,") || continue
    fields = split(line, ',')
    p = parse(Int, fields[3])
    c_inf = parse(Float64, fields[4])
    if haskey(finite_d4, p)
        c_fin = finite_d4[p]
        delta = c_inf - c_fin
        @printf(stderr, "  %d | %.10f | %.10f | %+.6f\n", p, c_inf, c_fin, delta)
    else
        @printf(stderr, "  %d | %.10f |      —       |\n", p, c_inf)
    end
end
