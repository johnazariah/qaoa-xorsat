#!/usr/bin/env julia
#
# Run all 15 (k,D) pairs from Stephen's table through p=p_max.
# Each pair runs a warm-started depth sequence with the standard settings.
# Results are preserved in individual run directories.
#
# Usage:
#   julia --project=. -t 12 scripts/run_full_table.jl [P_MAX]
#
# Default P_MAX=10. For the validation sweep, use P_MAX=8 (~30 min total).

using Dates
using Printf
using Random
using QaoaXorsat

p_max = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 10
seed = 1234
restarts = 2
maxiters = 320

# Stephen's table: all 15 (k,D) pairs
pairs = [
    (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
    (4, 5), (4, 6), (4, 7), (4, 8),
    (5, 6), (5, 7), (5, 8),
    (6, 7), (6, 8),
    (7, 8),
]

# JIT warmup
@info "JIT warmup..."
optimize_depth_sequence(3, 4, [1]; clause_sign=1, restarts=0, maxiters=5,
    autodiff=:adjoint, rng=MersenneTwister(0))
@info "JIT warmup complete"

# Summary CSV
summary_path = joinpath(@__DIR__, "..", ".project", "results", "full-table-summary.csv")
mkpath(dirname(summary_path))
open(summary_path, "w") do summary_io
    println(summary_io, "k,D,p,ctilde,delta_ctilde,wall_time_seconds,converged,g_abstol,gain_ratio")

    for (k, D) in pairs
        clause_sign = k == 2 ? -1 : 1
        rng = MersenneTwister(seed)

        @printf(stderr, "\n========================================\n")
        @printf(stderr, "=== (k=%d, D=%d) p=1..%d ===\n", k, D, p_max)
        @printf(stderr, "========================================\n")
        flush(stderr)

        prev_delta = NaN

        results = optimize_depth_sequence(
            k, D, 1:p_max;
            clause_sign, restarts, maxiters,
            autodiff=:adjoint, rng,
            on_result=result -> begin
                p = depth(result.angles)
                @printf(stderr, "  (k=%d,D=%d) p=%d: c̃=%.12f (%.3fs, %s, g_abstol=%.0e)\n",
                    k, D, p, result.value, result.wall_time_seconds,
                    result.converged ? "✓" : "✗", result.g_abstol)
                flush(stderr)
            end,
            on_evaluation=(si, ev, el, val, gn) -> begin
                @printf(stderr, "    start %d: %d evals, %.1fs, c̃=%.10f, g_norm=%.2e\n",
                    si, ev, el, val, gn)
                flush(stderr)
            end,
        )

        # Write summary rows
        prev_value = NaN
        for (i, result) in enumerate(results)
            p = depth(result.angles)
            delta = i > 1 ? result.value - results[i-1].value : NaN
            ratio = (i > 2 && !isnan(prev_delta) && prev_delta > 0) ?
                delta / prev_delta : NaN
            ratio_str = isnan(ratio) ? "" : @sprintf("%.4f", ratio)
            delta_str = isnan(delta) ? "" : @sprintf("%.6f", delta)
            @printf(summary_io, "%d,%d,%d,%.12f,%s,%.6f,%s,%.1e,%s\n",
                k, D, p, result.value, delta_str,
                result.wall_time_seconds, string(result.converged),
                result.g_abstol, ratio_str)
            prev_delta = delta
        end
        flush(summary_io)
    end
end

@printf(stderr, "\nSaved: %s\n", summary_path)
