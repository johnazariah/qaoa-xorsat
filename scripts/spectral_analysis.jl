#!/usr/bin/env julia
#
# Spectral analysis of the Basso branch-tensor iteration.
#
# Measures the WHT-domain spectrum of B^(t) at each iteration step to determine
# whether the branch tensor has low effective rank — i.e., whether a truncated
# iteration could break the 4^p scaling barrier.
#
# Usage:
#   julia --project=. -t auto scripts/spectral_analysis.jl [p_max]
#
# Outputs:
#   results/spectral/k3-D4-pXX-spectrum.csv    — full sorted magnitudes
#   results/spectral/k3-D4-pXX-ranks.csv       — effective ranks vs tolerance
#   results/spectral/summary.txt                — human-readable report
#
# Default p_max is 8 (runs in ~2 minutes on M4). Set higher for deeper analysis.

using QaoaXorsat
using Printf
using Dates

# ──────────────────────────────────────────────────────────────────────────────
# Known optimal angles from results/qaoa-best-values.csv
# These are the angles at which the branch tensor carries the actual
# interference pattern we care about.
# ──────────────────────────────────────────────────────────────────────────────

"""
    optimal_angles_k3_d4(p) -> QAOAAngles

Return optimal (or near-optimal) angles for (k=3, D=4) at depth p.

For p ≤ 6 we use quick on-the-fly optimisation. For higher p, extend from
the previous depth's optimum.
"""
function find_optimal_angles(params::TreeParams, p_max::Int)
    println(stderr, "Finding optimal angles for k=$(params.k), D=$(params.D), p=1…$p_max")

    angles_by_depth = Dict{Int,QAOAAngles{Float64}}()

    for p in 1:p_max
        tp = TreeParams(params.k, params.D, p)
        t_start = time()

        if p == 1
            result = optimize_angles(tp; restarts=10, autodiff=:adjoint)
        else
            prev = angles_by_depth[p-1]
            warm = extend_angles(prev, p)
            result = optimize_angles(tp; initial_guesses=[warm], restarts=5, autodiff=:adjoint)
        end

        elapsed = time() - t_start
        angles_by_depth[p] = result.angles
        @printf(stderr, "  p=%2d: c̃=%.10f  (%.1fs, %s)\n",
            p, result.value, elapsed, result.converged ? "converged" : "NOT converged")
    end

    angles_by_depth
end

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

function main()
    p_max = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 8
    k, D = 3, 4

    params = TreeParams(k, D, 1)  # p will vary per depth

    # Output directory
    out_dir = joinpath("results", "spectral")
    mkpath(out_dir)

    # Find optimal angles for each depth
    angles_map = find_optimal_angles(params, p_max)

    # Run spectral analysis at each depth
    summary_lines = String[]
    push!(summary_lines, "=" ^ 72)
    push!(summary_lines, "SPECTRAL ANALYSIS OF BASSO BRANCH-TENSOR ITERATION")
    push!(summary_lines, "k=$k, D=$D, p=1…$p_max")
    push!(summary_lines, "Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    push!(summary_lines, "=" ^ 72)
    push!(summary_lines, "")

    for p in 1:p_max
        tp = TreeParams(k, D, p)
        angles = angles_map[p]

        println(stderr, "\nSpectral analysis at p=$p...")
        _, profile = basso_branch_tensor_instrumented(tp, angles)

        # Write full spectrum CSV
        spectrum_path = joinpath(out_dir, "k$(k)-D$(D)-p$(lpad(p, 2, '0'))-spectrum.csv")
        open(spectrum_path, "w") do io
            write_spectral_csv(io, profile)
        end
        println(stderr, "  wrote $spectrum_path")

        # Write effective ranks CSV
        ranks_path = joinpath(out_dir, "k$(k)-D$(D)-p$(lpad(p, 2, '0'))-ranks.csv")
        open(ranks_path, "w") do io
            write_effective_ranks_csv(io, profile)
        end
        println(stderr, "  wrote $ranks_path")

        # Add to summary
        report = format_spectral_report(profile)
        push!(summary_lines, report)
        push!(summary_lines, "")

        # Print key finding for this depth
        final_snapshot = profile.snapshots[end]
        N = QaoaXorsat.basso_configuration_count(p)
        rank_1e10 = get(final_snapshot.effective_ranks, 1e-10, N)
        pct = round(100.0 * rank_1e10 / N; digits=2)
        rate, r² = spectral_decay_rate(final_snapshot)

        @printf(stderr, "  p=%d: effective rank at δ=1e-10: %d / %d (%.2f%%)\n", p, rank_1e10, N, pct)
        @printf(stderr, "  p=%d: decay rate=%.3f, R²=%.4f\n", p, rate, r²)
    end

    # Cross-depth summary: how does effective rank scale with p?
    push!(summary_lines, "=" ^ 72)
    push!(summary_lines, "SCALING SUMMARY")
    push!(summary_lines, "=" ^ 72)
    push!(summary_lines, "")
    push!(summary_lines, rpad("p", 5) *
        rpad("N=2^(2p+1)", 14) *
        rpad("rank@1e-4", 12) *
        rpad("rank@1e-8", 12) *
        rpad("rank@1e-10", 12) *
        rpad("decay_rate", 12) *
        "R²")
    push!(summary_lines, "-" ^ 72)

    for p in 1:p_max
        tp = TreeParams(k, D, p)
        angles = angles_map[p]
        _, profile = basso_branch_tensor_instrumented(tp, angles)
        final = profile.snapshots[end]
        N = QaoaXorsat.basso_configuration_count(p)
        rate, r² = spectral_decay_rate(final)

        r4 = get(final.effective_ranks, 1e-4, N)
        r8 = get(final.effective_ranks, 1e-8, N)
        r10 = get(final.effective_ranks, 1e-10, N)

        push!(summary_lines,
            rpad("$p", 5) *
            rpad("$N", 14) *
            rpad("$r4", 12) *
            rpad("$r8", 12) *
            rpad("$r10", 12) *
            rpad(@sprintf("%.3f", rate), 12) *
            @sprintf("%.4f", r²))
    end

    push!(summary_lines, "")
    push!(summary_lines, "If rank@δ grows polynomially while N grows as 4^p,")
    push!(summary_lines, "a truncated iteration can break the exponential barrier.")

    # Write summary
    summary_path = joinpath(out_dir, "summary.txt")
    open(summary_path, "w") do io
        println(io, join(summary_lines, "\n"))
    end
    println(stderr, "\nSummary written to $summary_path")

    # Print to stdout as well
    println(join(summary_lines, "\n"))
end

main()
