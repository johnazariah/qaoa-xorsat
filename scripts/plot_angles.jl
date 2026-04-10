#!/usr/bin/env julia
#
# Plot QAOA angles vs depth p for all 15 (k,D) pairs.
# Produces two plots: one for γ angles, one for β angles.
# Each pair is a separate colored line.
#
# Usage: julia --project=. scripts/plot_angles.jl
#
# Requires: Plots.jl (install with `using Pkg; Pkg.add("Plots")`)

using Printf
using DelimitedFiles

# Try to load Plots; if not available, generate CSV for external plotting
USE_PLOTS = try
    @eval using Plots
    true
catch
    @warn "Plots.jl not available — generating CSV data files instead"
    false
end

# ── Parse composite sheet ─────────────────────────────────────────────
csv_path = joinpath(@__DIR__, "..", "results", "composite-all-pairs.csv")
if !isfile(csv_path)
    # Fall back to best-values + swarm CSVs
    csv_path = joinpath(@__DIR__, "..", "results", "qaoa-best-values.csv")
end

# Data structure: (k,D) => [(p, γ_vector, β_vector), ...]
data = Dict{Tuple{Int,Int}, Vector{NamedTuple{(:p, :v, :gamma, :beta), Tuple{Int, Float64, Vector{Float64}, Vector{Float64}}}}}()

# Parse composite CSV
for line in eachline(csv_path)
    startswith(line, '#') && continue
    startswith(line, "k,") && continue
    fields = split(line, ',')

    # Composite format: k,D,p,ctilde,converged,source,gamma,beta
    # Best-values format: k,D,p_max,ctilde,wall_time,converged,g_abstol,machine
    if length(fields) >= 8 && occursin(';', fields[7])
        # Composite format
        k = tryparse(Int, fields[1]); k === nothing && continue
        D = tryparse(Int, fields[2]); D === nothing && continue
        p = tryparse(Int, fields[3]); p === nothing && continue
        v = tryparse(Float64, fields[4]); v === nothing && continue
        gamma = tryparse.(Float64, split(fields[7], ';'))
        beta = tryparse.(Float64, split(fields[8], ';'))
        any(isnothing, gamma) && continue
        any(isnothing, beta) && continue
        length(gamma) == p && length(beta) == p || continue
        v > 0.5 && v < 1.0 || continue

        key = (k, D)
        haskey(data, key) || (data[key] = [])
        push!(data[key], (p=p, v=v, gamma=Float64.(gamma), beta=Float64.(beta)))
    end
end

# Also parse swarm CSVs
for f in readdir(joinpath(@__DIR__, "..", "results"); join=true)
    (endswith(f, ".csv") && occursin("swarm", f)) || continue
    for line in eachline(f)
        startswith(line, '#') && continue
        startswith(line, "k,") && continue
        fields = split(line, ',')
        length(fields) >= 8 || continue
        k = tryparse(Int, fields[1]); k === nothing && continue
        D = tryparse(Int, fields[2]); D === nothing && continue
        p = tryparse(Int, fields[3]); p === nothing && continue
        v = tryparse(Float64, fields[4]); v === nothing && continue
        gamma = tryparse.(Float64, split(fields[7], ';'))
        beta = tryparse.(Float64, split(fields[8], ';'))
        any(isnothing, gamma) && continue
        any(isnothing, beta) && continue
        length(gamma) == p && length(beta) == p || continue
        v > 0.5 && v < 1.0 || continue

        key = (k, D)
        haskey(data, key) || (data[key] = [])
        # Only add if we don't have a better value at this depth
        existing = filter(e -> e.p == p, data[key])
        if isempty(existing) || v > existing[1].v
            filter!(e -> e.p != p, data[key])
            push!(data[key], (p=p, v=v, gamma=Float64.(gamma), beta=Float64.(beta)))
        end
    end
end

# Sort each pair's data by p
for key in keys(data)
    sort!(data[key], by=e -> e.p)
end

pairs = sort(collect(keys(data)))
@printf("Loaded angle data for %d (k,D) pairs\n", length(pairs))
for (k, D) in pairs
    depths = [e.p for e in data[(k, D)]]
    @printf("  (%d,%d): p=%s\n", k, D, join(string.(depths), ","))
end

# ── Generate CSV data files ───────────────────────────────────────────
# For each angle round r, write: k, D, p, angle_value
outdir = joinpath(@__DIR__, "..", "results", "angle-plots")
mkpath(outdir)

# Gamma angles
open(joinpath(outdir, "gamma_angles.csv"), "w") do io
    println(io, "k,D,p,round,gamma")
    for (k, D) in pairs
        for entry in data[(k, D)]
            for (r, g) in enumerate(entry.gamma)
                @printf(io, "%d,%d,%d,%d,%.12f\n", k, D, entry.p, r, g)
            end
        end
    end
end

# Beta angles
open(joinpath(outdir, "beta_angles.csv"), "w") do io
    println(io, "k,D,p,round,beta")
    for (k, D) in pairs
        for entry in data[(k, D)]
            for (r, b) in enumerate(entry.beta)
                @printf(io, "%d,%d,%d,%d,%.12f\n", k, D, entry.p, r, b)
            end
        end
    end
end

# One file per pair with all angles at the best depth
open(joinpath(outdir, "best_angles_summary.csv"), "w") do io
    println(io, "k,D,best_p,ctilde,gamma,beta")
    for (k, D) in pairs
        best = data[(k, D)][end]  # highest p
        gamma_str = join(string.(best.gamma), ';')
        beta_str = join(string.(best.beta), ';')
        @printf(io, "%d,%d,%d,%.12f,%s,%s\n", k, D, best.p, best.v, gamma_str, beta_str)
    end
end

println("\nCSV files written to: $outdir")

# ── Plot if Plots.jl available ────────────────────────────────────────
if USE_PLOTS
    # Color palette for 15 pairs
    colors = distinguishable_colors(15, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

    # ── Gamma plot: for each pair, plot γ_r vs r at best p ──────────
    p_gamma = plot(
        title="QAOA γ angles at best depth",
        xlabel="Round r",
        ylabel="γ_r",
        legend=:outertopright,
        size=(1000, 600),
        margin=5Plots.mm,
    )
    for (i, (k, D)) in enumerate(pairs)
        best = data[(k, D)][end]
        plot!(p_gamma, 1:best.p, best.gamma,
            label="($k,$D) p=$(best.p)",
            color=colors[i],
            marker=:circle, markersize=3,
            linewidth=1.5)
    end
    savefig(p_gamma, joinpath(outdir, "gamma_angles.png"))
    println("Saved: gamma_angles.png")

    # ── Beta plot ────────────────────────────────────────────────────
    p_beta = plot(
        title="QAOA β angles at best depth",
        xlabel="Round r",
        ylabel="β_r",
        legend=:outertopright,
        size=(1000, 600),
        margin=5Plots.mm,
    )
    for (i, (k, D)) in enumerate(pairs)
        best = data[(k, D)][end]
        plot!(p_beta, 1:best.p, best.beta,
            label="($k,$D) p=$(best.p)",
            color=colors[i],
            marker=:circle, markersize=3,
            linewidth=1.5)
    end
    savefig(p_beta, joinpath(outdir, "beta_angles.png"))
    println("Saved: beta_angles.png")

    println("\nPlots saved to: $outdir")
else
    println("\nTo generate plots, install Plots.jl:")
    println("  julia --project=. -e 'using Pkg; Pkg.add(\"Plots\")'")
    println("  julia --project=. scripts/plot_angles.jl")
    println("\nAlternatively, use the CSV files with Python/matplotlib or R.")
end
