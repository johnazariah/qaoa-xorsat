#!/usr/bin/env julia
#
# Build a comprehensive sheet of ALL best data per (k,D,p) across all sources.
# Outputs a single CSV with every depth for every pair, with angles and provenance.
#
# Usage: julia --project=. scripts/build-composite-sheet.jl
#
using Printf
using Dates

struct DataPoint
    k::Int
    D::Int
    p::Int
    value::Float64
    gamma::Vector{Float64}
    beta::Vector{Float64}
    source::String
    converged::Bool
end

all_data = Dict{Tuple{Int,Int,Int}, DataPoint}()

function try_ingest!(data, k, D, p, v, gamma, beta, source; converged=true)
    isfinite(v) && 0 < v < 1.001 || return
    length(gamma) == p && length(beta) == p || return
    key = (k, D, p)
    if !haskey(data, key) || v > data[key].value
        data[key] = DataPoint(k, D, p, v, Float64.(gamma), Float64.(beta), source, converged)
    end
end

# ── Local optimization runs ──────────────────────────────────────────────
runs_dir = joinpath(@__DIR__, "..", ".project", "results", "optimization", "runs")
if isdir(runs_dir)
    for run_name in readdir(runs_dir)
        csv = joinpath(runs_dir, run_name, "results.csv")
        isfile(csv) || continue
        for line in eachline(csv)
            startswith(line, "run_id") && continue
            fields = split(line, ',')
            length(fields) >= 26 || continue
            k = tryparse(Int, fields[8]); k === nothing && continue
            D = tryparse(Int, fields[9]); D === nothing && continue
            p = tryparse(Int, fields[10]); p === nothing && continue
            v = tryparse(Float64, fields[15]); v === nothing && continue
            gamma = tryparse.(Float64, split(fields[25], ';'))
            beta = tryparse.(Float64, split(fields[26], ';'))
            any(isnothing, gamma) && continue
            any(isnothing, beta) && continue
            conv = get(fields, 21, "true") == "true"
            try_ingest!(all_data, k, D, p, v, Float64.(gamma), Float64.(beta),
                "local/$run_name"; converged=conv)
        end
    end
end

# ── Local swarm results ──────────────────────────────────────────────────
results_dir = joinpath(@__DIR__, "..", "results")
if isdir(results_dir)
    for f in readdir(results_dir; join=true)
        endswith(f, ".csv") && occursin("swarm", f) || continue
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
            try_ingest!(all_data, k, D, p, v, Float64.(gamma), Float64.(beta),
                "swarm/$(basename(f))")
        end
    end
end

# ── Git branches (fleet, p710, stephen) ──────────────────────────────────
function ingest_branch_results!(data, branch)
    # Index CSV
    try
        lines = readlines(pipeline(`git show $(branch):.project/results/optimization/index.csv`, stderr=devnull))
        for line in lines
            startswith(line, "run_id") && continue
            fields = split(line, ',')
            length(fields) >= 26 || continue
            k = tryparse(Int, fields[8]); k === nothing && continue
            D = tryparse(Int, fields[9]); D === nothing && continue
            p = tryparse(Int, fields[10]); p === nothing && continue
            v = tryparse(Float64, fields[15]); v === nothing && continue
            gamma = tryparse.(Float64, split(fields[25], ';'))
            beta = tryparse.(Float64, split(fields[26], ';'))
            any(isnothing, gamma) && continue
            any(isnothing, beta) && continue
            conv = get(fields, 21, "true") == "true"
            try_ingest!(data, k, D, p, v, Float64.(gamma), Float64.(beta),
                branch; converged=conv)
        end
    catch; end

    # Individual run directories
    try
        dirs = strip.(readlines(pipeline(`git ls-tree -d --name-only $(branch):.project/results/optimization/runs/`, stderr=devnull)))
        for dir in dirs
            try
                lines = readlines(pipeline(`git show $(branch):.project/results/optimization/runs/$(dir)/results.csv`, stderr=devnull))
                for line in lines
                    startswith(line, "run_id") && continue
                    fields = split(line, ',')
                    length(fields) >= 26 || continue
                    k = tryparse(Int, fields[8]); k === nothing && continue
                    D = tryparse(Int, fields[9]); D === nothing && continue
                    p = tryparse(Int, fields[10]); p === nothing && continue
                    v = tryparse(Float64, fields[15]); v === nothing && continue
                    gamma = tryparse.(Float64, split(fields[25], ';'))
                    beta = tryparse.(Float64, split(fields[26], ';'))
                    any(isnothing, gamma) && continue
                    any(isnothing, beta) && continue
                    conv = get(fields, 21, "true") == "true"
                    try_ingest!(data, k, D, p, v, Float64.(gamma), Float64.(beta),
                        "$branch/$dir"; converged=conv)
                end
            catch; end
        end
    catch; end

    # Swarm CSVs on branches
    for kd in ["k5d7", "k5d8", "k6d7", "k6d8", "k7d8"]
        try
            lines = readlines(pipeline(`git show $(branch):results/swarm-$(kd).csv`, stderr=devnull))
            for line in lines
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
                try_ingest!(data, k, D, p, v, Float64.(gamma), Float64.(beta),
                    "$branch/swarm-$kd")
            end
        catch; end
    end
end

try
    branches = filter(b -> occursin("fleet", b) || occursin("p710", b) || occursin("stephen", b),
        strip.(readlines(pipeline(`git branch -r`, stderr=devnull))))
    for branch in branches
        @printf("Scanning %s...\n", strip(branch))
        ingest_branch_results!(all_data, strip(branch))
    end
catch e
    @warn "git branch scan failed" exception=e
end

# ── Write composite sheet ────────────────────────────────────────────────
outfile = joinpath(@__DIR__, "..", "results", "composite-all-pairs.csv")
open(outfile, "w") do io
    println(io, "# Composite best data per (k,D,p) — generated $(today())")
    println(io, "# Best c̃ at each depth from all sources (Mac, Azure fleet, P710 swarm, Stephen cluster)")
    println(io, "k,D,p,ctilde,converged,source,gamma,beta")

    pairs = sort(unique([(d.k, d.D) for d in values(all_data)]))
    for (k, D) in pairs
        depths = sort([(dp.p, dp) for dp in values(all_data) if dp.k == k && dp.D == D])
        for (p, dp) in depths
            @printf(io, "%d,%d,%d,%.12f,%s,%s,%s,%s\n",
                dp.k, dp.D, dp.p, dp.value, dp.converged, dp.source,
                join(string.(dp.gamma), ';'), join(string.(dp.beta), ';'))
        end
    end
end

# ── Print summary ────────────────────────────────────────────────────────
println("\n" * "="^100)
println("COMPOSITE SHEET — ALL PAIRS, ALL DEPTHS")
println("="^100)

pairs = sort(unique([(d.k, d.D) for d in values(all_data)]))
total_points = 0
for (k, D) in pairs
    depths = sort([(dp.p, dp) for dp in values(all_data) if dp.k == k && dp.D == D])
    @printf("\n(%d,%d): %d depths\n", k, D, length(depths))
    for (p, dp) in depths
        flag = dp.value < 0.55 ? " *** LOW" : ""
        @printf("  p=%2d  c̃=%.12f  %s  %s%s\n", p, dp.value,
            dp.converged ? "conv" : "NCON", dp.source, flag)
        global total_points += 1
    end
end

println("\n" * "-"^60)
@printf("Total: %d data points across %d (k,D) pairs\n", total_points, length(pairs))
@printf("Output: %s\n", outfile)
