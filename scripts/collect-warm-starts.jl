#!/usr/bin/env julia
# Collect best angles from all sources and write warm-start CSV
# Run from repo root: julia --project=. scripts/collect-warm-starts.jl

using Printf
using Dates

struct BestResult
    k::Int
    D::Int
    p::Int
    value::Float64
    gamma::Vector{Float64}
    beta::Vector{Float64}
    source::String
end

# Prefer highest depth with good value, then highest value at same depth.
# "Good" means > 0.65 — filters out bad basins (0.5-0.6) that would
# poison the warm-start chain.
is_better(new_p, new_v, old::BestResult) = begin
    new_valid = new_v > 0.65
    old_valid = old.value > 0.65
    if new_valid && !old_valid
        true
    elseif !new_valid && old_valid
        false
    elseif new_valid && old_valid
        new_p > old.p || (new_p == old.p && new_v > old.value)
    else
        false  # both invalid, keep first
    end
end

best = Dict{Tuple{Int,Int}, BestResult}()

function ingest_swarm_csv!(best, path, source)
    isfile(path) || return
    for line in eachline(path)
        startswith(line, '#') && continue
        startswith(line, "k,") && continue
        fields = split(line, ',')
        length(fields) >= 8 || continue
        k = tryparse(Int, fields[1]); k === nothing && continue
        D = tryparse(Int, fields[2]); D === nothing && continue
        p = tryparse(Int, fields[3]); p === nothing && continue
        v = tryparse(Float64, fields[4]); v === nothing && continue
        isfinite(v) && 0 < v < 1.001 || continue

        gamma = tryparse.(Float64, split(fields[7], ';'))
        beta = tryparse.(Float64, split(fields[8], ';'))
        any(isnothing, gamma) && continue
        any(isnothing, beta) && continue

        key = (k, D)
        if !haskey(best, key) || is_better(p, v, best[key])
            best[key] = BestResult(k, D, p, v, Float64.(gamma), Float64.(beta), source)
        end
    end
end

function ingest_results_csv!(best, path, source)
    isfile(path) || return
    for line in eachline(path)
        startswith(line, '#') && continue
        fields = split(line, ',')
        length(fields) >= 26 || continue
        k = tryparse(Int, fields[8]); k === nothing && continue
        D = tryparse(Int, fields[9]); D === nothing && continue
        p = tryparse(Int, fields[10]); p === nothing && continue
        v = tryparse(Float64, fields[15]); v === nothing && continue
        isfinite(v) && 0 < v < 1.001 || continue

        gamma = tryparse.(Float64, split(fields[25], ';'))
        beta = tryparse.(Float64, split(fields[26], ';'))
        any(isnothing, gamma) && continue
        any(isnothing, beta) && continue

        key = (k, D)
        if !haskey(best, key) || is_better(p, v, best[key])
            best[key] = BestResult(k, D, p, v, Float64.(gamma), Float64.(beta), source)
        end
    end
end

# Local swarm results
for f in readdir(joinpath(@__DIR__, "..", "results"); join=true)
    endswith(f, ".csv") && occursin("swarm", f) || continue
    ingest_swarm_csv!(best, f, "local-swarm")
end

# Local optimization runs
runs_dir = joinpath(@__DIR__, "..", ".project", "results", "optimization", "runs")
if isdir(runs_dir)
    for run_name in readdir(runs_dir)
        csv = joinpath(runs_dir, run_name, "results.csv")
        ingest_results_csv!(best, csv, "local-$run_name")
    end
end

# Fleet branches (via git show)
try
    branches = filter(b -> occursin("fleet", b) || occursin("p710", b) || occursin("stephen", b),
        strip.(readlines(pipeline(`git branch -r`, stderr=devnull))))
    for branch in branches
        branch = strip(branch)
        # Index CSV
        try
            lines = readlines(pipeline(`git show $(branch):.project/results/optimization/index.csv`, stderr=devnull))
            for line in lines
                startswith(line, '#') && continue
                fields = split(line, ',')
                length(fields) >= 26 || continue
                k = tryparse(Int, fields[8]); k === nothing && continue
                D = tryparse(Int, fields[9]); D === nothing && continue
                p = tryparse(Int, fields[10]); p === nothing && continue
                v = tryparse(Float64, fields[15]); v === nothing && continue
                isfinite(v) && 0 < v < 1.001 || continue
                gamma = tryparse.(Float64, split(fields[25], ';'))
                beta = tryparse.(Float64, split(fields[26], ';'))
                any(isnothing, gamma) && continue
                any(isnothing, beta) && continue
                key = (k, D)
                if !haskey(best, key) || is_better(p, v, best[key])
                    best[key] = BestResult(k, D, p, v, Float64.(gamma), Float64.(beta), branch)
                end
            end
        catch; end

        # Swarm CSVs on branches
        for fname in ["results/swarm-k5d7.csv", "results/swarm-k5d8.csv",
                       "results/swarm-k6d7.csv", "results/swarm-k6d8.csv",
                       "results/swarm-k7d8.csv"]
            try
                lines = readlines(pipeline(`git show $(branch):$(fname)`, stderr=devnull))
                for line in lines
                    startswith(line, '#') && continue
                    startswith(line, "k,") && continue
                    fields = split(line, ',')
                    length(fields) >= 8 || continue
                    k = tryparse(Int, fields[1]); k === nothing && continue
                    D = tryparse(Int, fields[2]); D === nothing && continue
                    p = tryparse(Int, fields[3]); p === nothing && continue
                    v = tryparse(Float64, fields[4]); v === nothing && continue
                    isfinite(v) && 0 < v < 1.001 || continue
                    gamma = tryparse.(Float64, split(fields[7], ';'))
                    beta = tryparse.(Float64, split(fields[8], ';'))
                    any(isnothing, gamma) && continue
                    any(isnothing, beta) && continue
                    key = (k, D)
                    if !haskey(best, key) || is_better(p, v, best[key])
                        best[key] = BestResult(k, D, p, v, Float64.(gamma), Float64.(beta), "$branch:$fname")
                    end
                end
            catch; end
        end
    end
catch; end

# Write output
outfile = joinpath(@__DIR__, "..", "results", "warm-start-angles.csv")
open(outfile, "w") do io
    println(io, "# Best QAOA angles for warm-starting — generated $(Dates.today())")
    println(io, "# Use with: julia --project=. scripts/swarm_chain.jl (reads this for warm start)")
    println(io, "k,D,p,ctilde,source,gamma,beta")
    for key in sort(collect(keys(best)))
        r = best[key]
        gamma_str = join(string.(r.gamma), ';')
        beta_str = join(string.(r.beta), ';')
        @printf(io, "%d,%d,%d,%.12f,%s,%s,%s\n", r.k, r.D, r.p, r.value, r.source, gamma_str, beta_str)
    end
end

# Print summary
println("\n=== WARM-START SUMMARY ===")
@printf("%-6s  %-4s  %-14s  %s\n", "(k,D)", "p", "c_tilde", "source")
@printf("%-6s  %-4s  %-14s  %s\n", "-----", "---", "-----------", "------")
for key in sort(collect(keys(best)))
    r = best[key]
    @printf("(%d,%d)   p=%-2d  %.12f  %s\n", r.k, r.D, r.p, r.value, r.source)
end
println("\nWrote: $outfile")
