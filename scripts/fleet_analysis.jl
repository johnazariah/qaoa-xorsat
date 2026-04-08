#!/usr/bin/env julia
using Printf

runs_dir = joinpath(@__DIR__, "..", ".project", "results", "optimization", "runs")

# Collect best value per (k,D,p) across all runs
best = Dict{Tuple{Int,Int,Int}, Float64}()

for run_name in readdir(runs_dir)
    csv_path = joinpath(runs_dir, run_name, "results.csv")
    isfile(csv_path) || continue
    lines = readlines(csv_path)
    length(lines) > 1 || continue
    for line in lines[2:end]
        fields = split(line, ',')
        length(fields) >= 15 || continue
        k = parse(Int, fields[8])
        D = parse(Int, fields[9])
        p = parse(Int, fields[10])
        v = parse(Float64, fields[15])
        key = (k, D, p)
        if !haskey(best, key) || v > best[key]
            best[key] = v
        end
    end
end

# Print per (k,D) all depths
pairs = sort(unique([(k,D) for (k,D,p) in keys(best)]))
summary = []

for (k,D) in pairs
    depths = sort([(p, best[(k,D,p)]) for (kk,dd,p) in keys(best) if kk==k && dd==D])
    @printf("\n(%d,%d):\n", k, D)
    last_good_p = 0
    last_good_v = 0.0
    for (p, v) in depths
        flag = ""
        if !isfinite(v)
            flag = " *** NaN/Inf"
        elseif v < -0.001 || v > 1.001
            flag = " *** OVERFLOW"
        elseif p > 1 && v < last_good_v - 0.005
            flag = " ** DROPPED"
        end
        if isempty(flag) && 0 <= v <= 1.001
            last_good_p = p
            last_good_v = v
        end
        @printf("  p=%2d: %24.12f%s\n", p, v, flag)
    end
    @printf("  -> BEST VALID: p=%d, c_tilde=%.12f\n", last_good_p, last_good_v)
    push!(summary, (k, D, last_good_p, last_good_v))
end

println("\n\n=== SUMMARY TABLE ===")
@printf("%-6s  %-6s  %-14s\n", "(k,D)", "p_max", "c_tilde")
@printf("%-6s  %-6s  %-14s\n", "-----", "-----", "-----------")
for (k, D, p, v) in summary
    @printf("(%d,%d)   p=%-3d   %.12f\n", k, D, p, v)
end
