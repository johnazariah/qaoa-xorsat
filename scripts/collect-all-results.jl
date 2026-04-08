#!/usr/bin/env julia
# Collect best results from all fleet branches + local runs
using Printf

all_data = Dict{Tuple{Int,Int,Int}, Float64}()

function ingest_csv_lines!(data, lines)
    for line in lines
        fields = split(line, ',')
        length(fields) >= 15 || continue
        k = tryparse(Int, fields[8]); k === nothing && continue
        D = tryparse(Int, fields[9]); D === nothing && continue
        p = tryparse(Int, fields[10]); p === nothing && continue
        v = tryparse(Float64, fields[15]); v === nothing && continue
        key = (k, D, p)
        if !haskey(data, key) || v > data[key]
            data[key] = v
        end
    end
end

# Fleet branches — read via git show
branches = try
    filter(b -> occursin("fleet", b), strip.(readlines(pipeline(`git branch -r`, stderr=devnull))))
catch
    String[]
end

for branch in branches
    branch = strip(branch)
    # Index CSV
    try
        lines = readlines(pipeline(`git show $(branch):.project/results/optimization/index.csv`, stderr=devnull))
        ingest_csv_lines!(all_data, lines)
    catch; end
    # Individual run directories
    try
        dirs = strip.(readlines(pipeline(`git ls-tree -d --name-only $(branch):.project/results/optimization/runs/`, stderr=devnull)))
        for dir in dirs
            try
                lines = readlines(pipeline(`git show $(branch):.project/results/optimization/runs/$(dir)/results.csv`, stderr=devnull))
                ingest_csv_lines!(all_data, lines)
            catch; end
        end
    catch; end
end

# Local results
runs_dir = joinpath(@__DIR__, "..", ".project", "results", "optimization", "runs")
if isdir(runs_dir)
    for run_name in readdir(runs_dir)
        csv_path = joinpath(runs_dir, run_name, "results.csv")
        isfile(csv_path) || continue
        ingest_csv_lines!(all_data, readlines(csv_path))
    end
end

println("Sources: $(length(branches)) fleet branches + local runs")
println("Total data points: $(length(all_data))")
println()

# Per-(k,D) detail with flags
pairs = sort(unique([(k,D) for (k,D,p) in keys(all_data)]))

for (k,D) in pairs
    depths = sort([(p, all_data[(k,D,p)]) for (kk,dd,p) in keys(all_data) if kk==k && dd==D])
    @printf("\n(%d,%d):\n", k, D)
    prev_v = -1.0
    for (p, v) in depths
        flag = ""
        if !isfinite(v); flag = " *** NaN/Inf"
        elseif v < -0.001 || v > 1.001; flag = " *** OVERFLOW"
        elseif p > 1 && v < prev_v - 0.005; flag = " ** DROPPED"
        end
        if isempty(flag) && isfinite(v) && 0 <= v <= 1.001
            prev_v = v
        end
        @printf("  p=%2d: %24.12f%s\n", p, v, flag)
    end
end

# Summary with monotonicity filter
println("\n\n=== SUMMARY TABLE (monotonicity-filtered) ===")
@printf("%-6s  %-6s  %-16s\n", "(k,D)", "p_max", "c_tilde")
@printf("%-6s  %-6s  %-16s\n", "-----", "-----", "-----------")
for (k,D) in pairs
    depths = sort([(p, all_data[(k,D,p)]) for (kk,dd,p) in keys(all_data) if kk==k && dd==D])
    last_good_p = 0; last_good_v = 0.0; prev_v = -1.0
    for (p, v) in depths
        valid = isfinite(v) && 0 < v < 1.001 && v >= prev_v - 0.005
        if valid
            last_good_p = p; last_good_v = v; prev_v = v
        else
            break
        end
    end
    @printf("(%d,%d)   p=%-3d   %.12f\n", k, D, last_good_p, last_good_v)
end
