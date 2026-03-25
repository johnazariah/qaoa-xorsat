#!/usr/bin/env julia
#
# Collect results from cluster runs into Stephen's comparison table.
#
# Usage:
#   julia scripts/collect-results.jl [RESULTS_DIR]
#
# Reads per-depth JSON files from results/cluster/k{K}-D{D}/p{NN}.json
# and produces a summary table.

using Printf

function collect_results(base_dir)
    # Known comparison data from Jordan et al.
    comparison = Dict(
        (3,4) => (prange=0.875, dqi_bp=0.87065, regev_fgum=0.89187, sa=0.9366),
        (3,5) => (prange=0.800, dqi_bp=0.81648, regev_fgum=0.83607, sa=0.9005),
        (3,6) => (prange=0.750, dqi_bp=0.77562, regev_fgum=0.78361, sa=0.8712),
        (3,7) => (prange=0.71428, dqi_bp=0.74727, regev_fgum=0.76024, sa=0.8492),
        (3,8) => (prange=0.6875, dqi_bp=0.72351, regev_fgum=0.72943, sa=0.8287),
        (4,5) => (prange=0.900, dqi_bp=0.8597, regev_fgum=0.92158, sa=0.9279),
        (4,6) => (prange=0.83333, dqi_bp=0.82062, regev_fgum=0.86144, sa=0.9024),
        (4,7) => (prange=0.78571, dqi_bp=0.78862, regev_fgum=0.82645, sa=0.8771),
        (4,8) => (prange=0.75, dqi_bp=0.76539, regev_fgum=0.79021, sa=0.8587),
        (5,6) => (prange=0.91667, dqi_bp=0.84305, regev_fgum=0.93123, sa=0.9190),
        (5,7) => (prange=0.85714, dqi_bp=0.81422, regev_fgum=0.88529, sa=0.8965),
        (5,8) => (prange=0.8125, dqi_bp=0.78752, regev_fgum=0.84403, sa=0.8740),
        (6,7) => (prange=0.92857, dqi_bp=0.82759, regev_fgum=0.94271, sa=0.9051),
        (6,8) => (prange=0.875, dqi_bp=0.80327, regev_fgum=0.89619, sa=0.8875),
        (7,8) => (prange=0.9375, dqi_bp=0.813, regev_fgum=0.94810, sa=0.8951),
    )

    # Scan for result directories
    results = Dict{Tuple{Int,Int}, Vector{NamedTuple}}()
    
    if !isdir(base_dir)
        println(stderr, "Results directory not found: $base_dir")
        return
    end

    for entry in readdir(base_dir)
        m = match(r"^k(\d+)-D(\d+)$", entry)
        m === nothing && continue
        k, D = parse(Int, m[1]), parse(Int, m[2])
        dir = joinpath(base_dir, entry)
        
        depths = NamedTuple[]
        for pfile in sort(readdir(dir))
            pm = match(r"^p(\d+)\.json$", pfile)
            pm === nothing && continue
            p = parse(Int, pm[1])
            content = read(joinpath(dir, pfile), String)
            vm = match(r"\"value\":\s*([\d.]+)", content)
            vm === nothing && continue
            value = parse(Float64, vm[1])
            push!(depths, (p=p, value=value))
        end
        
        if !isempty(depths)
            results[(k, D)] = depths
        end
    end

    # Print the full comparison table
    println("=" ^ 95)
    @printf("%-6s  %-7s  %-7s  %-10s  %-7s  %-12s\n",
        "(k,D)", "Prange", "DQI+BP", "Regev+FGUM", "SA", "QAOA (best p)")
    println("-" ^ 95)
    
    for (k, D) in sort(collect(keys(comparison)))
        comp = comparison[(k, D)]
        
        qaoa_str = "---"
        if haskey(results, (k, D))
            best = results[(k, D)][end]  # highest p
            qaoa_str = @sprintf("%.4f (p=%d)", best.value, best.p)
        end
        
        @printf("(%d,%d)   %.4f   %.4f   %.5f     %.4f   %s\n",
            k, D, comp.prange, comp.dqi_bp, comp.regev_fgum, comp.sa, qaoa_str)
    end
    println("=" ^ 95)
    
    # Also print per-depth detail for each (k,D) with results
    println("\nDetailed results by depth:")
    for (k, D) in sort(collect(keys(results)))
        println("\n  (k=$k, D=$D):")
        for r in results[(k, D)]
            @printf("    p=%2d: c̃ = %.10f\n", r.p, r.value)
        end
    end
end

base_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "results", "cluster")
collect_results(base_dir)
