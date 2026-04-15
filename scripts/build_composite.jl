#!/usr/bin/env julia
# Build composite best CSV from F64 (old) + D64 (new) results.
# For each (k,D,p), take the max c̃ across all sources, respecting
# the Float64 precision wall — F64 values above the wall are discarded.

using Printf

# Float64 precision wall: max reliable p per (k,D)
SAFE_P_MAX = Dict(
    (3,4) => 13, (3,5) => 13, (3,6) => 11, (3,7) => 11, (3,8) => 11,
    (4,5) => 11, (4,6) => 10, (4,7) => 10, (4,8) => 9,
    (5,6) => 9,  (5,7) => 9,  (5,8) => 9,
    (6,7) => 8,  (6,8) => 8,
    (7,8) => 7,
)

struct Entry
    ctilde::Float64
    converged::Bool
    source::String
    gamma::String
    beta::String
end

# (k,D,p) -> best Entry
best = Dict{Tuple{Int,Int,Int}, Entry}()

function maybe_update!(best, k, D, p, entry)
    key = (k, D, p)
    if !haskey(best, key) || entry.ctilde > best[key].ctilde
        best[key] = entry
    end
end

# ── Read F64 composite ────────────────────────────────────────────
f64_file = "/tmp/f64-composite.csv"
for line in eachline(f64_file)
    startswith(line, '#') && continue
    startswith(line, "k,") && continue
    fields = split(line, ',')
    length(fields) >= 8 || continue
    k = tryparse(Int, fields[1]); k === nothing && continue
    D = tryparse(Int, fields[2]); D === nothing && continue
    p = tryparse(Int, fields[3]); p === nothing && continue
    v = tryparse(Float64, fields[4]); v === nothing && continue
    
    # Skip if above precision wall
    safe_p = get(SAFE_P_MAX, (k, D), 8)
    p > safe_p && continue
    
    # Skip garbage values
    v < 0.5 && continue
    v > 1.0 && continue
    
    conv = fields[5] == "true"
    src = "f64/" * fields[6]
    gamma = fields[7]
    beta = fields[8]
    
    maybe_update!(best, k, D, p, Entry(v, conv, src, gamma, beta))
end

println("Loaded $(length(best)) F64 entries (below precision wall)")

# ── Read D64 results ──────────────────────────────────────────────
d64_dir = joinpath(@__DIR__, "..", "results")
for f in readdir(d64_dir)
    startswith(f, "swarm-d64-k") || continue
    endswith(f, ".csv") || continue
    path = joinpath(d64_dir, f)
    for line in eachline(path)
        startswith(line, '#') && continue
        startswith(line, "k,") && continue
        fields = split(line, ',')
        length(fields) >= 8 || continue
        k = tryparse(Int, fields[1]); k === nothing && continue
        D = tryparse(Int, fields[2]); D === nothing && continue
        p = tryparse(Int, fields[3]); p === nothing && continue
        v = tryparse(Float64, fields[4]); v === nothing && continue
        
        v < 0.5 && continue
        v > 1.0 && continue
        
        gamma = fields[7]
        beta = fields[8]
        
        maybe_update!(best, k, D, p, Entry(v, true, "d64/" * f, gamma, beta))
    end
end

println("Total $(length(best)) entries after D64 merge")

# ── Write composite ───────────────────────────────────────────────
out_file = joinpath(d64_dir, "composite-best.csv")
open(out_file, "w") do io
    println(io, "# Composite best c̃ per (k,D,p) — F64 (below precision wall) + D64")
    println(io, "# Generated $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM"))")
    println(io, "k,D,p,ctilde,source,gamma,beta")
    
    for key in sort(collect(keys(best)))
        k, D, p = key
        e = best[key]
        @printf(io, "%d,%d,%d,%.12f,%s,%s,%s\n",
            k, D, p, e.ctilde, e.source, e.gamma, e.beta)
    end
end

println("Written to $out_file")

# ── Summary: best per (k,D) ──────────────────────────────────────
println("\n=== Best per (k,D) ===")
pairs = sort(unique([(k,D) for (k,D,p) in keys(best)]))
for (k,D) in pairs
    entries = [(p, best[(k,D,p)]) for (kk,dd,p) in keys(best) if kk==k && dd==D]
    sort!(entries, by=x->x[1])
    p_best, e_best = entries[end]
    src_short = last(split(e_best.source, '/'))
    @printf("  (%d,%d) p=%d  c̃=%.6f  [%s]\n", k, D, p_best, e_best.ctilde, src_short)
end
