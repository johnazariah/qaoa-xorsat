#!/usr/bin/env julia
using Printf

function read_values(path)
    lines = readlines(path)
    d = Dict{Int,Float64}()
    for line in lines[2:end]
        fields = split(line, ",")
        p = parse(Int, fields[10])
        val = parse(Float64, fields[15])
        d[p] = val
    end
    d
end

base = joinpath(@__DIR__, "..", "..", ".project", "results", "optimization", "runs")
runs = [
    ("non-adjoint", joinpath(base, "20260324T092213-k3-d4-p1-13-r5-i80-s1234", "results.csv")),
    ("old-adjoint", joinpath(base, "20260324T213012-k3-d4-p1-13-r5-i80-s1234", "results.csv")),
    ("new-adjoint", joinpath(base, "20260325T090338-k3-d4-p1-13-r5-i80-s1234", "results.csv")),
]

data = [(name, read_values(path)) for (name, path) in runs]

println("  p │  non-adjoint  │  old-adjoint  │  new-adjoint  │    spread")
println("────┼───────────────┼───────────────┼───────────────┼──────────────")
for p in 1:11
    vals = Float64[]
    parts = String[]
    for (name, d) in data
        v = get(d, p, NaN)
        push!(parts, isnan(v) ? "      —      " : @sprintf("%.12f", v))
        isnan(v) || push!(vals, v)
    end
    spread = length(vals) > 1 ? maximum(vals) - minimum(vals) : NaN
    ss = isnan(spread) ? "     —" : @sprintf(" %.2e", spread)
    println(@sprintf(" %2d", p), " │ ", join(parts, " │ "), " │", ss)
end
