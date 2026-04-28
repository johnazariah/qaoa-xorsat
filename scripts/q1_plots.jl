#!/usr/bin/env julia
# Q1 plotting: reads the four Q1 CSVs and writes PNGs into figures/.
#
#   q1-angle-schedules.csv      → figures/q1-angle-schedules.png   (E1)
#   q1-intermediate-depth.csv   → figures/q1-intermediate-depth.png (E2)
#   q1-adiabatic-init.csv       → figures/q1-adiabatic-init.png    (E3)
#   q1-angle-curvature.csv      → figures/q1-angle-curvature.png   (E4)
#
# Usage: julia --project=. scripts/q1_plots.jl
#
# Requires Plots.jl. If not available the script aborts with a clear message
# rather than mutating Project.toml.

using Printf, DelimitedFiles

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
const FIGURES_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURES_DIR)

try
    @eval using Plots
catch e
    @error "Plots.jl is required to render Q1 figures. Install it once with " *
           "`julia --project=. -e 'using Pkg; Pkg.add(\"Plots\")'`."
    rethrow(e)
end

# ── Tiny CSV reader (skip lines starting with '#') ──────────────────────
function load_csv(path)
    lines = filter(l -> !startswith(l, '#') && !isempty(l), readlines(path))
    header = split(popfirst!(lines), ',')
    rows = [split(l, ',') for l in lines]
    return header, rows
end

function getcol(header, rows, name)
    idx = findfirst(==(name), header)
    idx === nothing && error("column $name not in $(header)")
    return [r[idx] for r in rows]
end

asfloat(xs) = parse.(Float64, xs)
asint(xs)   = parse.(Int,     xs)

# ── E1: angle schedules ────────────────────────────────────────────────

function plot_angle_schedules()
    path = joinpath(RESULTS_DIR, "q1-angle-schedules.csv")
    isfile(path) || (println("skip E1 (no $path)"); return)
    header, rows = load_csv(path)

    Ds = unique(asint(getcol(header, rows, "D")))
    γ_plt = plot(title="γ schedule: optimal vs linear adiabatic",
                 xlabel="step j / p", ylabel="γ", legend=:outertopright)
    β_plt = plot(title="β schedule: optimal vs linear adiabatic",
                 xlabel="step j / p", ylabel="β", legend=:outertopright)

    for D in sort(Ds)
        sel = [i for (i, r) in enumerate(rows) if parse(Int, r[2]) == D]
        isempty(sel) && continue
        sub = [rows[i] for i in sel]
        p   = parse(Int, sub[1][3])
        j   = asint(getcol(header, sub, "j"))
        x   = j ./ p
        γo  = asfloat(getcol(header, sub, "gamma_opt"))
        βo  = asfloat(getcol(header, sub, "beta_opt"))
        γa  = asfloat(getcol(header, sub, "gamma_adi"))
        βa  = asfloat(getcol(header, sub, "beta_adi"))
        plot!(γ_plt, x, γo, label="D=$D opt", lw=2, marker=:circle)
        plot!(γ_plt, x, γa, label="D=$D adi", ls=:dash)
        plot!(β_plt, x, βo, label="D=$D opt", lw=2, marker=:circle)
        plot!(β_plt, x, βa, label="D=$D adi", ls=:dash)
    end

    fig = plot(γ_plt, β_plt, layout=(1, 2), size=(1400, 500))
    out = joinpath(FIGURES_DIR, "q1-angle-schedules.png")
    savefig(fig, out)
    println("wrote $out")
end

# ── E2: intermediate-depth performance ─────────────────────────────────

function plot_intermediate_depth()
    path = joinpath(RESULTS_DIR, "q1-intermediate-depth.csv")
    isfile(path) || (println("skip E2 (no $path)"); return)
    header, rows = load_csv(path)

    Ds = sort(unique(asint(getcol(header, rows, "D"))))
    plt = plot(title="QAOA intermediate-depth performance",
               xlabel="depth t", ylabel="c̃",
               legend=:bottomright, ylim=(0.45, 0.92))

    for D in Ds
        sub = filter(r -> parse(Int, r[1]) == D, rows)
        t   = asint(getcol(header, sub, "t"))
        ct  = asfloat(getcol(header, sub, "ctilde_truncated"))
        co  = asfloat(getcol(header, sub, "ctilde_optimal_at_t"))
        ca  = asfloat(getcol(header, sub, "ctilde_adiabatic_at_t"))
        plot!(plt, t, co, label="D=$D optimal@t", lw=2,  marker=:circle)
        plot!(plt, t, ct, label="D=$D truncated", lw=1,  ls=:dash, marker=:utriangle)
        plot!(plt, t, ca, label="D=$D adiabatic", lw=1,  ls=:dot,  marker=:diamond)
    end
    hline!(plt, [0.5], color=:gray, ls=:dot, label="random (c̃=0.5)")

    out = joinpath(FIGURES_DIR, "q1-intermediate-depth.png")
    savefig(plt, out)
    println("wrote $out")
end

# ── E3: adiabatic-init optimization ────────────────────────────────────

function plot_adiabatic_init()
    path = joinpath(RESULTS_DIR, "q1-adiabatic-init.csv")
    isfile(path) || (println("skip E3 (no $path)"); return)
    header, rows = load_csv(path)

    Ds = sort(unique(asint(getcol(header, rows, "D"))))
    plt = plot(title="Adiabatic-initialised QAOA: best-of-grid vs warm-start",
               xlabel="D", ylabel="c̃",
               legend=:topright, xticks=Ds)

    best_vals  = Float64[]
    warm_vals  = Float64[]
    seed_bests = Float64[]
    for D in Ds
        sub = filter(r -> parse(Int, r[2]) == D, rows)
        cs  = asfloat(getcol(header, sub, "ctilde_seed"))
        cao = asfloat(getcol(header, sub, "ctilde_adi_opt"))
        cw  = asfloat(getcol(header, sub, "ctilde_warm"))[1]
        push!(best_vals,  maximum(cao))
        push!(warm_vals,  cw)
        push!(seed_bests, maximum(cs))
    end

    plot!(plt, Ds, warm_vals,  label="warm-start (c̃)",      lw=2, marker=:circle)
    plot!(plt, Ds, best_vals,  label="best adi-init (c̃)",   lw=2, marker=:utriangle)
    plot!(plt, Ds, seed_bests, label="best adi-seed (raw)", lw=1, ls=:dash, marker=:diamond)
    hline!(plt, [0.5], color=:gray, ls=:dot, label="random")

    out = joinpath(FIGURES_DIR, "q1-adiabatic-init.png")
    savefig(plt, out)
    println("wrote $out")
end

# ── E4: angle curvature (linear-fit r²) ────────────────────────────────

function plot_angle_curvature()
    path = joinpath(RESULTS_DIR, "q1-angle-curvature.csv")
    isfile(path) || (println("skip E4 (no $path)"); return)
    header, rows = load_csv(path)

    keep = filter(r -> parse(Int, r[5]) == 1, rows)  # deg = 1 only
    Ds   = sort(unique(asint(getcol(header, keep, "D"))))
    γ_r2 = Float64[]
    β_r2 = Float64[]
    for D in Ds
        sub_g = filter(r -> parse(Int, r[2]) == D && r[4] == "gamma", keep)
        sub_b = filter(r -> parse(Int, r[2]) == D && r[4] == "beta",  keep)
        push!(γ_r2, parse(Float64, sub_g[1][6]))
        push!(β_r2, parse(Float64, sub_b[1][6]))
    end

    plt = bar(Ds, γ_r2, label="γ", bar_position=:dodge,
              xlabel="D", ylabel="r²(linear fit)",
              title="Linear adiabatic fit quality vs D (deg=1)",
              ylim=(0, 1), xticks=Ds)
    bar!(plt, Ds .+ 0.35, β_r2, label="β", bar_width=0.35)
    hline!(plt, [1.0], color=:gray, ls=:dot, label="perfect linear")

    out = joinpath(FIGURES_DIR, "q1-angle-curvature.png")
    savefig(plt, out)
    println("wrote $out")
end

plot_angle_schedules()
plot_intermediate_depth()
plot_adiabatic_init()
plot_angle_curvature()
