#!/usr/bin/env julia
# Q1 Experiment 1: Angle Schedule Comparison
#
# For each (k=2, D), at the deepest available p, dump:
#   - The optimal QAOA angle schedule γ*_j, β*_j (unwrapped to [0,2π) and
#     [0,π) and unwrapped across j to remove modular jumps).
#   - The "linear adiabatic" schedule with the same endpoint magnitudes:
#       γ^adi_j = (j/p) · γ_max
#       β^adi_j = (1 - (j-1)/(p-1)) · β_max          (β_p = 0)
#     where γ_max, β_max are the maximum |·| values from the optimum.
#   - c̃ at the optimum and at the linear adiabatic angles
#     ("adiabatic fidelity" = how much performance is lost).
#
# Output:
#   results/q1-angle-schedules.csv   (per-step γ, β, both schedules)
#   results/q1-adiabatic-fidelity.csv  (one row per (D,p): c̃_opt vs c̃_adi)
#
# Usage: julia --project=. scripts/q1_angle_schedules.jl

using QaoaXorsat, Printf, Dates

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
const SCHED_FILE  = joinpath(RESULTS_DIR, "q1-angle-schedules.csv")
const FIDEL_FILE  = joinpath(RESULTS_DIR, "q1-adiabatic-fidelity.csv")

const CLAUSE_SIGN = -1
const K           = 2

# ── CSV helpers ─────────────────────────────────────────────────────────

function load_best_angles_for_p(file::AbstractString, k::Int, D::Int, p::Int)
    best_val = -Inf
    γ = Float64[]
    β = Float64[]
    isfile(file) || return (best_val, γ, β)
    for line in eachline(file)
        startswith(line, '#') && continue
        startswith(line, "k,") && continue
        fields = split(line, ',')
        length(fields) ≥ 7 || continue
        lk = tryparse(Int,     fields[1]); lk === nothing && continue
        lD = tryparse(Int,     fields[2]); lD === nothing && continue
        lp = tryparse(Int,     fields[3]); lp === nothing && continue
        lv = tryparse(Float64, fields[4]); lv === nothing && continue
        (lk == k && lD == D && lp == p) || continue
        if lv > best_val
            best_val = lv
            γ = parse.(Float64, split(fields[6], ';'))
            β = parse.(Float64, split(fields[7], ';'))
        end
    end
    return (best_val, γ, β)
end

function max_p_in_csv(file::AbstractString, k::Int, D::Int)
    p_max = 0
    for p_try in 1:20
        v, _, _ = load_best_angles_for_p(file, k, D, p_try)
        v > -Inf && (p_max = p_try)
    end
    return p_max
end

# QAOA: γ has period 2π, β has period π. Unwrap so the schedule shape is
# visible. Choose the wrap of each entry that minimises distance to the
# previous entry.
function unwrap_periodic(y::AbstractVector{<:Real}, period::Real)
    isempty(y) && return Float64[]
    out = Float64[float(y[1])]
    for j in 2:length(y)
        prev = out[end]
        cand = float(y[j])
        while cand - prev >  period / 2; cand -= period; end
        while cand - prev ≤ -period / 2; cand += period; end
        push!(out, cand)
    end
    return out
end

# ── Main ────────────────────────────────────────────────────────────────

function main()
    println("╔══════════════════════════════════════════════════════════╗")
    println("║  Q1 Experiment 1: Angle Schedule Comparison            ║")
    println("║  $(now())                            ║")
    println("╚══════════════════════════════════════════════════════════╝")
    println()

    mkpath(RESULTS_DIR)
    open(SCHED_FILE, "w") do io
        println(io, "# Q1: Optimal vs linear-adiabatic angle schedules — $(now())")
        println(io, "k,D,p,j,gamma_opt,beta_opt,gamma_adi,beta_adi")
    end
    open(FIDEL_FILE, "w") do io
        println(io, "# Q1: Adiabatic fidelity — $(now())")
        println(io, "# c̃_opt    = c̃ at optimal QAOA angles")
        println(io, "# c̃_adi    = c̃ at the linear-adiabatic schedule with matched magnitudes")
        println(io, "# delta    = c̃_opt - c̃_adi (performance lost by the adiabatic schedule)")
        println(io, "# rel_loss = (c̃_opt - c̃_adi) / (c̃_opt - 0.5) — relative to the random-guess baseline")
        println(io, "k,D,p,gamma_max,beta_max,ctilde_opt,ctilde_adi,delta,rel_loss")
    end

    @printf("  %-3s  %-4s  %-12s  %-12s  %-10s  %-10s\n",
            "D", "p", "c̃_opt", "c̃_adiabatic", "Δ", "relΔ")
    @printf("  %-3s  %-4s  %-12s  %-12s  %-10s  %-10s\n",
            "───", "────", "────────────", "────────────", "──────────", "──────────")

    for D in 3:8
        file  = joinpath(RESULTS_DIR, "maxcut-k2-d$(D)-sweep.csv")
        p_max = max_p_in_csv(file, K, D)
        p_max < 4 && continue

        c_opt, γ_raw, β_raw = load_best_angles_for_p(file, K, D, p_max)
        γ_un = unwrap_periodic(γ_raw, 2π)
        β_un = unwrap_periodic(β_raw, π)

        γ_max = maximum(abs, γ_un)
        β_max = maximum(abs, β_un)

        # Linear adiabatic schedule with matched endpoint magnitudes:
        # γ ramps 0 → γ_max, β ramps β_max → 0.
        γ_adi = [(j / p_max) * γ_max for j in 1:p_max]
        β_adi = p_max == 1 ? [β_max] :
                [(1 - (j - 1) / (p_max - 1)) * β_max for j in 1:p_max]

        c_adi = basso_expectation_normalized(
            TreeParams(K, D, p_max),
            QAOAAngles(γ_adi, β_adi);
            clause_sign=CLAUSE_SIGN,
        )

        Δ        = c_opt - c_adi
        rel_loss = c_opt > 0.5 ? Δ / (c_opt - 0.5) : NaN

        @printf("  %-3d  %-4d  %.10f  %.10f  %+.4e  %.4f\n",
                D, p_max, c_opt, c_adi, Δ, rel_loss)

        open(SCHED_FILE, "a") do io
            for j in 1:p_max
                @printf(io, "%d,%d,%d,%d,%.12f,%.12f,%.12f,%.12f\n",
                        K, D, p_max, j, γ_un[j], β_un[j], γ_adi[j], β_adi[j])
            end
        end
        open(FIDEL_FILE, "a") do io
            @printf(io, "%d,%d,%d,%.10f,%.10f,%.12f,%.12f,%+.12f,%.10f\n",
                    K, D, p_max, γ_max, β_max, c_opt, c_adi, Δ, rel_loss)
        end
    end

    println()
    println("Schedules → $SCHED_FILE")
    println("Fidelity  → $FIDEL_FILE")
end

main()
