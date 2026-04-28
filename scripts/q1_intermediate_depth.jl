#!/usr/bin/env julia
# Q1 Experiment 2: Intermediate-depth QAOA performance
#
# For each (D, p_max), take the optimal depth-p_max angles and evaluate
# the QAOA objective at intermediate depths t = 1, 2, ..., p_max using
# only the first t angle pairs.
#
# If QAOA is Trotterised adiabatic, c̃(t) should be monotonically increasing.
# If it dips or oscillates, QAOA is doing something non-adiabatic.
#
# Also compares against the linear adiabatic schedule (Experiment 1).
#
# Usage: julia --project=. scripts/q1_intermediate_depth.jl

using QaoaXorsat, Printf, Dates

clause_sign = -1
k = 2

function get_best_angles(file, k, D, target_p)
    best_val = -Inf
    best_gamma = Float64[]
    best_beta = Float64[]
    isfile(file) || return (best_val, best_gamma, best_beta)
    for line in eachline(file)
        startswith(line, '#') && continue
        startswith(line, "k,") && continue
        fields = split(line, ',')
        length(fields) >= 7 || continue
        lk = tryparse(Int, fields[1]); lk === nothing && continue
        lD = tryparse(Int, fields[2]); lD === nothing && continue
        lp = tryparse(Int, fields[3]); lp === nothing && continue
        lv = tryparse(Float64, fields[4]); lv === nothing && continue
        (lk == k && lD == D && lp == target_p) || continue
        if lv > best_val
            best_val = lv
            best_gamma = parse.(Float64, split(fields[6], ';'))
            best_beta = parse.(Float64, split(fields[7], ';'))
        end
    end
    return (best_val, best_gamma, best_beta)
end

println("╔══════════════════════════════════════════════════════════╗")
println("║  Q1: Is QAOA Trotterised Adiabatic? — Intermediate c̃  ║")
println("║  $(now())                            ║")
println("╚══════════════════════════════════════════════════════════╝")
println()

# Output CSV
output_file = joinpath(@__DIR__, "..", "results", "q1-intermediate-depth.csv")
mkpath(dirname(output_file))
open(output_file, "w") do io
    println(io, "# Q1: Intermediate-depth performance — $(now())")
    println(io, "D,p_max,t,ctilde_truncated,ctilde_optimal_at_t,ctilde_adiabatic_at_t")
end

for D in [3, 4, 5, 6]
    # Find the highest p we have
    results_file = joinpath(@__DIR__, "..", "results", "maxcut-k2-d$(D)-sweep.csv")
    p_max = 0
    for p_try in 1:15
        v, _, _ = get_best_angles(results_file, k, D, p_try)
        v > -Inf && (p_max = p_try)
    end
    p_max < 4 && continue

    println("━━━ D=$D, p_max=$p_max ━━━")

    # Get optimal angles at p_max
    _, gamma_opt, beta_opt = get_best_angles(results_file, k, D, p_max)

    @printf("  %-4s  %-14s  %-14s  %-14s  %-10s  %s\n",
            "t", "c̃(truncated)", "c̃(optimal@t)", "c̃(adiabatic)", "trunc-opt", "monotone?")
    @printf("  %-4s  %-14s  %-14s  %-14s  %-10s  %s\n",
            "──", "──────────────", "──────────────", "──────────────", "──────────", "────────")

    prev_trunc = 0.5
    for t in 1:p_max
        params_t = TreeParams(k, D, t)

        # 1. Truncated: first t angles from the p_max optimum
        trunc_angles = QAOAAngles(gamma_opt[1:t], beta_opt[1:t])
        c_trunc = basso_expectation_normalized(params_t, trunc_angles; clause_sign)

        # 2. Optimal at depth t (from CSV)
        v_opt, _, _ = get_best_angles(results_file, k, D, t)
        c_opt = v_opt > -Inf ? v_opt : NaN

        # 3. Linear adiabatic schedule: γ_j = j/t · γ_max, β_j = (1 - j/t) · β_max
        # Use the magnitude of the p_max angles as the adiabatic target
        γ_max = maximum(abs.(gamma_opt))
        β_max = maximum(abs.(beta_opt))
        γ_adiabatic = [j / t * γ_max for j in 1:t]
        β_adiabatic = [(1 - j / t) * β_max + 0.01 for j in 1:t]  # small offset to avoid exact 0
        adiab_angles = QAOAAngles(γ_adiabatic, β_adiabatic)
        c_adiab = basso_expectation_normalized(params_t, adiab_angles; clause_sign)

        delta = c_trunc - c_opt
        mono = c_trunc >= prev_trunc - 1e-10 ? "  ✓" : "  ✗ DIP"

        @printf("  t=%-2d  %.10f  %.10f  %.10f  %+.2e  %s\n",
                t, c_trunc, c_opt, c_adiab, delta, mono)

        open(output_file, "a") do io
            @printf(io, "%d,%d,%d,%.12f,%.12f,%.12f\n",
                    D, p_max, t, c_trunc, c_opt, c_adiab)
        end

        prev_trunc = c_trunc
    end
    println()
    flush(stdout)
end

println("Results written to $output_file")
