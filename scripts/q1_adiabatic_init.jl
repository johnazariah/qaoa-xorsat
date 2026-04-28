#!/usr/bin/env julia
# Q1 Experiment 3: Adiabatic-Initialised QAOA
#
# For each (k=2, D, p), seed L-BFGS from a linear adiabatic schedule
#   γ_j = (j/p) · γ_max, β_j = (1 - (j-1)/(p-1)) · β_max
# for several (γ_max, β_max) settings, with `restarts=0` so the
# adiabatic seed is the *only* starting point. Compare:
#
#   c̃_adi(start)   — c̃ at the adiabatic seed (raw)
#   c̃_adi(opt)     — c̃ after L-BFGS converges from the adiabatic seed
#   c̃_warm         — c̃ from the prior optimum (warm-start, our standard
#                    pipeline; loaded from the maxcut-k2-d{D}-sweep.csv)
#
# Result interpretation:
#   - If c̃_adi(opt) ≈ c̃_warm    → adiabatic init reaches the same basin
#   - If c̃_adi(opt)  < c̃_warm   → adiabatic init falls into a worse basin
#
# Output:  results/q1-adiabatic-init.csv
#
# Usage: julia --project=. -t 16 scripts/q1_adiabatic_init.jl

using QaoaXorsat, Printf, Dates, Random

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
const OUT_FILE    = joinpath(RESULTS_DIR, "q1-adiabatic-init.csv")

const CLAUSE_SIGN = -1
const K           = 2

# Test these (D, p) pairs. p kept moderate so runtime stays under control.
const TARGETS = [(3, 8), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8)]

# Adiabatic-magnitude grid. Covers small (γ_max≈π/4, β_max≈π/8) up to
# textbook full-bang values (γ_max=2π, β_max=π).
const GAMMA_MAXES = [π / 2, π, 3π / 2, 2π]
const BETA_MAXES  = [π / 4, π / 2, 3π / 4]

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

linear_adi(p::Int, γ_max, β_max) = QAOAAngles(
    [(j / p) * γ_max for j in 1:p],
    p == 1 ? [β_max] : [(1 - (j - 1) / (p - 1)) * β_max for j in 1:p],
)

# ── Main ────────────────────────────────────────────────────────────────

function main()
    println("╔══════════════════════════════════════════════════════════╗")
    println("║  Q1 Experiment 3: Adiabatic-Initialised QAOA           ║")
    println("║  $(now())                            ║")
    println("╚══════════════════════════════════════════════════════════╝")
    println()

    mkpath(RESULTS_DIR)
    open(OUT_FILE, "w") do io
        println(io, "# Q1 Exp 3: Adiabatic-init vs warm-start optimization — $(now())")
        println(io, "k,D,p,gamma_max,beta_max,ctilde_seed,ctilde_adi_opt,ctilde_warm,delta_warm_minus_adi,iterations,wall_seconds,converged")
    end

    rng = Random.MersenneTwister(20260429)

    for (D, p) in TARGETS
        params = TreeParams(K, D, p)
        sweep_file = joinpath(RESULTS_DIR, "maxcut-k2-d$(D)-sweep.csv")
        c_warm, _, _ = load_best_angles_for_p(sweep_file, K, D, p)

        @printf("━━━ D=%d, p=%d  (warm-start c̃ = %.6f) ━━━\n",
                D, p, c_warm)
        @printf("  %-7s  %-7s  %-12s  %-12s  %-10s  %-7s  %-9s\n",
                "γ_max", "β_max", "c̃(seed)", "c̃(adi-opt)", "warm-Δ", "iters", "secs")

        for γm in GAMMA_MAXES, βm in BETA_MAXES
            seed = linear_adi(p, γm, βm)
            c_seed = basso_expectation_normalized(params, seed; clause_sign=CLAUSE_SIGN)

            t0 = time()
            result = optimize_angles(
                params;
                clause_sign=CLAUSE_SIGN,
                initial_guesses=[seed],
                initial_guess_kind=:adiabatic,
                restarts=0,
                rng=rng,
                maxiters=400,
            )
            elapsed = time() - t0

            c_adi_opt = result.value
            iters     = result.iterations
            δ         = c_warm - c_adi_opt

            @printf("  %.4f  %.4f  %.10f  %.10f  %+.4e  %-7d  %.2f\n",
                    γm, βm, c_seed, c_adi_opt, δ, iters, elapsed)

            open(OUT_FILE, "a") do io
                @printf(io, "%d,%d,%d,%.10f,%.10f,%.12f,%.12f,%.12f,%+.12f,%d,%.2f,%s\n",
                        K, D, p, γm, βm, c_seed, c_adi_opt, c_warm, δ,
                        iters, elapsed, result.converged ? "true" : "false")
            end
        end
        println()
        flush(stdout)
    end

    println("Results → $OUT_FILE")
end

main()
