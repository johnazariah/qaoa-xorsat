#!/usr/bin/env julia
# Q1.5 Tier 2: Non-linear CD-QAOA via Wurtz–Love Eq. 29 + Eq. 45
#
# Constructs a piecewise-linear λ(t) on [0, T] with nodes
#   λ_0 = 0, λ_1, ..., λ_p = 1
# such that the inferred auxiliary field s̄_q vanishes at every step.
#
# At lowest BCH order with uniform τ = T/p, the constraint s̄_q = 0 gives
#   τ² · (λ_{q-1} + λ_q) · (2 - λ_{q-1} - λ_q) = -8 (λ_q - λ_{q-1}) α(λ_q; D)
# where α(λ; ν) is Wurtz–Love Eq. 45 for ν-regular triangle-free graphs.
#
# Procedure per (D, p):
#   - Outer bisection on T:
#       given T, set τ = T/p, λ_0 = 0
#       for q = 1..p: solve the constraint for λ_q ∈ (λ_{q-1}, 1]
#       record λ_p(T)
#       T* := T such that λ_p(T*) = 1 (within tolerance)
#   - With T* and {λ_q}: γ_q = τ(λ_{q-1}+λ_q)/2, β_q = τ(2-λ_{q-1}-λ_q)/2
#   - Evaluate c̃ via basso_expectation_normalized.
#
# Usage: julia --project=. -t auto scripts/q15_cd_qaoa_tier2.jl

using QaoaXorsat, Printf, Dates

const k = 2
const clause_sign = -1
const ROOT = joinpath(@__DIR__, "..")

# Wurtz–Love Eq. 45: α(λ; ν) for ν-regular graph with no triangles.
function alpha_wurtz_love(λ::Float64, ν::Int)
    one_mλ = 1 - λ
    num = -32 * one_mλ^2 - 8 * (3ν - 2) * λ^2
    den_term1 = (one_mλ^2 + 4 * (3ν - 2) * λ^2)^2
    den = 256 * den_term1 + 256 * λ^2 * one_mλ^2 * (ν - 1) + 96 * (ν - 1) * (ν - 2) * λ^4
    return num / den
end

# Constraint residual for fixed λ_prev, τ, D as a function of λ_q.
# F(u) = (λ_prev + u)(2 - λ_prev - u) τ² + 8 (u - λ_prev) α(u; D)
# (rewritten so RHS = 0 means the s̄_q = 0 condition).
function constraint_residual(u::Float64, λ_prev::Float64, τ::Float64, D::Int)
    return (λ_prev + u) * (2 - λ_prev - u) * τ^2 + 8 * (u - λ_prev) * alpha_wurtz_love(u, D)
end

# Bisection over u ∈ (λ_prev, u_hi] to find the smallest root > λ_prev.
function solve_step(λ_prev::Float64, τ::Float64, D::Int; u_hi::Float64=1.0,
                   tol::Float64=1e-12, maxiter::Int=200)
    f_prev = constraint_residual(λ_prev, λ_prev, τ, D)
    # f at u = λ_prev: (2λ_prev)(2 - 2λ_prev) τ² ≥ 0.  Equals 0 only at boundary.
    # As u increases, the first term decreases (eventually negative around u=1)
    # and the second term is negative (α<0, u>λ_prev).
    f_hi = constraint_residual(u_hi, λ_prev, τ, D)
    if f_hi >= 0
        # No sign change in [λ_prev, u_hi]; cannot reach λ_p = 1.
        return NaN
    end
    if f_prev <= 0
        # Degenerate: at λ_prev itself residual is non-positive.  Step is zero.
        return λ_prev
    end
    a, b = λ_prev, u_hi
    fa, fb = f_prev, f_hi
    for _ in 1:maxiter
        m = 0.5 * (a + b)
        fm = constraint_residual(m, λ_prev, τ, D)
        if abs(fm) < tol || (b - a) < tol
            return m
        end
        if fa * fm < 0
            b, fb = m, fm
        else
            a, fa = m, fm
        end
    end
    return 0.5 * (a + b)
end

# Forward sweep: given T, return λ_p (or NaN if any step fails).
function forward_sweep(T::Float64, p::Int, D::Int)
    τ = T / p
    λ = zeros(Float64, p + 1)
    for q in 1:p
        λ_q = solve_step(λ[q], τ, D; u_hi=1.0)
        if isnan(λ_q) || λ_q <= λ[q]
            return NaN, λ
        end
        λ[q + 1] = min(λ_q, 1.0)
        if λ[q + 1] >= 1.0
            # We hit the boundary; record and continue with frozen value
            λ[q + 1] = 1.0
        end
    end
    return λ[p + 1], λ
end

# Find T* such that λ_p(T*) = 1.  Monotone increasing in T.
function find_T_star(p::Int, D::Int; tol::Float64=1e-8, maxiter::Int=80)
    # Lower bound: τ² ≥ 1/2 ⇒ T ≥ p/√2 (from α(0) = -1/8 step-1 root condition)
    T_lo = p / sqrt(2) + 1e-3
    # Bracket: scan upward.
    λ_lo, _ = forward_sweep(T_lo, p, D)
    if !isnan(λ_lo) && λ_lo >= 1.0
        # Even at the smallest viable T we overshoot — try smaller.
        # Fallback: start from a tiny T and bracket up.
        T_lo = 0.01
    end
    # Find T_hi such that λ_p(T_hi) ≥ 1.
    T_hi = max(T_lo * 2, 1.0)
    for _ in 1:60
        λ_hi, _ = forward_sweep(T_hi, p, D)
        if !isnan(λ_hi) && λ_hi >= 1.0
            break
        end
        T_hi *= 1.5
        if T_hi > 1e4
            return NaN, Float64[]
        end
    end
    # Bisect.
    for _ in 1:maxiter
        T_mid = 0.5 * (T_lo + T_hi)
        λ_mid, λ_arr = forward_sweep(T_mid, p, D)
        if isnan(λ_mid)
            T_lo = T_mid
            continue
        end
        if abs(λ_mid - 1.0) < tol
            return T_mid, λ_arr
        end
        if λ_mid < 1.0
            T_lo = T_mid
        else
            T_hi = T_mid
        end
    end
    T_final = 0.5 * (T_lo + T_hi)
    _, λ_final = forward_sweep(T_final, p, D)
    return T_final, λ_final
end

function tier2_angles(T_star::Float64, λ::Vector{Float64}, p::Int)
    τ = T_star / p
    γ = [τ * (λ[q] + λ[q + 1]) / 2 for q in 1:p]
    β = [τ * (2 - λ[q] - λ[q + 1]) / 2 for q in 1:p]
    return γ, β
end

function get_warm_start(file, k, D, p)
    best_val = -Inf
    isfile(file) || return best_val
    for line in eachline(file)
        startswith(line, '#') && continue
        startswith(line, "k,") && continue
        fields = split(line, ',')
        length(fields) >= 4 || continue
        lk = tryparse(Int, fields[1]); lk === nothing && continue
        lD = tryparse(Int, fields[2]); lD === nothing && continue
        lp = tryparse(Int, fields[3]); lp === nothing && continue
        lv = tryparse(Float64, fields[4]); lv === nothing && continue
        (lk == k && lD == D && lp == p) || continue
        lv > best_val && (best_val = lv)
    end
    return best_val
end

println("╔══════════════════════════════════════════════════════════╗")
println("║  Q1.5 Tier 2: Non-linear CD-QAOA (Wurtz–Love Eqs 29+45) ║")
println("║  $(now())                            ║")
println("╚══════════════════════════════════════════════════════════╝")

output_file = joinpath(ROOT, "results", "q15-cd-qaoa-tier2.csv")
mkpath(dirname(output_file))
open(output_file, "w") do io
    println(io, "# Q1.5 Tier 2: non-linear CD-QAOA — $(now())")
    println(io, "D,p,T_star,lambda_p_residual,ctilde_tier2,ctilde_warm,gap,lambda_nodes,gamma,beta")
end

D_range = 3:8
p_range = 1:12

for D in D_range
    warm_file = joinpath(ROOT, "results", "maxcut-k2-d$(D)-sweep.csv")
    println("\n━━━ D=$D ━━━")
    @printf("  %-4s  %-10s  %-12s  %-14s  %-14s  %-10s\n",
            "p", "T*", "λ_p err", "c̃(tier2)", "c̃(warm)", "gap")
    @printf("  %-4s  %-10s  %-12s  %-14s  %-14s  %-10s\n",
            "──", "──────────", "────────────", "──────────────", "──────────────", "──────────")

    for p in p_range
        params = TreeParams(k, D, p)
        c_warm = get_warm_start(warm_file, k, D, p)

        T_star, λ_arr = find_T_star(p, D)
        if isnan(T_star) || isempty(λ_arr) || length(λ_arr) != p + 1
            @printf("  p=%-2d  FAILED to bracket T*\n", p)
            continue
        end
        λ_residual = abs(λ_arr[end] - 1.0)
        γ, β = tier2_angles(T_star, λ_arr, p)
        c_tier2 = basso_expectation_normalized(params, QAOAAngles(γ, β); clause_sign)
        gap = c_warm > -Inf ? c_warm - c_tier2 : NaN

        @printf("  p=%-2d  %.6f  %.4e  %.10f  %.10f  %+.2e\n",
                p, T_star, λ_residual, c_tier2, c_warm, gap)

        open(output_file, "a") do io
            λstr = join((@sprintf("%.10f", x) for x in λ_arr), ';')
            γstr = join((@sprintf("%.10f", x) for x in γ), ';')
            βstr = join((@sprintf("%.10f", x) for x in β), ';')
            cw = c_warm > -Inf ? @sprintf("%.12f", c_warm) : "NaN"
            gp = isnan(gap) ? "NaN" : @sprintf("%+.6e", gap)
            @printf(io, "%d,%d,%.10f,%.6e,%.12f,%s,%s,%s,%s,%s\n",
                    D, p, T_star, λ_residual, c_tier2, cw, gp, λstr, γstr, βstr)
        end
        flush(stdout)
    end
end

println("\nResults: $output_file")
