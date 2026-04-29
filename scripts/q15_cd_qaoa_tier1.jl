#!/usr/bin/env julia
# Q1.5 Tier 1: Optimal-T Linear Schedule (Wurtz–Love lowest-order CD-QAOA
# with linear λ(t) = t/T)
#
# At lowest BCH order with λ(t) = t/T linear and uniform τ_q = T/p:
#   γ_q = T(q-1/2)/p²
#   β_q = T(p-q+1/2)/p²
#
# These are LR-QAOA shape; α(λ) does not enter the angles at this order.
# We optimise T to maximise c̃ for each (D, p) cell.
#
# Compares to:
#   - warm-start optimum (from results/maxcut-k2-d{D}-sweep.csv)
#   - linear-adi at warm-start magnitudes (E1 baseline)
#
# Usage: julia --project=. -t auto scripts/q15_cd_qaoa_tier1.jl

using QaoaXorsat, Printf, Dates
using Optim

const k = 2
const clause_sign = -1
const ROOT = joinpath(@__DIR__, "..")

function get_warm_start(file, k, D, p)
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
        (lk == k && lD == D && lp == p) || continue
        if lv > best_val
            best_val = lv
            best_gamma = parse.(Float64, split(fields[6], ';'))
            best_beta = parse.(Float64, split(fields[7], ';'))
        end
    end
    return (best_val, best_gamma, best_beta)
end

# Tier-1 angles for given total time T and depth p.
function tier1_angles(T::Float64, p::Int)
    γ = [T * (q - 0.5) / p^2 for q in 1:p]
    β = [T * (p - q + 0.5) / p^2 for q in 1:p]
    return γ, β
end

# Negative c̃ for minimisation.
function neg_ctilde(T::Float64, params)
    γ, β = tier1_angles(T, params.p)
    return -basso_expectation_normalized(params, QAOAAngles(γ, β); clause_sign)
end

# Grid scan + Brent refine, to avoid Brent local traps on a multi-modal
# function (which the linear schedule c̃(T) appears to be at higher p).
function find_T_star(params; T_min::Float64=0.1, T_max::Float64=20.0,
                     n_grid::Int=40)
    Ts = range(T_min, T_max; length=n_grid)
    vals = Float64[neg_ctilde(T, params) for T in Ts]
    idx = argmin(vals)
    # Refine around idx using Brent within neighbouring grid cells.
    lo = idx > 1 ? Float64(Ts[idx - 1]) : T_min
    hi = idx < length(Ts) ? Float64(Ts[idx + 1]) : T_max
    res = optimize(T -> neg_ctilde(T, params), lo, hi,
                   Brent(); rel_tol=1e-10, abs_tol=1e-10)
    T_star = Optim.minimizer(res)
    c_star = -Optim.minimum(res)
    # Safety: also check the grid best (sometimes Brent worsens).
    c_grid = -vals[idx]
    if c_grid > c_star
        return Float64(Ts[idx]), c_grid
    end
    return T_star, c_star
end

println("╔══════════════════════════════════════════════════════════╗")
println("║  Q1.5 Tier 1: CD-QAOA with optimal T (linear λ)         ║")
println("║  $(now())                            ║")
println("╚══════════════════════════════════════════════════════════╝")

output_file = joinpath(ROOT, "results", "q15-cd-qaoa-tier1.csv")
mkpath(dirname(output_file))
open(output_file, "w") do io
    println(io, "# Q1.5 Tier 1: optimal-T linear CD-QAOA — $(now())")
    println(io, "D,p,T_star,ctilde_tier1,ctilde_warm,gap,gamma,beta")
end

D_range = 3:8
p_range = 1:12

for D in D_range
    warm_file = joinpath(ROOT, "results", "maxcut-k2-d$(D)-sweep.csv")
    println("\n━━━ D=$D ━━━")
    @printf("  %-4s  %-10s  %-14s  %-14s  %-10s\n",
            "p", "T*", "c̃(tier1)", "c̃(warm)", "gap")
    @printf("  %-4s  %-10s  %-14s  %-14s  %-10s\n",
            "──", "──────────", "──────────────", "──────────────", "──────────")

    for p in p_range
        params = TreeParams(k, D, p)
        c_warm, _, _ = get_warm_start(warm_file, k, D, p)

        T_star, c_tier1 = find_T_star(params)
        γ, β = tier1_angles(T_star, p)
        gap = c_warm > -Inf ? c_warm - c_tier1 : NaN

        @printf("  p=%-2d  %.6f  %.10f  %.10f  %+.2e\n",
                p, T_star, c_tier1, c_warm, gap)

        open(output_file, "a") do io
            γstr = join((@sprintf("%.10f", x) for x in γ), ';')
            βstr = join((@sprintf("%.10f", x) for x in β), ';')
            cw = c_warm > -Inf ? @sprintf("%.12f", c_warm) : "NaN"
            gp = isnan(gap) ? "NaN" : @sprintf("%+.6e", gap)
            @printf(io, "%d,%d,%.10f,%.12f,%s,%s,%s,%s\n",
                    D, p, T_star, c_tier1, cw, gp, γstr, βstr)
        end
        flush(stdout)
    end
end

println("\nResults: $output_file")
