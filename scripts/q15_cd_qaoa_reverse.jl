#!/usr/bin/env julia
# Q1.5 Tier 2 (REVERSE): Wurtz–Love induced counterdiabatic protocol
# from warm-start QAOA angles.
#
# This is W–L's procedure of §7 Fig. 6 (B → C): given variationally
# optimal angles {γ*, β*}, compute the *induced* continuous-time
# counterdiabatic protocol (λ(t), s(t)) per their Eq. 29.
#
# Per layer q ∈ 1..p:
#   τ_q       = γ_q + β_q                      (step duration)
#   λ̄_q      = γ_q / τ_q                       (mean λ over step)
#   s̄_q      = -γ_q β_q / (2 τ_q)
#               - (λ̄_q - λ̄_{q-1}) / τ_q · α(λ̄_q; D)
#                                              (mean auxiliary field)
# where α(λ; ν) is W–L Eq. 45 for ν-regular triangle-free graphs and
# we substitute ν = D (the infinite tree of degree D is triangle-free).
#
# Interpretation:
#   - If λ̄_q is monotonically increasing in q from ~0 to ~1, and s̄_q
#     stays small (or smoothly varying), then the warm-start QAOA
#     optimum *is* a Trotterised counterdiabatic protocol — W–L wins.
#   - If λ̄_q is non-monotonic or jumps wildly, or s̄_q has large
#     amplitude with sign flips, then there is NO smooth counterdiabatic
#     interpretation of the QAOA optimum — W–L loses on this problem.
#
# Reads warm-start angles from results/maxcut-k2-d{D}-sweep.csv.
#
# Usage: julia --project=. scripts/q15_cd_qaoa_reverse.jl

using QaoaXorsat, Printf, Dates, Statistics

const k = 2
const ROOT = joinpath(@__DIR__, "..")

# Wurtz–Love Eq. 45.
function alpha_wurtz_love(λ::Float64, ν::Int)
    one_mλ = 1 - λ
    num = -32 * one_mλ^2 - 8 * (3ν - 2) * λ^2
    den_term1 = (one_mλ^2 + 4 * (3ν - 2) * λ^2)^2
    den = 256 * den_term1 + 256 * λ^2 * one_mλ^2 * (ν - 1) + 96 * (ν - 1) * (ν - 2) * λ^4
    return num / den
end

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

# Reduce QAOA angles modulo periodicity to [0, π) so that γ_q+β_q is
# meaningful as a physical "time" (β has period π, γ effectively has
# period 2π for k=2 MaxCut but we use π/2 mod π convention).
# We adopt the conservative choice: take absolute values so that
# τ_q = |γ_q| + |β_q|.  This matches W–L's "angle budget" definition.
function reduce_angles(γ::Vector{Float64}, β::Vector{Float64})
    return abs.(γ), abs.(β)
end

function induced_protocol(γ_raw::Vector{Float64}, β_raw::Vector{Float64}, D::Int)
    γ, β = reduce_angles(γ_raw, β_raw)
    p = length(γ)
    τ = γ .+ β
    λ̄ = γ ./ τ
    # cumulative time at end of step q
    t_end = cumsum(τ)
    # mean λ at step q-1, with λ̄_0 = 0 by convention (W–L)
    λ̄_prev = vcat(0.0, λ̄[1:end-1])
    s̄ = similar(λ̄)
    for q in 1:p
        s̄[q] = -γ[q] * β[q] / (2 * τ[q]) -
               (λ̄[q] - λ̄_prev[q]) / τ[q] * alpha_wurtz_love(λ̄[q], D)
    end
    return τ, λ̄, t_end, s̄
end

# Diagnostics for "is this a smooth counterdiabatic protocol?":
#   - monotonicity violations of λ̄_q
#   - sign flips of s̄_q
#   - amplitude of s̄ relative to λ̄
function diagnose(λ̄, s̄)
    p = length(λ̄)
    n_descent = sum(λ̄[q+1] < λ̄[q] - 1e-12 for q in 1:p-1; init=0)
    n_sign_flip_s = 0
    for q in 1:p-1
        if s̄[q] * s̄[q+1] < 0
            n_sign_flip_s += 1
        end
    end
    s_ampl = maximum(abs, s̄)
    s_mean = mean(s̄)
    return (n_descent=n_descent, n_sign_flip_s=n_sign_flip_s,
            s_ampl=s_ampl, s_mean=s_mean)
end

println("╔══════════════════════════════════════════════════════════╗")
println("║  Q1.5 Tier 2 REVERSE: induced (λ̄, s̄) from warm-start    ║")
println("║  $(now())                            ║")
println("╚══════════════════════════════════════════════════════════╝")

output_file = joinpath(ROOT, "results", "q15-cd-qaoa-reverse.csv")
mkpath(dirname(output_file))
open(output_file, "w") do io
    println(io, "# Q1.5 Tier 2 reverse: induced (λ̄, s̄) — $(now())")
    println(io, "D,p,ctilde_warm,n_lambda_descent,n_s_sign_flip,s_amplitude,s_mean,tau_seq,lambdabar_seq,sbar_seq")
end

D_range = 3:8
p_range = 1:12

for D in D_range
    warm_file = joinpath(ROOT, "results", "maxcut-k2-d$(D)-sweep.csv")
    println("\n━━━ D=$D ━━━")
    @printf("  %-4s  %-14s  %-8s  %-8s  %-12s  %-12s\n",
            "p", "c̃(warm)", "↓λ̄", "±s̄ flips", "max|s̄|", "mean(s̄)")
    @printf("  %-4s  %-14s  %-8s  %-8s  %-12s  %-12s\n",
            "──", "──────────────", "────────", "────────", "────────────", "────────────")

    for p in p_range
        c_warm, γ, β = get_warm_start(warm_file, k, D, p)
        c_warm == -Inf && continue
        length(γ) == p || (println("  p=$p: bad warm-start length"); continue)

        τ, λ̄, _, s̄ = induced_protocol(γ, β, D)
        d = diagnose(λ̄, s̄)

        @printf("  p=%-2d  %.10f  %-8d  %-8d  %+.4e  %+.4e\n",
                p, c_warm, d.n_descent, d.n_sign_flip_s, d.s_ampl, d.s_mean)

        open(output_file, "a") do io
            τstr = join((@sprintf("%.10f", x) for x in τ), ';')
            λstr = join((@sprintf("%.10f", x) for x in λ̄), ';')
            sstr = join((@sprintf("%.10e", x) for x in s̄), ';')
            @printf(io, "%d,%d,%.12f,%d,%d,%.6e,%.6e,%s,%s,%s\n",
                    D, p, c_warm, d.n_descent, d.n_sign_flip_s,
                    d.s_ampl, d.s_mean, τstr, λstr, sstr)
        end
    end
end

println("\nResults: $output_file")
