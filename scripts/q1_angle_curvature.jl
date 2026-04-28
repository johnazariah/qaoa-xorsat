#!/usr/bin/env julia
# Q1 Experiment 4: Angle-profile curvature analysis
#
# A linear adiabatic schedule predicts γ_j ∝ j/p and β_j ∝ 1 - j/p in the
# continuum limit. We fit polynomials of degree 1, 2, 3 to the optimal QAOA
# angle profiles γ*(j/p), β*(j/p) and report:
#
#   - r²(deg=1): how much variance is captured by a *linear* fit
#   - rmse(deg=1): the linear-fit residual magnitude
#   - quadratic_curvature: the quadratic coefficient (sign + magnitude)
#   - cubic_curvature:    the cubic coefficient
#   - improvement Δr² from deg-1 → deg-3
#
# Large quadratic / cubic coefficients and a poor linear fit are evidence
# that the optimal QAOA schedule is *not* a Trotterised linear adiabatic.
#
# Usage: julia --project=. scripts/q1_angle_curvature.jl

using Printf, Dates

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
const OUT_FILE    = joinpath(RESULTS_DIR, "q1-angle-curvature.csv")

# ── Helpers ─────────────────────────────────────────────────────────────

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

# QAOA angles live on a torus: γ has period 2π, β has period π. The CSVs
# store canonicalised angles in [0, 2π) and [0, π); successive values that
# straddle a wrap appear as discontinuities. We "unwrap" by adding ±period
# whenever the next angle jumps by more than period/2, recovering the
# smooth schedule shape that's relevant for curvature analysis.
function unwrap_periodic(y::AbstractVector{<:Real}, period::Real)
    isempty(y) && return Float64[]
    out = Float64[float(y[1])]
    for j in 2:length(y)
        prev = out[end]
        cand = float(y[j])
        # bring cand within (prev - period/2, prev + period/2]
        while cand - prev > period / 2;  cand -= period; end
        while cand - prev ≤ -period / 2; cand += period; end
        push!(out, cand)
    end
    return out
end

# Vandermonde-style least-squares polynomial fit; returns the coefficients
# c so that y ≈ c[1] + c[2]·x + c[3]·x² + … + c[deg+1]·x^deg.
function polyfit(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, deg::Int)
    n = length(x)
    @assert n == length(y) "x and y must have equal length"
    @assert n ≥ deg + 1 "need ≥ deg+1 points"
    V = [xi^j for xi in x, j in 0:deg]
    return V \ collect(y)
end

polyeval(c::AbstractVector{<:Real}, x::Real) =
    sum(c[i+1] * x^i for i in 0:(length(c) - 1))

function fit_quality(x, y, deg)
    c    = polyfit(x, y, deg)
    yhat = [polyeval(c, xi) for xi in x]
    res  = y .- yhat
    ss_res = sum(abs2, res)
    ȳ      = sum(y) / length(y)
    ss_tot = sum(abs2, y .- ȳ)
    r²     = ss_tot > 0 ? 1 - ss_res / ss_tot : 1.0
    rmse   = sqrt(ss_res / length(y))
    return (coeffs = c, r² = r², rmse = rmse)
end

# ── Main ────────────────────────────────────────────────────────────────

function main()
    println("╔══════════════════════════════════════════════════════════╗")
    println("║  Q1 Experiment 4: Angle-Profile Curvature Analysis     ║")
    println("║  $(now())                            ║")
    println("╚══════════════════════════════════════════════════════════╝")
    println()

    mkpath(dirname(OUT_FILE))
    open(OUT_FILE, "w") do io
        println(io, "# Q1: Angle profile curvature — $(now())")
        println(io, "# Fits γ_j and β_j (j=1..p) against x = j/p with polynomials of deg 1,2,3.")
        println(io, "k,D,p,profile,deg,r2,rmse,c0,c1,c2,c3")
    end

    k = 2
    summary_rows = Tuple{Int,Int,Int,Float64,Float64,Float64,Float64}[]

    for D in 3:8
        file  = joinpath(RESULTS_DIR, "maxcut-k2-d$(D)-sweep.csv")
        p_max = max_p_in_csv(file, k, D)
        p_max < 4 && continue

        _, γ, β = load_best_angles_for_p(file, k, D, p_max)
        # Unwrap before fitting so the polynomial captures shape, not
        # mod-2π / mod-π wraps.
        γ = unwrap_periodic(γ, 2π)
        β = unwrap_periodic(β, π)
        x = [j / p_max for j in 1:p_max]

        @printf("━━━ D=%d, p_max=%d ━━━\n", D, p_max)
        @printf("  %-7s  %-4s  %-10s  %-10s  %-10s  %-10s  %-10s\n",
                "profile", "deg", "r²", "rmse", "c0", "c1", "c2/c3")
        @printf("  %-7s  %-4s  %-10s  %-10s  %-10s  %-10s  %-10s\n",
                "───────", "───", "──────────", "──────────", "──────────",
                "──────────", "──────────")

        for (label, y) in (("gamma", γ), ("beta", β))
            r2_lin = NaN
            r2_cub = NaN
            quad_c = NaN
            cubic_c = NaN
            for deg in 1:3
                fit = fit_quality(x, y, deg)
                pad = vcat(fit.coeffs, fill(NaN, 4 - length(fit.coeffs)))
                @printf("  %-7s  %-4d  %.6f  %.6f  %+.5f  %+.5f  %+.5f\n",
                        label, deg, fit.r², fit.rmse, pad[1], pad[2],
                        deg == 1 ? NaN : pad[3])
                open(OUT_FILE, "a") do io
                    @printf(io, "%d,%d,%d,%s,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n",
                            k, D, p_max, label, deg, fit.r², fit.rmse,
                            pad[1], pad[2], pad[3], pad[4])
                end
                deg == 1 && (r2_lin  = fit.r²)
                deg == 2 && (quad_c  = fit.coeffs[3])
                deg == 3 && begin
                    r2_cub  = fit.r²
                    cubic_c = fit.coeffs[4]
                end
            end
            push!(summary_rows, (D, p_max, label == "gamma" ? 1 : 2,
                                 r2_lin, r2_cub - r2_lin, quad_c, cubic_c))
        end
        println()
    end

    println("━━━ Summary ━━━")
    @printf("  %-3s  %-5s  %-7s  %-10s  %-10s  %-10s  %-10s\n",
            "D", "p", "profile", "r²(lin)", "Δr²(→cubic)", "quad coef", "cubic coef")
    for r in summary_rows
        D, p, prof, r2_lin, dr2, qc, cc = r
        @printf("  %-3d  %-5d  %-7s  %.6f  %+.6f      %+.5f    %+.5f\n",
                D, p, prof == 1 ? "gamma" : "beta", r2_lin, dr2, qc, cc)
    end
    println()
    println("Results written to $OUT_FILE")
end

main()
