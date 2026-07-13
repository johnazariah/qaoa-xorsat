#!/usr/bin/env julia
#
# Stage 0 smoothness probe for the Heisenberg (XYZ) cost pivot.
#
# Runs an INTERP (layer-by-layer) optimisation of the depth-p Heisenberg QAOA
# energy density on the 1D chain, for p = 1..P_MAX, and records the optimal
# angle curves (gamma_l, beta_l) vs layer l. The point is to eyeball whether the
# curves come out smooth (Stephen's conjecture / Mele et al.). This uses the
# self-contained statevector oracle in src/heisenberg_statevector.jl; no fold
# engine, no cluster.
#
# Feasibility (chain length L = 6p+2): p<=4 fits a laptop (~1 GB at p=4),
# p=5 needs a large-memory node (~68 GB). See .project/SPEC-heisenberg-cost.md.
#
# Usage:
#   julia --project=. scripts/heisenberg_smoothness.jl LABEL P_MAX [SEED]
#     LABEL in {iso, xxz, ising, all}   (default all)
#     P_MAX  maximum depth               (default 4)
#
# Output: results/heisenberg-smoothness-<label>.csv

using QaoaXorsat
using Optim
using Printf
using Random
using Dates

label_arg = length(ARGS) ≥ 1 ? ARGS[1] : "all"
p_max     = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 4
seed      = length(ARGS) ≥ 3 ? parse(Int, ARGS[3]) : 2026

const COUPLINGS = Dict(
    "iso"   => HeisenbergCouplings(1.0, 1.0, 1.0),   # isotropic AFM Heisenberg (entangled ground state)
    "xxz"   => HeisenbergCouplings(1.0, 1.0, 0.5),   # XXZ, U(1) symmetric, antiferromagnetic
    "ising" => HeisenbergCouplings(0.0, 0.0, 1.0),   # AFM Ising / MaxCut limit
)

labels = label_arg == "all" ? ["iso", "xxz", "ising"] : [label_arg]
all(l -> haskey(COUPLINGS, l), labels) ||
    error("LABEL must be one of: iso, xxz, ising, all")

"""Linearly resample a length-`n` schedule to length `m` (INTERP seed)."""
function resample(v::Vector{Float64}, m::Int)
    n = length(v)
    n == m && return copy(v)
    n == 1 && return fill(v[1], m)
    out = Vector{Float64}(undef, m)
    for i in 1:m
        x = m == 1 ? 0.0 : (i - 1) / (m - 1)      # in [0,1]
        t = x * (n - 1)                            # in [0, n-1]
        lo = clamp(floor(Int, t), 0, n - 2)
        frac = t - lo
        out[i] = (1 - frac) * v[lo+1] + frac * v[lo+2]
    end
    out
end

"""Energy-density objective for a flat parameter vector `x = [γ; β]`."""
function make_objective(J::HeisenbergCouplings, p::Int)
    L = exact_chain_length(p)
    function (x::Vector{Float64})
        angles = QAOAAngles(x[1:p], x[p+1:2p])
        heisenberg_energy_density(angles, J; L)
    end
end

"""A bounded random start: γ ∈ [0, 2π), β ∈ [0, π) (mixer period is π)."""
random_start(rng, p::Int) = vcat(2π .* rand(rng, p), π .* rand(rng, p))

"""Optimise `2p` angles from a seed plus random restarts; return (x, ε)."""
function optimise_depth(J, p, seed_x, restarts, rng)
    f = make_objective(J, p)
    opts = Optim.Options(iterations=2000, g_tol=1e-10)
    best_x, best_ε = seed_x, f(seed_x)
    starts = vcat([seed_x], [random_start(rng, p) for _ in 1:restarts])
    for x0 in starts
        res = optimize(f, x0, NelderMead(), opts)
        if Optim.minimum(res) < best_ε
            best_ε = Optim.minimum(res)
            best_x = Optim.minimizer(res)
        end
    end
    best_x, best_ε
end

function run_label(label, J, p_max, rng, results_file)
    @printf(stderr, "\n=== %s  J=(%.2f, %.2f, %.2f)  p=1..%d ===\n",
        label, J.Jx, J.Jy, J.Jz, p_max)
    prev_x = Float64[]
    for p in 1:p_max
        seed_x = isempty(prev_x) ? random_start(rng, 1) :
                 vcat(resample(prev_x[1:p-1], p), resample(prev_x[p:end], p))
        restarts = p == 1 ? 64 : (p == 2 ? 48 : (p == 3 ? 32 : 16))
        t0 = time()
        x, ε = optimise_depth(J, p, seed_x, restarts, rng)
        wall = time() - t0
        prev_x = x
        γ, β = x[1:p], x[p+1:2p]

        @printf(stderr, "  p=%d: ε=%.10f (%.1fs)\n", p, ε, wall)
        @printf(stderr, "        γ = [%s]\n", join((@sprintf("%.4f", g) for g in γ), ", "))
        @printf(stderr, "        β = [%s]\n", join((@sprintf("%.4f", b) for b in β), ", "))

        open(results_file, "a") do io
            for l in 1:p
                @printf(io, "%s,%.4f,%.4f,%.4f,%d,%d,%.15f,%.15f,%.15f\n",
                    label, J.Jx, J.Jy, J.Jz, p, l, γ[l], β[l], ε)
            end
        end
    end
end

rng = MersenneTwister(seed)
for label in labels
    J = COUPLINGS[label]
    results_file = joinpath(@__DIR__, "..", "results", "heisenberg-smoothness-$(label).csv")
    mkpath(dirname(results_file))
    open(results_file, "w") do io
        println(io, "# Heisenberg smoothness sweep ($label) — $(now())")
        println(io, "label,Jx,Jy,Jz,p,layer,gamma,beta,epsilon")
    end
    run_label(label, J, p_max, rng, results_file)
    @printf(stderr, "Done. Results -> %s\n", results_file)
end
