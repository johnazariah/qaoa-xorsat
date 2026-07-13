#!/usr/bin/env julia
#
# Stephen's assignment (2026-07-01 meeting): QAOA on a random 3-regular
# antiferromagnetic Heisenberg model with the transverse-field (sum X) driver,
# looking for UNIVERSAL optimal-angle curves.
#
# Uses the matrix-free Schrodinger evolver (src/sparse_qaoa.jl): exact
# e^{-i gamma H_C} via RK4, full 2^N statevector, any edge list, any mixer.
# This is the brute-force primary engine of the Non-variational Eigensolver plan
# (.project/SPEC-nve.md in the research repo).
#
# What it does, for each random 3-regular graph instance:
#   INTERP (layer-by-layer) optimisation of the depth-p Heisenberg QAOA energy
#   density for p = 1..P_MAX, recording the optimal (gamma_l, beta_l) curves.
#   Running several instances lets us eyeball BOTH universality notions we can
#   reach at fixed N: across depth p (smooth curve in l), and "sideways" across
#   instances (do the curves cluster?).
#
# Feasibility (full 2^N ComplexF64 statevector):
#   N=10 -> 16 KB, N=16 -> 1 MB, N=20 -> 16 MB, N=24 -> 268 MB, N=28 -> 4.3 GB.
#   Depth p is cheap; N is the wall. Exact ground-state reference only for N<=12
#   (dense diagonalisation). See SPEC-nve.md sec 10.
#
# WARNING: a full sweep does thousands of statevector evaluations. N<=16 is fine
#   inline; N>=20 should be launched deliberately (background / node) with logged
#   output, per repo experiment protocol.
#
# Usage:
#   julia --project=. scripts/nve_heisenberg_3regular.jl [N] [P_MAX] [N_INSTANCES] [LABEL] [SEED]
#     N            number of qubits, even         (default 10)
#     P_MAX        maximum depth                  (default 4)
#     N_INSTANCES  random 3-regular graphs        (default 3)
#     LABEL        iso | xxz | ising              (default iso)
#     SEED         RNG seed                       (default 2026)
#
# Output: results/nve-3regular-<label>-N<N>.csv

using QaoaXorsat
using Optim
using Printf
using Random
using Dates
using LinearAlgebra

N           = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 10
p_max       = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 4
n_instances = length(ARGS) ≥ 3 ? parse(Int, ARGS[3]) : 3
label       = length(ARGS) ≥ 4 ? ARGS[4] : "iso"
seed        = length(ARGS) ≥ 5 ? parse(Int, ARGS[5]) : 2026

const COUPLINGS = Dict(
    "iso"   => HeisenbergCouplings(1.0, 1.0, 1.0),   # isotropic AFM Heisenberg
    "xxz"   => HeisenbergCouplings(1.0, 1.0, 0.5),   # XXZ (U(1) symmetric), AFM
    "ising" => HeisenbergCouplings(0.0, 0.0, 1.0),   # AFM Ising / MaxCut limit
)
haskey(COUPLINGS, label) || error("LABEL must be one of: iso, xxz, ising")
iseven(N) || error("a 3-regular graph needs an even number of vertices (got N=$N)")
N ≥ 4 || error("need N >= 4")
N ≤ 26 || error("N=$N exceeds the inline statevector ceiling; launch on a node")

const J = COUPLINGS[label]

# ── Random 3-regular graph (configuration / pairing model with rejection) ─────

"""Return the edge list of a random simple 3-regular graph on `N` vertices."""
function random_3regular_edges(N::Int, rng::AbstractRNG; max_tries::Int=10_000)
    for _ in 1:max_tries
        stubs = shuffle!(rng, repeat(1:N, inner=3))   # 3 stubs per vertex
        edges = Tuple{Int,Int}[]
        seen = Set{Tuple{Int,Int}}()
        ok = true
        for i in 1:2:length(stubs)
            a, b = stubs[i], stubs[i+1]
            a == b && (ok = false; break)               # reject self-loop
            e = a < b ? (a, b) : (b, a)
            e in seen && (ok = false; break)            # reject multi-edge
            push!(seen, e); push!(edges, e)
        end
        ok && return edges
    end
    error("failed to build a simple 3-regular graph on N=$N after $max_tries tries")
end

# ── INTERP helpers ────────────────────────────────────────────────────────────

"""Linearly resample a length-`n` schedule to length `m` (INTERP seed)."""
function resample(v::Vector{Float64}, m::Int)
    n = length(v)
    n == m && return copy(v)
    n == 1 && return fill(v[1], m)
    out = Vector{Float64}(undef, m)
    for i in 1:m
        x = m == 1 ? 0.0 : (i - 1) / (m - 1)
        t = x * (n - 1)
        lo = clamp(floor(Int, t), 0, n - 2)
        frac = t - lo
        out[i] = (1 - frac) * v[lo+1] + frac * v[lo+2]
    end
    out
end

"""Bounded random start: γ ∈ [0, 2π), β ∈ [0, π) (X-mixer period is π)."""
random_start(rng, p::Int) = vcat(2π .* rand(rng, p), π .* rand(rng, p))

"""Energy-density objective (large-N RK4 fallback) for flat `x = [γ; β]`."""
function make_objective_rk4(cost_terms, mixer_terms, n_edges::Int, p::Int)
    function (x::Vector{Float64})
        angles = QAOAAngles(x[1:p], x[p+1:2p])
        sparse_qaoa_energy(N, cost_terms, mixer_terms, angles) / n_edges
    end
end

# For small N we diagonalise the cost and mixer ONCE and apply the EXACT layer
# exponentials e^{-iθH} = Q e^{-iθΛ} Q† (no RK4, no integration error). This is
# fast and exact up to N≈12; beyond that dense diagonalisation is infeasible and
# the run must move to a node with the iterative RK4 engine.
const MAX_DENSE = 12

"""Dense Hermitian matrix of a Pauli-sum, built column-by-column, matrix-free."""
function dense_hamiltonian(terms, N::Int)
    dim = 2^N
    H = zeros(ComplexF64, dim, dim)
    e = zeros(ComplexF64, dim); col = similar(e)
    for j in 1:dim
        fill!(e, 0); e[j] = 1
        QaoaXorsat.apply_terms!(col, terms, e, N)
        @views H[:, j] .= col
    end
    Hermitian(H)
end

"""Exact exponential applier e^{-iθH} via a cached eigendecomposition of H."""
struct ExpApplier
    Q::Matrix{ComplexF64}
    λ::Vector{Float64}
end
ExpApplier(H) = (F = eigen(H); ExpApplier(Matrix{ComplexF64}(F.vectors), Vector{Float64}(F.values)))
apply_exp(a::ExpApplier, θ::Real, ψ::AbstractVector) = a.Q * (cis.(-θ .* a.λ) .* (a.Q' * ψ))
expval(a::ExpApplier, ψ::AbstractVector) = real(sum(a.λ .* abs2.(a.Q' * ψ)))

"""|+⟩^N as a dense statevector."""
plus_state_dense(N::Int) = fill(ComplexF64(1 / sqrt(2.0^N)), 2^N)

"""Exact energy-density objective for flat `x = [γ; β]` (small-N dense path)."""
function make_objective_dense(cost::ExpApplier, mix::ExpApplier, ψ0, n_edges::Int, p::Int)
    function (x::Vector{Float64})
        ψ = ψ0
        @inbounds for l in 1:p
            ψ = apply_exp(cost, x[l], ψ)
            ψ = apply_exp(mix, x[p+l], ψ)
        end
        expval(cost, ψ) / n_edges
    end
end

"""Optimise `2p` angles from an INTERP seed plus random restarts; return (x, ε)."""
function optimise_depth(f, p, seed_x, restarts, rng)
    opts = Optim.Options(iterations=2000)
    best_x, best_ε = seed_x, f(seed_x)
    for x0 in vcat([seed_x], [random_start(rng, p) for _ in 1:restarts])
        res = optimize(f, x0, NelderMead(), opts)
        if Optim.minimum(res) < best_ε
            best_ε = Optim.minimum(res)
            best_x = Optim.minimizer(res)
        end
    end
    best_x, best_ε
end

"""|+⟩^N as a dense statevector."""
plus_state_dense(N::Int) = fill(ComplexF64(1 / sqrt(2.0^N)), 2^N)

# ── Sweep ─────────────────────────────────────────────────────────────────────

function run_instance(inst, edges, results_file, rng)
    n_edges = length(edges)
    cost_terms  = heisenberg_terms(edges, J)
    mixer_terms = x_mixer_terms(N)

    if N ≤ MAX_DENSE
        cost = ExpApplier(dense_hamiltonian(cost_terms, N))
        mix  = ExpApplier(dense_hamiltonian(mixer_terms, N))
        ψ0   = plus_state_dense(N)
        gs   = minimum(cost.λ) / n_edges
        objective = p -> make_objective_dense(cost, mix, ψ0, n_edges, p)
    else
        @warn "N=$N > $MAX_DENSE: dense diagonalisation infeasible; using the iterative RK4 engine (slow; intended for a node)."
        gs = NaN
        objective = p -> make_objective_rk4(cost_terms, mixer_terms, n_edges, p)
    end

    @printf(stderr, "\n--- instance %d: N=%d, %d edges, %s J=(%.2f,%.2f,%.2f)%s ---\n",
        inst, N, n_edges, label, J.Jx, J.Jy, J.Jz,
        isnan(gs) ? "" : @sprintf("  exact ε₀=%.6f", gs))

    prev_x = Float64[]
    for p in 1:p_max
        seed_x = isempty(prev_x) ? random_start(rng, 1) :
                 vcat(resample(prev_x[1:p-1], p), resample(prev_x[p:end], p))
        restarts = p == 1 ? 32 : (p == 2 ? 24 : (p == 3 ? 16 : 8))
        f = objective(p)
        t0 = time()
        x, ε = optimise_depth(f, p, seed_x, restarts, rng)
        wall = time() - t0
        prev_x = x
        γ, β = x[1:p], x[p+1:2p]

        @printf(stderr, "  p=%d: ε=%.8f%s (%.1fs)\n", p, ε,
            isnan(gs) ? "" : @sprintf("  (ε-ε₀=%.4f)", ε - gs), wall)
        @printf(stderr, "        γ = [%s]\n", join((@sprintf("%.4f", g) for g in γ), ", "))
        @printf(stderr, "        β = [%s]\n", join((@sprintf("%.4f", b) for b in β), ", "))

        open(results_file, "a") do io
            for l in 1:p
                @printf(io, "%s,%d,%d,%d,%d,%.15f,%.15f,%.15f,%.15f\n",
                    label, N, inst, p, l, γ[l], β[l], ε, gs)
            end
        end
    end
end

rng = MersenneTwister(seed)
results_file = joinpath(@__DIR__, "..", "results", "nve-3regular-$(label)-N$(N).csv")
mkpath(dirname(results_file))
open(results_file, "w") do io
    println(io, "# NVE random 3-regular Heisenberg QAOA (X-mixer) — $(now())")
    println(io, "# N=$N, P_MAX=$p_max, instances=$n_instances, label=$label, seed=$seed")
    println(io, "label,N,instance,p,layer,gamma,beta,epsilon,gs_density")
end

for inst in 1:n_instances
    edges = random_3regular_edges(N, rng)
    run_instance(inst, edges, results_file, rng)
end
@printf(stderr, "\nDone. Results -> %s\n", results_file)
