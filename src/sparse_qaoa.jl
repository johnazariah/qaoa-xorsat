"""
Stage 0.5 of the Heisenberg pivot: a matrix-free (sparse) Schrödinger-picture
QAOA simulator that applies each layer by integrating the Schrödinger equation
`d|ψ⟩/ds = -iH|ψ⟩` with RK4 (or Euler), following Stephen's sparse-matrix +
RK4 recipe.

Unlike `heisenberg_statevector.jl` (which hard-codes the Eq. 4.1 gate sequence
on a chain), this works for ANY edge list (chains and random D-regular graphs),
ANY couplings, and ANY of the X / XY / SWAP mixers — no gate decomposition and
no Trotter split. It integrates the *exact* `e^{-iγH_C}` to the chosen accuracy.

Cost: full 2^N statevector, so N ≲ 28 qubits. The Hamiltonians are applied
matrix-free as sums of Pauli-string terms (O(nnz) per mat-vec), so no matrix is
ever stored. See `.project/SPEC-heisenberg-cost.md` (Stage 0.5).
"""

"""
    PauliTerm(kind, i, j, coeff)

A single weighted Pauli term. `kind ∈ (:x, :z, :xx, :yy, :zz)`; `i`, `j` are
1-indexed qubits (`j` ignored for one-body `:x` and `:z` terms); `coeff` is the
real coefficient.
"""
struct PauliTerm
    kind::Symbol
    i::Int
    j::Int
    coeff::Float64
end

x_term(i::Int, coeff::Real=1.0) = PauliTerm(:x, i, i, Float64(coeff))
z_term(i::Int, coeff::Real=1.0) = PauliTerm(:z, i, i, Float64(coeff))
xx_term(i::Int, j::Int, coeff::Real=1.0) = PauliTerm(:xx, i, j, Float64(coeff))
yy_term(i::Int, j::Int, coeff::Real=1.0) = PauliTerm(:yy, i, j, Float64(coeff))
zz_term(i::Int, j::Int, coeff::Real=1.0) = PauliTerm(:zz, i, j, Float64(coeff))

# ── Hamiltonian builders ─────────────────────────────────────────────────────

"""Heisenberg cost `Σ_{⟨ij⟩} (Jx XX + Jy YY + Jz ZZ)` over `edges`."""
function heisenberg_terms(edges, J::HeisenbergCouplings)
    terms = PauliTerm[]
    for e in edges
        iszero(J.Jx) || push!(terms, xx_term(e[1], e[2], J.Jx))
        iszero(J.Jy) || push!(terms, yy_term(e[1], e[2], J.Jy))
        iszero(J.Jz) || push!(terms, zz_term(e[1], e[2], J.Jz))
    end
    terms
end

"""Transverse-field mixer `Σ_j X_j` on `N` qubits."""
x_mixer_terms(N::Int) = [x_term(q) for q in 1:N]

"""XY mixer `Σ_{⟨ij⟩} (XX + YY)` over `edges` (conserves total Sᶻ, U(1))."""
xy_mixer_terms(edges) = vcat(([xx_term(e[1], e[2]), yy_term(e[1], e[2])] for e in edges)...)

"""SWAP/Heisenberg mixer `Σ_{⟨ij⟩} (XX + YY + ZZ)` over `edges` (conserves SU(2))."""
swap_mixer_terms(edges) =
    vcat(([xx_term(e[1], e[2]), yy_term(e[1], e[2]), zz_term(e[1], e[2])] for e in edges)...)

# ── Matrix-free application  out = H·ψ ───────────────────────────────────────

"""
    apply_terms!(out, terms, ψ, N)

Compute `out = H·ψ` matrix-free, where `H = Σ terms`. Cost O(#terms · 2ᴺ).
"""
function apply_terms!(out::Vector{ComplexF64}, terms, ψ::Vector{ComplexF64}, N::Int)
    fill!(out, zero(ComplexF64))
    dim = length(ψ)
    for t in terms
        c = ComplexF64(t.coeff)
        if t.kind === :z
            @inbounds for b in 0:dim-1
                zi = z_eigenvalue((b >> (t.i - 1)) & 1)
                out[b+1] += c * zi * ψ[b+1]
            end
        elseif t.kind === :zz
            @inbounds for b in 0:dim-1
                zi = z_eigenvalue((b >> (t.i - 1)) & 1)
                zj = z_eigenvalue((b >> (t.j - 1)) & 1)
                out[b+1] += c * (zi * zj) * ψ[b+1]
            end
        elseif t.kind === :x
            m = one(Int) << (t.i - 1)
            @inbounds for b in 0:dim-1
                out[b+1] += c * ψ[(b ⊻ m)+1]
            end
        elseif t.kind === :xx
            flip = (one(Int) << (t.i - 1)) | (one(Int) << (t.j - 1))
            @inbounds for b in 0:dim-1
                out[b+1] += c * ψ[(b ⊻ flip)+1]
            end
        elseif t.kind === :yy
            flip = (one(Int) << (t.i - 1)) | (one(Int) << (t.j - 1))
            @inbounds for b in 0:dim-1
                zi = z_eigenvalue((b >> (t.i - 1)) & 1)
                zj = z_eigenvalue((b >> (t.j - 1)) & 1)
                # (Yᵢ Yⱼ ψ)[b] = -zᵢ zⱼ · ψ[b ⊕ flip]
                out[b+1] += c * (-(zi * zj)) * ψ[(b ⊻ flip)+1]
            end
        else
            throw(ArgumentError("unknown PauliTerm kind :$(t.kind)"))
        end
    end
    out
end

"""`⟨ψ|H|ψ⟩` (real) for `H = Σ terms`."""
function hamiltonian_expectation(terms, ψ::Vector{ComplexF64}, N::Int)
    Hψ = similar(ψ)
    apply_terms!(Hψ, terms, ψ, N)
    acc = zero(ComplexF64)
    @inbounds for k in eachindex(ψ)
        acc += conj(ψ[k]) * Hψ[k]
    end
    real(acc)
end

# ── Time integrators for  ψ ← e^{-iθH} ψ  ────────────────────────────────────

"""Conservative default step count so that `Δs · ‖H‖₁ ≲ 0.1` (RK4 accurate to ~1e-4)."""
function _default_steps(terms, θ::Real)
    Hnorm = sum(abs(t.coeff) for t in terms; init=0.0)
    max(32, ceil(Int, abs(θ) * Hnorm / 0.1))
end

"""
    evolve_rk4!(ψ, terms, θ, N; steps)

Apply `e^{-iθH} ψ` in place by 4th-order Runge–Kutta integration of
`dψ/ds = -iHψ` over `s ∈ [0, θ]`. Norm-preserving to O(Δs⁵) per step.
"""
function evolve_rk4!(ψ::Vector{ComplexF64}, terms, θ::Real, N::Int;
    steps::Int=_default_steps(terms, θ))
    iszero(θ) && return ψ
    Δ = Float64(θ) / steps
    k1 = similar(ψ)
    k2 = similar(ψ)
    k3 = similar(ψ)
    k4 = similar(ψ)
    tmp = similar(ψ)
    for _ in 1:steps
        apply_terms!(k1, terms, ψ, N)
        k1 .*= -im
        @. tmp = ψ + (Δ / 2) * k1
        apply_terms!(k2, terms, tmp, N)
        k2 .*= -im
        @. tmp = ψ + (Δ / 2) * k2
        apply_terms!(k3, terms, tmp, N)
        k3 .*= -im
        @. tmp = ψ + Δ * k3
        apply_terms!(k4, terms, tmp, N)
        k4 .*= -im
        @. ψ += (Δ / 6) * (k1 + 2k2 + 2k3 + k4)
    end
    ψ
end

"""
    evolve_euler!(ψ, terms, θ, N; steps)

Apply `e^{-iθH} ψ` by forward Euler (`ψ ← ψ - iΔs·Hψ`). First order and NOT
norm-preserving; included for comparison with RK4. Needs many small steps.
"""
function evolve_euler!(ψ::Vector{ComplexF64}, terms, θ::Real, N::Int;
    steps::Int=_default_steps(terms, θ))
    iszero(θ) && return ψ
    Δ = Float64(θ) / steps
    Hψ = similar(ψ)
    for _ in 1:steps
        apply_terms!(Hψ, terms, ψ, N)
        @. ψ += (-im * Δ) * Hψ
    end
    ψ
end

# ── QAOA driver ──────────────────────────────────────────────────────────────

"""
    sparse_qaoa_state(N, cost_terms, mixer_terms, angles; kwargs...) -> Vector{ComplexF64}

Depth-`p` QAOA state on `N` qubits: start from `|+⟩^{⊗N}` (override with
`initial_state`) and, for each round, apply `e^{-iγ H_cost}` then
`e^{-iβ H_mixer}` by numerical integration.

Keyword args: `cost_steps`, `mixer_steps` (integration steps; auto by default),
`method` (`:rk4` or `:euler`), `initial_state`.
"""
function sparse_qaoa_state(
    N::Int, cost_terms, mixer_terms, angles::QAOAAngles;
    cost_steps::Union{Int,Nothing}=nothing,
    mixer_steps::Union{Int,Nothing}=nothing,
    method::Symbol=:rk4,
    initial_state::Union{Vector{ComplexF64},Nothing}=nothing,
)
    method in (:rk4, :euler) || throw(ArgumentError("method must be :rk4 or :euler"))
    evolve! = method === :rk4 ? evolve_rk4! : evolve_euler!
    ψ = initial_state === nothing ? plus_state(N) : copy(initial_state)
    for r in 1:depth(angles)
        γ = angles.γ[r]
        cs = cost_steps === nothing ? _default_steps(cost_terms, γ) : cost_steps
        evolve!(ψ, cost_terms, γ, N; steps=cs)
        β = angles.β[r]
        ms = mixer_steps === nothing ? _default_steps(mixer_terms, β) : mixer_steps
        evolve!(ψ, mixer_terms, β, N; steps=ms)
    end
    ψ
end

"""
    sparse_qaoa_energy(N, cost_terms, mixer_terms, angles; kwargs...) -> Float64

Total cost energy `⟨H_cost⟩` of the depth-`p` QAOA state. Divide by the number
of edges for an energy density.
"""
function sparse_qaoa_energy(N::Int, cost_terms, mixer_terms, angles::QAOAAngles; kwargs...)
    ψ = sparse_qaoa_state(N, cost_terms, mixer_terms, angles; kwargs...)
    hamiltonian_expectation(cost_terms, ψ, N)
end
