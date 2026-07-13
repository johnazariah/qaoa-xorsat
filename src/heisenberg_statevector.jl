"""
Stage 0 of the Heisenberg (XYZ) cost pivot: a self-contained exact statevector
simulator for depth-`p` Heisenberg QAOA on a 1D open chain.

This deliberately does NOT use the Basso fold engine. It is a small, obviously
correct oracle used to (a) test the reduction in
`.project/learning/23-heisenberg-cost-derivation.md` §4, and (b) probe whether
the optimal angle curves come out smooth. See
`.project/SPEC-heisenberg-cost.md` for the full plan.

Model: cost `H_C = Σ_{⟨ij⟩} (Jx XᵢXⱼ + Jy YᵢYⱼ + Jz ZᵢZⱼ)`, mixer `B = Σⱼ Xⱼ`,
reference state `|+⟩^{⊗L}`. Each cost layer is the first-order Pauli split
(Eq. 4.1): three rotated `ZZ` phase layers wrapped in fixed Clifford dressing.
"""

"""
    HeisenbergCouplings(Jx, Jy, Jz)

XYZ Heisenberg edge couplings. `(0, 0, Jz)` is the Ising/MaxCut limit;
`(J, J, J)` is the isotropic point; `(J, J, Δ)` is XXZ.
"""
struct HeisenbergCouplings
    Jx::Float64
    Jy::Float64
    Jz::Float64
end

HeisenbergCouplings(; Jx=0.0, Jy=0.0, Jz=0.0) = HeisenbergCouplings(Jx, Jy, Jz)

"""Nearest-neighbour edges of an open chain of `L` sites: `[1,2], [2,3], …`."""
chain_edges(L::Int) = [[i, i + 1] for i in 1:(L-1)]

"""Central edge `(L÷2, L÷2 + 1)` of an `L`-site chain, as the observable edge."""
central_edge(L::Int) = (L ÷ 2, L ÷ 2 + 1)

"""Chain length that makes the depth-`3p` causal cone exact: `L = 6p + 2`."""
exact_chain_length(p::Int) = 6p + 2

# ── Single-qubit gate primitives ─────────────────────────────────────────────

const _INV_SQRT2 = inv(sqrt(2.0))

# 2×2 gates returned as (u11, u12, u21, u22).
_hadamard() = (ComplexF64(_INV_SQRT2), ComplexF64(_INV_SQRT2),
    ComplexF64(_INV_SQRT2), ComplexF64(-_INV_SQRT2))
# R_Y = S·H with S = diag(1, i):     (1/√2)[[1, 1], [i, -i]]
_ry() = (ComplexF64(_INV_SQRT2), ComplexF64(_INV_SQRT2),
    ComplexF64(0.0, _INV_SQRT2), ComplexF64(0.0, -_INV_SQRT2))
# R_Y† = H·S† :                       (1/√2)[[1, -i], [1, i]]
_ry_dag() = (ComplexF64(_INV_SQRT2), ComplexF64(0.0, -_INV_SQRT2),
    ComplexF64(_INV_SQRT2), ComplexF64(0.0, _INV_SQRT2))

"""
    apply_single_qubit!(state, u11, u12, u21, u22, qubit)

Apply the 2×2 gate `[u11 u12; u21 u22]` to `qubit` (1-indexed) of `state`.
"""
function apply_single_qubit!(
    state::Vector{ComplexF64},
    u11::ComplexF64, u12::ComplexF64, u21::ComplexF64, u22::ComplexF64,
    qubit::Int,
)
    mask = one(Int) << (qubit - 1)
    @inbounds for base in 0:length(state)-1
        iszero(base & mask) || continue
        i0 = base + 1
        i1 = (base | mask) + 1
        a0 = state[i0]
        a1 = state[i1]
        state[i0] = u11 * a0 + u12 * a1
        state[i1] = u21 * a0 + u22 * a1
    end
    state
end

"""Apply the same single-qubit gate to every qubit `1:L`."""
function apply_single_qubit_all!(
    state::Vector{ComplexF64},
    gate::NTuple{4,ComplexF64},
    L::Int,
)
    for qubit in 1:L
        apply_single_qubit!(state, gate..., qubit)
    end
    state
end

# ── Two-qubit ZZ phase layer ─────────────────────────────────────────────────

"""
    apply_zz_layer!(state, edges, θ)

Apply the diagonal phase `exp(-iθ Σ_{⟨ij⟩} ZᵢZⱼ)` over `edges`.
"""
function apply_zz_layer!(state::Vector{ComplexF64}, edges, θ::Real)
    iszero(θ) && return state
    θf = Float64(θ)
    @inbounds for base in 0:length(state)-1
        spin_sum = 0
        for edge in edges
            zi = z_eigenvalue((base >> (edge[1] - 1)) & 1)
            zj = z_eigenvalue((base >> (edge[2] - 1)) & 1)
            spin_sum += zi * zj
        end
        state[base+1] *= cis(-θf * spin_sum)
    end
    state
end

# ── Heisenberg cost layer (Eq. 4.1) and the full circuit ─────────────────────

"""
    apply_heisenberg_cost_layer!(state, edges, L, γ, J)

Apply one first-order Pauli-split Heisenberg cost layer (doc 23, Eq. 4.1):

`R_X U_ZZ(γJx) R_X† · R_Y U_ZZ(γJy) R_Y† · U_ZZ(γJz)`

with `R_X = ∏ H`, `R_Y = ∏ SH`. Operators act right-to-left. When
`Jx = Jy = 0` the Clifford dressing cancels and this reduces to `U_ZZ(γJz)`.
"""
function apply_heisenberg_cost_layer!(
    state::Vector{ComplexF64},
    edges,
    L::Int,
    γ::Real,
    J::HeisenbergCouplings,
)
    H = _hadamard()
    RY = _ry()
    RYd = _ry_dag()

    apply_zz_layer!(state, edges, γ * J.Jz)          # U_ZZ(γ Jz)
    apply_single_qubit_all!(state, RYd, L)           # R_Y†
    apply_zz_layer!(state, edges, γ * J.Jy)          # U_ZZ(γ Jy)
    apply_single_qubit_all!(state, RY, L)            # R_Y
    apply_single_qubit_all!(state, H, L)             # R_X†  (= H)
    apply_zz_layer!(state, edges, γ * J.Jx)          # U_ZZ(γ Jx)
    apply_single_qubit_all!(state, H, L)             # R_X   (= H)
    state
end

"""
    heisenberg_chain_state(angles, J; L=exact_chain_length(depth(angles)))

Prepare the depth-`p` Heisenberg QAOA state on an `L`-site open chain, starting
from `|+⟩^{⊗L}` and applying `p` rounds of `[cost layer, mixer]`.

The default `L = 6p + 2` makes the central-edge observable exact (the depth-`3p`
causal cone does not reach the boundary).
"""
function heisenberg_chain_state(
    angles::QAOAAngles,
    J::HeisenbergCouplings;
    L::Int=exact_chain_length(depth(angles)),
)
    p = depth(angles)
    L ≥ 2 || throw(ArgumentError("chain needs L ≥ 2, got $L"))
    L ≥ exact_chain_length(p) || throw(ArgumentError(
        "L=$L is too short for exact depth-$p evaluation; need L ≥ $(exact_chain_length(p))"))

    edges = chain_edges(L)
    state = plus_state(L)
    for round in 1:p
        apply_heisenberg_cost_layer!(state, edges, L, angles.γ[round], J)
        apply_mixer_layer!(state, angles.β[round], L)
    end
    state
end

# ── Edge two-point correlators ───────────────────────────────────────────────

"""`⟨Zₐ Z_b⟩` for a normalised `state`."""
function edge_zz(state::Vector{ComplexF64}, a::Int, b::Int)
    total = 0.0
    @inbounds for base in 0:length(state)-1
        za = z_eigenvalue((base >> (a - 1)) & 1)
        zb = z_eigenvalue((base >> (b - 1)) & 1)
        total += abs2(state[base+1]) * za * zb
    end
    total
end

"""`⟨Xₐ X_b⟩` for a normalised `state`."""
function edge_xx(state::Vector{ComplexF64}, a::Int, b::Int)
    flip = (one(Int) << (a - 1)) | (one(Int) << (b - 1))
    acc = ComplexF64(0)
    @inbounds for base in 0:length(state)-1
        acc += conj(state[base+1]) * state[(base⊻flip)+1]
    end
    real(acc)
end

"""`⟨Yₐ Y_b⟩` for a normalised `state`."""
function edge_yy(state::Vector{ComplexF64}, a::Int, b::Int)
    flip = (one(Int) << (a - 1)) | (one(Int) << (b - 1))
    acc = ComplexF64(0)
    @inbounds for base in 0:length(state)-1
        za = z_eigenvalue((base >> (a - 1)) & 1)
        zb = z_eigenvalue((base >> (b - 1)) & 1)
        # Yₐ Y_b |base⟩ = -zₐ z_b |base ⊕ flip⟩
        acc += conj(state[base+1]) * (-(za * zb)) * state[(base⊻flip)+1]
    end
    real(acc)
end

"""
    heisenberg_edge_correlators(angles, J; L=…) -> (xx, yy, zz)

The central-edge correlators `(⟨XX⟩, ⟨YY⟩, ⟨ZZ⟩)` of the depth-`p` Heisenberg
QAOA state.
"""
function heisenberg_edge_correlators(
    angles::QAOAAngles,
    J::HeisenbergCouplings;
    L::Int=exact_chain_length(depth(angles)),
)
    state = heisenberg_chain_state(angles, J; L)
    a, b = central_edge(L)
    (edge_xx(state, a, b), edge_yy(state, a, b), edge_zz(state, a, b))
end

"""
    heisenberg_energy_density(angles, J; L=…) -> Float64

Per-edge energy density `ε = Jx⟨XX⟩ + Jy⟨YY⟩ + Jz⟨ZZ⟩` on the central edge,
i.e. the QAOA objective ⟨H_C⟩/M for the depth-`p` Heisenberg ansatz.
"""
function heisenberg_energy_density(
    angles::QAOAAngles,
    J::HeisenbergCouplings;
    L::Int=exact_chain_length(depth(angles)),
)
    xx, yy, zz = heisenberg_edge_correlators(angles, J; L)
    J.Jx * xx + J.Jy * yy + J.Jz * zz
end
