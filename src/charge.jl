# Charge decomposition evaluator — O(p·4^p) branch tensor contraction.
#
# Translates JPM's rank-4 charge decomposition (QOKit add-max-k-xor-sat branch)
# into the qaoa-xorsat Julia codebase.  The key idea: decompose the doubled
# density matrix into 4 charge channels via a 2×2 WHT butterfly, fusing the
# mixer into each contraction step.
#
# Compared to the Basso (2p+1)-bit basis evaluator:
#   - Works in 4^p hyperindex space directly (not 2^(2p+1))
#   - Each branch level adds one round: branch grows from 4^ℓ to 4^(ℓ+1)
#   - Total cost: O(p·4^p) vs O(p²·4^p)
#
# The evaluator computes ⟨Z^⊗k⟩ (the parity correlator), same as QOKit's
# contract_symmetric_tree.
#
# ── Layout convention ─────────────────────────────────────────────────────
#
# All flat vectors use C-order (row-major) indexing to match the original
# QOKit implementation.  Julia's column-major `reshape` is corrected with
# `_reshape_c` at the few sites that build or consume multi-dimensional
# tensors.  The multi-dimensional code (mode products, wht_charge_contract)
# remains clean and readable; the layout adapters are isolated.
#
# ── γ convention ──────────────────────────────────────────────────────────
#
# Our codebase uses the γ/2 phase convention (the physical gate is
# exp(-iγ/2 · Z^⊗k)), while the charge primitives assume the full-angle
# convention (exp(-iγ · Z^⊗k)).  The public API `charge_parity_expectation`
# halves γ before passing to the internal routines.

# ──────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ──────────────────────────────────────────────────────────────────────────────

"""C-order (row-major) reshape: last axis varies fastest in memory."""
_reshape_c(A::AbstractVector, dims::Int...) =
    permutedims(reshape(A, reverse(dims)...), length(dims):-1:1)

"""C-order flatten: inverse of `_reshape_c`."""
_vec_c(A::AbstractArray) = vec(permutedims(A, ndims(A):-1:1))

# ──────────────────────────────────────────────────────────────────────────────
# Charge primitives
# ──────────────────────────────────────────────────────────────────────────────

"""
    CHARGE_DIAG

4×4 charge diagonal: `CHARGE_DIAG[a+1, σ+1] = (-1)^{bit pattern}`.

- a=0: identity
- a=1: Z_bra
- a=2: Z_ket
- a=3: Z_ket·Z_bra
"""
const CHARGE_DIAG = [
    1  1  1  1;
    1 -1  1 -1;
    1  1 -1 -1;
    1 -1 -1  1
]

"""
    doubled_mixer(β)

4×4 doubled mixer `M[σ_out, σ_in] = Rx[sk_o,sk_i] ⊗ Rx*[sb_o,sb_i]`.
"""
function doubled_mixer(β::T) where T<:Real
    c = cos(β)
    s = sin(β)
    Rx = Complex{T}[c -im*s; -im*s c]
    kron(Rx, conj.(Rx))
end

"""
    charge_weight_matrix(γ)

4×4 weight matrix `W[h+1, a+1]` for the k-body phase gate charge decomposition.
"""
function charge_weight_matrix(γ::T) where T<:Real
    c = cos(γ)
    s = sin(γ)
    c2 = c * c
    s2 = s * s
    ics = im * c * s
    zk = Complex{T}[1, 1, -1, -1]
    zb = Complex{T}[1, -1, 1, -1]
    hcat(fill(complex(c2), 4), ics .* zb, -ics .* zk, s2 .* zk .* zb)
end

"""
    root_charge_weights(γ)

Root charge weights `u = [cos²γ, i·c·s, -i·c·s, sin²γ]`.
"""
function root_charge_weights(γ::T) where T<:Real
    c = cos(γ)
    s = sin(γ)
    Complex{T}[c*c, im*c*s, -im*c*s, s*s]
end

"""
    wht_charge_contract(M, T_tensor)

WHT butterfly charge contraction for all 4 channels simultaneously.

Given base mixer `M` (4×4) and tensor `T_tensor` of shape `(n, 4, 4, rest)`,
computes `out[a][i, b, r] = Σ_σ CHARGE_DIAG[a,σ] · M[b,σ] · T[i,σ,b,r]`
for all 4 charge channels using a 2×2 WHT butterfly.

Cost: 4 multiplies + 8 adds per element (vs 16 muls + 12 adds naive).

Returns a tuple of 4 arrays, each of shape `(n, 4, rest)`.
"""
function wht_charge_contract(M::AbstractMatrix, T_tensor::AbstractArray{CT,4}) where CT
    n = size(T_tensor, 1)
    rest = size(T_tensor, 4)

    # e[σ][i, b, r] = M[b, σ] · T[i, σ, b, r]
    e = ntuple(4) do s
        M_col = reshape(M[:, s], 1, 4, 1)
        T_slice = reshape(T_tensor[:, s, :, :], n, 4, rest)
        M_col .* T_slice
    end

    # WHT butterfly (H₂ ⊗ H₂)
    p02 = e[1] .+ e[3]
    q02 = e[1] .- e[3]
    p13 = e[2] .+ e[4]
    q13 = e[2] .- e[4]

    (
        p02 .+ p13,  # a=0: +1,+1,+1,+1
        p02 .- p13,  # a=1: +1,-1,+1,-1
        q02 .+ q13,  # a=2: +1,+1,-1,-1
        q02 .- q13,  # a=3: +1,-1,-1,+1
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Branch tensor contraction
# ──────────────────────────────────────────────────────────────────────────────

"""
    _charge_hyperedge_branch(γs, βs, num_rounds, k; child_branch=nothing)

Branch tensor for one hyperedge using charge decomposition.

Returns a flat `Complex{T}` vector of `4^num_rounds` entries in C-order.
"""
function _charge_hyperedge_branch(
    γs::AbstractVector{T},
    βs::AbstractVector{T},
    num_rounds::Int,
    k::Int;
    child_branch::Union{Nothing, AbstractVector}=nothing,
) where T<:Real
    CT = Complex{T}
    m = k - 1  # children per hyperedge

    # Precompute modified mixer matrices: MD[ℓ][a] = M(β_ℓ) · diag(CHARGE_DIAG[a,:])
    MD = [let M = doubled_mixer(βs[ℓ])
        [M .* CT.(CHARGE_DIAG[a, :]') for a in 1:4]
    end for ℓ in 1:num_rounds]

    # Charge weight matrices
    W = [charge_weight_matrix(γs[ℓ]) for ℓ in 1:num_rounds]

    child_rounds = child_branch !== nothing ? num_rounds - 1 : 0

    # ── Phase 1: coupled contractions consuming child branch ──
    if child_branch !== nothing && child_rounds ≥ 2
        # V is a matrix (n_ch, entries_per_row) stored as a flat C-order vector.
        # Each iteration of the loop applies wht_charge_contract to expand
        # n_ch by 4× while shrinking entries_per_row by 4×.
        V_flat = CT(0.5) .* copy(child_branch)  # flat C-order, length 4^child_rounds
        n_ch = 1
        entries = length(V_flat)  # = n_ch * entries_per_row
        for ℓ in 1:child_rounds - 1
            entries_per_row = div(entries, n_ch)
            rest = div(entries_per_row, 16)
            # C-order reshape to (n_ch, 4, 4, rest)
            fi = _reshape_c(V_flat, n_ch, 4, 4, rest)
            channels = wht_charge_contract(doubled_mixer(βs[ℓ]), fi)
            # Each channel is (n_ch, 4, rest) — flatten to C-order and concatenate
            V_flat = vcat([_vec_c(ch) for ch in channels]...)
            n_ch *= 4
            entries = length(V_flat)
        end
        # Final reshape: V_flat is C-order (n_ch * 4) entries → (n_ch, 4)
        V = _reshape_c(V_flat, n_ch, 4)
    elseif child_branch !== nothing
        V = reshape(CT(0.5) .* child_branch, 1, 4)
    else
        V = fill(CT(0.5), 1, 4)
    end

    # ── Phase 2: fused mixer + trace ──
    # Build flat C-order vector via recursive expansion.
    # NB: use transpose() (not ') — Python .T is plain transpose.
    start_mv = max(child_rounds - 1, 0)

    trace_vecs = [MD[num_rounds][a][1, :] .+ MD[num_rounds][a][4, :] for a in 1:4]
    trace_matrix = hcat(trace_vecs...)  # (4, 4)

    function _phase2_trace(V_local, ℓ_0)
        if ℓ_0 == num_rounds - 1
            result = V_local * trace_matrix
            return vcat([result[:, a] for a in 1:4]...)
        end
        ℓ = ℓ_0 + 1  # 1-based index into MD
        parts = [_phase2_trace(V_local * transpose(MD[ℓ][a]), ℓ_0 + 1) for a in 1:4]
        return vcat(parts...)
    end

    t_flat = _phase2_trace(V, start_mv)
    t = _reshape_c(t_flat, ntuple(_ -> 4, num_rounds)...)

    # Reorder axes (matches QOKit permutation)
    remaining = num_rounds - start_mv
    if num_rounds > 1
        perm = vcat(collect(num_rounds:-1:remaining+1), collect(1:remaining))
        t = permutedims(t, perm)
    end

    # Entrywise (k-1) power with normalization
    if m > 1
        t_max = maximum(abs, t)
        t_max > 0 && (t = t ./ t_max)
    end
    F = t .^ m

    # Mode products: contract each axis ℓ with W[ℓ]
    for ℓ in 1:num_rounds
        perm = vcat(ℓ, setdiff(1:num_rounds, ℓ))
        F_perm = permutedims(F, perm)
        shape = size(F_perm)
        F_perm = reshape(W[ℓ] * reshape(F_perm, 4, :), shape)
        F = permutedims(F_perm, sortperm(perm))
    end

    _vec_c(F)
end

# ──────────────────────────────────────────────────────────────────────────────
# Root contraction
# ──────────────────────────────────────────────────────────────────────────────

"""
    _charge_root_contract(rb, γs, βs, p, D, k)

Root contraction using factored rank-1 representation.

Returns the parity expectation ⟨Z^⊗k⟩.
"""
function _charge_root_contract(
    rb::AbstractVector{CT},
    γs::AbstractVector{T},
    βs::AbstractVector{T},
    p::Int, D::Int, k::Int,
) where {T<:Real, CT<:Complex{T}}
    coeffs = CT[CT(0.5)^k]
    factor = rb  # flat, length 4^p in C-order
    R = 1

    for ℓ in 1:p-1
        M = doubled_mixer(βs[ℓ])
        u = root_charge_weights(γs[ℓ])
        rest = div(length(factor), R * 16)
        # C-order reshape to (R, 4, 4, rest), fix σ/b axis order
        fi = _reshape_c(factor, R, 4, 4, rest)
        channels = wht_charge_contract(M, fi)
        # Each channel is (R, 4, rest) — flatten back to C-order rows
        factor = vcat([_vec_c(ch) for ch in channels]...)
        coeffs = vcat([u[a] .* coeffs for a in 1:4]...)
        R *= 4
    end

    # Final round + Z measurement
    M = doubled_mixer(βs[p])
    u = root_charge_weights(γs[p])

    entries_per_row = div(length(factor), R)
    result = zero(CT)
    for a in 1:4
        K = M .* CT.(CHARGE_DIAG[a, :]')
        tv = K[1, :] .- K[4, :]  # Z trace vector
        z = Vector{CT}(undef, R)
        for i in 0:R-1
            z[i+1] = sum(j -> factor[i * entries_per_row + j] * tv[j], 1:4)
        end
        result += u[a] * sum(coeffs .* z .^ k)
    end

    real(result)
end

# ──────────────────────────────────────────────────────────────────────────────
# Full contraction — public API
# ──────────────────────────────────────────────────────────────────────────────

"""
    charge_parity_expectation(params, angles; clause_sign=1)

Exact ⟨Z^⊗k⟩ for depth-p QAOA on a D-regular k-uniform tree using charge
decomposition.  Cost: O(p·4^p).

This is the charge-decomposed analogue of `basso_parity_expectation`.
"""
function charge_parity_expectation(
    params::TreeParams,
    angles::QAOAAngles{T};
    clause_sign::Int=1,
) where T
    p = params.p
    k = params.k
    D = params.D
    depth(angles) == p || throw(ArgumentError("angle depth must match tree depth"))
    validate_clause_sign(clause_sign)

    # charge primitives use full-angle; our convention is γ/2
    # clause_sign flips the phase gate direction
    γs = T(clause_sign) .* T.(angles.γ) ./ 2
    βs = T.(angles.β)

    log_scale = zero(T)

    # Level 1: leaf (no child branch)
    F = _charge_hyperedge_branch(γs, βs, 1, k)

    # Levels 2..p
    for level in 2:p
        F_max = maximum(abs, F)
        if F_max > 0
            F = F ./ F_max
            log_scale += (D - 1) * log(F_max)
        end
        child = F .^ (D - 1)
        F = _charge_hyperedge_branch(γs, βs, level, k; child_branch=child)
    end

    # Final normalization before (D-1) power
    F_max = maximum(abs, F)
    if F_max > 0
        F = F ./ F_max
        log_scale += (D - 1) * log(F_max)
    end
    rb = F .^ (D - 1)

    # Root contraction
    raw = _charge_root_contract(rb, γs, βs, p, D, k)

    # Apply accumulated scale
    raw * exp(k * log_scale)
end

"""
    charge_expectation(params, angles; clause_sign=1)

Expected satisfaction fraction using charge decomposition.

`(1 + clause_sign · ⟨Z^⊗k⟩) / 2`
"""
function charge_expectation(
    params::TreeParams,
    angles::QAOAAngles;
    clause_sign::Int=1,
)
    validate_clause_sign(clause_sign)
    parity = charge_parity_expectation(params, angles; clause_sign)
    (1 + clause_sign * parity) / 2
end
