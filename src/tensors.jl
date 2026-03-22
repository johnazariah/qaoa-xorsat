"""
    QAOAAngles(γ, β)

QAOA angle parameters for depth `p`.

- `γ`: problem angles, length `p`
- `β`: mixer angles, length `p`
"""
struct QAOAAngles
    γ::Vector{Float64}
    β::Vector{Float64}

    function QAOAAngles(γ::AbstractVector{<:Real}, β::AbstractVector{<:Real})
        length(γ) == length(β) ||
            throw(ArgumentError("γ and β must have same length"))
        !isempty(γ) || throw(ArgumentError("need at least p=1"))
        new(Float64.(γ), Float64.(β))
    end
end

"""QAOA depth `p`."""
depth(angles::QAOAAngles) = length(angles.γ)

validate_depth(p::Int) =
    p ≥ 1 || throw(ArgumentError("p must be ≥ 1, got $p"))

validate_round(round::Int, p::Int) =
    1 ≤ round ≤ p || throw(ArgumentError("round must be in 1:$p, got $round"))

validate_slice(slice::Int, p::Int) =
    1 ≤ slice ≤ p || throw(ArgumentError("slice must be in 1:$p, got $slice"))

validate_hyperindex(σ::Integer) =
    σ ≥ 0 || throw(ArgumentError("hyperindex must be non-negative, got $σ"))

function validate_clause_sign(clause_sign::Int)
    clause_sign in (-1, 1) ||
        throw(ArgumentError("clause_sign must be ±1, got $clause_sign"))
    clause_sign
end

"""
    hyperindex_dimension(p)

Number of hyperindex values for depth `p`, namely `4^p = 2^(2p)`.
"""
function hyperindex_dimension(p::Int)
    validate_depth(p)
    one(Int) << (2p)
end

"""
    slice_from_physical_round(round, p)

Map a physical QAOA round index to the corresponding hyperindex slice.

Physical rounds are ordered from the initial state toward the observable:

- physical round `1` is the outermost slice
- physical round `p` is the root/innermost slice
"""
function slice_from_physical_round(round::Int, p::Int)
    validate_depth(p)
    validate_round(round, p)
    p - round + 1
end

"""
    physical_round_from_slice(slice, p)

Map a hyperindex slice index back to the corresponding physical QAOA round.
"""
function physical_round_from_slice(slice::Int, p::Int)
    validate_depth(p)
    validate_slice(slice, p)
    p - slice + 1
end

"""
    slice_bit_positions(slice, p)

Return the `(ket_bit, bra_bit)` positions for the requested root-to-leaf
hyperindex slice under the interleaved convention

`(ket₁, bra₁, ket₂, bra₂, …, ket_p, bra_p)`.
"""
function slice_bit_positions(slice::Int, p::Int)
    validate_depth(p)
    validate_slice(slice, p)
    (2slice - 1, 2slice)
end

"""
    round_bit_positions(slice, p)

Backward-compatible alias for `slice_bit_positions`. Despite the historical name,
the argument is a root-to-leaf slice index, not a physical QAOA round.
"""
round_bit_positions(slice::Int, p::Int) = slice_bit_positions(slice, p)

"""
    hyperindex_bit(σ, ℓ)

Extract bit `ℓ` (1-indexed) from hyperindex `σ`.
"""
function hyperindex_bit(σ::Integer, ℓ::Int)
    validate_hyperindex(σ)
    ℓ ≥ 1 || throw(ArgumentError("bit position must be ≥ 1, got $ℓ"))
    (Int(σ) >> (ℓ - 1)) & 1
end

"""
    hyperindex_parity(σ, positions)

Compute the XOR parity of the bits of hyperindex `σ` at the requested
`positions`.
"""
function hyperindex_parity(σ::Integer, positions)
    foldl(⊻, (hyperindex_bit(σ, position) for position in positions); init=0)
end

z_eigenvalue(bit::Int) = bit == 0 ? 1 : -1

function mixer_gate_entry(output::Int, input::Int, β::Float64)
    output == input ? ComplexF64(cos(β)) : ComplexF64(0.0, -sin(β))
end

ket_bit(σ::Int, round::Int, p::Int) = hyperindex_bit(σ, first(round_bit_positions(round, p)))
bra_bit(σ::Int, round::Int, p::Int) = hyperindex_bit(σ, last(round_bit_positions(round, p)))

"""
    leaf_tensor(angles)

Build the leaf tensor for a single boundary variable node.

With the interleaved hyperindex convention, each round contributes

`⟨+|e^{iβ_ℓ X}|b_ℓ⟩ ⟨k_ℓ|e^{-iβ_ℓ X}|+⟩ = 1/2`

because `|+⟩` is an eigenstate of `X`. The full leaf tensor is therefore the
angle-independent vector with every entry equal to `2^{-p}`.
"""
function leaf_tensor(angles::QAOAAngles)::Vector{Float64}
    p = depth(angles)
    fill(exp2(-p), hyperindex_dimension(p))
end

"""
    mixer_tensor(β, slice, p)

Build the raw single-qubit mixer superoperator for the requested hyperindex
`slice`.

The returned matrix has size `4^p × 4^p` and acts on the full hyperindex space.
Only the `(ket_slice, bra_slice)` pair is transformed; all other slice-pairs are
left unchanged.
"""
function mixer_tensor(β::Real, slice::Int, p::Int)::Matrix{ComplexF64}
    validate_depth(p)
    validate_slice(slice, p)

    βf = Float64(β)
    dim = hyperindex_dimension(p)
    ket_position, bra_position = slice_bit_positions(slice, p)
    ket_mask = one(Int) << (ket_position - 1)
    bra_mask = one(Int) << (bra_position - 1)
    keep_mask = xor(typemax(Int), ket_mask | bra_mask)

    tensor = zeros(ComplexF64, dim, dim)
    for input in 0:dim-1
        base = input & keep_mask
        input_ket = hyperindex_bit(input, ket_position)
        input_bra = hyperindex_bit(input, bra_position)

        for output_ket in 0:1, output_bra in 0:1
            output = base | (output_ket << (ket_position - 1)) | (output_bra << (bra_position - 1))
            tensor[output+1, input+1] =
                mixer_gate_entry(output_ket, input_ket, βf) *
                conj(mixer_gate_entry(output_bra, input_bra, βf))
        end
    end

    tensor
end

function parity_sign(configuration, position::Int)
    foldl(*, (z_eigenvalue(hyperindex_bit(σ, position)) for σ in configuration); init=1)
end

function problem_phase(configuration, γ::Float64, slice::Int, p::Int; clause_sign::Int=1)
    validate_clause_sign(clause_sign)
    ket_position, bra_position = slice_bit_positions(slice, p)
    ket_sign = parity_sign(configuration, ket_position)
    bra_sign = parity_sign(configuration, bra_position)
    cis(clause_sign * γ * (bra_sign - ket_sign) / 2)
end

"""
    problem_tensor(k, γ, slice, p; clause_sign=1)

Build the flattened diagonal of the raw `k`-body problem-gate tensor for the
requested root-to-leaf `slice`.

The returned vector has length `(4^p)^k`. Reshape it with

`reshape(problem_tensor(k, γ, slice, p), ntuple(_ -> 4^p, k)...)`

to recover the diagonal weights indexed by the `k` qubit hyperindices.

`clause_sign = 1` corresponds to the even clause `(1 + Z₁⋯Z_k)/2`, while
`clause_sign = -1` corresponds to the odd clause `(1 - Z₁⋯Z_k)/2` used by MaxCut.
"""
function problem_tensor(
    k::Int,
    γ::Real,
    slice::Int,
    p::Int;
    clause_sign::Int=1,
)::Vector{ComplexF64}
    k ≥ 2 || throw(ArgumentError("k must be ≥ 2, got $k"))
    validate_depth(p)
    validate_slice(slice, p)
    validate_clause_sign(clause_sign)

    γf = Float64(γ)
    dim = hyperindex_dimension(p)
    ranges = ntuple(_ -> 0:dim-1, k)
    tensor = Vector{ComplexF64}(undef, dim^k)

    for (index, configuration) in enumerate(Iterators.product(ranges...))
        tensor[index] =
            problem_phase(configuration, γf, slice, p; clause_sign)
    end

    tensor
end

function parity_observable_weight(configuration, p::Int)
    all(ket_bit(σ, 1, p) == bra_bit(σ, 1, p) for σ in configuration) || return 0.0

    foldl(*, (z_eigenvalue(ket_bit(σ, 1, p)) for σ in configuration); init=1)
end

function identity_observable_weight(configuration, p::Int)
    all(ket_bit(σ, 1, p) == bra_bit(σ, 1, p) for σ in configuration) ? 1.0 : 0.0
end

"""
    identity_observable_tensor(k, p)

Build the flattened diagonal of the root identity observable on the innermost
slice.

Only configurations with matching ket/bra bits on the root slice contribute.
This is the denominator tensor for normalized root expectation values in the raw
hyperindex representation.
"""
function identity_observable_tensor(k::Int, p::Int)::Vector{Float64}
    k ≥ 2 || throw(ArgumentError("k must be ≥ 2, got $k"))
    validate_depth(p)

    dim = hyperindex_dimension(p)
    ranges = ntuple(_ -> 0:dim-1, k)
    tensor = Vector{Float64}(undef, dim^k)

    for (index, configuration) in enumerate(Iterators.product(ranges...))
        tensor[index] = identity_observable_weight(configuration, p)
    end

    tensor
end

"""
    parity_observable_tensor(k, p)

Build the flattened diagonal of the raw parity correlator observable

`Z₁⋯Z_k`

using the innermost root slice `(ket₁, bra₁)`. Off-diagonal bra/ket
configurations vanish because the observable is diagonal in the computational
basis.
"""
function parity_observable_tensor(k::Int, p::Int)::Vector{Float64}
    k ≥ 2 || throw(ArgumentError("k must be ≥ 2, got $k"))
    validate_depth(p)

    dim = hyperindex_dimension(p)
    ranges = ntuple(_ -> 0:dim-1, k)
    tensor = Vector{Float64}(undef, dim^k)

    for (index, configuration) in enumerate(Iterators.product(ranges...))
        tensor[index] = parity_observable_weight(configuration, p)
    end

    tensor
end

"""
    observable_tensor(k, p; clause_sign=1)

Build the flattened diagonal of the root observable

`C_α = (1 + clause_sign * Z₁⋯Zₖ)/2`

using the innermost `(ket₁, bra₁)` hyperindex pair. Off-diagonal bra/ket
configurations vanish because the observable is diagonal in the computational
basis.
"""
function observable_tensor(k::Int, p::Int; clause_sign::Int=1)::Vector{Float64}
    k ≥ 2 || throw(ArgumentError("k must be ≥ 2, got $k"))
    validate_depth(p)
    validate_clause_sign(clause_sign)

    dim = hyperindex_dimension(p)
    ranges = ntuple(_ -> 0:dim-1, k)
    tensor = Vector{Float64}(undef, dim^k)

    for (index, configuration) in enumerate(Iterators.product(ranges...))
        parity = parity_observable_weight(configuration, p)
        tensor[index] = iszero(parity) ? 0.0 : 0.5 * (1 + clause_sign * parity)
    end

    tensor
end
