"""
    ReducedBasis(p)

Precomputed index mappings for the 4× reduced branch-tensor iteration.

The branch tensor `B(a)` on `(2p+1)`-bit configurations has two exact symmetries:

1. **Root-bit independence:** `B(a) = B(a ⊕ eᵣ)` where `eᵣ` flips only the root bit
   `a^[0]`. This follows from `Γ_root = 0` in the phase vector, making the constraint
   kernel independent of the root bit.

2. **Complement invariance:** `B(a) = B(ā)` where `ā` flips all non-root bits.
   This follows from the mirror symmetry `f(a) = conj(f(mirror(a)))` and the sign
   structure of the gamma vector.

Together these define `H^⊥ = ⟨eᵣ, v_c⟩ ≤ Z₂^{2p+1}` of order 4, and `B` is constant
on `H^⊥`-cosets. The iteration can be reformulated on the quotient space of size
`M = 2^{2p-1} = N/4`, using a standard `(2p-1)`-bit Walsh–Hadamard transform.

Coset representatives are chosen with `fwd₁ = 0` (bit 0) and `root = 0` (bit p),
leaving `2p-1` free bits at positions `{1, …, p-1, p+1, …, 2p}`.
"""
struct ReducedBasis
    p::Int
    M::Int                      # 2^(2p-1)
    N::Int                      # 2^(2p+1)
    root_mask::Int              # 1 << p
    complement_mask::Int        # all non-root bits
    free_positions::Vector{Int} # 0-indexed bit positions of free bits
end

function ReducedBasis(p::Int)
    validate_depth(p)
    n_bits = basso_bit_count(p)
    N = basso_configuration_count(p)
    M = one(Int) << (2p - 1)
    root_mask = one(Int) << p
    complement_mask = (N - 1) ⊻ root_mask
    free_positions = [collect(1:p-1); collect(p+1:2p)]

    ReducedBasis(p, M, N, root_mask, complement_mask, free_positions)
end

"""Map a reduced index (0-based) to a full `(2p+1)`-bit configuration (0-based)."""
function reduced_to_full(basis::ReducedBasis, j::Int)
    full = 0
    @inbounds for (k, pos) in enumerate(basis.free_positions)
        full |= ((j >> (k - 1)) & 1) << pos
    end
    full
end

"""Return the 4 full-space indices in the `H^⊥`-coset of representative `r`."""
function coset_elements(basis::ReducedBasis, r::Int)
    (r,
     r ⊻ basis.root_mask,
     r ⊻ basis.complement_mask,
     r ⊻ basis.root_mask ⊻ basis.complement_mask)
end

"""
    reduce_coset_sum(full_vector, basis)

Sum `full_vector` over each `H^⊥`-coset, producing a length-`M` vector.

Use for `f_table`, which does NOT have `H^⊥` symmetry: the correct reduced
representation is `f_red(r) = Σ_{v ∈ H^⊥} f(repr(r) ⊕ v)`.
"""
function reduce_coset_sum(full_vector::AbstractVector, basis::ReducedBasis)
    length(full_vector) == basis.N || throw(ArgumentError(
        "expected length $(basis.N), got $(length(full_vector))",
    ))
    result = Vector{eltype(full_vector)}(undef, basis.M)
    @inbounds for j in 0:basis.M-1
        r = reduced_to_full(basis, j)
        a0, a1, a2, a3 = coset_elements(basis, r)
        result[j+1] = full_vector[a0+1] + full_vector[a1+1] +
                       full_vector[a2+1] + full_vector[a3+1]
    end
    result
end

"""
    reduce_sample(full_vector, basis)

Sample `full_vector` at coset representatives, producing a length-`M` vector.

Use for vectors with `H^⊥` symmetry (constraint kernel, branch tensor), where
all 4 coset elements have the same value.
"""
function reduce_sample(full_vector::AbstractVector, basis::ReducedBasis)
    length(full_vector) == basis.N || throw(ArgumentError(
        "expected length $(basis.N), got $(length(full_vector))",
    ))
    result = Vector{eltype(full_vector)}(undef, basis.M)
    @inbounds for j in 0:basis.M-1
        result[j+1] = full_vector[reduced_to_full(basis, j)+1]
    end
    result
end

"""
    expand_symmetric(reduced, basis)

Expand a reduced vector back to full `N`-length by copying each value to all 4
elements of its `H^⊥`-coset.
"""
function expand_symmetric(reduced::AbstractVector, basis::ReducedBasis)
    length(reduced) == basis.M || throw(ArgumentError(
        "expected length $(basis.M), got $(length(reduced))",
    ))
    full = Vector{eltype(reduced)}(undef, basis.N)
    @inbounds for j in 0:basis.M-1
        r = reduced_to_full(basis, j)
        a0, a1, a2, a3 = coset_elements(basis, r)
        val = reduced[j+1]
        full[a0+1] = val
        full[a1+1] = val
        full[a2+1] = val
        full[a3+1] = val
    end
    full
end

"""
    basso_branch_tensor_reduced(params, angles; f_table) -> (B_red, basis)

Compute the Basso branch tensor using the 4× reduced iteration.

The iteration runs entirely in the quotient space `Z₂^{2p+1} / H^⊥` of size
`M = 2^{2p-1}`, using a standard `(2p-1)`-bit WHT. This gives identical results
to `basso_branch_tensor` but with 4× less memory and 4× fewer operations per WHT.

Returns `(B_red, basis)` where `B_red` has length `M` and `basis` is the
`ReducedBasis` needed to expand back to full space via `expand_symmetric`.
"""
function basso_branch_tensor_reduced(
    params::TreeParams,
    angles::QAOAAngles{T};
    f_table::AbstractVector=basso_f_table(angles),
) where T
    depth(angles) == params.p || throw(ArgumentError("angle depth must match tree depth"))

    p = params.p
    basis = ReducedBasis(p)
    M = basis.M

    kernel = basso_constraint_kernel(angles, basso_branching_degree(params))
    f_red = reduce_coset_sum(complex.(f_table), basis)
    kernel_red = reduce_sample(complex.(kernel), basis)
    kernel_hat_red = wht(kernel_red)

    child_arity = params.k - 1
    branch_degree = basso_branching_degree(params)

    B_red = ones(Complex{T}, M)
    for _ in 1:p
        child_hat_red = wht(f_red .* B_red)
        folded_red = iwht(kernel_hat_red .* (child_hat_red .^ child_arity))
        B_red = folded_red .^ branch_degree
    end

    (B_red, basis)
end

"""
    basso_expectation_reduced(params, angles; clause_sign) -> Float64

Evaluate exact QAOA expectation using the 4× reduced branch-tensor iteration.

The `p`-step branch iteration runs in the reduced space (`2^{2p-1}` elements).
The single root fold expands to full space (done once, not performance-critical).
"""
function basso_expectation_reduced(
    params::TreeParams,
    angles::QAOAAngles{T};
    clause_sign::Int=default_clause_sign(params.k),
) where T
    depth(angles) == params.p || throw(ArgumentError("angle depth must match tree depth"))
    validate_clause_sign(clause_sign)

    f_table = basso_f_table(angles)
    B_red, basis = basso_branch_tensor_reduced(params, angles; f_table)
    B_full = expand_symmetric(B_red, basis)

    parity_sum = basso_root_parity_sum(params, angles, B_full, f_table; clause_sign)
    (1 + clause_sign * real(parity_sum)) / 2
end
