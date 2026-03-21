# Tensor Derivation Notes for P1.2

This note records the tensor conventions used by the Julia implementation in
`src/tensors.jl` and the contraction-ordering facts that P1.3 will depend on.

## Hyperindex convention

We use an **interleaved** hyperindex ordering

`(ket₁, bra₁, ket₂, bra₂, ..., ket_p, bra_p)`

rather than grouping all ket bits before all bra bits. This keeps every QAOA
round local to one adjacent bit-pair:

- round `ℓ` uses bit positions `(2ℓ-1, 2ℓ)`
- the innermost/root slice is round `1`
- the outermost/leaf slice is round `p`

This is a small clarification of the draft spec rather than a conceptual
change: each qubit still has `2p` binary components and therefore `4^p`
possible hyperindex values.

## Raw tensor primitives

The code implements **raw local tensors**. It does **not** yet collapse them
into the effective branch-transfer objects used by the leaf-to-root contraction.
That distinction matters:

- the leaf tensor is real and angle-independent
- the observable tensor is real
- the mixer and problem tensors are naturally **complex-valued**

This matches the paper's Eq. (13), where the raw gate tensors carry the complex
phases and the eventual expectation value becomes real only after the full
contraction.

## Leaf tensor

For each round `ℓ`, the boundary contribution is

`⟨+|e^{iβ_ℓ X}|b_ℓ⟩ ⟨k_ℓ|e^{-iβ_ℓ X}|+⟩`.

Since `X|+⟩ = |+⟩`, both factors contribute `1 / √2` up to cancelling phases,
so every round contributes exactly `1/2`. Therefore

`L(σ) = 2^{-p}`

for every hyperindex `σ`. This explains why `leaf_tensor` is independent of
both `γ` and `β`.

At `p = 1`, the four entries are

`(1/2, 1/2, 1/2, 1/2)`.

## Mixer tensor

Let

`U_X(β) = e^{-iβX} = [cos β   -i sin β; -i sin β   cos β]`.

For one round and one qubit, the raw bra-ket superoperator on the pair
`(ket_ℓ, bra_ℓ)` is

`M((k, b), (k', b')) = U_X(β)_{k, k'} * conj(U_X(β)_{b, b'})`.

The full `4^p × 4^p` mixer tensor acts as this `4 × 4` block on the chosen round
and as the identity on all other rounds.

At `β = 0`, `U_X(0) = I`, so the full mixer tensor is exactly the identity.

## Problem tensor

For a `k`-body constraint with standard XORSAT phase

`exp(-iγ Z₁⋯Z_k / 2)`,

the raw diagonal bra-ket weight for hyperindices `σ₁, ..., σ_k` at round `ℓ` is

`exp(-iγ z_ket / 2) * exp(+iγ z_bra / 2)`,

where

- `z_ket = ∏_{j=1}^k (1 - 2 * ket_bit(σ_j, ℓ))`
- `z_bra = ∏_{j=1}^k (1 - 2 * bra_bit(σ_j, ℓ))`.

Equivalently,

`P = cis(γ * (z_bra - z_ket) / 2)`.

The implementation stores the **flattened diagonal** of this raw tensor, so
`problem_tensor(k, γ, ℓ, p)` has length `(4^p)^k`.

For `p = 1`, fixing the remaining legs to hyperindex `0 = (ket=0, bra=0)`, the
four parent entries are

`(1, e^{iγ}, e^{-iγ}, 1)`.

That is the `p = 1` golden slice used in the tests.

## Observable tensor

The root observable in Spec P1.2 is

`C_α = (1 + Z₁⋯Z_k) / 2`.

Because it is diagonal, bra and ket must match on the root slice. Under our
interleaved convention the root slice is `(ket₁, bra₁)`, so the observable
weight is

- `0` if any qubit has `ket₁ ≠ bra₁`
- otherwise `0.5 * (1 + ∏_j (1 - 2 * ket₁(σ_j)))`.

This yields entries in `{0, 1}` for the flattened diagonal stored by
`observable_tensor(k, p)`.

## Contraction ordering (for P1.3)

Farhi et al. 2025, Eq. (14) and Fig. 5(b), show that the contraction proceeds by
contracting one colored box, then raising the resulting branch tensor entrywise
to the branching multiplicity before moving one level toward the root.

Mapped to our notation:

- **leaf level** uses round `p`
- then the contraction moves inward through rounds `p-1, p-2, ..., 1`
- at a variable node, the multiplicity is `D-1`
- at a constraint node, the multiplicity is `k-1`

So for `p = 2`, `k = 2`, `D = 3`:

1. start from the leaf tensor on round `2`
2. contract the deepest round-`2` mixer/problem structure
3. raise the resulting branch entries to the power `D-1 = 2`
4. move inward to round `1`
5. apply the root observable on the round-`1` slice

For general `k`, the same root-to-leaf indexing applies; only the
constraint-node multiplicity changes from `1` child branch (`k = 2`) to
`k-1` child branches.

## Zero-angle check

At `γ = β = 0`:

- `leaf_tensor` is still the constant vector `2^{-p}`
- `mixer_tensor` is the identity
- `problem_tensor` is the all-ones diagonal

The root observable then averages uniformly over the computational-basis
configurations on the root slice, so the expectation value reduces to the random
baseline `1/2`, as required by Spec P1.3.
