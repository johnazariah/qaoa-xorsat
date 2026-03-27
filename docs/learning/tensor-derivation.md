# Tensor Derivation Notes for P1.2

This note records the tensor conventions used by the Julia implementation in
`src/tensors.jl` and the contraction-ordering facts that P1.3 will depend on.

## Hyperindex convention

We use an **interleaved** hyperindex ordering

`(ket₁, bra₁, ket₂, bra₂, ..., ket_p, bra_p)`

rather than grouping all ket bits before all bra bits. This keeps every QAOA
slice local to one adjacent bit-pair:

- slice `s` uses bit positions `(2s-1, 2s)`
- slice `1` is the innermost/root slice
- slice `p` is the outermost/leaf slice

This is a small clarification of the draft spec rather than a conceptual
change: each qubit still has `2p` binary components and therefore `4^p`
possible hyperindex values.

The **physical QAOA round** index runs in the opposite direction:

- physical round `1` is the outermost slice
- physical round `p` is the innermost/root slice

So the explicit mapping is

`slice = p - round + 1`.

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

## Contraction ordering and current blocker (for P1.3)

The light cone at physical depth `p` contains the root clause plus `p`
additional constraint shells and a boundary variable shell. For MaxCut
`(k=2, D=3, p=1)` that means **6 qubits and 5 edges**, not the smaller
three-node draft tree.

### Contraction Ordering

The indexing used in the code is now fixed:

- root-to-leaf **slice** index `s = 1, ..., p`
- physical QAOA **round** index `r = 1, ..., p`
- mapping `s = p - r + 1`

So:

- physical round `1` is the **outermost** slice
- physical round `p` is the **innermost** slice next to the observable

For the concrete example `(k=2, D=3, p=2)`:

- round `1` lives on slice `2` (closest to the `|+⟩` boundary)
- round `2` lives on slice `1` (closest to the root observable)

The exact reference evaluator in `src/qaoa.jl` applies the physical circuit in
that forward order:

1. problem layer with `γ₁`, then mixer layer with `β₁`
2. problem layer with `γ₂`, then mixer layer with `β₂`
3. measure the root observable

This agrees with the adopted hyperindex convention from the P1.2 raw tensors:
the innermost root slice is `(ket₁, bra₁)`, while the outermost slice is
`(ket_p, bra_p)`.

What is still **not** derived is the effective branch-transfer object that lets
one contract those raw tensors in `O(4^p)` while preserving this ordering.

The raw P1.2 tensors are still useful **local oracles**, but they are not yet a
complete derivation of the intended O(4^p) branch-transfer recursion. Two facts
are now fixed:

1. **Only variable branching exponentiates entrywise.** At a variable node the
   `D-1` identical child constraints contribute an entrywise power.
2. **Constraint contraction is multilinear in `k-1` child messages.** The draft
   `.^ (k-1)` rule at non-root constraints is wrong in general.

We also have the corrected MaxCut p=1 validation target:

`⟨Z_u Z_v⟩ = -sin(4β) cos²(γ) sin(γ)`

and therefore

`c_edge = (1 - ⟨Z_u Z_v⟩) / 2 = 1/2 + √3/9 ≈ 0.69245`

at the optimum.

What remains blocked is the **effective transfer object** that connects the raw
P1.2 tensors to the paper's Eq. (14) style O(4^p) recursion without double
counting boundary/mixer structure. The repository now carries an exact
light-cone statevector evaluator as a correctness reference for small trees,
while keeping the raw tensor API and slice/round mapping explicit for the next
derivation step.

## Zero-angle check

At `γ = β = 0`:

- `leaf_tensor` is still the constant vector `2^{-p}`
- `mixer_tensor` is the identity
- `problem_tensor` is the all-ones diagonal

The root observable then averages uniformly over the computational-basis
configurations on the root slice, so the expectation value reduces to the random
baseline `1/2`, as required by Spec P1.3.

The exact evaluator now checks this explicitly for `(k=3, D=4, p=1)`.
