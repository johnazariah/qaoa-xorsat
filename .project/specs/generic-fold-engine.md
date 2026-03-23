# Spec: Generic QAOA Fold Engine

**Phase**: Architecture refactor
**Status**: Design
**Depends on**: P1.3 (contraction), Phase 4 (optimisation) — both complete
**Blocks**: Generalisation to other CSP families, multi-(k,D) sweeps

---

## Motivation

The current codebase computes QAOA expectation values for Max-k-XORSAT. But the
fold algorithm itself is problem-agnostic — it depends on the cost function only
through two pluggable components: a **constraint kernel** and a **root
observable**. Everything else (the tree structure, leaf tensor, mixer, WHT
acceleration, angle optimisation) is identical across problems.

This spec extracts the problem-specific parts into a clean interface, turning the
codebase from a one-off XORSAT calculator into a reusable QAOA evaluation engine
for any CSP on a regular, locally-tree-like hypergraph.

---

## The Abstraction

The fold evaluates:

```
leaf → [constraint fold → variable fold → mixer]^p → root observable → scalar
```

The **problem** determines two things:
1. What happens inside the constraint fold (the kernel $\kappa$)
2. What happens at the root (the observable)

Everything else is structural.

---

## Interface

### The `CostAlgebra` type

```julia
"""
    CostAlgebra{K}

Defines a k-local cost function for QAOA evaluation on D-regular k-uniform
hypergraphs. The type parameter K is the constraint arity.

A CostAlgebra must provide:
- A constraint kernel: how the k-body problem gate affects the bra-ket sandwich
- A root observable: how to extract the expectation value at the root constraint

# Required properties (see documentation for proofs and counterexamples)

**P1 (Diagonal)**: The cost operator C_α is diagonal in the computational basis.
    Every classical cost function satisfies this.

**P2 (k-local)**: Each term C_α involves exactly k variables.

**P3 (Uniform)**: All constraints have identical gate structure (up to variable
    relabelling). Mixed-arity constraints require separate CostAlgebra instances.

**P4 (Group function)**: The constraint kernel κ(d) depends on the spin
    configuration d ∈ {±1}^{2p+1} only through its group element in Z₂^{2p+1}.
    This is what makes the WHT applicable. Any function that depends only on
    the bra-ket sandwich bits satisfies this automatically.

The fold engine additionally requires:

**P5 (High girth)**: The instance hypergraph has girth > 2p, so the light cone
    is a tree. This is a property of the instance class, not the cost function.

**P6 (Regularity)**: Every variable has degree D, every constraint has arity k.
    This collapses the tree to a single representative branch.
"""
abstract type CostAlgebra{K} end

"""Constraint arity."""
arity(::CostAlgebra{K}) where {K} = K
```

### Required methods

```julia
"""
    constraint_kernel(algebra, angles, D) -> Vector{ComplexF64}

Build the constraint kernel κ(d) for all d ∈ {±1}^{2p+1}, encoded as a vector
of length 2^{2p+1} indexed by the binary encoding of the spin configuration.

The kernel encodes the bra-ket sandwich contribution of the problem gate
e^{-iγ C_α} at the constraint node. It is convolved with the child branch
tensors during the constraint fold step.

The returned vector must have length `basso_configuration_count(depth(angles))`.
"""
function constraint_kernel end

"""
    root_expectation(algebra, angles, D, branch_tensors) -> Float64

Compute the scalar expectation value from the k root-variable branch tensors.

`branch_tensors` is a length-k vector of identical branch tensors (each a
Vector{ComplexF64} of length 2^{2p+1}). The function combines them with the
root constraint's problem gate and observable to produce the satisfaction
fraction.

Must return a value in [0, 1].
"""
function root_expectation end

"""
    default_clause_sign(algebra) -> Int

Return the default clause sign convention (+1 for even-parity constraints
like XORSAT, -1 for odd-parity like MaxCut).
"""
function default_clause_sign end
```

---

## Concrete Instances

### XORSATAlgebra

```julia
"""
Max-k-XORSAT on D-regular k-uniform hypergraphs.

Cost operator: C_α = (1 + clause_sign · Z_{i₁}···Z_{iₖ}) / 2

Kernel: κ(d) = cos(Γ · spins(d) / √D)  (from the Basso finite-D iteration)
"""
struct XORSATAlgebra{K} <: CostAlgebra{K}
    clause_sign::Int
end

XORSATAlgebra(k::Int; clause_sign::Int=1) = XORSATAlgebra{k}(clause_sign)

default_clause_sign(::XORSATAlgebra) = 1

# MaxCut is the k=2, clause_sign=-1 special case
MaxCutAlgebra() = XORSATAlgebra(2; clause_sign=-1)
```

### Future: SATAlgebra

```julia
"""
Max-k-SAT on D-regular k-uniform hypergraphs.

Each clause has a "forbidden assignment" f ∈ {0,1}^k. The clause is satisfied
by all assignments except f. The cost operator:

  C_α = I - |f⟩⟨f|

The kernel differs from XORSAT: instead of a cosine (parity structure), the
phase depends on whether each configuration matches the forbidden pattern.

Kernel: κ(d) = 1 - e^{-iγ} · δ(d matches forbidden pattern)
"""
struct SATAlgebra{K} <: CostAlgebra{K}
    forbidden::NTuple{K, Int}  # the forbidden assignment (0s and 1s)
end
```

### Future: GraphColouringAlgebra

```julia
"""
Max-q-COL on D-regular graphs.

Each edge constraint: "adjacent vertices have different colours."
Variables are q-valued (not binary). Requires q-ary hyperindex (dimension q^{2p}
instead of 4^p) and DFT over Z_q instead of WHT over Z_2.

This is a larger refactor — the fold engine's inner types change from
Vector{ComplexF64} of length 2^{2p+1} to length q^{2p+1}.
"""
# Deferred — requires q-ary generalisation of the engine.
```

---

## Refactored Fold Engine

### Public API

```julia
"""
    qaoa_expectation(algebra, params, angles) -> Float64

Evaluate the QAOA expectation value for a single constraint, using the
WHT-accelerated fold on the light-cone tree.

This is the main entry point. It is parameterised by the CostAlgebra,
which determines the problem-specific kernel and observable.
"""
function qaoa_expectation(
    algebra::CostAlgebra,
    params::TreeParams,
    angles::QAOAAngles,
)::Float64
    depth(angles) == params.p ||
        throw(ArgumentError("angle depth must match tree depth"))
    arity(algebra) == params.k ||
        throw(ArgumentError("algebra arity must match tree arity"))

    κ = constraint_kernel(algebra, angles, params.D)
    branch = fold_tree(κ, angles, params)
    root_expectation(algebra, angles, params.D, branch)
end

"""
    optimize_angles(algebra, params; kwargs...) -> AngleOptimizationResult

Optimise QAOA angles for the given problem and tree parameters.
Delegates to L-BFGS with multistart.
"""
function optimize_angles(
    algebra::CostAlgebra,
    params::TreeParams;
    kwargs...,
)::AngleOptimizationResult
    # ... wraps qaoa_expectation(algebra, params, angles) as objective
end
```

### Internal fold (problem-agnostic)

```julia
"""
    fold_tree(κ, angles, params) -> Vector{ComplexF64}

Execute the WHT-accelerated fold from leaves to root.

Returns the final branch tensor at the root level (before observable
application). This function is entirely problem-agnostic — the problem
enters only through the kernel κ.

Steps per round (from outermost p to innermost 1):
1. Constraint fold: Ŝ = κ̂ · ĝ^{k-1}  (WHT domain)
2. Variable fold:   B .^= (D-1)
3. Mixer:           B = M(β) * B
"""
function fold_tree(
    κ::Vector{ComplexF64},
    angles::QAOAAngles,
    params::TreeParams,
)::Vector{ComplexF64}
    # ... the generic fold loop, identical for all problems
end
```

---

## Migration Path

### Phase 1: Extract (non-breaking)

1. Define `CostAlgebra` abstract type and the two method signatures
2. Implement `XORSATAlgebra` wrapping the existing kernel/observable code
3. Add `MaxCutAlgebra() = XORSATAlgebra(2; clause_sign=-1)` alias
4. Add a new method `qaoa_expectation(algebra, params, angles)` that delegates
   to the existing implementation
5. Keep the old `qaoa_expectation(params, angles; clause_sign)` as a convenience
   that constructs the algebra internally

**Tests**: all existing tests pass unchanged. Add new tests that call the
algebra-parameterised API and verify identical results.

### Phase 2: Refactor internals

1. Move kernel construction from `basso_finite_d.jl` into `XORSATAlgebra`'s
   `constraint_kernel` method
2. Move root observable into `root_expectation`
3. Make `fold_tree` call the kernel as a parameter, not hardcoded
4. Delete the old hardcoded kernel construction

**Tests**: cross-validate old and new paths at all (k, D, p) combinations.
Then remove old path.

### Phase 3: Second instance (MaxCut as proof of abstraction)

1. Verify `MaxCutAlgebra()` produces identical results to existing MaxCut tests
2. If the Farhi 2025 direct transfer-matrix method (which doesn't use Basso's
   formulation) can also be expressed as a CostAlgebra, implement it as
   `MaxCutTransferAlgebra` — this validates that the abstraction supports
   multiple evaluation strategies for the same problem

### Phase 4: Documentation

1. Write docstrings for all six properties (P1-P6) with:
   - Mathematical statement
   - One-paragraph explanation of why it's required
   - What breaks if violated (with concrete example)
   - How to verify the property for a new cost function
2. Add a "How to add a new problem" guide with a worked example

---

## Property Documentation (to include in CostAlgebra docstring)

### P1 — Diagonal in the computational basis

**Required**: $C_\alpha$ is diagonal, i.e.,
$\langle x | C_\alpha | y \rangle = 0$ for $x \neq y$.

**Why**: the hyperindex representation merges bra and ket into a single index.
This is only valid when the problem gate doesn't create off-diagonal coherences.
If violated, the bra-ket sandwich doesn't reduce to a phase function and the
entire fold formulation breaks.

**Satisfied by**: any classical cost function (function of the bit string).
Violated by quantum cost functions with off-diagonal terms (e.g., QAOA applied
to a quantum Hamiltonian with transverse-field terms in the cost).

### P2 — k-local

**Required**: each $C_\alpha$ involves exactly k variables.

**Why**: determines the tree arity and the kernel dimensionality. The kernel
is a function on $\{-1,+1\}^{2p+1}$ encoding the combined effect on k qubits.
If different constraints have different arities, you'd need multiple kernel
types (one per arity class).

### P3 — Uniform structure

**Required**: all constraints have the same gate structure up to variable
relabelling.

**Why**: this is what makes all branches at the same depth identical, enabling
the "compute once, exponentiate" trick. If constraint #47 has a different
gate than constraint #48, their subtrees produce different branch tensors
and you can't fold with a single representative.

**Generalisation**: for a finite number of constraint types, you could track
one branch tensor per type. Cost multiplies by the number of types.

### P4 — Group function on $\mathbb{Z}_2^{2p+1}$

**Required**: $\kappa(d)$ depends on $d$ only through the group structure
(i.e., it's well-defined as a function $\mathbb{Z}_2^{2p+1} \to \mathbb{C}$).

**Why**: the WHT diagonalises convolution on this group. If the kernel depended
on additional structure (e.g., an ordering of the bits that breaks the group
symmetry), the convolution theorem wouldn't apply and the WHT acceleration
would be invalid.

**In practice**: any kernel derived from a diagonal problem gate automatically
satisfies this, because the bra-ket sandwich structure naturally produces a
function of the spin configuration $d \in \{-1,+1\}^{2p+1}$.

### P5 — High girth

**Required**: the factor graph has girth > 2p.

**Why**: ensures the light cone is a tree. If there are short cycles, two
branches share variables and the branch-independence assumption fails. The
fold would double-count contributions from shared variables.

**Not a property of the cost function** — it's a property of the instance
class (the hypergraph). In practice, large random D-regular k-uniform
hypergraphs have girth $O(\log n)$, so any fixed p is covered for large
enough n.

### P6 — Regularity

**Required**: every variable has degree D, every constraint has arity k.

**Why**: regularity makes all branches at the same depth isomorphic. Without
it, different branches have different shapes and different branch tensors.
You can't collapse to a single representative.

**Generalisation**: for graphs with a bounded number of distinct local
structures (degree profiles), you could track one branch tensor per profile.
Cost multiplies by the number of profiles.

---

## Acceptance Criteria

1. `CostAlgebra` abstract type and methods defined
2. `XORSATAlgebra` passes all existing tests
3. `MaxCutAlgebra()` produces identical results to existing MaxCut tests
4. `fold_tree` is problem-agnostic (no XORSAT-specific code)
5. `qaoa_expectation(algebra, params, angles)` is the primary public API
6. Old API `qaoa_expectation(params, angles; clause_sign)` still works (convenience wrapper)
7. All six properties documented with mathematical statements and explanations
8. "How to add a new problem" guide exists with worked example
9. No performance regression (fold_tree benchmarks identical to current code)
