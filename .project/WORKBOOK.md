# QAOA-XORSAT Workbook

> **Purpose**: A self-contained walkthrough of what we are doing, why each step
> matters, and which tests prove it works. Written so that John can explain every
> decision to Stephen Jordan from first principles.

---

## 1. The Problem We Are Solving

Stephen Jordan wants to know: **how well does QAOA perform on D-regular
Max-k-XORSAT, at specific small (k, D)?**

The primary target is **(k=3, D=4)**. Stephen has performance numbers for
DQI, simulated annealing, and other algorithms at this point. We are computing
the QAOA column to complete the comparison.

### Why this matters

DQI (Decoded Quantum Interferometry) and QAOA are fundamentally different
quantum approaches to combinatorial optimisation. Comparing them at specific
(k, D) values reveals which structural features each algorithm exploits.
At (k=3, D=4), DQI is weak (blocked by OGP on random instances), SA is
strong (0.9366), and QAOA's performance is unknown. Computing it fills a
gap in the literature.

### The competition at (k=3, D=4)

| Algorithm | Satisfaction fraction | Status |
|-----------|----------------------|--------|
| Random | 0.5000 | Trivial baseline |
| DQI+BP | 0.87065 | Weak at this (k,D) |
| Prange | 0.875 | Trivial DQI baseline |
| Regev+FGUM | 0.89187 | Quantum-inspired |
| **SA** | **0.9366** | **The bar to clear** |
| **QAOA(p)** | **???** | **Our computation** |

### Repository layout

The codebase is on branch **`feature/p1.3-contraction`**, one commit ahead of
`main`. The merged work (P1.1 tree, P1.2 tensors) is on `main`; the current
branch adds `src/qaoa.jl` (brute-force light-cone simulator) and
`test/test_qaoa.jl`. All **269 tests pass**.

### What "QAOA performance at depth p" means precisely

For a single root constraint α of a D-regular k-uniform hypergraph with girth > 2p, the QAOA expected satisfaction fraction is:

$$\tilde{c}(p) = \max_{\gamma_1,\ldots,\gamma_p,\,\beta_1,\ldots,\beta_p}
\langle\gamma,\beta\,|\,C_\alpha\,|\,\gamma,\beta\rangle$$

where C_α = (1 + Z₁⋯Zₖ)/2 is the constraint observable, and:

$$|\gamma,\beta\rangle = \prod_{\ell=1}^{p}
e^{-i\beta_\ell \sum_i X_i}\;e^{-i\gamma_\ell \sum_\alpha C_\alpha}
\;|{+}\rangle^{\otimes n}$$

On a locally-tree-like graph (girth > 2p), this expectation value depends only
on the tree-shaped neighbourhood of the root constraint out to depth p —
the **light-cone tree**.

---

## 2. The Method: Tensor Network Contraction on the Light-Cone Tree

### 2.1 Why we can work on a tree

The QAOA state at depth p is created by p rounds of local unitaries (problem
gates on constraints, mixer gates on variables). The expectation value of
a single constraint depends only on qubits within distance p in the factor
graph. On a graph with girth > 2p, this neighbourhood is a **tree** — there
are no cycles.

This is the "light-cone" argument (Farhi et al. 2014, §III):
the global optimisation on an n-qubit graph reduces to a local
computation on a finite tree.

### 2.2 The tree structure

The light-cone tree is a **bipartite factor graph** alternating between
constraint nodes (hyperedges) and variable nodes (qubits):

```
Level 0:   [root constraint]           ← 1 constraint
            /      |      \
Level 1:  (x₁)   (x₂)   (x₃)         ← k variable nodes
          /|\     /|\     /|\
Level 2: [·]×3  [·]×3   [·]×3         ← k(D-1) = 9 constraint nodes
         /|\    /|\     /|\
Level 3: (·)×2 per constraint          ← k(D-1)(k-1) = 18 variable nodes
         ...
```

At each two-level step, the tree branches by factor **b = (D-1)(k-1)**.

For (k=3, D=4): b = 6. For MaxCut (k=2, D=3): b = 2.

**This is implemented and tested** (merged to `main`). The `TreeParams` struct
captures (k, D, p) and all counting functions are derived from it.

> **Code**: `src/tree.jl` — `TreeParams`, `branching_factor`, `variable_count_at_level`,
> `constraint_count_at_level`, `total_variables`, `total_constraints`, `leaf_count`
>
> **Tests**: `test/test_tree.jl` — **120 tests passing**
>
> | What's tested | How | Test names |
> |---------------|-----|------------|
> | Parameter validation | k<2, D<2, p<1 all rejected | `TreeParams > construction` |
> | Branching factor | b=2 for MaxCut, b=6 for (3,4), b=12 for (4,5) | `TreeParams > branching_factor` |
> | MaxCut node counts | Exact match to spec table for p=1..4 | `TreeParams > physical light-cone counts k=2, D=3` |
> | Target node counts | Exact match to spec table for p=1..5 | `TreeParams > physical light-cone counts k=3, D=4` |
> | Leaf counts | k·bᵖ verified | `TreeParams > leaf_count` |
> | Monotonicity | total_nodes grows with p for all tested (k,D) | `TreeParams > monotonicity` |
> | Bounds checking | Out-of-range level indices throw errors | `TreeParams > level count bounds` |

**Key validation data** (from spec P1.1, matches the paper):

| p | Variables (k=2,D=3) | Variables (k=3,D=4) |
|---|---------------------|---------------------|
| 1 | 6 | 21 |
| 2 | 14 | 129 |
| 3 | 30 | 777 |
| 4 | 62 | 4,665 |

### 2.3 The bra-ket sandwich and hyperindex representation

To compute ⟨γ,β|C_α|γ,β⟩, we expand the circuit into a tensor network.
Each qubit contributes a **ket** (forward circuit) and **bra** (conjugate
circuit). Because the problem gates are diagonal in the Z basis and the
initial state |+⟩ is separable, we can combine ket and bra into a single
**hyperindex** per qubit.

For each qubit, the hyperindex σ is a 2p-bit string:

$$\sigma = (\text{ket}_1, \text{bra}_1, \text{ket}_2, \text{bra}_2,
\ldots, \text{ket}_p, \text{bra}_p)$$

This gives 4ᵖ possible values per qubit. We chose **interleaved ordering**
(ket₁ and bra₁ adjacent) so that each QAOA round occupies a contiguous
bit-pair — important for efficient contraction.

**This mapping is implemented and tested** (merged to `main`).

> **Code**: `src/tensors.jl` — `hyperindex_dimension`, `slice_bit_positions`,
> `hyperindex_bit`, `hyperindex_parity`
>
> **Tests**: `test/test_tensors.jl` — **25 hyperindex tests passing**
>
> | What's tested | How |
> |---------------|-----|
> | Dimension: 4ᵖ | hyperindex_dimension(1)=4, (3)=64 |
> | Bit positions | slice 1 → bits (1,2); slice 3 → bits (5,6) |
> | Round↔slice mapping | round 1 at depth 3 → slice 3 (outermost); round 3 → slice 1 (root) |
> | Bit extraction | hyperindex_bit(0b1010, 2) = 1 |
> | Parity | hyperindex_parity(0b1010, [1,2]) = 1 (XOR of bits) |

### 2.4 The four tensor primitives

The tensor network has four types of local tensors:

#### Leaf tensor (boundary qubits in |+⟩)

Each boundary qubit starts in |+⟩. Since X|+⟩ = |+⟩, the mixer's action on
the initial state produces a constant: at each round ℓ, the bra-ket
contribution is exactly 1/2 (the phases cancel). Over p rounds:

$$L(\sigma) = 2^{-p} \quad \text{for all } \sigma$$

The leaf tensor is **angle-independent** — this is a non-trivial fact that
falls out of |+⟩ being an eigenstate of X.

> **Code**: `src/tensors.jl` — `leaf_tensor(angles)`
>
> **Tests**: `test/test_tensors.jl` — **19 leaf tensor tests passing**
>
> | What's tested | How |
> |---------------|-----|
> | Correct dimension | length = 4ᵖ for p=1..4 |
> | Angle independence | Random angles give same tensor |
> | Correct value | Every entry ≈ 2⁻ᵖ |
> | Real-valued | eltype is Real |

#### Mixer tensor (single-qubit X rotation)

The mixer gate e^{-iβX} acts on one qubit at one QAOA round. In the
hyperindex picture, it becomes a 4×4 superoperator block on the (ketₗ, braₗ)
bits for that round, tensored with the identity on all other round-bits.

The 4×4 block (for round ℓ) has entries:

$$M_{(k',b'),(k,b)} = \langle k'|e^{-i\beta X}|k\rangle \cdot
\overline{\langle b'|e^{-i\beta X}|b\rangle}$$

At β=0 this is the identity. The full tensor is a 4ᵖ × 4ᵖ matrix.

> **Code**: `src/tensors.jl` — `mixer_tensor(β, slice, p)`
>
> **Tests**: `test/test_tensors.jl` — **22 mixer tensor tests passing**
>
> | What's tested | How |
> |---------------|-----|
> | Correct dimension | 4ᵖ × 4ᵖ for p=1..4 |
> | Identity at β=0 | mixer_tensor(0,…) ≈ I |
> | Golden values at β=π/6 | First column matches hand-derived (cos²β, -i·cosβ·sinβ, +i·cosβ·sinβ, sin²β) |
> | Round locality | Round 2 at p=2 affects only bits 3,4; bits 1,2 unchanged |
> | Unitarity | M·M† ≈ I for random β, p=1..3 |
> | Periodicity | mixer_tensor(β) = mixer_tensor(β+2π) |

#### Problem tensor (k-body diagonal phase)

The problem gate exp(-iγ Z₁⋯Zₖ/2) is diagonal — it multiplies each
computational basis state by a phase depending on the parity Z₁⋯Zₖ.
In the bra-ket sandwich:

$$P(\sigma_1, \ldots, \sigma_k) = \text{cis}\!\bigl(\gamma\,(z_{\text{bra}} - z_{\text{ket}})/2\bigr)$$

where z_ket and z_bra are the product of Z eigenvalues on the ket and bra
sides respectively. The tensor is stored as a flattened diagonal of length
(4ᵖ)ᵏ.

At γ=0 every entry is 1. The `clause_sign` parameter handles even (+1) vs
odd (-1) parity constraints (MaxCut uses odd).

> **Code**: `src/tensors.jl` — `problem_tensor(k, γ, slice, p; clause_sign)`
>
> **Tests**: `test/test_tensors.jl` — **26 problem tensor tests passing**
>
> | What's tested | How |
> |---------------|-----|
> | Correct dimension | length = (4ᵖ)ᵏ for k=2,3 and p=1,2 |
> | All-ones at γ=0 | No phase at zero angle |
> | MaxCut golden values | k=2, γ=π/3 entries: (1, cis(γ), cis(-γ), 1) match hand derivation |
> | 3-XORSAT golden values | k=3, γ=π/3 entries match hand derivation |
> | Odd-clause sign | clause_sign=-1 swaps cis(γ) ↔ cis(-γ) |
> | Periodicity | 2π-periodic |

#### Observable tensor (root constraint measurement)

The root observable C_α = (1 + clause_sign · Z₁⋯Zₖ)/2 is diagonal. The
tensor is non-zero only when ket₁ = bra₁ for all root qubits (because the
observable is in the computational basis). Values are 0 or 1.

> **Code**: `src/tensors.jl` — `observable_tensor(k, p; clause_sign)`,
> `parity_observable_tensor(k, p)`
>
> **Tests**: `test/test_tensors.jl` — **42 observable tests passing** (22 parity + 20 full)
>
> | What's tested | How |
> |---------------|-----|
> | Correct dimension | length = (4ᵖ)ᵏ |
> | Parity values | Z₁Z₂ gives ±1 on diagonal, 0 off-diagonal |
> | Even clause | (1+Z₁⋯Zₖ)/2: entries in {0, 1} |
> | Odd clause | (1−Z₁⋯Zₖ)/2: complementary to even |
> | Completeness | Sum of parity tensor = 0 (balanced) |

---

## 3. The Reference Implementation: Brute-Force Statevector Simulation

> **Branch**: `feature/p1.3-contraction` — added in this branch, not yet merged to `main`.

Before implementing the efficient O(4ᵖ) contraction, we built a
**correctness oracle** — a brute-force simulation that explicitly constructs
the light-cone tree, creates a 2ⁿ-amplitude statevector, and applies every
gate qubit-by-qubit.

This is intentionally slow (exponential in the total number of qubits) but
**unambiguously correct** — there is no cleverness to get wrong. It serves
as the ground truth for validating the eventual efficient contraction.

### How it works

1. **Build the explicit tree** (`build_light_cone_tree`): assigns qubit indices
   1, 2, 3, … to every variable node, records every constraint as a list of
   qubit indices. Asserts the counts match `TreeParams`.

2. **Initialise |+⟩** (`plus_state`): uniform superposition over all 2ⁿ
   basis states.

3. **Apply p rounds**:
   - Problem layer: for each basis state, compute the total parity across all
     constraints, multiply the amplitude by cis(-γₗ · parity_sum / 2).
   - Mixer layer: for each qubit, apply the 2×2 rotation e^{-iβₗX}.

4. **Measure**: compute ⟨Z₁⋯Zₖ⟩ on the root constraint by summing
   |amplitude|² × (Z eigenvalue product) over all basis states.

### Safety guard

The implementation is **guarded to ≤ 22 qubits** (`MAX_EXACT_LIGHTCONE_QUBITS`).
Beyond this, the statevector would be too large. This means:

- MaxCut (k=2, D=3): feasible at p=1 (6 qubits), p=2 (14 qubits), p=3 (30 — **blocked**)
- Target (k=3, D=4): feasible at p=1 (21 qubits), p=2 (129 — **blocked**)

> **Code**: `src/qaoa.jl` — `build_light_cone_tree`, `simulate_light_cone_state`,
> `parity_expectation`, `qaoa_expectation`
>
> **Tests**: `test/test_qaoa.jl` — **10 QAOA evaluation tests passing**
>
> | What's tested | Why it matters |
> |---------------|----------------|
> | Zero-angle baseline: ⟨Z₁⋯Zₖ⟩ = 0, c̃ = 0.5 | At γ=β=0, QAOA is just random guessing |
> | MaxCut p=1 formula: ⟨ZZ⟩ = -sin(4β)cos²(γ)sin(γ) | Matches the known analytical result from Farhi 2014 (two angle points tested) |
> | **MaxCut p=1 optimum: c̃ = ½ + √3/9 ≈ 0.6924** | **The foundational validation target** — this is the first result in the QAOA literature (Farhi et al. 2014, §III) |
> | MaxCut p=2 cross-check against independent reference | An independent MaxCut-specific statevector simulator (defined in test_qaoa.jl) produces the same answer at specific (γ, β) angles — validates that our general k-XORSAT code agrees with a MaxCut-only implementation |
> | Guard test: (k=3, D=4, p=2) throws ArgumentError | Confirms the 22-qubit safety limit is enforced — 129 qubits is too large for brute force |

### The MaxCut p=1 validation in detail

This is the most important single test. The analytical formula for 3-regular
MaxCut at p=1 is:

$$\langle Z_u Z_v \rangle = -\sin(4\beta)\cos^2(\gamma)\sin(\gamma)$$

This is a function of a 6-qubit tree (root edge + 2 vertices, each with 2
additional neighbours). Maximising over (γ, β):

- Optimal angles: γ* = arctan(1/√2), β* = π/8
- Optimal parity: ⟨ZZ⟩* = -√3/9
- Optimal cut fraction: c̃* = (1 - ⟨ZZ⟩*)/2 = ½ + √3/9 ≈ **0.6924**

Our test evaluates `qaoa_expectation(TreeParams(2,3,1), QAOAAngles([γ*],[β*]); clause_sign=-1)`
and checks it matches 0.6924 to 10 decimal places. **This passes.**

---

## 4. What Remains: The O(4ᵖ) Contraction (Spec P1.3)

### The problem with brute force

The brute-force approach stores 2ⁿ amplitudes. For (k=3, D=4) at p=2,
n = 129 qubits → 2¹²⁹ amplitudes. That's more atoms than in the observable
universe. Brute force is fundamentally impossible beyond tiny trees.

### The key insight: branch symmetry + element-wise exponentiation

On a regular tree, **every branch at the same depth is identical** (by the
regularity of the hypergraph). Instead of contracting each branch separately,
we contract **one representative branch** and raise it to the appropriate power.

This is the element-wise exponentiation trick from Farhi et al. 2025:

1. Start at the **leaves** with the leaf tensor (a vector of 4ᵖ entries)
2. Contract inward one level at a time, alternating:
   - **Variable step**: raise the branch tensor element-wise to the (D-1)th
     power (accounts for D-1 identical child constraints), then apply the
     mixer tensor for this round
   - **Constraint step**: combine (k-1) identical child variable branches
     (this is a multilinear operation, not simple exponentiation for k>2),
     then apply the problem gate tensor for this round
3. After p variable steps and p constraint steps, apply the **root observable**
   to get the scalar expectation value

### Cost

Each step processes a tensor of 4ᵖ entries. There are 2p steps total.

**Total: O(p · 4ᵖ) time, O(4ᵖ) space** — independent of k and D.

This is the crucial result: k and D only affect the exponents in the
element-wise operations, not the tensor dimensions.

| p | 4ᵖ | Memory (8 bytes/entry) | Feasibility |
|---|-----|------------------------|-------------|
| 5 | 1,024 | 8 KB | Trivial |
| 10 | 1,048,576 | 8 MB | Easy |
| 12 | 16,777,216 | 128 MB | Comfortable |
| 15 | 1,073,741,824 | 8 GB | Needs care |
| 17 | 17,179,869,184 | 128 GB | HPC |

Farhi et al. 2025 pushed to **p=17** for MaxCut using C++ with OpenMP.

### Why this hasn't been implemented yet

The spec (P1.3) is written and the algorithm is clear in outline. The
remaining subtlety is the **constraint-step contraction for k > 2**.

For MaxCut (k=2), the root edge has 2 variables, and the constraint step
is a simple element-wise exponentiation by (k-1) = 1 (trivially the identity).
The entire contraction is just variable steps.

For k=3, each non-root constraint connects to (k-1) = 2 child variables.
The combination of 2 child branch tensors is **multilinear** (a tensor
product followed by contraction with the problem gate), not a simple
element-wise power. The tensor derivation notes flag this:

> "Constraint contraction is multilinear in k-1 child messages.
> The draft .^ (k-1) rule at non-root constraints is wrong in general."

This requires careful derivation of the effective transfer object that
maps raw P1.2 tensors into the branch-transfer recursion of Eq. (14) in
Farhi 2025, generalised from k=2 to k=3. That derivation is the next
piece of mathematical work.

### How we will validate P1.3

The brute-force reference exists precisely for this purpose:

1. **MaxCut p=1**: contraction must give c̃ ≈ 0.6924 (already validated by brute force)
2. **MaxCut p=2**: contraction must match the brute-force result at specific angles (cross-check with independent MaxCut reference already exists in tests)
3. **(k=3, D=4) p=1**: contraction must match the brute-force result (21 qubits — within the guard limit)
4. **MaxCut p=5**: c̃ ≈ 0.8333 (Farhi 2025, Table 1) — this is the first value unreachable by brute force
5. **MaxCut p=7**: c̃ ≈ 0.8536 (Farhi 2025, Table 1)

Targets 1–3 validate against our own reference oracle. Targets 4–5 validate
against published literature values.

---

## 5. After Contraction: Angle Optimisation (Phase 4)

Once P1.3 gives us a function `qaoa_expectation(params, angles)` that runs
in O(p · 4ᵖ), we optimise over the 2p angles (γ₁,…,γₚ, β₁,…,βₚ) to find
the maximum satisfaction fraction at each depth.

**Method**: L-BFGS with multiple random restarts, seeded from optimal angles
at smaller p. This is standard practice in the QAOA literature (Basso et al.
2021, Farhi et al. 2025).

The output is a table:

| p | c̃(p) for (k=3, D=4) | Optimal γ | Optimal β |
|---|----------------------|-----------|-----------|
| 1 | ??? | … | … |
| 2 | ??? | … | … |
| … | … | … | … |
| p_max | ??? | … | … |

We keep pushing p until c̃(p) is high enough to compare meaningfully against
SA's 0.9366, or until we hit computational limits.

---

## 6. The Full Pipeline, End to End

```
    (k, D, p, γ, β)
         │
         ▼
┌─────────────────┐
│  TreeParams      │  ← "How big is the light cone?"
│  (counting only) │     120 tests validate all sizes
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Raw tensors     │  ← "What does each gate look like in the hyperindex picture?"
│  leaf, mixer,    │     134 tests validate all tensor primitives
│  problem, obs.   │
└────────┬────────┘
         │
         ▼  (NOT YET IMPLEMENTED)
┌─────────────────┐
│  O(4ᵖ)           │  ← "Contract the tree efficiently"
│  contraction     │     Will validate against brute-force reference
└────────┬────────┘
         │
         ▼  (NOT YET IMPLEMENTED)
┌─────────────────┐
│  Angle           │  ← "Find the best (γ, β)"
│  optimisation    │     L-BFGS with restarts
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Comparison      │  ← "How does QAOA compare at (k=3, D=4)?"
│  table           │     vs. SA, DQI+BP, Regev+FGUM, Prange
└─────────────────┘
```

---

## 7. Summary of Completed Work

| Step | What | Where | Tests | Status |
|------|------|-------|-------|--------|
| Tree counting | TreeParams, branching, node counts | src/tree.jl | 120 pass | **Merged** (P1.1) |
| Hyperindex scheme | Interleaved (ket₁,bra₁,…) bit layout | src/tensors.jl | 25 pass | **Merged** (P1.2) |
| Leaf tensor | L(σ) = 2⁻ᵖ ∀σ (angle-independent) | src/tensors.jl | 19 pass | **Merged** (P1.2) |
| Mixer tensor | 4ᵖ×4ᵖ superoperator, unitary, round-local | src/tensors.jl | 22 pass | **Merged** (P1.2) |
| Problem tensor | k-body diagonal phase cis(γ(z_bra−z_ket)/2) | src/tensors.jl | 26 pass | **Merged** (P1.2) |
| Observable tensor | Root constraint (1+Z₁⋯Zₖ)/2 | src/tensors.jl | 42 pass | **Merged** (P1.2) |
| Brute-force reference | Full statevector simulation ≤22 qubits | src/qaoa.jl | 10 pass | **WIP** (P1.3 branch) |
| MaxCut p=1 validation | c̃ = ½+√3/9 ≈ 0.6924 | test/test_qaoa.jl | 1 pass | **WIP** (P1.3 branch) |
| MaxCut p=2 cross-check | Matches independent reference | test/test_qaoa.jl | 2 pass | **WIP** (P1.3 branch) |

**Total: 269 tests, all passing.**

## 8. What Comes Next

| Step | What | Depends on | Validates against |
|------|------|------------|-------------------|
| **P1.3 contraction** | O(4ᵖ) tree contraction algorithm | P1.2 tensors | Brute force at p=1,2; literature at p=5,7 |
| Angle optimisation | L-BFGS over 2p angles | P1.3 | MaxCut published values |
| (k=3,D=4) computation | Push p as high as possible | Angle opt | Stephen's comparison table |
| Comparison | Add QAOA column to the table | Computation | — |

---

## Appendix A — Key Equations to Know

### The QAOA state

$$|\gamma,\beta\rangle = \prod_{\ell=1}^{p}
e^{-i\beta_\ell \sum_i X_i}\;e^{-i\gamma_\ell \sum_\alpha C_\alpha}
\;|{+}\rangle^{\otimes n}$$

### The k-XORSAT cost operator

$$C_\alpha = \frac{1 + (-1)^{b_\alpha} Z_{i_1} Z_{i_2} \cdots Z_{i_k}}{2}$$

For even parity (b_α=0): C_α = (1 + Z₁⋯Zₖ)/2.
For odd parity (b_α=1): C_α = (1 − Z₁⋯Zₖ)/2 (this is the MaxCut convention).

### MaxCut p=1 analytical formula (3-regular)

$$\langle Z_u Z_v \rangle = -\sin(4\beta)\cos^2(\gamma)\sin(\gamma)$$

$$\tilde{c}_{\text{edge}} = \frac{1}{2} + \frac{\sqrt{3}}{9} \approx 0.6924$$

### Branching factor

$$b = (D-1)(k-1)$$

### Tree size

Variable nodes at shell j: $k \cdot b^j$. Leaf count: $k \cdot b^p$.

### Contraction cost

$$O(p \cdot 4^p) \text{ time}, \quad O(4^p) \text{ space}$$

---

## Appendix B — Glossary for Conversations with Stephen

| Term | Meaning |
|------|---------|
| Light cone | Tree-shaped neighbourhood of a constraint, out to depth p |
| Hyperindex | 2p-bit index combining ket and bra paths through the QAOA sandwich |
| Element-wise exponentiation | Raising branch tensor entries to the (D-1)th power to account for identical siblings |
| Transfer tensor / branch tensor | The contracted representation of a subtree, a vector of 4ᵖ entries |
| Clause sign | +1 for even parity (XORSAT), -1 for odd parity (MaxCut) |
| Prange bound | 1/2 + k/(2D) — trivial DQI baseline from random decoding |
| Semicircle law | DQI's performance ceiling: √(ℓ/m · (1-ℓ/m)), governed by decoding radius ℓ |
| OGP | Overlap gap property — topological barrier that blocks "stable" algorithms on random instances |
| SA | Simulated annealing — the classical benchmark (0.9366 at (3,4)) |
| DQI+BP | DQI with belief propagation decoder — weak at (3,4) because BP fails on random LDPC |

---

## Appendix C — How to Run Everything

```bash
# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run all 269 tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Quick interactive check of the MaxCut p=1 optimum
julia --project=. -e '
using QaoaXorsat
params = TreeParams(2, 3, 1)
γ, β = atan(1/sqrt(2)), π/8
angles = QAOAAngles([γ], [β])
println("MaxCut p=1 optimum: ", qaoa_expectation(params, angles; clause_sign=-1))
println("Expected:           ", 0.5 + sqrt(3)/9)
'
```
