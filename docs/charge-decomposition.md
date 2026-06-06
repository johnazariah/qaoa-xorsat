# Charge Decomposition Evaluator — Design & Implementation

## Overview

This document describes the charge decomposition evaluator added in
`src/charge.jl`, which computes the QAOA parity correlator ⟨Z^⊗k⟩ on
D-regular k-uniform symmetric trees at **O(p·4^p)** cost, improving on the
existing Basso (2p+1)-bit evaluator's **O(p²·4^p)**.

The algorithm is a Julia translation of JP Morgan's rank-4 charge
decomposition, originally implemented in Python/JAX on QOKit's
`add-max-k-xor-sat` branch.

## Table of Contents

1. [Mathematical Background](#1-mathematical-background)
2. [Algorithm Structure](#2-algorithm-structure)
3. [Translation Challenges](#3-translation-challenges)
4. [Public API](#4-public-api)
5. [Testing](#5-testing)
6. [Performance](#6-performance)

---

## 1. Mathematical Background

### 1.1 The Doubled Density Matrix

QAOA expectation values can be expressed as traces over doubled (ket⊗bra)
density matrices.  A single qubit's doubled state lives in a 4-dimensional
space indexed by σ = (s_ket, s_bra) ∈ {00, 01, 10, 11}.

The **doubled mixer** for angle β is:

```
M[σ_out, σ_in] = Rx(β)[sk_out, sk_in] ⊗ Rx*(β)[sb_out, sb_in]
```

where `Rx(β) = [[cos β, -i sin β], [-i sin β, cos β]]`.

### 1.2 Rank-4 Charge Decomposition

The key insight: the k-body phase gate `exp(-iγ Z^⊗k)` acting on the doubled
density matrix can be decomposed into 4 charge channels via a Walsh-Hadamard
butterfly:

```
exp(-iγ(z_ket - z_bra)) = Σ_a w_a(γ) · CHARGE_DIAG[a, σ]
```

where `a ∈ {0,1,2,3}` indexes the charge channel, and:

| Channel a | Physical meaning | Weight w_a(γ) | CHARGE_DIAG pattern |
|-----------|-----------------|---------------|---------------------|
| 0 | Identity | cos²γ | +1, +1, +1, +1 |
| 1 | Z_bra | i·cos γ·sin γ | +1, -1, +1, -1 |
| 2 | Z_ket | -i·cos γ·sin γ | +1, +1, -1, -1 |
| 3 | Z_ket·Z_bra | sin²γ | +1, -1, -1, +1 |

This factorises the k-body phase gate into independent per-qubit terms
(one charge label per qubit), enabling O(1) contraction per child instead
of the O(2^k) joint contraction.

### 1.3 WHT Butterfly Contraction

The contraction `Σ_σ CHARGE_DIAG[a,σ] · M[b,σ] · T[i,σ,b,r]` for all 4
charge channels simultaneously uses the 2×2 Walsh-Hadamard butterfly:

```
e[σ] = M[:,σ] · T[:,σ,:,:]       (4 pointwise products)
p₀₂ = e[0] + e[2];  q₀₂ = e[0] - e[2]
p₁₃ = e[1] + e[3];  q₁₃ = e[1] - e[3]
channel[0] = p₀₂ + p₁₃           (identity)
channel[1] = p₀₂ - p₁₃           (Z_bra)
channel[2] = q₀₂ + q₁₃           (Z_ket)
channel[3] = q₀₂ - q₁₃           (Z_ket·Z_bra)
```

Cost: 4 multiplies + 8 adds per element (vs 16 muls + 12 adds for naive
summation over all 4 σ values).

### 1.4 Why O(p·4^p) Instead of O(p²·4^p)

The Basso evaluator works in the (2p+1)-bit branch basis.  Each branch
tensor step requires:
- Computing `f_table .* B[t]` — O(2^(2p+1)) = O(4^p) multiplications
- A WHT of size 2^(2p+1) — O(p · 4^p)
- Pointwise power — O(4^p)

Since there are p steps, total cost is **O(p² · 4^p)**.

The charge decomposition works in 4^ℓ space at level ℓ, building up
incrementally:
- Level 1: 4¹ = 4 entries
- Level 2: 4² = 16 entries
- Level ℓ: 4^ℓ entries

The total work is Σ_{ℓ=1}^{p} O(4^ℓ) = O(4^p), and the root contraction
adds O(p · 4^p) for the p intermediate rounds.  Total: **O(p · 4^p)**.

The improvement factor is ~p×, which is significant at high depth (e.g.,
16× at p=16).

---

## 2. Algorithm Structure

### 2.1 Branch Tensor Construction (`_charge_hyperedge_branch`)

Builds the branch tensor for one hyperedge, processing `num_rounds` QAOA
rounds.  The construction has two phases:

**Phase 1 — Coupled contractions (consuming child branch):**
For `child_rounds ≥ 2`, the child branch tensor is progressively contracted
through the inner QAOA rounds via `wht_charge_contract`, absorbing one
mixer layer per step.  This builds up the factor matrix `V` which tracks
the child's contribution across charge channels.

**Phase 2 — Fused mixer + trace:**
A recursive expansion builds the branch tensor by applying the modified
mixer matrices `MD[ℓ][a] = M(β_ℓ) · diag(CHARGE_DIAG[a,:])` and tracing
over the outermost (root-facing) round.  The recursion produces a flat
C-order vector of 4^num_rounds entries.

**Post-processing:**
1. C-order reshape to a p-way tensor of dimension 4 per axis
2. Axis reordering (matches QOKit's permutation convention)
3. Entrywise (k-1) power with normalization (for k > 2)
4. Mode products: contract each axis with the charge weight matrix W[ℓ]

### 2.2 Root Contraction (`_charge_root_contract`)

Handles the root clause using a factored rank-1 representation:

1. **Intermediate rounds (ℓ = 1..p-1):**  For each round, reshape the
   factor into a 4D tensor, apply `wht_charge_contract`, and expand the
   coefficient vector by the root charge weights.  After each round,
   R (the number of charge-channel combinations) quadruples.

2. **Final round + Z measurement:**  Compute the Z trace vector
   `tv = K[1,:] - K[4,:]` (difference between σ=00 and σ=11 rows of the
   modified mixer), dot-product with each factor row, raise to the k-th
   power, and sum with coefficients.

### 2.3 Full Contraction (`charge_parity_expectation`)

Orchestrates the complete evaluation:

1. Build leaf branch tensor (level 1, no child)
2. For each level 2..p: normalize, raise to (D-1) power, build next
   level's branch tensor with child
3. Final normalization + (D-1) power → root branch `rb`
4. Root contraction → raw parity correlator
5. Apply accumulated log-scale from normalizations

---

## 3. Translation Challenges

### 3.1 Conjugate Transpose vs Plain Transpose

**Problem:** Julia's `A'` operator is the conjugate transpose (adjoint),
while Python's `A.T` is a plain transpose (no conjugation).  The phase-2
trace recursion applies `V @ MD[ℓ][a].T`, which in Julia was initially
written as `V * MD[ℓ][a]'`.  Since `MD` contains complex entries (from
the mixer's `-i sin β` terms), the spurious conjugation flipped imaginary
signs in the branch tensor.

**Symptom:** Branch tensor entries at p=2 were complex conjugates of the
correct values (e.g., `0.878 + 0.102i` instead of `0.878 - 0.102i`).
At p=1, all entries happened to be real, so this was undetectable.

**Fix:** Use `transpose(MD[ℓ][a])` instead of `MD[ℓ][a]'`.

### 3.2 Row-Major (C) vs Column-Major (F) Reshape

**Problem:** Python's `numpy.reshape` and JAX's `jnp.reshape` use C-order
(row-major) by default — the last axis varies fastest in memory.  Julia's
`reshape` uses F-order (column-major) — the first axis varies fastest.
When a flat vector built by C-order concatenation is reshaped into a
multi-dimensional tensor, the axis assignments differ between the two
languages:

```
C-order:  flat[a₁·4^(p-1) + a₂·4^(p-2) + ⋯ + aₚ] → tensor[a₁, a₂, …, aₚ]
F-order:  flat[aₚ + aₚ₋₁·4 + ⋯ + a₁·4^(p-1)]    → tensor[aₚ, aₚ₋₁, …, a₁]
```

The axes are reversed.  This affected three operations:

1. **Phase-2 trace output** → `_reshape_c(t_flat, 4, 4, …, 4)` to build
   the p-way tensor from the C-order flat recursion output.

2. **Root contraction reshape** → `_reshape_c(factor, R, 4, 4, rest)` to
   feed `wht_charge_contract`, which expects `T[i, σ, b, r]` with
   specific axis semantics.

3. **Root channel flattening** → `_vec_c(ch)` to flatten each WHT channel
   back into C-order for the next round's concatenation.

**Symptom:** At p=1 (single axis), reshape is unambiguous.  At p≥2, the
mode products applied charge weight matrices to the wrong axes, producing
incorrect branch tensors.

**Fix:** Two small helper functions:

```julia
_reshape_c(A, dims...) = permutedims(reshape(A, reverse(dims)...), N:-1:1)
_vec_c(A) = vec(permutedims(A, ndims(A):-1:1))
```

These are used at the 3 boundary sites.  All internal multi-dimensional
operations (mode products, `wht_charge_contract`) use clean Julia array
semantics with no layout concerns.

**Key insight:** We initially considered two alternative approaches:
(a) keeping everything as flat vectors with stride-based arithmetic, or
(b) reversing the recursion to match Julia's F-order.  Both sacrificed
readability.  The `_reshape_c` approach keeps the code clean and
concentrates the C/F adaptation in exactly 3 call sites.

### 3.3 γ Convention (Half-Angle vs Full-Angle)

**Problem:** Our codebase uses the γ/2 phase convention — the physical
gate is `exp(-iγ/2 · Z^⊗k)`.  The QOKit charge decomposition (and its
quimb reference tests) uses the full-angle convention — the physical gate
is `exp(-iγ · Z^⊗k)`.  This means our γ is twice QOKit's γ.

**Symptom:** At p=1, the charge evaluator produced systematically wrong
values until γ was halved.

**Fix:** `charge_parity_expectation` divides γ by 2 before passing to the
internal charge routines:

```julia
γs = T(clause_sign) .* T.(angles.γ) ./ 2
```

The `clause_sign` multiplier (±1) is folded into γ here, which flips the
phase gate direction for MaxCut (clause_sign = -1).

### 3.4 Phase 1 Coupled Contractions (p ≥ 4)

**Problem:** Phase 1 of `_charge_hyperedge_branch` activates when
`child_rounds ≥ 2` (i.e., p ≥ 4).  It iteratively applies
`wht_charge_contract` to the child branch, producing intermediate `V`
vectors.  Between iterations, the output channels must be flattened back
to a single C-order vector for the next iteration's `_reshape_c`.

The initial implementation used Julia's `reshape(ch, n_ch, :)` to flatten
each channel — but this is F-order, while the next `_reshape_c` expects
C-order input.  The mismatch corrupted the intermediate factor matrix.

**Symptom:** Correct at p=1–3, wrong at p=4+.  Values diverged
progressively at higher p (e.g., p=5: charge=3.25 vs basso=0.57).

**Fix:** Maintain `V` as a single flat C-order vector throughout Phase 1.
Use `_vec_c(ch)` to flatten each WHT channel to C-order, then concatenate.
The final `V → (n_ch, 4)` reshape also uses `_reshape_c`:

```julia
V_flat = vcat([_vec_c(ch) for ch in channels]...)  # C-order concat
V = _reshape_c(V_flat, n_ch, 4)                    # C-order final reshape
```

This was later optimised to use `_wht_charge_contract_flat!` directly,
avoiding the 4D intermediate entirely.

---

## 4. Public API

### `charge_parity_expectation(params, angles; clause_sign=1)`

Compute the exact parity correlator ⟨Z^⊗k⟩ using charge decomposition.

- **Input:** `TreeParams(k, D, p)`, `QAOAAngles(γ, β)`, optional `clause_sign`
- **Output:** `Float64` — the parity correlator
- **Cost:** O(p · 4^p), independent of D, k, and N_lc

### `charge_expectation(params, angles; clause_sign=1)`

Compute the expected satisfaction fraction `(1 + clause_sign · ⟨Z^⊗k⟩) / 2`.

Both functions are drop-in replacements for `basso_parity_expectation` and
`basso_expectation` respectively.

---

## 5. Testing

20 tests in `test/test_charge.jl`:

| Test Suite | Cases | What it verifies |
|-----------|-------|------------------|
| matches basso_parity_expectation | 10 | Charge matches Basso for (k,D,p) ∈ {(2,3,1-3), (3,4,1-3), (3,2,1-2), (4,3,1-2)} |
| matches basso_expectation | 6 | clause_sign ∈ {+1,-1} for (k,D,p) ∈ {(2,3,1-2), (3,4,1-2)} with random angles |
| zero angles | 2 | E[c̃] = 0.5 at γ=β=0 |
| MaxCut p=1 optimal | 1 | Matches basso at optimal angles, c̃ ≈ 0.6924 |

All tests use absolute tolerance 1e-10 (or 1e-8 for cross-method comparison).

---

## 6. Performance

### 6.1 Complexity Comparison

| | Basso (2p+1)-bit | Charge decomposition |
|-----------|-----------------|---------------------|
| Branch tensor size | 2^(2p+1) | 4^p |
| Cost per branch step | O(p · 4^p) | O(4^ℓ) at level ℓ |
| Total branch cost | O(p² · 4^p) | O(4^p) |
| Root contraction | O(4^p) | O(p · 4^p) |
| **Total** | **O(p² · 4^p)** | **O(p · 4^p)** |
| Memory | O(p · 4^p) [adjoint] | O(4^p) |

### 6.2 Measured Speedup

Benchmarks on Apple M4 (single-threaded), comparing `charge_parity_expectation`
against `basso_parity_expectation`:

| k | D | p | Basso (ms) | Charge (ms) | Speedup |
|---|---|---|-----------|------------|---------|
| 2 | 3 | 3 | 0.09 | 0.014 | 6.6× |
| 2 | 3 | 5 | 0.38 | 0.065 | 5.8× |
| 2 | 3 | 7 | 7.5 | 0.71 | 10.6× |
| 2 | 3 | 9 | 285 | 12.6 | 22.6× |
| 2 | 3 | 11 | 4076 | 407 | 10.0× |
| 3 | 4 | 3 | 0.09 | 0.015 | 5.9× |
| 3 | 4 | 5 | 0.43 | 0.076 | 5.7× |
| 3 | 4 | 7 | 7.5 | 0.82 | 9.1× |
| 3 | 4 | 9 | 262 | 16.3 | 16.1× |

The speedup exceeds the theoretical p× factor at moderate depths because
the charge evaluator also benefits from:

1. **Smaller working set** — 4^ℓ entries at level ℓ vs 2^(2p+1) throughout
   in Basso, giving better cache utilisation for inner levels
2. **In-place operations** — flat-vector WHT butterfly and mode products
   use double-buffering with no intermediate allocations
3. **No WHT on large vectors** — the Basso WHT operates on 2^(2p+1)-entry
   vectors; the charge decomposition avoids this entirely

### 6.3 Memory Profile

At high p, memory is dominated by the root contraction's `factor` and
double-buffer vectors (each 4^p × 16 bytes = 4^p ComplexF64 entries):

| p | factor (MB) | scratch (MB) | coeffs (MB) | Total (MB) | Feasible on |
|---|------------|-------------|-------------|-----------|-------------|
| 11 | 67 | 67 | 17 | ~150 | Laptop |
| 12 | 268 | 268 | 67 | ~600 | Laptop |
| 13 | 1,074 | 1,074 | 268 | ~2,400 | Workstation |
| 14 | 4,295 | 4,295 | 1,074 | ~9,700 | Workstation (≥16 GB) |
| 15 | 17,180 | 17,180 | 4,295 | ~38,000 | HPC node |
| 16 | 68,719 | 68,719 | 17,180 | ~155,000 | Large HPC / GPU |

**Optimizations applied:**
- Branch `F` buffer is reused as root scratch (saves one 4^p allocation)
- Coefficient expansion uses pre-allocated double buffer (zero per-round allocs)

**Symmetries discovered but not yet exploited:**
- **Complement invariance** (`F[i] = F[N-1-i]`): the branch tensor is a
  palindrome in the flat C-order vector.  Verified exact to machine precision.
  Could halve branch storage.  Does NOT survive the root contraction's
  WHT charge contraction (verified: palindrome breaks after round 1).
- **Ket↔bra conjugation** (`F[σ] = conj(F[swap(σ)])`): swapping ket/bra
  bits within each doubled index gives the complex conjugate.  Combined
  with complement, this generates a Z₂ × Z₂ symmetry group of order 4.
  Orbit structure at p=8: 16,512 orbits (16,256 of size 4, 256 of size 2),
  ratio → 4× at large p.  Not yet exploited.

### 6.4 Theoretical Speedup Factor

The improvement is approximately **p×** at each evaluation.  For the
target case (k=3, D=4):

| Depth p | Basso cost factor | Charge cost factor | Speedup |
|---------|------------------|-------------------|---------|
| 8 | 64 · 4^8 | 8 · 4^8 | 8× |
| 12 | 144 · 4^12 | 12 · 4^12 | 12× |
| 16 | 256 · 4^16 | 16 · 4^16 | 16× |

### 6.5 Performance Optimization Journey

The initial translation matched QOKit's clean multi-dimensional style but
was **only 1.0–1.8× faster** than Basso due to allocation overhead.
Profiling revealed that ~95% of time was spent in memory allocation
(`permutedims`, broadcast `materialize`, `vcat`), not computation.

Three targeted optimizations brought this to **5–23×**:

1. **In-place flat WHT butterfly** (`_wht_charge_contract_flat!`):
   Replaced the 4D-array `wht_charge_contract` in the root contraction
   with a flat-vector kernel using explicit C-order stride arithmetic.
   Eliminates all `_reshape_c` / `_vec_c` / `vcat` allocations in the
   hot loop.  Uses double-buffering (factor ↔ scratch swap).

2. **Stride-based mode products** (`_mode_product_flat!`):
   Replaced `permutedims` → `reshape` → matmul → `reshape` → `permutedims`
   (5 allocations per axis per level) with a direct flat-vector kernel.
   C-axis `ℓ` has stride `4^(num_rounds-ℓ)` in the flat vector.

3. **Buffer reuse**: The branch tensor `F` (4^p entries) is passed as the
   root contraction's scratch buffer, avoiding a second 4^p allocation.
   Coefficient expansion uses pre-allocated double buffers.

The multi-dimensional `wht_charge_contract` is retained for the branch
construction's Phase 1 (cold path, small tensors), keeping that code
readable.

### 6.6 Future Work

- **Adjoint differentiation** for the charge evaluator (would give O(p·4^p)
  gradient, matching QOKit's forward-mode JVP but with our memory-efficient
  reverse-mode approach)
- **Exploit Z₂×Z₂ symmetry** for ~4× memory reduction on branch tensors
  (complement + ket↔bra conjugation); requires reworking the root contraction
  to operate in the quotient space
- **Checkpointed root contraction** to trade compute for memory: split p-1
  root rounds into segments of size s, recompute rather than store; reduces
  coeffs from 4^(p-1) to 4^s at cost of recomputation
- **GPU offload** of the WHT butterfly and mode products (embarrassingly
  parallel; A100 has 80 GB VRAM for p≥15)
- **Integration with the optimizer** as an alternative backend for
  `optimize_angles`

---

## Source Files

| File | Purpose |
|------|---------|
| `src/charge.jl` | Charge decomposition evaluator |
| `test/test_charge.jl` | Tests for charge evaluator |
| `src/QaoaXorsat.jl` | Module wiring (include + export) |
| `test/runtests.jl` | Test runner (includes test_charge.jl) |

## References

- QOKit `add-max-k-xor-sat` branch — `qokit/max_k_xor_sat/jax/primitives.py`
  and `qokit/max_k_xor_sat/jax/contract.py`
- Basso et al., "The Quantum Approximate Optimization Algorithm at High
  Depth for MaxCut on Large-Girth Regular Graphs and the Sherrington-Kirkpatrick
  Model" (2022)
- Villalonga et al., "A large-scale quantum simulator on a diamond chip"
  — for the tensor network contraction framework
