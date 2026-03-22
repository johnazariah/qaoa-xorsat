# Testing Register — QAOA-XORSAT

**Last updated**: 22 March 2026
**Total tests**: 659 (runtime, including parametric expansion)
**Status**: All passing
**Coverage**: 100% line coverage on `src/`

---

## Summary by Component

| Test File | Component | Static @test | Runtime Tests | Status |
|-----------|-----------|-------------|---------------|--------|
| test_tree.jl | P1.1 Factor Tree | 19 | ~85 | ✅ |
| test_tensors.jl | P1.2 Tensor Network | 85 | ~200 | ✅ |
| test_basso_finite_d.jl | P1.3 Basso Finite-D | 35 | ~180 | ✅ |
| test_transfer_oracles.jl | P1.3 Transfer Oracles | 4 | ~30 | ✅ |
| test_maxcut_transfer.jl | P1.3 MaxCut Transfer | 16 | ~20 | ✅ |
| test_qaoa.jl | P1.3 Brute-Force QAOA | 9 | ~50 | ✅ |
| test_wht_factorisation.jl | P1.3 WHT Acceleration | 6 | ~94 | ✅ |
| **Total** | | **174** | **659** | **✅** |

Note: runtime count exceeds static count due to parametric `@testset for` loops
that expand over multiple (k, D, p) values and random angle samples.

---

## test_tree.jl — Factor Tree Construction (P1.1)

Tests for `TreeParams` and the combinatorial tree structure.

### Construction validation
- `TreeParams(k, D, p)` constructs successfully for valid parameters
- Throws `ArgumentError` for k < 2, D < 2, p < 1

### Branching factor
- `branching_factor(TreeParams(2, 3, 1)) == 2` (MaxCut, D=3)
- `branching_factor(TreeParams(3, 4, 1)) == 6` (our target)
- `branching_factor(TreeParams(4, 5, 1)) == 12`

### Golden values — MaxCut (k=2, D=3)
Parametric over p=1..4. For each p, verifies:
- `total_variables` matches hand-computed table (2, 6, 14, 30)
- `total_constraints` matches (1, 5, 13, 29)
- `total_nodes` matches (3, 11, 27, 59)
- `leaf_count` matches (2, 4, 8, 16)

### Golden values — XORSAT (k=3, D=4)
Parametric over p=1..5. For each p, verifies:
- `total_variables` matches (3, 21, 129, 777, 4665)
- `total_constraints` matches (1, 10, 64, 388, 2332)
- `total_nodes` matches (4, 31, 193, 1165, 6997)

### Leaf count spot checks
- `leaf_count(TreeParams(2, 3, 1)) == 2`
- `leaf_count(TreeParams(3, 4, 1)) == 3`
- `leaf_count(TreeParams(3, 4, 2)) == 18`

### Level count bounds
- Out-of-range level indices throw `ArgumentError`

### Monotonicity
- Parametric over k ∈ {2,3,4}, D ∈ {3,4,5}, p=1..4
- `total_variables` and `total_nodes` strictly increase with p

---

## test_tensors.jl — Tensor Network Primitives (P1.2)

Tests for `QAOAAngles`, hyperindex utilities, and raw tensor construction.

### QAOAAngles construction
- Valid construction with matching γ, β lengths
- Auto-promotes integer inputs to Float64
- Throws for mismatched lengths or empty arrays

### Hyperindex utilities
- `hyperindex_bit`: correct bit extraction from binary integers
- `hyperindex_parity`: correct XOR parity over selected positions
- Boundary checks: negative hyperindex and zero-position throw errors

### Leaf tensor
- **Dimensions**: length == 4^p for p=1..4
- **Constant value**: all entries equal 2^{-p} (angle-independent, since |+⟩ is X eigenstate)
- **Golden values**: p=1 at arbitrary angles → all entries 0.5
- **Periodicity**: invariant under γ → γ+2π and β → β+2π

### Mixer tensor
- **Dimensions**: 4^p × 4^p matrix for p=1..4
- **Identity at zero angle**: mixer at β=0 is the identity matrix
- **Local p=1 block**: known cos²/sin² structure at β=π/6
- **Periodicity**: invariant under β → β+2π
- **Multi-round isolation**: round 2 mixer at p=2 acts only on bits 3,4 (not 1,2)
- **Superoperator unitarity**: M·M† = I for random β at p=1..3

### Problem tensor
- **Dimensions**: (4^p)^k entries for p=1..2, k=2,3
- **Zero angle**: all entries equal 1 (identity at γ=0)
- **Golden values MaxCut p=1**: known phase pattern at γ=π/3 for k=2
- **Odd-clause sign handling**: clause_sign=-1 flips phase correctly
- **Golden values XORSAT k=3 p=1**: known phase pattern for 3-body gate
- **Periodicity**: invariant under γ → γ+2π

### Parity observable tensor
- **Dimensions**: (4^p)^k entries for p=1..2, k=2,3
- **p=1 values**: known Z₁Z₂ and Z₁Z₂Z₃ eigenvalue structure
- **Completeness**: correct count of nonzero entries

### Observable tensor
- **Dimensions**: (4^p)^k entries for p=1..2, k=2,3
- **Even-clause (XORSAT)**: C = (1 + Z₁···Zₖ)/2 values
- **Odd-clause (MaxCut)**: C = (1 - Z₁Z₂)/2 values

---

## test_basso_finite_d.jl — Basso Finite-D Iteration (P1.3 Tier 2)

Tests for the Basso Eq. 8.7 implementation and helper functions.

### Gamma vector construction
- Correct length 2p+1 for the Basso Γ vector
- Correct angular scaling convention

### Bit counts and configuration space
- `basso_bit_count(p)` returns 2p+1
- `basso_configuration_count(p)` returns 2^{2p+1}

### Decode bits
- Integer-to-spin conversion: 0 → {-1,+1}^n representation

### f(a) mixer function
- Zero beta: f returns 1 for all-plus configuration, 0 for flipped configurations
- p=1 values: verified against manual cos/sin products

### Phase argument
- Correct dot product of Γ with spin configuration

### Branch tensor
- Initial state: all entries 1.0
- Zero angles: branch tensor stays uniform after iteration steps

### Exact branch step verification
- k=2: single iteration step matches manual sum over configurations
- k=3: single iteration step matches manual sum over (k-1)-tuples

### Root kernel decomposition
- Parity kernel correctly extracts Z₁···Zₖ eigenvalue

### Branch-to-hyperindex mapping
- Basso branch-tensor indices map correctly to P1.2 hyperindex convention

### Root local factor
- Parametric over (k, p, clause_sign) combinations
- Root fold factors match raw tensor semantics for both even and odd clauses

### Zero-angle root parity
- Parametric over (k, D, p) combinations
- Full Basso iteration at zero angles yields zero parity (→ 0.5 satisfaction)

---

## test_transfer_oracles.jl — Transfer Oracle Cross-Validation (P1.3)

Tests verifying the transfer contraction agrees with brute-force.

### k=2 reduction
- Transfer oracle matches brute-force for MaxCut at multiple p values and angles

### Multilinearity for k=3
- Transfer oracle matches brute-force for k=3 at p=1

### Zero-angle factorization
- Transfer oracle correctly returns 0.5 at zero angles

### Smallest finite-D target
- Cross-validation at the smallest non-trivial (k, D, p) parameters

---

## test_maxcut_transfer.jl — MaxCut Compact Transfer Matrix (P1.3)

Tests for the Julia port of the Basso/Villalonga MaxCut transfer recursion.

### p=1 regression
- (2p+1)×(2p+1) matrix entries match upstream reference values to 1e-12
- Transfer objective matches reference scalar

### p=2 regression
- 5×5 matrix entries match upstream reference
- Transfer objective matches reference

### Corner symmetry
- Parametric: matrix satisfies the transpose and conjugation symmetry relations
  across all elements for random angles at p=2, D=4

---

## test_qaoa.jl — Brute-Force Light-Cone Simulator (P1.3 Tier 1)

Tests for the exact state-vector QAOA evaluation — the reference oracle.

### Zero-angle baseline
- `parity_expectation` = 0.0 at γ=β=0 for (k=3, D=4, p=1)
- `qaoa_expectation` = 0.5 at γ=β=0

### MaxCut p=1 exact formula
- Parametric over multiple (γ, β) pairs
- Parity matches analytical formula: ⟨Z₁Z₂⟩ = -sin(4β)cos²(γ)sin(γ)
- Satisfaction fraction matches (1 - parity)/2

### MaxCut p=1 optimum
- At γ* = atan(1/√2), β* = π/8: satisfaction = 1/2 + √3/9 ≈ 0.6924

### MaxCut p=2 cross-reference
- Matches independent state-vector simulation at specific angles

### k=3 exact cross-reference
- k=3, D=2, p=1 at multiple (γ, β, clause_sign) combinations
- Matches independent state-vector reference

### Exact light-cone guard
- `TreeParams(3, 4, 2)` (129 qubits) correctly throws `ArgumentError`

---

## test_wht_factorisation.jl — Walsh-Hadamard Acceleration (P1.3)

Tests for the WHT-based constraint fold optimisation.

### WHT round trip
- Parametric over n=3,5: IWHT(WHT(x)) recovers x to machine precision

### XOR convolution theorem
- Parametric over n=3,5: WHT(f ⊛ g) = WHT(f) · WHT(g)
- Verified for random f, g vectors

### XOR autoconvolution theorem
- Parametric over n=3,5: WHT of self-convolution = WHT(f)²
- Verified for random f vectors

### Naive vs WHT constraint fold agreement
- Parametric over arity ∈ {2,3}, p=1..3, D ∈ {3,4}
- Max |S_naive - S_wht| < 10⁻¹⁴ across multiple random-angle trials
- This is the critical validation that the WHT acceleration is correct

---

## Validation Targets (Non-Negotiable)

These are published results that must always pass:

| Case | Expected | Source | Test Location |
|------|----------|--------|---------------|
| MaxCut k=2, D=3, p=1 optimum | 0.5 + √3/9 ≈ 0.6924 | Farhi 2014 | test_qaoa.jl |
| MaxCut k=2, D=3, p=1 formula | -sin(4β)cos²(γ)sin(γ) | Farhi 2014 §4 | test_qaoa.jl |
| Zero angles → 0.5 | 0.5 ± 1e-10 | By construction | test_qaoa.jl, test_basso_finite_d.jl |
| WHT ≡ naive fold | < 1e-14 | Self-consistency | test_wht_factorisation.jl |
| Tier 2 ≡ Tier 1 | < 1e-10 | Self-consistency | test_transfer_oracles.jl |
