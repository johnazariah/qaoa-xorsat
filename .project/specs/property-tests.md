# Spec: Property-Based Tests

**Status**: Ready for implementation
**Target file**: `test/test_properties.jl`
**Branch**: implement on `feature/phase4-optimization` worktree
**Include in**: `test/runtests.jl`

---

## Overview

Add property-based tests that verify mathematical invariants of the QAOA
computation across random inputs. These complement the existing golden-value
and cross-validation tests by checking structural laws that must hold for
ALL valid inputs, not just specific test points.

No external libraries needed. Use parameterised `@testset` loops with random
angles.

---

## Properties to Implement

### 1. Clause Sign Complement Identity

**Mathematical law**: For any (k, D, p) and any angles (γ, β):

$$\text{qaoa}(\text{clause\_sign}=+1) + \text{qaoa}(\text{clause\_sign}=-1) = 1$$

Even-parity and odd-parity constraints partition the outcome space.

**Test spec**:
```julia
@testset "clause sign complement" begin
    @testset "k=$k, D=$D, p=$p" for k in [2, 3], D in [3, 4], p in 1:2
        for _ in 1:50
            angles = QAOAAngles(π .* rand(p), (π / 2) .* rand(p))
            params = TreeParams(k, D, p)
            even = basso_expectation(params, angles; clause_sign = 1)
            odd  = basso_expectation(params, angles; clause_sign = -1)
            @test even + odd ≈ 1.0 atol = 1e-10
        end
    end
end
```

**Why it matters**: catches sign errors in the problem gate, observable, or
clause_sign propagation. A single flipped sign would break this.

---

### 2. Boundedness at Higher p

**Mathematical law**: For any (k, D, p) and any angles:

$$0 \le \text{basso\_expectation}(\ldots) \le 1$$

The expectation value is a probability — must be in [0, 1].

**Test spec**:
```julia
@testset "boundedness via Basso" begin
    @testset "k=$k, D=$D, p=$p" for (k, D) in [(2, 3), (3, 4)],
                                     p in [1, 2, 3, 5]
        for _ in 1:100
            angles = QAOAAngles(π .* rand(p), (π / 2) .* rand(p))
            params = TreeParams(k, D, p)
            val = basso_expectation(params, angles)
            @test 0.0 ≤ val ≤ 1.0
        end
    end
end
```

**Why it matters**: overflow, normalisation errors, and WHT scaling bugs
may only appear at higher p where values are larger. Testing at p=5
exercises the full fold pipeline.

**Note**: p=5 at k=3 uses the WHT path. If boundedness holds there, the
WHT is producing sensible results at non-trivial depth.

---

### 3. Angle Periodicity

**Mathematical law**: The QAOA expectation is periodic:
- In each γᵢ with period 2π (from $e^{-iγC}$ where C has integer eigenvalues)
- In each βᵢ with period π (from $e^{-iβX}$ where $X^2 = I$)

**Test spec**:
```julia
@testset "angle periodicity" begin
    @testset "γ periodicity" begin
        @testset "k=$k, D=$D, p=$p" for (k, D) in [(2, 3), (3, 4)], p in 1:3
            for _ in 1:20
                γ = π .* rand(p)
                β = (π / 2) .* rand(p)
                params = TreeParams(k, D, p)
                base = basso_expectation(params, QAOAAngles(γ, β))
                for i in 1:p
                    γ_shifted = copy(γ)
                    γ_shifted[i] += 2π
                    shifted = basso_expectation(params, QAOAAngles(γ_shifted, β))
                    @test shifted ≈ base atol = 1e-10
                end
            end
        end
    end

    @testset "β periodicity" begin
        @testset "k=$k, D=$D, p=$p" for (k, D) in [(2, 3), (3, 4)], p in 1:3
            for _ in 1:20
                γ = π .* rand(p)
                β = (π / 2) .* rand(p)
                params = TreeParams(k, D, p)
                base = basso_expectation(params, QAOAAngles(γ, β))
                for i in 1:p
                    β_shifted = copy(β)
                    β_shifted[i] += π
                    shifted = basso_expectation(params, QAOAAngles(γ, β_shifted))
                    @test shifted ≈ base atol = 1e-10
                end
            end
        end
    end
end
```

**Why it matters**: periodicity errors indicate incorrect scaling of angles
in the Γ vector, the f-function, or the cosine kernel. The β period is π
(not 2π) because the Basso formulation uses a doubled-angle convention
internally — verify this is correct, and adjust to 2π if the test fails
at π.

---

### 4. Random Baseline (extended)

**Mathematical law**: At γ = β = 0, the expectation is exactly 0.5.

**Test spec**:
```julia
@testset "zero-angle baseline via Basso" begin
    @testset "k=$k, D=$D, p=$p" for k in [2, 3, 4], D in [3, 4, 5], p in 1:4
        params = TreeParams(k, D, p)
        angles = QAOAAngles(zeros(p), zeros(p))
        @test basso_expectation(params, angles) ≈ 0.5 atol = 1e-12
    end
end
```

**Why it matters**: checks normalisation for a wide range of (k, D, p)
combinations that the existing tests don't cover. Each combination
exercises a different tree shape.

---

### 5. Parity-to-Expectation Consistency

**Mathematical law**: `qaoa_expectation == 0.5 * (1 + clause_sign * parity_expectation)`.

**Test spec**:
```julia
@testset "parity-to-expectation identity" begin
    @testset "k=$k, D=$D, p=$p, sign=$sign" for (k, D) in [(2, 3), (3, 4)],
                                                  p in 1:2,
                                                  sign in [-1, 1]
        for _ in 1:30
            angles = QAOAAngles(π .* rand(p), (π / 2) .* rand(p))
            params = TreeParams(k, D, p)
            parity = basso_parity_expectation(params, angles; clause_sign = sign)
            full   = basso_expectation(params, angles; clause_sign = sign)
            @test full ≈ 0.5 * (1 + sign * parity) atol = 1e-10
        end
    end
end
```

**Why it matters**: verifies the relationship between the raw parity
correlator and the satisfaction fraction. These use different code paths
internally.

---

## Implementation Notes

- Import `QaoaXorsat` and `Test` only. No external deps.
- Use a fixed `Random.seed!` at the top of the file for reproducibility
  in CI, but still use `rand()` for the actual sampling.
- The β periodicity might be 2π rather than π depending on the convention
  in the Basso iteration. If the test fails at π, try 2π and document which
  is correct.
- p=5 tests for k=3, D=4 use the WHT-accelerated path. This is intentional
  — it exercises the production code path.
- Expected test count: ~5 testsets × ~200-400 individual @test calls ≈ 1000-2000
  new tests. Runtime should be under 10 seconds.

## Acceptance Criteria

1. All property tests pass
2. File included in `runtests.jl`
3. Total test suite still passes
4. β periodicity period documented (π or 2π) based on which one works
