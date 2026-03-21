---
applyTo: "test/**"
---

# Testing Standards — QAOA-XORSAT

## Test Organisation

```
test/
  runtests.jl              # Entry point — includes all test files
  test_tree.jl             # Tree construction tests
  test_tensors.jl          # Tensor building and contraction tests
  test_qaoa.jl             # QAOA evaluation tests
  test_optimization.jl     # Angle optimization tests
  test_aqua.jl             # Package quality (Aqua.jl)
```

Each source file in `src/` has a corresponding test file. `runtests.jl` includes them all.

## Test Structure

Use nested `@testset` blocks for clarity:

```julia
@testset "FactorTree" begin
    @testset "construction" begin
        @testset "k=$k, D=$D" for k in [2, 3, 4], D in [3, 4, 5]
            tree = build_factor_tree(k, D, 1)
            @test tree_depth(tree) == 1
            @test tree.k == k
            @test tree.D == D
        end
    end

    @testset "size" begin
        # Known sizes for specific (k, D, p)
        @test tree_size(build_factor_tree(2, 3, 1)) == 5   # MaxCut, D=3, p=1
        @test tree_size(build_factor_tree(3, 4, 1)) == 13
    end
end
```

## Validation Targets

These are non-negotiable correctness checks from the literature:

| Case | Expected | Source |
|------|----------|--------|
| MaxCut k=2, D=3, p=1 | c̃_edge ≈ 0.7500 | Farhi et al. 2014, Table I |
| MaxCut k=2, D=3, p=5 | c̃_edge ≈ 0.8333 | Farhi et al. 2025, Table I |
| MaxCut k=2, D=3, p=7 | c̃_edge ≈ 0.8536 | Farhi et al. 2025, Table I |

Every new feature must preserve these validation results. They serve as regression tests.

## Test Categories

1. **Unit tests** — individual functions produce correct output for known inputs
2. **Property tests** — invariants hold across parameterised ranges:
   - `tree_size(k, D, p) > tree_size(k, D, p-1)` for all valid params
   - `0 ≤ qaoa_expectation(k, D, p, γ, β) ≤ 1` for all angles
   - Contraction of a single leaf is the identity operation
3. **Validation tests** — reproduce published results (the table above)
4. **Package quality** — Aqua.jl checks (no ambiguities, no unbound args, no piracy)

## Numeric Tolerances

- Validation tests: `@test result ≈ expected atol=1e-4` (papers report 4 decimal places)
- Internal consistency: `atol=1e-10` or `rtol=1e-8` for floating-point comparisons
- Optimisation results: `atol=1e-6` (L-BFGS convergence threshold)

## Test Dependencies

Add in `[extras]` and `[targets]` sections of `Project.toml`:

```toml
[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
Aqua = "4c88cf16-eb10-579e-8560-4a9242c79595"

[targets]
test = ["Test", "Aqua"]
```

## Coverage

- Aim for **line coverage ≥ 80%** on `src/` code
- All exported functions must have at least one test
- All error paths must be tested (`@test_throws`)
- Use Julia's built-in coverage: `julia --project=. --code-coverage=user -e 'using Pkg; Pkg.test()'`
