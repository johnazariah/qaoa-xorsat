# Spec — Threaded Evaluation Internals (Approach B)

**Phase**: 4 (Optimisation — Performance)
**Status**: Active
**Depends on**: autodiff-generics (complete), precompute f_table (complete)
**Blocks**: Higher-p feasibility (p=9+)

---

## Goal

Thread the embarrassingly parallel comprehensions inside the Basso evaluator so
that a single evaluation uses all available CPU cores, not just one.

## Motivation

At p=8, a single evaluation takes 416ms (Float64) / 1370ms (ForwardDiff).
Three comprehensions dominate the cost:

| Component | Float64 time | % of total | Parallelizable? |
|-----------|-------------|------------|-----------------|
| `basso_f_table` | 145ms | 35% | Yes — independent per config |
| `basso_constraint_kernel` | 51ms | 12% | Yes — independent per config |
| `basso_root_problem_kernel` | ~50ms | 12% | Yes — independent per config |
| WHT (per step) | 11ms | 3% | Partially — stages sequential, butterflies parallel |
| Pointwise ops | 48ms | 12% | Yes — element-wise |

The comprehensions are embarrassingly parallel: each configuration's value
depends only on the angles and the configuration index, not on any other entry.

## Design

### Thread the three big comprehensions

Replace serial comprehensions with pre-allocated vectors + `Threads.@threads`:

```julia
# Before
f_table = [f_function(angles, config) for config in 0:N-1]

# After
f_table = Vector{Complex{T}}(undef, N)
Threads.@threads for config in 0:N-1
    @inbounds f_table[config+1] = f_function(angles, config)
end
```

Same pattern for `basso_constraint_kernel` and `basso_root_problem_kernel`.

### Do NOT thread the WHT (yet)

The WHT butterfly has stage-sequential dependencies. At p=8 (131K elements),
the per-stage work is ~8K butterflies × 2 additions — too small for thread
spawn overhead to pay off. Defer to p≥12 or GPU.

### Do NOT thread pointwise ops

Julia's broadcast (`.*`, `.^`) already uses SIMD. Threading 131K-element
broadcasts adds overhead for negligible gain. At p≥12 (16M elements), revisit.

## Files changed

| File | Changes |
|------|---------|
| `src/basso_finite_d.jl` | Thread `basso_f_table`, `basso_constraint_kernel`, `basso_root_problem_kernel` |

## Expected speedup

With 10 threads on M4 (p=8):

| Component | Before | After | Savings |
|-----------|--------|-------|---------|
| `f_table` | 145ms | ~18ms | 127ms |
| `constraint_kernel` | 51ms | ~7ms | 44ms |
| `root_problem_kernel` | ~50ms | ~7ms | 43ms |
| **Per-eval total** | 416ms | ~230ms | ~45% |
| **ForwardDiff gradient** | 1370ms | ~850ms | ~38% |

## Validation

1. All 690 existing tests pass unchanged
2. Numeric values identical to serial (deterministic, no order-dependent accumulation)
3. Coverage remains 100%
4. Benchmark at p=7,8 shows measurable improvement

## Risk

Low. Each thread writes to a unique index — no data races, no locks needed.
Thread overhead (~1μs spawn) is negligible for 131K work items.
`@inbounds` is safe because indices are bounded by the loop range.
