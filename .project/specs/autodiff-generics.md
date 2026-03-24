# Spec — ForwardDiff-Compatible Generic Numeric Types

**Phase**: 4 (Optimisation)
**Status**: Active
**Depends on**: P1.3 (Evaluation)
**Blocks**: Higher-p convergence improvements

---

## Goal

Make the Basso Tier 2 evaluation pipeline generic over the angle element type `T <: Real`,
so that ForwardDiff.jl dual numbers propagate through the full objective function.
This gives Optim.jl exact analytic gradients instead of finite differences.

## Motivation

The L-BFGS optimizer currently uses finite-difference gradients (Optim.jl default).
At depth p, each gradient requires ~2p+1 function evaluations. With ForwardDiff:

- **Exact gradients** in a single forward pass (cost ~3× one evaluation, not 2p+1×)
- **No FD noise floor** — allows tighter `g_abstol` (1e-8 vs 1e-6)
- **Better convergence** at high p where FD noise dominates

At p=12 (128s per eval), saving ~23 evaluations per gradient step is significant.

## Design

### Parametric `QAOAAngles{T}`

```julia
struct QAOAAngles{T<:Real}
    γ::Vector{T}
    β::Vector{T}
end
```

Inner constructor promotes both vectors to a common type via `promote_type`.
The old `QAOAAngles([0.3], [0.5])` infers `T=Float64` — fully backward-compatible.

### Type promotion strategy

Every function in the Basso pipeline derives its working type from `eltype(angles.γ)`:
- `T` for real intermediate arrays
- `Complex{T}` for complex intermediate arrays
- Return type annotations removed or made parametric

The element type `T` is the **single source of truth**. No function creates
`Float64` or `ComplexF64` literals directly — they use `zero(T)`, `one(T)`,
`complex(...)`, etc.

### `wht.jl` — no changes needed (already generic)

### Pattern catalogue

| Before | After |
|--------|-------|
| `zeros(Float64, n)` | `zeros(T, n)` |
| `zeros(ComplexF64, m, n)` | `zeros(Complex{T}, m, n)` |
| `ones(ComplexF64, n)` | `ones(Complex{T}, n)` |
| `ComplexF64(0.0, sin(β))` | `complex(zero(T), sin(β))` |
| `ComplexF64(0.5)` | `complex(one(T) / 2)` |
| `ComplexF64[expr for ...]` | `[complex(expr) for ...]` or `Complex{T}[...]` |
| `wht(ComplexF64.(x))` | `wht(complex.(x))` |
| `::Vector{ComplexF64}` return | remove annotation |
| `::Vector{Float64}` return | remove annotation |
| `::Float64` return | remove annotation |
| `::ComplexF64` return | remove annotation |
| `Float64(γ)` | deleted (γ is already `T`) |
| `im * sin(...)` | `complex(zero(T), sin(...))` |

### Files changed

| File | Nature of changes |
|------|------------------|
| `src/tensors.jl` | Parametric `QAOAAngles{T}`; inner constructor promotes |
| `src/basso_finite_d.jl` | All ~55 type barriers → generic `T` / `Complex{T}` |
| `src/qaoa.jl` | Remove `::Float64` return annotations; accept `QAOAAngles{T}` |
| `src/optimization.jl` | Remove `::QAOAAngles` return annotation on `angles_from_vector`; add `autodiff=AutoForwardDiff()` to `Optim.optimize`; add `ForwardDiff`, `ADTypes` imports |
| `Project.toml` | Add `ForwardDiff` and `ADTypes` as direct dependencies |

### Files NOT changed

| File | Why |
|------|-----|
| `src/wht.jl` | Already fully generic |
| `src/tree.jl` | Pure integer graph structure, no angle types |
| `src/transfer_oracles.jl` | Not on the optimizer hot path |
| `src/maxcut_transfer.jl` | Not on the optimizer hot path |

## Validation

1. All 690 existing tests must pass unchanged
2. New test: ForwardDiff gradient of `basso_expectation` matches finite-difference gradient
3. New test: `QAOAAngles{Float32}` and `QAOAAngles{Float64}` both work
4. Coverage remains 100%
5. Cross-validate optimized MaxCut k=2,D=3,p=1 result (0.7500) still holds

## Risk

Medium — this is a mechanical refactor of the core evaluator. The key risks are:
- Accidentally changing numeric values via type promotion (mitigated by existing tests)
- ForwardDiff not supporting some operation in the pipeline (e.g. `cis` — but it does)
- Performance regression from type instability (mitigated by specialization on `T=Float64`)
