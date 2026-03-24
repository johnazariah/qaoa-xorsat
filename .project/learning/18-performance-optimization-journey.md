# Implementation Note: Performance Optimization Journey

**Date**: 24 March 2026
**Author**: John S Azariah (PI) + AI agent session
**Context**: QAOA-XORSAT exact evaluation at (k=3, D=4) depths p=1–12

---

## Starting Point

The project computes exact QAOA performance on Max-3-XORSAT at D=4 regular
hypergraphs using the Basso finite-D branch-tensor iteration, accelerated by
a Walsh-Hadamard factorisation of the constraint fold. The WHT reduces the
per-evaluation cost from O(4^{kp}) to O(p²·4^p).

At the start of this session, the code ran on a Linux devcontainer with:
- Serial optimizer restarts
- Finite-difference gradients (Optim.jl default)
- No precomputation of angle-dependent tables
- Results computed through p=5 only (p=6+ never completed)

The goal: push to p=8+ for comparison against DQI (0.8707).

---

## Migration to Bare Metal (M4 Mac)

Migrated from devcontainer (Debian 12, emulated) to native Apple Silicon M4.
Julia 1.12.5, 12 logical cores, 10 performance cores.

**Impact**: ~2–4× faster per evaluation from native execution alone.

---

## Optimization 1: Thread-Parallel Restarts

**Commit**: `a312e2f`

The L-BFGS optimizer runs multiple restarts with different random initial angles
to avoid local minima. These restarts are fully independent.

**Before**: Serial `for guess in guesses` loop — 1 core busy.
**After**: `Threads.@threads for i in eachindex(guesses)` — all restarts run
simultaneously on separate cores, results collected into a pre-allocated vector.

**Impact**: With 3 restarts at p≥5 (budget-limited), uses 3 cores. Modest
wall-clock improvement since the per-restart time dominates. More significant
at lower p with 9+ restarts.

---

## Optimization 2: ForwardDiff Exact Gradients (QAOAAngles{T})

**Commit**: `a312e2f`

The biggest conceptual change. Made the entire evaluation pipeline generic over
the angle element type `T <: Real` so that ForwardDiff.jl dual numbers propagate
end-to-end, giving Optim.jl exact analytic gradients instead of finite differences.

### The refactor

`QAOAAngles` became `QAOAAngles{T<:Real}` with a promote-type constructor.
~55 type barriers across `basso_finite_d.jl` were converted:

| Pattern | Replacement |
|---------|-------------|
| `zeros(Float64, n)` | `zeros(T, n)` |
| `ComplexF64(0.0, sin(β))` | `complex(zero(T), sin(β))` |
| `ComplexF64(0.5)` | `complex(one(T) / 2)` |
| `ComplexF64[expr for ...]` | `[expr for ...]` or `Complex{T}[...]` |
| `wht(ComplexF64.(x))` | `wht(complex.(x))` |
| `::Vector{ComplexF64}` return | removed annotation |

`wht.jl` needed no changes — it was already fully generic.

### The autodiff experiment

We added a toggle (`autodiff` kwarg) and ran head-to-head comparisons:

| p | ForwardDiff | Finite Differences | FD converged? |
|---|-------------|-------------------|---------------|
| 1 | 1.4s | 1.2s | ✅ |
| 2 | 0.2s | 0.01s | ✅ |
| 3 | 0.3s | 0.3s | ✅ |
| 4 | 0.8s | 8.8s | ❌ (retry) |
| 5 | 2.9s | 91.1s | ❌ (retry) |
| 6+ | 21s | never finishes | ❌ |

**Key finding**: Finite differences cannot converge at p≥4 due to gradient noise
hitting the `g_abstol=1e-6` tolerance floor. ForwardDiff gives exact gradients
that L-BFGS can actually use. At p=5, ForwardDiff is 31× faster and converges
in 17 iterations vs FD's 320 (maxed out, unconverged).

**The crossover hypothesis was wrong**: We initially expected ForwardDiff to
become slower than FD at high p because each Dual number carries 2p partials.
In reality, FD's noisy gradients cause so many more iterations that it can
never compete — even at p where ForwardDiff is expensive, FD is more expensive
*and* doesn't converge.

---

## Optimization 3: Precomputed Tables

**Commit**: `1a6bfc4`

Profiling revealed that `basso_branch_tensor_step` was recomputing two
angle-dependent tables from scratch on every one of p iterations:

1. **`f_table`** (mixer weights): 131K complex trig products at p=8, ~145ms each
2. **`constraint_kernel`** (phase kernel): 131K cos evaluations, ~51ms each

Both depend only on the angles, not on the branch tensor state — they are
identical across all p iterations.

**Fix**: Added `basso_f_table()` as a precomputation step. `basso_branch_tensor`
computes both tables once and passes them to each step. `basso_parity_expectation`
shares the `f_table` between the branch iteration and the root message.

Also inlined the `configuration_spins` function to eliminate 131K × 17-element
vector allocations per kernel computation.

**Impact at p=7**: 270s → 81s (3.3× speedup).

---

## Optimization 4: Threaded Evaluation Comprehensions

**Commit**: `7ea6209`

The three large comprehensions (`basso_f_table`, `basso_constraint_kernel`,
`basso_root_problem_kernel`) iterate over all 2^(2p+1) configurations
independently. Each configuration's value depends only on the angles and the
configuration index.

**Fix**: Replaced serial comprehensions with `Threads.@threads` over pre-allocated
output vectors. Used `@inbounds` since indices are loop-bounded.

**Impact at p=8 (10 threads)**:

| | Float64 eval | ForwardDiff gradient |
|---|---|---|
| Before | 416ms | 1370ms |
| After | **44ms** | **847ms** |
| Speedup | **9.5×** | **1.6×** |

The Float64 path gets nearly linear speedup (10 threads → 9.5×). ForwardDiff
gets less because the serial WHT and pointwise operations (which can't easily
be threaded) dominate the Dual-number path.

---

## Optimization 5: Manual Adjoint Differentiation

**Commit**: `dc7d8da` (on `feature/adjoint-differentiation`)

The ForwardDiff overhead scales as ~2p× per evaluation (each operation carries
2p dual partials). At p=8 this is 19× overhead. At p=12 it would be ~24×.

The manual adjoint computes exact gradients via reverse-mode differentiation
at cost ~1.6× a single Float64 evaluation, independent of p.

### How it works

**Forward pass**: Run the standard Basso evaluation but cache all intermediates:
- All p branch tensors B₀, B₁, ..., Bₚ
- All p WHT results (child_hat) and folded tensors
- Phase arguments for the constraint and root kernels
- The f_table, trig_table, and bits_table

**Backward pass**: Propagate cotangents (∂L/∂z for each intermediate z) from
the output back through each operation using these adjoint rules:

| Forward operation | Adjoint rule |
|-------------------|-------------|
| `z = WHT(x)` | `x̄ += WHT(z̄)` (WHT is self-adjoint) |
| `z = iWHT(x)` | `x̄ += iWHT(z̄)` |
| `z = x .* y` | `x̄ += conj(y) .* z̄`; `ȳ += conj(x) .* z̄` |
| `z = x .^ n` | `x̄ += n · conj(x.^(n-1)) .* z̄` |
| `z = sum(x .* y)` | `x̄ += conj(y) .* z̄`; `ȳ += conj(x) .* z̄` |

The branch-tensor backward recurrence runs in reverse: for t = p down to 1,
undo the power, iWHT, pointwise multiply, and WHT, accumulating cotangents
into `kernel_hat_bar` (across all steps) and propagating `B_bar` backward.

**γ gradient**: The constraint kernel `cos(½ Σ γ·spin)` and root kernel
`i·sin(½·cs·Σ γ·spin)` have straightforward trig derivatives. The γ_full
cotangent is mapped back to γ via the mirrored indexing convention.

**β gradient**: The mixer weight `f(a) = ½ ∏ trigs[Δ+1, j]` is a product of
2p complex trig factors. Using the log-derivative trick:
- Δ=0 factor: `cos(β)` → log-derivative = `-tan(β)`
- Δ=1 factor: `i·sin(β)` → log-derivative = `cot(β)`

Both forward and mirror positions have the same log-derivative (the mirror
signs cancel). So: `∂f/∂β_r = f · (logderiv[Δ_fwd] + logderiv[Δ_bwd])`.

### The cos(-β) derivative bug

The initial implementation had a sign error in the mirror position derivative:
`d cos(-β)/dβ` was coded as `+sin(β)` but the correct value is `-sin(β)`.

Derivation: `d/dβ cos(-β) = -sin(-β) · d(-β)/dβ = -sin(-β) · (-1) = sin(-β) = -sin(β)`.

The γ gradient was unaffected (it doesn't go through trig products). The bug
was caught immediately by cross-validation against ForwardDiff at p=1.

### Performance results

| p | Plain eval | ForwardDiff | **Adjoint** | Adj speedup over FD |
|---|-----------|-------------|-------------|---------------------|
| 5 | 0.6ms | 9.5ms | **0.85ms** | **11×** |
| 7 | 10ms | 211ms | **17ms** | **12×** |
| 8 | 51ms | 971ms | **81ms** | **12×** |

The adjoint overhead is constant at ~1.6× regardless of p (vs ForwardDiff's
~2p×). At p=12 the projected advantage is ~24×.

### Cotangent convention

We use the R² cotangent convention throughout: for complex intermediate z,
`z̄ = ∂L/∂Re(z) + i·∂L/∂Im(z)`. This is consistent with `conj(y) .* z̄` for
multiply adjoints. For real parameters θ:
`∂L/∂θ = Re(conj(z̄) · ∂z/∂θ)`.

---

## Cumulative Impact

**p=8 optimization wall time**:

| Configuration | Estimated time |
|---------------|---------------|
| Devcontainer, serial FD, no precompute | **Never completes** (FD can't converge at p≥4) |
| Devcontainer, g_abstol fix, FD | ~7 hours (extrapolated from p=5 timing) |
| M4 bare metal + ForwardDiff | 698s (11.6 min) — first ever p=8 completion |
| + precomputed tables | ~400s (est.) |
| + threaded comprehensions | ~200s (est.) |
| **+ manual adjoint** | **~50–100s** (projected, 12× gradient speedup) |

**p=8 per-gradient cost**:

| Method | Cost | Can converge? |
|--------|------|--------------|
| Finite differences (2p+1 evals) | 17 × 44ms = 748ms | ❌ at p≥4 |
| ForwardDiff (1 Dual eval) | 971ms | ✅ |
| **Manual adjoint** | **81ms** | ✅ |

---

## XORSAT Results (k=3, D=4)

| p | c̃(p) | Δc̃ | Gap to SA (0.9366) | Converged |
|---|-------|------|-------------------|-----------|
| 1 | 0.6761 | — | 0.2606 | ✅ |
| 2 | 0.7391 | +0.0630 | 0.1975 | ✅ |
| 3 | 0.7771 | +0.0380 | 0.1595 | ✅ |
| 4 | 0.8022 | +0.0251 | 0.1344 | ✅ |
| 5 | 0.8205 | +0.0183 | 0.1161 | ✅ |
| 6 | 0.8344 | +0.0139 | 0.1022 | ✅ |
| 7 | 0.8453 | +0.0109 | 0.0913 | ✅ |
| 8 | 0.8541 | +0.0088 | 0.0825 | ✅ |

Per-step gain decays at ratio ~0.75. Projected plateau ~0.89–0.90.
QAOA likely crosses DQI+BP (0.8707) around p=11 and Prange (0.875) around p=12.

---

## Architecture After Optimizations

```
Optimizer (optimize_angles)
  ├── Thread-parallel restarts (Threads.@threads over starts)
  │     └── Per-start: Optim.LBFGS with gradient via one of:
  │           ├── :adjoint  → basso_expectation_and_gradient()  [default, fastest]
  │           ├── :forward  → ForwardDiff.gradient()            [generic, slower at high p]
  │           └── :finite   → Optim finite differences          [broken at p≥4]
  │
  └── basso_expectation(params, angles)
        ├── Precompute: f_table, constraint_kernel (threaded, reused)
        ├── Branch tensor iteration (p steps, precomputed tables passed in)
        │     └── Per step: f.*B → WHT → .^arity → .*kernel_hat → iWHT → .^degree
        └── Root fold: root_message → WHT → .^k → iWHT → ·root_kernel → sum

Adjoint backward pass (basso_expectation_and_gradient):
  ├── Forward: same as above, caching B[0:p], child_hat[1:p], folded[1:p]
  ├── Backward: cotangent propagation in reverse through cached graph
  │     ├── Root fold adjoint (WHT/iWHT self-adjoint, power/multiply rules)
  │     ├── Branch recurrence backward (t = p, ..., 1)
  │     └── Table cotangent accumulation (f_table_bar, kernel_hat_bar)
  └── Angle gradients: γ from kernel phases, β from log-derivative trick
```

---

## Files Changed

| File | Key changes |
|------|-------------|
| `src/tensors.jl` | `QAOAAngles{T<:Real}` parametric struct |
| `src/basso_finite_d.jl` | Generic types, precomputed tables, threaded comprehensions |
| `src/adjoint.jl` | **New** — forward pass with caching + backward pass + angle gradients |
| `src/optimization.jl` | Thread-parallel restarts, autodiff toggle (:adjoint/:forward/:finite) |
| `src/qaoa.jl` | Removed ::Float64 return annotations |
| `src/QaoaXorsat.jl` | Added adjoint.jl include + export |
| `scripts/optimize_qaoa.jl` | CLI autodiff flag (adjoint/forward/finite) |
| `Project.toml` | Added ForwardDiff, ADTypes dependencies |
| `test/test_adjoint.jl` | **New** — cross-validation against ForwardDiff |
| `test/test_qaoa.jl` | Tier 1 coverage tests |
| `test/test_optimization.jl` | merge_optimization_results tests |

---

## Lessons Learned

1. **ForwardDiff overhead is O(2p)** — each operation carries 2p dual partials. At p=8
   this is 16× overhead on every multiply, sin, cos, and WHT butterfly. The manual
   adjoint avoids this entirely.

2. **FD gradient noise is a hard wall** — the `g_abstol=1e-6` threshold that works
   for the optimizer is above the FD noise floor at p≥4. ForwardDiff or the adjoint
   give exact gradients that converge cleanly.

3. **Precomputation is free** — the f_table and kernel depend only on angles, not on
   the iteration state. Computing them once and reusing across p steps is obvious in
   hindsight but gave 3.3× at p=7.

4. **WHT is self-adjoint** — this makes the backward pass through the constraint fold
   trivially simple. The Hadamard matrix is symmetric and orthogonal (up to scaling).

5. **The log-derivative trick** for the β gradient avoids complex division (f/trig) and
   reduces to simple tan/cot. Both forward and mirror positions have the same
   log-derivative despite different signs in the trig arguments.

6. **cos(-β) derivative gotcha** — `d/dβ cos(-β) = -sin(β)`, not `+sin(β)`. The double
   negation from chain rule through `-β` is easy to get wrong. Cross-validation against
   ForwardDiff at small p caught this immediately.
