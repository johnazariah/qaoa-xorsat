# Spec — Metal.jl GPU Acceleration (Approach C)

**Phase**: 5+ (Future — Performance Research)
**Status**: Design only — not yet implementable
**Depends on**: threaded-eval (approach B), autodiff-generics
**Blocks**: p≥13 feasibility

---

## Goal

Move the vector-level operations in the Basso evaluator onto the M4's GPU via
Metal.jl, achieving 20-40× speedup on the Float64 evaluation path.

## Motivation

At p=12, vectors have $2^{25}$ = 33M entries. The WHT, element-wise powers, and
broadcasts are perfect GPU workloads — massive SIMD with streaming access. The
M4's unified memory eliminates explicit CPU↔GPU transfers.

Projected per-eval times at p=12:

| | CPU (current) | GPU (projected) |
|---|---|---|
| Float64 | ~128s | ~3-6s |
| With gradient | unknown | unknown |

## Architecture

### Two-tier: GPU eval + CPU gradient

```
Optimizer (CPU)
  │
  ├── ForwardDiff gradient (CPU, for p ≤ crossover)
  │     └── basso_expectation(QAOAAngles{Dual})  ← current path
  │
  └── GPU eval + adjoint gradient (for p > crossover)
        ├── forward pass: basso_expectation on MtlArray{Float64}
        └── backward pass: manual adjoint or Enzyme.jl
```

### Forward pass (GPU)

```julia
using Metal

function basso_branch_tensor_gpu(params, angles; steps=params.p)
    # Move precomputed tables to GPU once
    f_table_gpu = MtlArray(basso_f_table(angles))
    kernel_gpu = MtlArray(basso_constraint_kernel(angles, ...))

    current = Metal.ones(ComplexF64, basso_configuration_count(params.p))

    for _ in 1:steps
        child_weights = f_table_gpu .* current
        child_hat = wht_gpu!(child_weights)      # custom Metal kernel
        kernel_hat = wht_gpu!(copy(kernel_gpu))
        folded = iwht_gpu!(kernel_hat .* (child_hat .^ child_arity))
        current = folded .^ branch_degree
    end

    Array(current)  # pull back to CPU for root fold
end
```

### GPU WHT kernel

The Walsh-Hadamard transform maps to a standard parallel butterfly:

```
for each stage s = 0, 1, ..., log2(N)-1:
    block_size = 2^s
    for each butterfly (independent, parallel):
        left, right = paired elements
        new_left  = left + right
        new_right = left - right
```

Each stage is a GPU kernel launch. At 131K elements (p=8), that's 17 stages ×
65K butterflies per stage. At 33M elements (p=12), it's 25 stages × 16M
butterflies. Metal.jl's `@metal` macro or pre-written compute shaders handle this.

### The gradient problem

**This is the blocking issue.** Metal.jl cannot execute ForwardDiff dual numbers.
Three paths forward:

#### Option C1: Manual adjoint (medium effort, medium risk)

Derive and implement the backward pass for the Basso pipeline by hand.

The chain rule through the hot path:

```
∂L/∂angles = ∂L/∂branch_tensor × ∂branch_tensor/∂angles

branch_tensor = (iwht(wht(f⊙prev) ⊙ wht(kernel)^arity))^degree
```

Each operation has a known adjoint:
- `.^n` → `n .* x.^(n-1) .* adj`
- `wht` → `wht(adj)` (self-adjoint up to scaling)
- `iwht` → `iwht(adj)`
- `.*` → element-wise: `adj_left = adj .* right`, `adj_right = adj .* left`
- `f_table(angles)` → trig derivatives (chain rule through sin/cos products)

**Effort**: ~200-400 lines. One-time derivation + implementation + testing.
**Risk**: Correctness — must cross-validate against ForwardDiff on CPU at small p.

#### Option C2: Enzyme.jl (low effort, high risk)

Enzyme performs source-to-source AD that can compile to GPU targets.

```julia
using Enzyme
grad = Enzyme.gradient(Reverse, basso_expectation_gpu, params, angles)
```

**Effort**: ~20 lines if it works.
**Risk**: Enzyme + Metal.jl integration is experimental (as of 2026). May not
support all operations (complex arithmetic, WHT butterfly, etc.). Likely to
require workarounds or fail entirely.

#### Option C3: Hybrid — GPU eval + CPU-side finite-difference (not viable)

Use GPU-accelerated Float64 evaluations with CPU-side finite differences for
gradients. Each gradient needs 2p+1 GPU evaluations.

**Not viable**: We showed FD doesn't converge at p≥4. This path is dead unless
we also implement the GPU adjoint.

## Implementation phases

### Phase C.1: GPU forward pass (no gradients)

- Implement `wht_gpu!` / `iwht_gpu!` as Metal compute kernels
- Port `basso_branch_tensor` to use `MtlArray`
- Port `basso_root_fold` (xor_convolution_power on GPU)
- Benchmark at p=8,10,12 — establish the GPU eval baseline

**Effort**: ~150 lines + Metal kernel
**Deliverable**: Fast Float64 evaluation, no optimization (no gradients)
**Useful for**: Exploring the landscape, computing tables at fixed angles

### Phase C.2: Manual adjoint

- Derive adjoint rules for each operator in the pipeline
- Implement `basso_expectation_and_gradient_gpu(params, angles)`
- Cross-validate against ForwardDiff at p=1-5
- Benchmark gradient computation at p=8-12

**Effort**: ~300 lines
**Deliverable**: GPU-accelerated optimization
**Risk**: Correctness of manual adjoint — one sign error breaks everything

### Phase C.3 (alternative): Enzyme.jl exploration

- Test Enzyme.jl on the CPU Basso pipeline first
- If it works, test with Metal.jl arrays
- File issues for any unsupported operations

**Effort**: ~20 lines to try, unknown to fix issues
**Deliverable**: Possibly GPU-accelerated optimization with minimal code

## Decision criteria

Implement C only if:
1. Approach B (threaded eval) is insufficient for p=10-12
2. p=10+ results are scientifically needed (not just nice-to-have)
3. The comparison against DQI requires depths beyond what CPU can reach

If the c̃ curve clearly plateaus below DQI+BP=0.8707 by p=8-9, there may be
no scientific need for p≥12, making GPU acceleration unnecessary.

## Files affected (Phase C.1)

| File | Changes |
|------|---------|
| `src/wht_gpu.jl` (new) | Metal WHT/iWHT kernels |
| `src/basso_gpu.jl` (new) | GPU-accelerated branch tensor + root fold |
| `Project.toml` | Add Metal.jl dependency |
| `test/test_basso_gpu.jl` (new) | Cross-validate GPU vs CPU at p=1-5 |

## Hardware requirements

- Apple Silicon Mac (M1/M2/M3/M4) with Metal support
- Metal.jl ≥ 1.0 (check `using Metal; Metal.functional()`)
- At p=12: ~256MB GPU memory for branch tensor (well within M4's unified pool)
