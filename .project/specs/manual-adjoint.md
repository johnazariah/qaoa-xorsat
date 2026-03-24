# Spec — Manual Adjoint (Reverse-Mode) Differentiation

**Phase**: 4 (Optimisation — Performance)
**Status**: Active
**Depends on**: precompute f_table, threaded-eval
**Blocks**: p≥9 feasibility, GPU acceleration (approach C)

---

## Goal

Implement reverse-mode (adjoint) differentiation for `basso_expectation`, giving
exact gradients ∂E/∂γ and ∂E/∂β at cost ~2× a single Float64 evaluation —
independent of p. This replaces ForwardDiff, whose cost scales as ~2p× per eval.

## Motivation

| Method | Gradient cost at p=8 | Gradient cost at p=12 |
|--------|---------------------|----------------------|
| Finite differences | 17 × 44ms = 748ms (doesn't converge) | 25 × ~4s = 100s |
| ForwardDiff | 847ms (16 dual partials) | ~24 × eval (enormous) |
| **Manual adjoint** | **~88ms** (2 × 44ms) | **~8s** (2 × ~4s) |

The adjoint is **~10× faster than ForwardDiff at p=8** and the advantage grows
linearly with p. It also works on GPU (float arrays, no dual numbers).

## Mathematical Derivation

### Forward pass (existing code, annotated)

```
Inputs: γ[1:p], β[1:p]          ← the 2p optimization variables
Params: k, D, p, clause_sign

Step 1: f_table[a] = f(angles, a)           for a = 0:N-1    (N = 2^(2p+1))
         Depends on: β
         Each entry is a product of 2p trig factors:
           f(a) = ½ ∏_{j=1}^{2p} trigs[Δ(a,j)+1, j]
         where Δ(a,j) = a_j ⊕ a_{j+1} and
           trigs[1,j] = cos(β_j),  trigs[2,j] = i·sin(β_j)
         (with mirrored β convention for j > p)

Step 2: kernel[a] = cos(½ Σ_i γ_full[i] · spin(a,i))     for a = 0:N-1
         Depends on: γ
         γ_full is the (2p+1)-entry vector with zeros at the root bit

Step 3: Branch tensor iteration (p steps):
         B₀ = ones(N)
         For t = 1:p:
           child_weights_t = f_table .* B_{t-1}
           child_hat_t     = WHT(child_weights_t)
           folded_t        = iWHT(WHT(kernel) .* child_hat_t .^ arity)
           B_t             = folded_t .^ degree
         where arity = k-1, degree = D-1

Step 4: Root message:
         root_msg[a] = root_parity(a) · f_table[a] · B_p[a]

Step 5: Root kernel:
         root_kernel[a] = i·sin(½·cs · Σ_i γ_full[i] · spin(a,i))
         Depends on: γ

Step 6: Root fold:
         S = Σ_a root_kernel[a] · xor_conv_power(root_msg, k)[a]
         where xor_conv_power(v, k) = iWHT(WHT(v) .^ k)

Step 7: Output:
         E = (1 + cs · Re(S)) / 2
```

### Backward pass (adjoint rules)

We propagate the cotangent (adjoint) `Ē = 1` backward through each step.
For complex intermediates z, we track the real cotangent
`z̄ = ∂E/∂Re(z) + i·∂E/∂Im(z)`.

**Step 7 → 6**: `S̄ = cs/2` (real part only contributes)

**Step 6 → 5,4**: Root fold adjoint.
Let `conv = xor_conv_power(root_msg, k)`.
```
root_kernel̄ = S̄ · conj(conv)
conv̄ = S̄ · conj(root_kernel)
```
The xor_conv_power adjoint: if `conv = iWHT(WHT(v).^k)`:
```
v_hat = WHT(v)
v_hat̄ = k · conj(v_hat .^ (k-1)) .* WHT(conv̄)    [pointwise power adjoint]
root_msḡ = WHT(v_hat̄)                              [WHT is self-adjoint]
```
Wait — need care with WHT adjoint scaling.

**WHT adjoint**: WHT is its own transpose (symmetric matrix). So:
  - If z = WHT(x), then x̄ += WHT(z̄)
  - If z = iWHT(x) = WHT(x)/N, then x̄ += iWHT(z̄)

**Actually**: Let W be the WHT matrix. z = Wx ⟹ x̄ = Wᵀz̄ = Wz̄ (W symmetric).
iWHT: z = (1/N)Wx ⟹ x̄ = (1/N)Wz̄ = iWHT(z̄). ✓

**Element-wise power adjoint**: If z = x.^n then x̄ += n · x.^(n-1) .* z̄
(using real chain rule applied to each component of the complex number).

**Caution**: for complex x.^n, the derivative is n·x^(n-1) (complex derivative).
Since our pipeline ultimately produces a real output, the Wirtinger calculus
simplifies: we can use `x̄ += n · conj(x.^(n-1)) .* z̄` for the cotangent.

Actually, let me be precise. For real-valued loss L and complex intermediate
z = f(x) where x is also complex:
  ∂L/∂Re(x) = ∂L/∂Re(z) · ∂Re(z)/∂Re(x) + ∂L/∂Im(z) · ∂Im(z)/∂Re(x)

For z = x^n (complex): ∂z/∂x = n·x^{n-1} in the Wirtinger sense.
The cotangent propagation rule: x̄ += real(conj(∂z/∂x) · z̄) where z̄ is complex.

Hmm, this needs care. Let me use a simpler framework: treat everything as
real vectors of length 2N (real and imaginary parts interleaved). Then all
operations are R→R and standard reverse-mode applies. The WHT, element-wise
multiply, and power all have well-defined real Jacobians.

### Simplified adjoint formulas (real-valued chain rule)

For each primitive, given output cotangent z̄ (complex = pair of reals):

| Primitive | Forward: z = ... | Adjoint: x̄ += ... |
|-----------|-----------------|-------------------|
| `z = x .* y` | element-wise multiply | `x̄ += conj(y) .* z̄`; `ȳ += conj(x) .* z̄` |
| `z = x .^ n` (int n) | element-wise power | `x̄ += n .* conj(x .^ (n-1)) .* z̄` |
| `z = WHT(x)` | Walsh-Hadamard | `x̄ += WHT(z̄)` |
| `z = iWHT(x)` | inverse WHT | `x̄ += iWHT(z̄)` |
| `z = cos(x)` (real x) | cosine | `x̄ += -sin(x) .* Re(z̄)` |
| `z = sin(x)` (real x) | sine | `x̄ +=  cos(x) .* Re(z̄)` |
| `z = complex(0, sin(x))` | pure imaginary | `x̄ += cos(x) .* Im(z̄)` |
| `z = sum(x .* y)` | dot product | `x̄ += conj(y) .* z̄`; `ȳ += conj(x) .* z̄` |

**Note**: The `conj()` terms arise because we're differentiating a real-valued
loss through complex intermediates. When the loss is `Re(S)`, the initial seed
is real, and the conj() terms ensure correct real gradient recovery.

### The branch tensor backward recurrence

Forward stores: `B_0, B_1, ..., B_p` (save all intermediates)

Given `B̄_p` (from root fold adjoint), propagate backward for t = p, p-1, ..., 1:

```
# Undo Step 3.d: B_t = folded_t .^ degree
folded̄_t = degree .* conj(folded_t .^ (degree - 1)) .* B̄_t

# Undo Step 3.c: folded_t = iWHT(kernel_hat .* child_hat_t .^ arity)
product̄ = iWHT(folded̄_t)                    # adjoint of iWHT
kernel_hat̄ += conj(child_hat_t .^ arity) .* product̄     # accumulate across steps
child_hat̄_t = arity .* conj(child_hat_t .^ (arity - 1)) .* conj(kernel_hat) .* product̄

# Undo Step 3.b: child_hat_t = WHT(child_weights_t)
child_weights̄_t = WHT(child_hat̄_t)          # adjoint of WHT

# Undo Step 3.a: child_weights_t = f_table .* B_{t-1}
f_tablē += conj(B_{t-1}) .* child_weights̄_t    # accumulate across steps
B̄_{t-1} = conj(f_table) .* child_weights̄_t     # propagate to previous step
```

After all p steps: we have accumulated `f_tablē` and `kernel_hat̄`.

### Angle gradients from table cotangents

**γ gradient** (from kernel̄ and root_kernel̄):

The constraint kernel: `kernel[a] = cos(½ Σ_i γ_full[i] · spin(a,i))`

```
∂kernel[a]/∂γ_full[i] = -½ · sin(½ Σ_j γ_full[j] · spin(a,j)) · spin(a,i)
```

So: `γ_full̄[i] += Σ_a Re(kernel̄[a]) · (-½ · sin(phase[a]) · spin(a,i))`

where `phase[a] = ½ Σ_j γ_full[j] · spin(a,j)` (saved from forward pass).

Similarly for root_problem_kernel (replace cos with i·sin).

Then `γ̄[r]` is assembled from `γ_full̄` via the mirrored indexing.

**β gradient** (from f_tablē):

`f_table[a] = ½ ∏_{j=1}^{2p} trigs[Δ(a,j)+1, j]`

The derivative of a product with respect to one factor:
```
∂f_table[a]/∂trigs[d,j] = f_table[a] / trigs[d,j]    (if Δ(a,j) == d)
```

And `∂trigs[1,j]/∂β_r = -sin(β_r)`, `∂trigs[2,j]/∂β_r = i·cos(β_r)`
(for j corresponding to round r; negate for the mirror).

Assembling: for each round r, sum over all configurations a and the two positions
(round r and mirror 2p-r+1) the contribution through the trig factor.

## Implementation Structure

### New file: `src/adjoint.jl`

```julia
"""
    basso_expectation_and_gradient(params, angles; clause_sign) -> (value, γ_grad, β_grad)

Compute the expectation value AND its gradient with respect to γ and β in a
single forward+backward pass. Cost ≈ 2× a single Float64 evaluation.
"""
function basso_expectation_and_gradient(params, angles; clause_sign=1)
    # --- Forward pass (save intermediates) ---
    f_table, phase_cache = basso_f_table_with_cache(angles)
    kernel, kernel_phase = basso_constraint_kernel_with_cache(angles, ...)
    kernel_hat = wht(complex.(kernel))

    B = Vector{Vector{ComplexF64}}(undef, params.p + 1)
    B[1] = ones(ComplexF64, N)
    folded = similar(B[1])  # reuse
    for t in 1:params.p
        child_weights = f_table .* B[t]
        child_hat = wht(child_weights)
        folded = iwht(kernel_hat .* child_hat .^ arity)
        B[t+1] = folded .^ degree
    end

    root_msg = root_parity .* f_table .* B[end]
    root_kernel, root_phase = basso_root_kernel_with_cache(angles, ...; clause_sign)
    msg_hat = wht(complex.(root_msg))
    conv = iwht(msg_hat .^ k)
    S = sum(root_kernel .* conv)
    value = (1 + clause_sign * real(S)) / 2

    # --- Backward pass ---
    S̄ = clause_sign / 2

    # Root fold backward
    root_kernel̄ = S̄ .* conj(conv)
    conv̄ = S̄ .* conj(root_kernel)

    # xor_conv_power backward
    msg_hat̄ = k .* conj(msg_hat .^ (k-1)) .* wht(conv̄)
    root_msḡ = wht(msg_hat̄)

    # Root message backward
    f_tablē = conj(root_parity .* B[end]) .* root_msḡ
    B̄ = conj(root_parity .* f_table) .* root_msḡ

    # Branch tensor backward (p steps, reversed)
    kernel_hat̄ = zeros(ComplexF64, N)
    for t in params.p:-1:1
        # Undo power
        folded_t = ...  # recompute or cache
        folded̄ = degree .* conj(folded_t .^ (degree-1)) .* B̄

        # Undo iwht
        product̄ = iwht(folded̄)

        # Undo pointwise
        child_hat_t = ...  # recompute
        kernel_hat̄ .+= conj(child_hat_t .^ arity) .* product̄
        child_hat̄ = arity .* conj(child_hat_t .^ (arity-1)) .* conj(kernel_hat) .* product̄

        # Undo wht
        child_weights̄ = wht(child_hat̄)

        # Undo f_table .* B[t]
        f_tablē .+= conj(B[t]) .* child_weights̄
        B̄ = conj(f_table) .* child_weights̄
    end

    # Convert table cotangents to angle gradients
    γ_grad = kernel_to_gamma_grad(kernel̄, kernel_hat̄, kernel_phase, angles) +
             root_kernel_to_gamma_grad(root_kernel̄, root_phase, angles)
    β_grad = f_table_to_beta_grad(f_tablē, phase_cache, angles)

    (value, γ_grad, β_grad)
end
```

### Integration with Optim.jl

```julia
function objective_and_gradient!(F, G, values, params, clause_sign)
    angles = QAOAAngles(values[1:p], values[p+1:2p])
    if G !== nothing
        val, γ_grad, β_grad = basso_expectation_and_gradient(params, angles; clause_sign)
        G[1:p] .= -γ_grad    # negate because we minimize -E
        G[p+1:2p] .= -β_grad
        if F !== nothing
            return -val
        end
    elseif F !== nothing
        return -basso_expectation(params, angles; clause_sign)
    end
end

# In optimize_angles:
result = Optim.optimize(
    Optim.only_fg!(
        (F, G, x) -> objective_and_gradient!(F, G, x, params, clause_sign)
    ),
    x0,
    Optim.LBFGS(),
    Optim.Options(...)
)
```

## Testing Strategy

1. **Cross-validate against ForwardDiff** at p=1,2,3: both methods compute
   the same gradient to ~1e-12 relative tolerance
2. **Cross-validate against finite differences** at p=1: FD with ε=1e-7,
   match to ~1e-5
3. **All 690 existing tests pass** (adjoint is additive, doesn't change existing code)
4. **Optimizer produces same c̃ values** for MaxCut k=2,D=3,p=1 (0.7500)
5. **Coverage**: new adjoint code fully covered

## Files

| File | Changes |
|------|---------|
| `src/adjoint.jl` (new) | ~200-250 lines: forward+backward pass, angle grad assembly |
| `src/QaoaXorsat.jl` | Add `include("adjoint.jl")`, export |
| `src/optimization.jl` | Use `Optim.only_fg!` with adjoint when `autodiff=:adjoint` |
| `test/test_adjoint.jl` (new) | ~80 lines: cross-validation tests |
| `test/runtests.jl` | Include test_adjoint.jl |

## Expected Performance

| p | Float64 eval | ForwardDiff grad | Adjoint grad | Adjoint speedup |
|---|-------------|-----------------|-------------|----------------|
| 7 | 10ms | 190ms | ~20ms | **9.5×** |
| 8 | 44ms | 847ms | ~88ms | **9.6×** |
| 10 | ~400ms | ~8s (est.) | ~800ms | **10×** |
| 12 | ~4s | ~96s (est.) | ~8s | **12×** |

The adjoint cost is ~2× the Float64 eval, independent of p. The speedup over
ForwardDiff grows linearly with p (because ForwardDiff's overhead is ~2p×).

## Memory

The backward pass stores all p intermediate branch tensors `B_0, ..., B_p`.
Each is a `Vector{ComplexF64}` of size N = 2^(2p+1).

| p | N | Storage for B array | Total memory |
|---|---|-------------------|-------------|
| 8 | 131K | 8 × 2MB = 16MB | ~32MB (+ working vectors) |
| 10 | 2M | 10 × 32MB = 320MB | ~640MB |
| 12 | 33M | 12 × 512MB = 6GB | ~12GB |

At p=12 this is tight on 16GB machines. Checkpointing (recompute rather than
store) can trade memory for ~2× more compute. Not needed until p≥11.

## Risk

Medium. The mathematical derivation requires care with complex adjoint rules.
The cross-validation against ForwardDiff at small p is the safety net — any
sign error or missing conjugate will show up as a gradient mismatch.

## Future: GPU compatibility

The adjoint pass uses only: WHT, iWHT, element-wise multiply, element-wise
power, and summation. All of these work on `MtlArray{ComplexF64}`. This means
the adjoint can run entirely on GPU once Phase C.1 (GPU forward pass) lands.
No dual numbers, no ForwardDiff — just float arrays.
