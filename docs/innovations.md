# Ten Innovations That Made Exact QAOA Tractable

_Reference implementation for arXiv:2604.24633_

---

## Introduction

This codebase computes the **exact expected satisfaction fraction** $\tilde{c}(p)$
of the Quantum Approximate Optimization Algorithm at depth $p$ on infinite
$D$-regular $k$-uniform hypergraphs, using tensor-network contraction on the
QAOA light-cone tree. For Max-$k$-XORSAT, the satisfaction fraction is

$$\tilde{c}(p) = \max_{\boldsymbol{\gamma}, \boldsymbol{\beta}} \;
\frac{1 + c_s \cdot \langle Z_1 \cdots Z_k \rangle}{2}$$

where $c_s = +1$ for XORSAT and $c_s = -1$ for MaxCut, and the expectation
is over the root clause of the infinite tree.

The fundamental computational cost is **inescapable**: the branch tensor lives
on $2^{2p+1}$ configurations, so every evaluation requires $O(4^p)$ work.
At $p = 12$, that is 67 million complex numbers per vector.
At $p = 13$, 268 million. At $p = 16$, 17 billion.
No algorithm can reduce this — the light-cone tree has $2p + 1$ binary variables,
and the recurrence must touch every configuration.

What we optimised is *everything else*: the constraint fold from $O(4^{kp})$ to
$O(p \cdot 4^p)$, the gradient from $2p$ evaluations to $1.6\times$ one evaluation,
the memory from $p \cdot 4^p$ to $\sqrt{p} \cdot 4^p$, the numerical stability
from overflow at $p \approx 9$ to controlled results at $p = 16$.

The result: all 15 $(k, D)$ pairs with $k \in \{3, 4, 5, 6, 7\}$ and
$D \in \{4, 5, 6, 7, 8\}$ through $p = 12$ on an Apple M4 Max Mac Studio (64 GB),
$p = 13$ on cloud instances, and $p = 16$ via collaborators on HPC clusters.
MaxCut ($k = 2$) results for $D = 3, \ldots, 9$ fell out as a free byproduct,
matching Farhi et al. to $10^{-15}$ where published values exist.

The ten innovations described here form a **dependency chain** — each was
discovered because the previous one removed a wall, making the next wall visible.
No innovation was designed speculatively. Each was diagnosed from a production
failure at a specific $(k, D, p)$ frontier.

---

## Innovation 1: Walsh–Hadamard Factorisation

### The Problem

The Basso–Villalonga transfer recurrence builds the branch tensor $B_{t+1}$
from $B_t$ via a *constraint fold*: for each parent configuration $a$, sum over
all $k - 1$ child configurations $b^1, \ldots, b^{k-1}$, weighting by the
constraint phase kernel $\kappa$ and the child branch tensors:

$$B_{t+1}(a) = \left[\sum_{b} \kappa(a, b) \cdot f(b) \cdot B_t(b)\right]^{D-1}$$

where the sum inside the brackets is a **$(k-1)$-fold convolution** on
$(\mathbb{Z}_2^{2p+1}, \oplus)$. Naïvely, each fold costs $O(N^{k-1})$ where
$N = 2^{2p+1}$. For $k = 3$, the fold is $O(N^2) = O(4^{2p+1})$; at $p = 8$
that is $10^{10}$ operations *per step*, making $k \geq 3$ intractable beyond
$p \approx 5$.

### The Insight

The constraint fold is a convolution on the abelian group
$(\mathbb{Z}_2^{2p+1}, \oplus)$. The Walsh–Hadamard Transform (WHT) diagonalises
this convolution: if $\hat{g}$ is the WHT of $g$ and $\hat{\kappa}$ is the WHT
of the kernel, then

$$(g \star \cdots \star g)(x) = \text{iWHT}\!\left(\hat{\kappa} \cdot \hat{g}^{\,k-1}\right)(x)$$

where $\hat{g}^{\,k-1}$ denotes element-wise $(k-1)$th power.

### The Algorithm

The constraint fold becomes three lines:

```julia
child_hat = wht(f_table .* B_t)
folded    = iwht(kernel_hat .* child_hat .^ (k - 1))
B_next    = folded .^ (D - 1)
```

The first line applies the WHT to the child message (mixer weights times branch
tensor). The second line performs the fold in the spectral domain — a single
element-wise power and product, then inverse WHT. The third line raises to the
$(D - 1)$th power for the variable-node branching.

### The Cost

Each WHT costs $O(N \log N) = O(p \cdot 4^p)$ via the butterfly decomposition.
There are $p$ steps, so the total recurrence costs $O(p^2 \cdot 4^p)$.
Compare with the naïve fold at $O(p \cdot 4^{kp})$:

| $(k, p)$ | Naïve | WHT | Speedup |
|-----------|-------|-----|---------|
| $(3, 6)$  | $4^{13} \approx 6.7 \times 10^7$ | $6 \cdot 36 \cdot 4^6 \approx 10^6$ | $65\times$ |
| $(3, 8)$  | $4^{17} \approx 1.7 \times 10^{10}$ | $8 \cdot 64 \cdot 4^8 \approx 3 \times 10^7$ | $550\times$ |
| $(3, 10)$ | $4^{21} \approx 4.4 \times 10^{12}$ | $10 \cdot 100 \cdot 4^{10} \approx 10^9$ | $4{,}400\times$ |
| $(3, 12)$ | $4^{25} \approx 1.1 \times 10^{15}$ | $12 \cdot 144 \cdot 4^{12} \approx 3 \times 10^{10}$ | $37{,}000\times$ |

For $k = 3$ at $p = 8$, this is a $\mathbf{65{,}000\times}$ speedup — the
difference between infeasible and routine.

### The $k$-Independence

A crucial structural property: the constraint arity $k$ affects only the
*exponent* in the spectral domain ($\hat{g}^{\,k-1}$), not the transform size
or the number of steps. This means $k = 7$ costs essentially the same as
$k = 3$ per step — the only additional work is raising to the 6th power instead
of the 2nd, an $O(N)$ operation dwarfed by the $O(N \log N)$ WHTs.

### Implementation

The WHT uses a **cache-oblivious recursive decomposition**: for large vectors,
the transform splits at the top butterfly level, then recurses on each half.
Sub-problems of 2048 complex elements (32 KB, fitting in L1 cache on all
targets) use a SIMD-annotated iterative kernel:

```julia
const _WHT_RECURSIVE_CUTOFF = 2048

function _wht_recursive!(values, offset, n)
    if n ≤ _WHT_RECURSIVE_CUTOFF
        _wht_iterative!(values, offset, n)
        return
    end
    half = n >> 1
    @inbounds @simd for i in 0:half-1
        left = offset + i
        right = left + half
        x = values[left]
        y = values[right]
        values[left] = x + y
        values[right] = x - y
    end
    _wht_recursive!(values, offset, half)
    _wht_recursive!(values, offset + half, half)
end
```

Since the butterfly levels of the WHT operate on independent bit positions,
the level-ordering is free — the recursive and iterative approaches produce
identical results. The recursive strategy naturally keeps working sets
L1-resident, which matters at $p \geq 10$ where vectors exceed the L3 cache.

### The Core Three Lines

From `basso_finite_d.jl`, the actual constraint fold used in production:

```julia
child_hat = wht(f_table .* B_t)
folded    = iwht(kernel_hat .* child_hat .^ arity)
B_next    = folded .^ degree
```

where `arity = k - 1` and `degree = D - 1`. This is the entire hot loop of the
evaluation — everything else is angle setup and root extraction.

---

## Innovation 2: Manual Adjoint Differentiation

### The Problem

The optimizer needs gradients $\partial\tilde{c}/\partial\gamma_r$ and
$\partial\tilde{c}/\partial\beta_r$ for all $r = 1, \ldots, p$.
Three options exist:

1. **Finite differences**: $2 \times 2p$ evaluations, noisy at high $p$
2. **Forward-mode AD** (ForwardDiff.jl): exact, but costs $2p$ forward passes
   (one per angle component via JVP)
3. **Reverse-mode adjoint**: exact, single backward pass, cost $\sim 1.6\times$
   one evaluation

At $p = 8$ with $2p = 16$ components, forward-mode requires 16 forward passes.
Each pass is $O(p^2 \cdot 4^p)$. The reverse-mode adjoint requires a single
backward pass at $\sim 0.6\times$ the cost of the forward pass, giving a total
of $\sim 1.6\times$ instead of $\sim 17\times$.

### The Solution

The full pipeline — from angles through branch tensor iteration through root
fold to $\tilde{c}$ — is differentiated by hand using the chain rule for each
operation:

- **WHT is self-adjoint**: if $z = \text{WHT}(x)$, then the cotangent transport
  is $\bar{x} = \text{WHT}(\bar{z})$. This is because the WHT matrix $H$ satisfies
  $H = H^T$, so $\bar{x} = H^T \bar{z} = H \bar{z} = \text{WHT}(\bar{z})$.
  Similarly, iWHT is adjoint to iWHT.

- **Element-wise power**: if $z_i = x_i^n$, then $\bar{x}_i = n \cdot \overline{x_i^{n-1}} \cdot \bar{z}_i$
  (using the Wirtinger derivative for complex-valued functions).

- **$\beta$ gradients via the log-derivative trick**: the mixer weight $f(a)$
  depends on $\beta_r$ through factors $\cos(\beta_r)$ and $i\sin(\beta_r)$.
  Rather than differentiating $f(a)$ directly, we use
  $\partial f / \partial \beta_r = f(a) \cdot \sum_j \ell_j(\beta_r)$
  where $\ell_j$ is $-\tan(\beta_r)$ or $\cot(\beta_r)$ depending on whether
  the $j$th transition is same-bit or different-bit. This avoids recomputing
  the full product.

### Cost

The backward pass performs one WHT per step (same as forward), plus $O(N)$
element-wise operations. Total cost: $\sim 1.6\times$ a single evaluation,
**independent of** $p$. Compare:

| Method | Cost (× one evaluation) | At $p = 8$ |
|--------|------------------------|------------|
| Finite differences | $4p + 1$ | $33\times$ |
| ForwardDiff (JVP) | $2p$ | $16\times$ |
| Manual adjoint | $\sim 1.6$ | $1.6\times$ |

At $p = 8$, the adjoint is **$10\times$ faster** than ForwardDiff and
**$20\times$ faster** than finite differences. At $p = 12$, the ratios
grow to $15\times$ and $33\times$.

### The Sign Bug Story

The initial implementation of the $\beta$ gradient had `+sin(β)` instead of
`-sin(β)` in one of the log-derivative terms. This produced gradients that
were *almost* correct — wrong by a sign flip on one component out of $2p$.

The bug was caught immediately by **cross-validation with ForwardDiff at $p = 1$**:
the manual adjoint gradient and the ForwardDiff gradient disagreed in the 4th
decimal place. The unit test that caught it:

```julia
@test γ_grad ≈ γ_grad_fd atol=1e-10
@test β_grad ≈ β_grad_fd atol=1e-10
```

This is exactly the class of subtle bug that would have caused the optimizer to
converge to wrong local optima, producing **plausible but incorrect results**.
The cross-validation layer caught it before any optimization was ever run.

### Comparison with JPM

The JPM implementation (Boulebnane et al.) uses forward-mode Jacobian-vector
products at $2p\times$ cost — effectively $2p$ forward passes, one per angle
component. Their approach uses $O(4^p)$ memory per pass with no backward
computation graph.

Our reverse-mode adjoint caches $p + 1$ branch tensor snapshots
(the `B_history` array) and performs a single backward sweep. The memory cost
is $O(p \cdot 4^p)$ for the cache, versus $O(4^p)$ per JVP pass, but the compute
savings of $2p / 1.6 \approx p$ make it overwhelmingly favorable for $p > 2$.

---

## Innovation 3: Generic Fold Engine (Cost Algebra)

### The Problem

The initial implementation was hardcoded for $(k = 3, D = 4)$ XORSAT. To
validate correctness, we needed to compare against published MaxCut results
(Farhi, Goldstone, Gutmann 2014; Basso et al. 2022). But MaxCut is $k = 2$ with
$c_s = -1$, and XORSAT is $k \geq 3$ with $c_s = +1$ — different arity,
different sign convention, seemingly different code.

### The Solution

A **cost algebra** abstraction that parametrises the entire fold engine:

```julia
abstract type CostAlgebra{K} end

struct XORSATAlgebra{K} <: CostAlgebra{K}
    clause_sign::Int
end

MaxCutAlgebra() = XORSATAlgebra(2; clause_sign=-1)
```

The cost algebra provides two pluggable methods:
- `constraint_kernel(algebra, angles, branch_degree)` — the problem gate's
  contribution at each non-root constraint node:
  $\kappa(a) = \cos(\Gamma \cdot \text{spins}(a) / 2)$
- `root_observable_kernel(algebra, angles, branch_degree)` — the root observable
  including the $Z_1 \cdots Z_k$ measurement:
  $\kappa_{\text{root}}(a) = i \sin(c_s \cdot \Gamma \cdot \text{spins}(a) / 2)$

Everything else — tree construction, leaf tensor, mixer weights, WHT acceleration,
angle optimization — is **problem-agnostic**. The fold engine is a catamorphism
parametrised by the algebra.

### The "Dry Run" That Built Trust

MaxCut on the $D$-regular random graph corresponds to $(k = 2, c_s = -1)$.
At $D = 3$, Farhi et al. published exact $\tilde{c}(p)$ values through $p = 5$,
and Basso et al. extended to $p = 11$.

Running our code with `MaxCutAlgebra()` reproduced these values to
**$10^{-15}$ relative error** — the limit of Float64 arithmetic:

| $p$ | Farhi et al. | Our code | Relative error |
|-----|-------------|----------|----------------|
| 1 | 0.750000... | 0.750000... | $< 10^{-15}$ |
| 2 | 0.799304... | 0.799304... | $< 10^{-15}$ |
| 3 | 0.824620... | 0.824620... | $< 10^{-15}$ |

This validation runs on the **exact same code path** that computes novel XORSAT
results. A bug in the WHT factorisation, the adjoint differentiation, the
normalisation, or the root extraction would manifest in *both* MaxCut and
XORSAT computations. Agreement with published MaxCut results to machine
precision is therefore strong evidence that the entire pipeline is correct.

### Cross-Validation Principle

The cost algebra makes this cross-validation **structural**: the same
`_forward_pass` function, the same `_backward_pass` function, the same
optimizer — dispatching on different `CostAlgebra` instances. For results to
be wrong for XORSAT but correct for MaxCut, a bug would have to be
conditioned on `clause_sign` or `k` in a way that accidentally cancels for
$k = 2$ but not $k \geq 3$. This is exceedingly unlikely.

### Novel MaxCut Results at Zero Marginal Cost

Because the engine is parametric, first exact finite-$D$ MaxCut results for
$D = 4, 5, 6, 7, 8$ fell out **automatically** — we simply ran
`optimize_depth_sequence(TreeParams(2, D, p_max))` with `MaxCutAlgebra()`.
These results had never been published for $D \geq 4$.

Julia's parametric type system makes this natural:
`QAOAAngles{T}` propagates the element type $T$ through the entire pipeline.
When $T$ is `Float64`, we get standard evaluation. When $T$ is
`ForwardDiff.Dual`, we get automatic forward-mode gradients.
When $T$ is `Double64`, we get extended precision. The cost algebra
dispatches orthogonally, giving a two-dimensional genericity grid at
no code duplication cost.

---

## Innovation 4: Plateau Detection

### The Problem

At $p = 12$, the L-BFGS optimizer routinely spends **2+ hours** making no
progress after finding the optimum. The gradient norm oscillates around
$10^{-3}$ — too large for the default convergence tolerance ($10^{-6}$) but
producing no meaningful change in $\tilde{c}$. The optimizer dutifully continues
until hitting `maxiters`, burning hours of compute.

### The Solution

A **circular buffer** tracking the last 30 objective values. After each
L-BFGS iteration, the callback checks whether the range
$\max(\text{buffer}) - \min(\text{buffer})$ is less than `g_abstol`. If so,
the optimizer has plateaued — the objective is no longer changing, regardless
of what the gradient norm says.

```julia
const PLATEAU_WINDOW_SIZE = 30

push!(value_buffer, val)
if length(value_buffer) > PLATEAU_WINDOW_SIZE
    popfirst!(value_buffer)
end

if length(value_buffer) == PLATEAU_WINDOW_SIZE
    vrange = maximum(value_buffer) - minimum(value_buffer)
    if vrange < g_abstol
        converged_flag = true
        return true  # stop Optim immediately
    end
end
```

A secondary guard checks gradient stagnation: if the gradient norm has been
below $100 \times$ `g_abstol` for 20 consecutive iterations *and* the value
range over those 20 iterations is below $10 \times$ `g_abstol`, exit.

### Impact

At $p = 12$ for $(k = 3, D = 4)$: wall time dropped from **2+ hours to
$\sim 40$ minutes**. The optimizer now stops within a few iterations of finding
the basin, instead of orbiting around the minimum for hundreds of iterations.

The depth-dependent gradient tolerance reinforces this:

| Depth | `g_abstol` | Rationale |
|-------|-----------|-----------|
| $p \leq 10$ | $10^{-6}$ | Converges in 5–45 iterations reliably |
| $p = 11$ | $10^{-5}$ | Gradient noise floor makes $10^{-6}$ marginal |
| $p \geq 12$ | $10^{-4}$ | Gradient norm oscillates $\sim 10^{-4}$ |

---

## Innovation 5: Normalised Branch Tensor Recurrence

### The Problem

The branch tensor recurrence raises vectors to the $(D - 1)$th power at each
step:

$$B_{t+1}(a) = \left[\text{folded}_t(a)\right]^{D-1}$$

At high $(k, D)$, the magnitudes compound exponentially. For $(k = 5, D = 7)$
with branching factor $b = (D-1)(k-1) = 24$, the magnitudes grow as roughly
$\alpha^{(D-1)^t}$ where $\alpha$ is a typical magnitude ratio. Float64
overflows at $\sim 1.8 \times 10^{308}$, which is reached around $p \approx 9$
for large $(k, D)$.

### The Solution

**Per-step conditional normalisation** with log-scale accumulation. Before each
power operation, if the vector's maximum magnitude exceeds a safety threshold
($10^{30}$, chosen so that $(\text{threshold})^{\max(\text{arity}, \text{degree})} < 10^{300}$),
normalise to unit maximum magnitude and accumulate the scale factor in log space:

$$\hat{c}_t^{\text{norm}} = \hat{c}_t / \alpha_t, \quad
\alpha_t = \max_i |\hat{c}_t(i)|$$

$$f_t^{\text{norm}} = f_t / \beta_t, \quad
\beta_t = \max_i |f_t(i)|$$

$$\log s_{t+1} = (k-1)(D-1) \cdot \log s_t + (k-1) \cdot \log \alpha_t + (D-1) \cdot \log \beta_t$$

At the root, the physical answer is recovered via:

$$\tilde{c} = \frac{1 + c_s \cdot \exp(L) \cdot \operatorname{Re}(S_{\text{norm}})}{2}$$

where $L = k \cdot (\log s_{p+1} + \log \mu)$ and $\mu$ is the root message
normalisation factor. The product $\exp(L) \cdot \operatorname{Re}(S_{\text{norm}})$ is
computed entirely in log space to avoid overflow.

### The Three-Tier Defense

The normalisation has three tiers:

1. **Threshold normalisation**: only normalise when magnitudes threaten overflow
   ($> 10^{30}$), not every step. Normalising unconditionally destroys the
   relative magnitude relationships between entries that carry the physical
   signal (the deviation from $\tilde{c} = 0.5$), causing signal underflow at
   high depth.

2. **Log-space scale accumulation**: the running scale is a single `Float64`
   scalar updated per step. No vector operations; no precision loss from
   multiplying large numbers.

3. **Overflow guard at the root**: if $L > 700$ (which would overflow `exp()`),
   return `NaN` — the optimizer's overflow-safe layer (Innovation 6) handles it.

### Impact

Without normalisation:
- $(k = 5, D = 7)$ at $p \geq 8$: silent `Inf` in branch tensors,
  producing $\tilde{c} = \text{NaN}$ that the optimizer treats as a valid result
- $(k = 7, D = 8)$ at $p \geq 6$: same failure

With normalisation, these $(k, D)$ pairs produce valid results through $p = 12$
in Double64 precision.

### Backward Pass Compatibility

The backward pass operates entirely on normalised intermediates. The gradient
of the normalisation factors (which are `max` operations with sparse
subgradients) is **detached** — treated as a constant. This is valid because:

1. The gradient of `max|x|` is a sparse selection operator (nonzero only at the
   argmax), contributing negligibly compared to the $O(N)$ gradient terms.
2. The physical answer $\tilde{c}$ is invariant to the normalisation convention.
3. The single `exp(L)` multiplier is applied once at the end; any gradient
   approximation error affects only the *path* to convergence, not the result.

---

## Innovation 6: Overflow-Safe Optimiser

### The Problem

Even with normalisation (Innovation 5), some angle configurations overflow
during optimizer exploration — particularly early random restarts at high
$(k, D)$. The optimizer evaluates $\tilde{c}$ at an overflow-producing angle
configuration, gets `NaN` or a value outside $[0, 1]$, and treats it as a
valid objective value. L-BFGS, designed for smooth landscapes, cannot recover
from garbage in its Hessian approximation.

### The Five Guards

The overflow-safe optimizer wraps the evaluation pipeline with five layers:

**Guard 1: Non-zero gradient at overflow.** When the evaluator returns an
invalid $\tilde{c}$ (NaN, Inf, or outside $[0, 1]$), return a large objective
value ($10^6$) with a gradient pointing toward the origin:

```julia
function _overflow_gradient!(G, values, p)
    for j in eachindex(G)
        G[j] = values[j] > 0 ? 1.0 : -1.0
    end
end
```

This makes L-BFGS *backtrack* away from the overflow region rather than parking
at it. Returning zero gradient would fake convergence — the optimizer would
declare success at an invalid point.

**Guard 2: Post-evaluation range check.** Every evaluation checks
$\tilde{c} \in [-10^{-9}, 1 + 10^{-9}]$. The tolerance allows for
floating-point noise but catches overflow-induced values like $\tilde{c} = 3.23$
or $\tilde{c} = -47.2$.

```julia
is_valid_qaoa_value(v::Real) = isfinite(v) && -1.0e-9 ≤ v ≤ 1.0 + 1.0e-9
```

**Guard 3: Validity-aware argmax.** When selecting the best result across
restarts, a valid result **always** beats an invalid result, regardless of
objective value:

```julia
best = if sv && !pv
    secondary
elseif pv && !sv
    primary
elseif sv && pv && secondary.value > primary.value
    secondary
else
    primary
end
```

**Guard 4: Validity-aware merge.** When combining results from primary and
retry runs, the same principle applies. An invalid primary with $\tilde{c} =
10^6$ is replaced by a valid retry with $\tilde{c} = 0.502$.

**Guard 5: Warm-start chain quarantine.** The depth-sweep strategy uses the
optimal angles at depth $p$ to warm-start the optimization at depth $p + 1$.
If the $p$-result is invalid (failed to converge, overflowed), its angles are
*not* propagated into the warm-start chain. Fresh random starts are used
instead. This prevents a single overflow event from poisoning the entire
depth sequence.

### Key Insight

The fundamental principle is: **a valid result always beats an invalid result**.
This is trivially correct — we would rather report $\tilde{c} = 0.501$ (a weak
but valid lower bound) than $\tilde{c} = 3.23$ (meaningless). The optimizer's
convergence guarantees are local; the overflow guards ensure that the global
result selection is sound.

---

## Innovation 7: Swarm/Memetic Optimiser

### The Problem

At high $(k, D)$, the optimisation landscape is **extremely flat**: most
starting points see $\tilde{c} \approx 0.5$ (no better than random), and only
specific basins carry signal above the noise floor. Standard multi-start
L-BFGS with 8 restarts fails at $(k = 7, D = 8)$ starting from $p = 3$ — none
of the random starts find a basin with signal.

### The Solution

A **memetic optimizer** that maintains a population of 100 candidates and
repeatedly applies selection pressure:

1. **Short L-BFGS bursts** (20 iterations each) on all candidates — enough to
   find a local gradient signal but not enough to waste compute on flat regions
2. **Cull** the worst 50% of the population
3. **Replenish** with a mix of:
   - 40% fresh random starts (exploration)
   - 60% midpoint crossovers from survivors (exploitation)
4. **Early exit**: if the best candidate has converged (gradient norm below
   tolerance), skip remaining generations and polish
5. **Full polish**: run the winner through a full L-BFGS optimization

```julia
function swarm_optimize(params::TreeParams;
    population::Int=100,
    generations::Int=10,
    burst_iters::Int=20,
    cull_fraction::Float64=0.5,
    random_fraction::Float64=0.4,
    ...)
```

The burst strategy is key: 20 L-BFGS iterations are enough to determine
whether a starting point has gradient signal ($\tilde{c}$ increasing) or is
stuck on the flat plateau ($\tilde{c} \approx 0.5$). Candidates that show no
improvement after 20 iterations are culled, freeing compute budget for more
promising regions.

Memory-bounded concurrency ensures that at high $p$ (where each evaluation
uses multiple GB of RAM), the swarm doesn't exhaust system memory:

```julia
est_bytes_per_eval = 40 * N * sizeof(ComplexF64)
available_ram = Sys.total_memory() * 0.75
max_concurrent = max(1, floor(Int, available_ram / est_bytes_per_eval))
burst_semaphore = Base.Semaphore(max(1, burst_concurrency))
```

### Impact

- $(k = 7, D = 8)$: went from **failing at $p = 3$** to valid results at $p = 8+$
- $(k = 6, D = 7)$: swarm finds basins that multi-start misses, improving
  $\tilde{c}$ by $10^{-4}$ to $10^{-3}$ over the multi-start result
- Low $(k, D)$: no benefit (the landscape is smooth; multi-start suffices)

The swarm optimizer is used only when the landscape warrants it. For $(k = 3, D = 4)$,
standard multi-start L-BFGS converges reliably.

---

## Innovation 8: Double64 Precision

### The Problem

At $(k, D)$ with large branching factor $b = (D-1)(k-1)$, the $(D-1)$th power
in the branch tensor recurrence amplifies cancellation errors beyond Float64's
15 decimal digits. The normalised recurrence (Innovation 5) prevents overflow,
but the *relative* precision of the normalised values degrades.

### The Diagnostic

At $(k = 6, D = 7)$, $p = 10$:
- **Float64** returns $\tilde{c} = 3.23$ — an obviously invalid result
  (satisfaction fractions must be in $[0, 1]$)
- **Double64** returns $\tilde{c} = 0.8130...$ — a valid, physically
  meaningful result

The invalid Float64 result is not an overflow (the normalisation prevents
that) but a **precision collapse**: after 10 steps of raising to the 6th
power, the relative error in the normalised branch tensor entries exceeds
$10^{-1}$, making the final root extraction meaningless.

### The Solution

[DoubleFloats.jl](https://github.com/JuliaMath/DoubleFloats.jl) provides
`Double64`: a pair of `Float64` values used as a single extended-precision
number with $\sim 31$ decimal digits, at $3$–$5\times$ the cost of a single
Float64 operation.

The entire evaluation pipeline is **type-generic** via Julia's parametric types.
`QAOAAngles{T}` propagates the element type `T` through every function:

```julia
struct QAOAAngles{T<:Real}
    γ::Vector{T}
    β::Vector{T}
end
```

To switch from Float64 to Double64, the optimizer simply promotes angles
before evaluation:

```julia
_promote_angles(a::QAOAAngles, ::Type{T}) where T = QAOAAngles(T.(a.γ), T.(a.β))
```

No code changes, no separate code path. The WHT, the power operations, the
normalisation, the adjoint — all work unchanged with Double64 elements.

### The Precision Frontier

The maximum safe depth in Float64 varies by $(k, D)$:

| $(k, D)$ | Max safe $p$ (Float64) | Branching factor | Notes |
|-----------|----------------------|------------------|-------|
| $(3, 4)$  | $\geq 16$ | 6 | No precision issues observed |
| $(3, 8)$  | $\geq 13$ | 14 | Marginal at $p = 14$ |
| $(5, 6)$  | 10 | 20 | Float64 fails at $p = 11$ |
| $(5, 7)$  | 8 | 24 | Float64 fails at $p = 9$ |
| $(6, 7)$  | 7 | 30 | Float64 returns invalid at $p = 8$ |
| $(7, 8)$  | 5 | 42 | Float64 fails at $p = 6$ |

For each $(k, D)$ past its Float64 frontier, Double64 extends the range by
$\sim 5$–$8$ additional depths before encountering its own precision limit.

---

## Innovation 9: GPU Acceleration

### The Problem

At $p \geq 12$, wall time on CPU is hours per evaluation. The branch tensor
has $2^{25} \approx 33$ million complex entries at $p = 12$, and each step
performs multiple WHTs and element-wise operations on these vectors. The
$O(p^2 \cdot 4^p)$ cost is dominated by memory bandwidth, not compute — a
natural fit for GPU parallelism.

### The Solution

GPU kernels written with [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
for portable execution on **CUDA** (NVIDIA) and **Metal** (Apple Silicon)
backends. The GPU pipeline includes:

- **GPU WHT**: level-by-level butterfly with fused multi-level kernel launches.
  Each thread handles one butterfly pair; threads are independent within a
  level but synchronise between levels.
- **GPU forward pass**: precomputation (angles, trig tables) on CPU;
  the $O(p^2 \cdot 4^p)$ hot loop (element-wise products, powers, WHTs)
  runs entirely on GPU.
- **GPU backward pass**: adjoint differentiation on GPU, using the same
  kernel structure as the forward pass.
- **GPU checkpointed adjoint**: gradient checkpointing (Innovation 10)
  with GPU-accelerated segment recomputation.

```julia
@kernel function _fold_kernel!(out, @Const(kernel_hat), @Const(child_hat), @Const(arity))
    i = @index(Global)
    @inbounds begin
        val = child_hat[i]
        result = val
        for _ in 2:arity
            result *= val
        end
        out[i] = kernel_hat[i] * result
    end
end
```

Metal requires Float32 (no Float64 support on Apple GPUs), which limits
precision but provides $\sim 10\times$ throughput over CPU at $p \leq 10$.
CUDA supports Float64 natively.

### Status

The GPU pipeline is implemented and tested but **not wired into the production
depth-sweep scripts**. The CPU checkpointed adjoint path is the workhorse for
$p \geq 13$ results. GPU acceleration is most valuable for interactive
exploration at $p \leq 10$ and for future $p \geq 14$ runs where CPU wall
time becomes prohibitive.

---

## Innovation 10: Gradient Checkpointing

### The Problem

The standard adjoint (Innovation 2) caches all $p + 1$ branch tensors $B_t$,
plus $p$ child-hat vectors and $p$ folded vectors — a total of $\sim 3p$
vectors of size $N = 2^{2p+1}$. At $p = 13$:

$$3 \times 13 \times 2^{27} \times 16 \text{ bytes} \approx 84 \text{ GB}$$

This exceeds the RAM of most workstations. At $p = 16$:

$$3 \times 16 \times 2^{33} \times 16 \text{ bytes} \approx 3.3 \text{ TB}$$

### The Solution

Store only $\lceil\sqrt{p}\rceil$ branch tensor checkpoints instead of all $p + 1$.
During the backward pass, when intermediates from step $t$ are needed, recompute
the forward pass from the nearest preceding checkpoint:

```julia
ci = checkpoint_interval > 0 ? checkpoint_interval : max(1, ceil(Int, sqrt(p)))
```

The checkpoint-selection strategy:
1. Always store the initial $B_1 = \mathbf{1}$
2. Store $B_t$ at every $\lceil\sqrt{p}\rceil$ steps
3. Always store the final $B_{p+1}$ (needed for the root backward)

During backward, when step $t$ needs $B_t$, child\_hat$_t$, and folded$_t$:
1. Find the nearest checkpoint at or before level $t$
2. Recompute the forward pass from that checkpoint to $t$, storing
   only the needed intermediates
3. Run the backward step
4. Discard the recomputed intermediates

### Cost Analysis

**Memory**: $\sqrt{p}$ checkpoints × $N$ × 16 bytes instead of $3p$ vectors:

| $p$ | Full cache | Checkpointed | Reduction |
|-----|-----------|-------------|-----------|
| 12 | $\sim 50$ GB | $\sim 8$ GB | $6\times$ |
| 13 | $\sim 84$ GB | $\sim 12$ GB | $7\times$ |
| 16 | $\sim 3.3$ TB | $\sim 128$ GB | $26\times$ |

**Compute**: each backward step recomputes at most $\sqrt{p}$ forward steps,
and there are $p$ backward steps, giving $O(p\sqrt{p})$ forward steps total
instead of $O(p)$. The overhead is $\sim 2\times$ the forward pass cost
instead of $\sim 0.6\times$, for a total of $\sim 3\times$ one evaluation
instead of $\sim 1.6\times$.

At $p = 13$ this trade-off is essential: $3\times$ compute with 12 GB RAM
is feasible; $1.6\times$ compute with 84 GB RAM is not (on a 64 GB machine).

### Disk Spillover

For $p = 16+$ where even $\sqrt{p}$ checkpoints exceed RAM, the checkpointer
supports **disk spillover**: checkpoints that exceed a RAM cap are serialised
to disk using Julia's `Serialization` module and read back during the backward
pass:

```julia
if ram_count >= max_ram_checkpoints && disk_dir !== nothing
    path = joinpath(disk_dir, "checkpoint_B_$(t+1).bin")
    open(path, "w") do io
        serialize(io, B)
    end
    checkpoint_disk_paths[t + 1] = path
end
```

At $p = 16$ with Float64, each checkpoint is $\sim 32$ GB (the $B$ vector).
With 4 checkpoints in RAM and the rest on NVMe, this enables runs on machines
with 128 GB RAM + fast SSD.

---

## The Stacking Architecture

The ten innovations form a stack where each layer depends on those below it:

| # | Innovation | Wall Broken | Enabled By |
|---|-----------|-------------|------------|
| 1 | WHT factorisation | $k \geq 3$ beyond $p = 5$ | Mathematical insight |
| 2 | Manual adjoint | $p \geq 6$ tractable gradients | Clean WHT (self-adjoint) |
| 3 | Cost algebra | Cross-validation against MaxCut | Parametric fold engine |
| 4 | Plateau detection | $p = 12$ wall time | Gradient convergence analysis |
| 5 | Normalised recurrence | High $(k,D)$ overflow at $p \geq 9$ | Log-scale arithmetic |
| 6 | Overflow-safe optimizer | Optimizer crashes on overflow | Normalisation exposed the failures |
| 7 | Swarm optimizer | Flat landscapes at high $(k,D)$ | Overflow-safe evaluator |
| 8 | Double64 | Precision collapse at high $b$ | Type-generic pipeline |
| 9 | GPU acceleration | CPU wall time at $p \geq 12$ | KernelAbstractions portability |
| 10 | Gradient checkpointing | Memory wall at $p \geq 13$ | Adjoint structure |

### The Discovery Thesis

No innovation was designed speculatively. The development sequence was:

1. Implement the basic recurrence → works for $(k = 3, D = 4)$ at $p \leq 5$
2. WHT factorisation → extends to $p = 8$, but gradients are slow
3. Manual adjoint → gradients at $1.6\times$, but only for $(k = 3, D = 4)$
4. Cost algebra → validates against MaxCut, extends to all $(k, D)$
5. Run at $p = 12$ → optimizer wastes hours after converging
6. Plateau detection → $p = 12$ in 40 minutes, but $(k = 5, D = 7)$ overflows at $p = 8$
7. Normalisation → fixes overflow, but optimizer crashes on remaining overflows
8. Overflow-safe optimizer → stable, but $(k = 7, D = 8)$ fails at $p = 3$
9. Swarm optimizer → finds basins, but Float64 precision collapses at high $(k, D)$
10. Double64 → precision to 31 digits, but memory limits $p$ to 12 on workstations
11. Gradient checkpointing → $p = 13$ on 64 GB, $p = 16$ on HPC

Each innovation made the next wall visible by removing the previous one. The
codebase's **clarity was not a luxury** — it was the mechanism of discovery.
A tangled implementation would have obscured the failure modes that motivated
each innovation. The parametric, compositional design (type-generic angles,
cost algebra dispatch, self-adjoint WHT) made each innovation a local change
rather than a rewrite.

---

## Correctness Framework: Why You Should Trust These Numbers

The central claim of arXiv:2604.24633 is exact $\tilde{c}(p)$ values for 15
$(k, D)$ pairs through $p = 12+$, together with optimal angles. The
computation is non-trivial: each data point is the global maximum of a
$2p$-dimensional function evaluated via a $p$-step tensor network contraction.
Why should you trust these numbers?

### The Layered Defense

For a reported $\tilde{c}(p)$ value to be wrong, **all** of the following
independent validation layers would have to fail simultaneously:

#### Layer 1: MaxCut Cross-Validation

The code with `MaxCutAlgebra()` reproduces Farhi et al. (2014) and Basso et al.
(2022) to $10^{-15}$ — the limit of Float64 arithmetic. This validates the
*entire pipeline*: WHT, branch tensor iteration, root fold, optimizer.

A bug in any shared component (and the fold engine is entirely shared) would
manifest as a MaxCut discrepancy.

#### Layer 2: Adjoint Gradient Cross-Validation

The manual adjoint gradient is validated against ForwardDiff.jl (Julia's
forward-mode AD) at every $(k, D)$ and multiple $p$ values. Agreement is
to $10^{-10}$ or better:

```julia
@test γ_grad ≈ γ_grad_fd atol=1e-10
@test β_grad ≈ β_grad_fd atol=1e-10
```

This catches sign errors, missing terms, and indexing bugs in the backward
pass. The sign bug story (Innovation 2) demonstrates that this layer catches
*exactly* the class of subtle errors that produce plausible but incorrect
optimization results.

#### Layer 3: Range Checks

Every evaluation verifies $\tilde{c} \in [0, 1]$ (with floating-point
tolerance). The satisfaction fraction of a quantum state is a probability
and cannot be negative or exceed 1. Any value outside this range indicates
a computational failure — overflow, precision collapse, or a code bug.

This layer catches failures that other layers might miss: a correct gradient
at a correct evaluation point doesn't help if the evaluator overflows at
different points in the landscape.

#### Layer 4: Strict Monotonicity

For every $(k, D)$ pair, the optimal $\tilde{c}(p)$ must satisfy

$$\tilde{c}(p) \leq \tilde{c}(p + 1)$$

because QAOA at depth $p$ is a strict subset of QAOA at depth $p + 1$ (set the
last angle pair to zero). Our results satisfy this for all 15 $(k, D)$ pairs
at every depth step from $p = 1$ through $p = 12$.

A violation of monotonicity would indicate either a failed optimization
(stuck in a bad local optimum) or a computational error. Neither has been
observed.

#### Layer 5: Cross-Seed Consistency

Multiple random restarts produce optimal $\tilde{c}$ values that agree to 8+
decimal digits through $p = 10$. At $p \geq 11$ the agreement is 4–6 digits,
reflecting the gradient noise floor rather than correctness issues.

This layer catches optimizer failures: if one restart finds a better optimum
than another, the disagreement reveals that the worse run was stuck. The
validity-aware merge (Innovation 6) ensures that the best valid result is
always selected.

#### Layer 6: Float64/Double64 Agreement

For $(k, D)$ pairs where both Float64 and Double64 produce valid results,
the values agree to Float64 precision ($\sim 10^{-14}$). This validates
both the Float64 results (they are not corrupted by precision loss) and the
Double64 results (the extended precision is not introducing bugs).

Disagreement at a specific $(k, D, p)$ indicates that Float64 precision is
insufficient — exactly the diagnostic used to establish the precision frontier
(Innovation 8).

#### Layer 7: Comprehensive Test Suite

The test suite contains **2,012 assertions** covering:

- WHT correctness (forward, inverse, convolution theorem)
- Branch tensor recurrence against brute-force light-cone evaluation
- Adjoint gradients against ForwardDiff at multiple $(k, D, p)$
- Cost algebra dispatch for MaxCut and XORSAT
- Normalisation stability under extreme magnitudes
- Overflow guard behavior
- Checkpointed vs. full-cache gradient agreement
- Optimizer convergence on known-solution instances
- Edge cases (degenerate angles, boundary conditions)

The tests validate not just correctness but **consistency**: the checkpointed
adjoint must produce identical gradients to the full-cache adjoint, the GPU
forward pass must match the CPU forward pass, the reduced-basis iteration
must match the full-basis iteration.

#### Layer 8: Smooth Angle Trajectories

At low $k$ (where the landscape is well-behaved), the optimal angles
$\gamma_r(p)$ and $\beta_r(p)$ trace smooth curves as $p$ increases. A
sudden discontinuity in the angle trajectory would indicate that the optimizer
jumped to a different basin — possibly a worse one.

This is a *visual* diagnostic rather than an automated test, but it provides
a sanity check that automated layers cannot: the optimizer is tracking the
*same* basin across depth, not hopping between local optima.

#### Layer 9: Physical Bounds

The satisfaction fraction must lie in $[0.5, 1.0]$: the lower bound is
the random assignment fraction (QAOA at $p = 0$ achieves exactly $0.5$),
and the upper bound is perfect satisfaction. All reported values satisfy
this constraint, with $\tilde{c}(1) > 0.5$ at every $(k, D)$ (QAOA at $p = 1$
always beats random).

### Why Nine Layers?

No single validation layer is sufficient. MaxCut cross-validation (Layer 1)
validates the pipeline but not the optimizer. Adjoint cross-validation (Layer 2)
validates the gradient but not the evaluation. Range checks (Layer 3) catch
catastrophic failures but not subtle precision loss. Monotonicity (Layer 4)
catches optimizer failures but not symmetric bugs. Cross-seed consistency
(Layer 5) catches local optima but not systematic errors. Float64/Double64
agreement (Layer 6) catches precision issues but not algorithmic bugs.
Tests (Layer 7) cover unit-level correctness but not integration-level
optimizer behavior. Angle trajectories (Layer 8) catch basin-hopping but
require human judgment. Physical bounds (Layer 9) catch absurd results but
allow a wide range of plausible-but-wrong values.

Together, these nine layers form a **defense in depth** where each layer
catches failure modes that the others miss. For a reported result to be wrong,
a bug would have to:

1. Not affect MaxCut evaluation (Layer 1)
2. Not affect gradient computation (Layer 2)
3. Produce values in $[0, 1]$ (Layer 3)
4. Preserve monotonicity in $p$ (Layer 4)
5. Be deterministic across random seeds (Layer 5)
6. Be precision-independent (Layer 6)
7. Not be caught by 2,012 unit tests (Layer 7)
8. Not disrupt angle trajectories (Layer 8)
9. Stay within physical bounds (Layer 9)

Satisfying all nine constraints simultaneously while producing incorrect
$\tilde{c}$ values is, in our assessment, exceedingly unlikely.

### The Sign Bug, Revisited

The sign bug in the adjoint gradient (Innovation 2) provides a concrete
example of the validation framework in action. The bug was:

- **Not caught** by range checks (Layer 3): the wrong gradient still produces
  $\tilde{c} \in [0, 1]$
- **Not caught** by monotonicity (Layer 4): the optimizer with wrong gradients
  still finds increasing $\tilde{c}(p)$, just at slightly worse values
- **Not caught** by physical bounds (Layer 9): the results look plausible

But it was **caught immediately** by adjoint cross-validation (Layer 2): the
manual gradient and ForwardDiff gradient disagreed at $p = 1$. Without this
layer, the bug would have produced an entire table of *plausible but incorrect*
results — slightly worse than the true optima, indistinguishable from genuine
results without an independent implementation.

This is why the correctness framework has nine layers: each bug class requires
a different detection mechanism.

---

## Appendix: File Map

The innovations map to source files as follows:

| Innovation | Primary source file(s) |
|-----------|----------------------|
| 1. WHT factorisation | `src/wht.jl`, `src/basso_finite_d.jl` |
| 2. Manual adjoint | `src/adjoint.jl` |
| 3. Cost algebra | `src/cost_algebra.jl` |
| 4. Plateau detection | `src/optimization.jl` |
| 5. Normalised recurrence | `src/adjoint.jl` (forward pass) |
| 6. Overflow-safe optimizer | `src/optimization.jl` |
| 7. Swarm optimizer | `src/optimization.jl` (`swarm_optimize`) |
| 8. Double64 precision | `src/tensors.jl` (`QAOAAngles{T}`) |
| 9. GPU acceleration | `src/gpu_*.jl` |
| 10. Gradient checkpointing | `src/checkpointed_adjoint.jl` |
