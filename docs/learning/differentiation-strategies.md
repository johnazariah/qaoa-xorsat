# Differentiation Strategies for QAOA Angle Optimisation

> **Purpose**: Explain why we need derivatives of the QAOA objective, the three
> strategies we use (finite differences, forward-mode AD, manual adjoint), and
> how each maps to the codebase.
> **Read after**: `03-explainer-farhi2025-maxcut-lower-bound.md` (L-BFGS
> overview), `15-wht-factorisation-discovery.md` (the evaluation pipeline being
> differentiated).
> **Date**: 24 March 2026

---

## Why We Need Gradients

The QAOA angle optimiser (L-BFGS) finds the angles
$(\gamma_1 \ldots \gamma_p, \beta_1 \ldots \beta_p)$ that maximise the expected
satisfaction fraction $E(\boldsymbol{\gamma}, \boldsymbol{\beta})$. L-BFGS is a
**quasi-Newton** method: at each step it needs the gradient

$$\nabla E = \left(\frac{\partial E}{\partial \gamma_1}, \ldots, \frac{\partial E}{\partial \gamma_p}, \frac{\partial E}{\partial \beta_1}, \ldots, \frac{\partial E}{\partial \beta_p}\right)$$

to choose a search direction and step size. The quality and cost of this gradient
directly determine how fast the optimiser converges and how deep (in $p$) we can
push.

The function $E$ is computed by `basso_expectation` — a pipeline of ~500 lines
of tensor operations, Walsh–Hadamard transforms, trigonometric functions, and
element-wise powers. We need to differentiate through all of it.

---

## Strategy 1: Finite Differences

### The idea

Perturb each angle individually, re-evaluate, take the slope:

$$\frac{\partial E}{\partial \gamma_i} \approx \frac{E(\gamma_i + \epsilon) - E(\gamma_i - \epsilon)}{2\epsilon}$$

### Cost

$2 \times 2p$ evaluations per gradient — two per angle, and we have $2p$ angles.

### Quality

Approximate. Choosing $\epsilon$ is a balancing act:
- Too large → truncation error (the secant doesn't match the tangent)
- Too small → catastrophic cancellation ($E(\gamma_i + \epsilon)$ and
  $E(\gamma_i - \epsilon)$ agree to most digits, and the difference is noise)

The practical accuracy floor is around $10^{-8}$. This is why we had to relax the
L-BFGS convergence tolerance `g_abstol` from $10^{-8}$ to $10^{-6}$ in Entry 18:
the optimiser was trying to follow a gradient below its noise floor and declaring
premature convergence.

### In the codebase

- `Optim.jl` with `ADTypes.AutoFiniteDiff()` in `src/optimization.jl`
- Activated when `autodiff=false` in `optimize_angles`
- Used as a fallback / debugging baseline

---

## Strategy 2: Forward-Mode AD (ForwardDiff.jl)

### The idea

Instead of perturbing the input numerically, replace every `Float64` with a
**dual number**:

$$a + b\varepsilon \quad \text{where } \varepsilon^2 = 0$$

When you compute $f(a + b\varepsilon)$, the result is
$f(a) + f'(a) \cdot b\varepsilon$ — the derivative rides along through every
operation (addition, multiplication, `sin`, `cos`, `exp`, etc.) via the standard
rules of differential arithmetic.

### Why it works in our code

`QAOAAngles{T<:Real}` is parametric. When `T = ForwardDiff.Dual`, the entire
Basso pipeline — trig functions, WHT butterfly loops, element-wise powers —
propagates derivatives automatically with **zero code changes**. Julia's type
specialisation compiles a monomorphised code path for dual numbers just as it
does for `Float64`.

### Cost

One "fat" evaluation per input direction. To get all $2p$ partial derivatives,
ForwardDiff runs $2p$ passes, each carrying one dual component. So the cost is:

$$\text{cost} = 2p \times (\text{one evaluation with Dual overhead})$$

The Dual overhead makes each evaluation roughly $2\text{–}3\times$ slower than
plain `Float64` (due to carrying the derivative alongside the value).

At $p = 8$ with 16 angles, the total gradient cost is roughly $16 \times 2.5
\approx 40\times$ one plain evaluation.

### Quality

**Exact** to machine precision. No $\epsilon$ to tune, no noise floor. L-BFGS
converges in fewer iterations because the search direction is trustworthy.

### In the codebase

- `Optim.jl` with `ADTypes.AutoForwardDiff()` in `src/optimization.jl`
- Activated when `autodiff=true` (the default)
- Enabled by the `{T<:Real}` generic signatures throughout `src/tensors.jl`,
  `src/wht.jl`, and `src/basso_finite_d.jl`

---

## Strategy 3: Manual Adjoint (Reverse-Mode) Differentiation

### The idea

Forward mode propagates derivatives *forward* through the computation, one input
direction at a time. The adjoint flips this: it propagates *backward* from the
output, computing **all** input derivatives simultaneously.

### How it works

**Forward pass**: Run `basso_expectation` normally, but **save all
intermediates** in a cache:
- Branch tensor values $B^{(0)}, B^{(1)}, \ldots, B^{(p)}$
- WHT-transformed arrays at each step
- Precomputed trig tables ($\cos$, $\sin$ of phase arguments)
- The $f$-table (mixer weights)

**Backward pass**: Starting from $\frac{\partial E}{\partial E} = 1$, apply the
chain rule in reverse through each saved operation:

1. **Root observable**: the observable is linear, so its adjoint is trivial
2. **Element-wise power** $x \mapsto x^n$: adjoint is $\bar{x}_i \mathrel{+}= n \cdot x_i^{n-1} \cdot \bar{y}_i$
3. **WHT**: the Walsh–Hadamard transform is its own transpose (self-adjoint up
   to a scale factor), so the adjoint of WHT is another WHT — no new code needed
4. **Cosine kernel** $\cos(\phi)$: adjoint is $-\sin(\phi)$, using the saved
   trig table
5. **Phase argument** $\phi = \sum_i \gamma_i z_i$: adjoint accumulates
   $\bar{\gamma}_i \mathrel{+}= z_i \cdot \bar{\phi}$
6. **Mixer weights** ($f$-table contributions): adjoint accumulates
   $\bar{\beta}$ contributions via the saved $f$-table derivatives

At the end, the accumulated cotangent vectors $\bar{\boldsymbol{\gamma}}$ and
$\bar{\boldsymbol{\beta}}$ are the exact gradient.

### Cost

$$\text{cost} \approx 2 \times (\text{one Float64 evaluation})$$

This is **independent of the number of angles**. Whether $p=4$ (8 angles) or
$p=12$ (24 angles), the backward pass has the same cost.

### Why the advantage grows with $p$

| $p$ | Angles ($2p$) | ForwardDiff cost | Adjoint cost | Adjoint speedup |
|-----|---------------|-------------------|--------------|-----------------|
| 4   | 8             | $\sim 20\times$   | $\sim 2\times$ | $10\times$   |
| 8   | 16            | $\sim 40\times$   | $\sim 2\times$ | $20\times$   |
| 12  | 24            | $\sim 60\times$   | $\sim 2\times$ | $30\times$   |

The adjoint is the key unlock for $p \geq 9$.

### Why not use an off-the-shelf reverse-mode AD?

Julia has reverse-mode AD packages (Zygote.jl, Enzyme.jl), but they struggle
with our pipeline:

- **Zygote.jl**: Cannot handle in-place mutations (`wht!`), bitwise integer
  operations, or the complex control flow in the Basso iteration
- **Enzyme.jl**: Requires LLVM-level instrumentation and has edge cases with
  Julia's type system and generic functions

Our pipeline has **exploitable structure** that a general-purpose tool cannot
leverage:
- The WHT is self-adjoint → the backward WHT is just another forward WHT
- Element-wise power has a trivially simple derivative
- The trig tables are already computed on the forward pass

Hand-writing the adjoint lets us exploit all of this while avoiding the overhead
and fragility of a general-purpose reverse-mode tool. The trade-off: ~420 lines
of differentiation code, but a gradient that is exact, fast, and fully under our
control.

### In the codebase

- `src/adjoint.jl` (on the `feature/adjoint-differentiation` branch)
- Public API: `basso_expectation_and_gradient(params, angles; clause_sign)`
- Spec: `.project/specs/manual-adjoint.md`
- **Status (24 March 2026)**: γ gradient validated against ForwardDiff to
  $10^{-10}$; β gradient has a known bug under investigation

---

## Summary: Three Strategies Compared

| Property              | Finite Differences    | ForwardDiff           | Manual Adjoint        |
|-----------------------|-----------------------|-----------------------|-----------------------|
| Cost per gradient     | $4p \times$ eval      | $\sim 2p \times$ eval | $\sim 2 \times$ eval  |
| At $p=8$              | $32\times$            | $\sim 40\times$       | $\sim 2\times$        |
| Gradient quality      | $\sim 10^{-8}$        | machine-ε             | machine-ε             |
| Code changes needed   | none                  | generics (`{T}`)      | ~420 LOC hand-written |
| Works with `Optim.jl` | yes (`AutoFiniteDiff`) | yes (`AutoForwardDiff`) | needs wrapper       |
| Fragility             | $\epsilon$ sensitive  | robust                | must maintain by hand |

The project evolved through these strategies in order:
1. **Finite differences** — got the pipeline working and producing initial results
   through $p=5$ (Entries 17–18)
2. **ForwardDiff** — exact gradients with minimal code changes, enabling reliable
   convergence and thread-parallel restarts (Entry 19)
3. **Manual adjoint** — depth-independent gradient cost, unlocking $p \geq 9$
   (Entry 19, in progress)

---

## Connection to the Literature

Farhi et al. 2025 (arXiv:2503.12789) pushed MaxCut to $p=17$ using C++ with
LBFGS++. Their paper does **not specify** the gradient computation method
(noted in `13-deep-dive-farhi2025.md`). Given their C++ stack, they likely used
either finite differences or a hand-written adjoint — general-purpose AD in C++
is less natural than in Julia.

Our approach has two structural advantages over what Farhi et al. likely used:
1. **ForwardDiff via parametric types**: Julia's type system makes forward-mode
   AD transparent, requiring only generic signatures — no code duplication
2. **WHT self-adjointness**: our manual adjoint exploits the fact that the
   dominant operation in the Basso pipeline (the Walsh–Hadamard transform) is
   its own transpose, halving the adjoint implementation effort
