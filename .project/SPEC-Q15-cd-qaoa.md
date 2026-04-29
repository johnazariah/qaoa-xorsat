# Q1.5: Test Wurtz–Love's Counterdiabatic QAOA on D-Regular MaxCut

## Status

Follow-up to Q1 ([SPEC-Q1-adiabatic.md](SPEC-Q1-adiabatic.md)).
Worktree: `/Users/johnaz/PhD/qaoa-xorsat-q1`, branch `q1-adiabatic`.

## Motivation

[learning/21-qaoa-vs-trotterised-adiabatic.md](learning/21-qaoa-vs-trotterised-adiabatic.md)
identified that our Q1 results refute the *strawman* "QAOA is Trotterised
adiabatic" claim (Paper 1, McDowall et al.) but do not directly test the
*non-linear* CD-QAOA prescription of Wurtz & Love (Paper 2,
[arXiv:2106.15645](https://arxiv.org/abs/2106.15645)).

Q1.5 closes that gap: we implement Wurtz–Love's lowest-order CD-QAOA
forward map (their Eq. 29) using the explicit $\alpha(\lambda)$ for
$\nu$-regular triangle-free graphs (their Eq. 45) on the infinite-girth
tree, and compare its performance to our warm-start optima.

## Wurtz–Love Construction

### Eq. 45 — adiabatic gauge potential coefficient

For a $\nu$-regular graph with no triangles:

$$
\alpha(\lambda; \nu) = \frac{-32(1-\lambda)^2 - 8(3\nu - 2)\lambda^2}
{256\bigl((1-\lambda)^2 + 4(3\nu-2)\lambda^2\bigr)^2 + 256\lambda^2(1-\lambda)^2(\nu-1) + 96(\nu-1)(\nu-2)\lambda^4}.
$$

The infinite-girth tree of degree $D$ satisfies the triangle-free condition,
so we substitute $\nu = D$.  $\alpha \leq 0$ for all $\lambda \in [0,1]$.

### Eq. 29 — lowest-order forward map (1st Magnus / 2nd BCH)

For each layer $q = 1, \ldots, p$ of a CD-QAOA evolution along a continuous
protocol $(\lambda(t), s(t))$ on $[0, T]$:

$$
\bar\lambda_q := \frac{1}{\tau_q}\int_{t_{q-1}}^{t_q}\lambda(t)\,dt = \frac{\gamma_q}{\gamma_q + \beta_q},
\qquad
\tau_q = \gamma_q + \beta_q,
$$

$$
\bar s_q := \frac{1}{\tau_q}\int_{t_{q-1}}^{t_q} s(t)\,dt
       = -\frac{\gamma_q \beta_q}{2 \tau_q} - \frac{\lambda_q - \lambda_{q-1}}{\tau_q} \alpha(\lambda_q).
$$

## Two Experiments

### Tier 1: Optimal-T Linear Schedule (LR-QAOA at best $T$)

With $\lambda(t) = t/T$ and uniform $\tau_q = T/p$:

$$
\gamma_q = \frac{T(q - \tfrac{1}{2})}{p^2}, \qquad
\beta_q = \frac{T(p - q + \tfrac{1}{2})}{p^2}.
$$

(α(λ) does not appear — at lowest order with linear λ, the angles are pure
LR-QAOA shape.)

**Procedure**: for each $(D, p)$ with $D \in \{3..8\}$, $p \in \{1..12\}$:

1. Scan $T \in [0.1, 10]$ on a grid of ~50 points + golden-section refinement
2. Evaluate $\tilde c[\text{Tier 1}](D, p, T)$ via `basso_expectation_normalized`
3. Record $T^*$ and $\tilde c^*$

**Hypothesis**: even with $T$ optimised, Tier 1 will fall well below
warm-start at $D \geq 5$ — strengthening E1.

### Tier 2: Non-Linear CD-QAOA via Eq. 45

We require the inferred $\bar s_q$ to vanish for all $q$ — i.e. the QAOA
faithfully reproduces a counterdiabatic evolution along $\lambda(t)$ alone,
with the BCH commutator term acting as the $-\dot\lambda \alpha(\lambda)$
counterdiabatic field.  With piecewise-linear $\lambda(t)$ through nodes
$\lambda_0 = 0, \lambda_1, \ldots, \lambda_p = 1$ and uniform $\tau = T/p$:

- $\bar\lambda_q = (\lambda_{q-1} + \lambda_q)/2$
- $\gamma_q = \tau (\lambda_{q-1} + \lambda_q)/2$
- $\beta_q = \tau (2 - \lambda_{q-1} - \lambda_q)/2$
- $\bar s_q = 0$ becomes:

$$
\tau^2 (\lambda_{q-1} + \lambda_q)(2 - \lambda_{q-1} - \lambda_q)
= -8 (\lambda_q - \lambda_{q-1})\,\alpha(\lambda_q; D).
\tag{*}
$$

**Procedure**: for each $(D, p)$:

1. Outer bisection on $T$:
   - Initialise $\lambda_0 = 0$.
   - For $q = 1, \ldots, p$: solve (*) for $\lambda_q$ given $\lambda_{q-1}, \tau, D$ via 1D root-finding (Brent / bisection on $[\lambda_{q-1}, 1]$).
   - Record $\lambda_p(T)$.
   - Bisect $T$ until $\lambda_p(T) = 1$ within tolerance $10^{-8}$.
2. With $T^*$ and $\{\lambda_q\}$, compute $\{\gamma_q, \beta_q\}$.
3. Evaluate $\tilde c[\text{Tier 2}]$.

**Numerical concern**: at $\lambda \to 0$, $\alpha(0) = -1/8$, equation (*)
gives $\lambda_1 = 2 - 1/(2\tau^2) \cdot (\dots)$ — there is a minimum
$\tau$ below which the prescription has no positive root.  Concretely
$\tau^2 \geq 1/2$ at the first step.  Need $T \geq p/\sqrt{2}$ as a lower
bracket for the bisection.

**Hypothesis**: Tier 2 will improve over Tier 1 (strictly stronger
construction) but **still fall short of warm-start** at higher $D$.  This
is the direct test of Wurtz–Love's construction on D-regular MaxCut.

### Tier 3 (deferred)

Higher-order BCH/Magnus expansion (W-L's general Eq. 27) requires
deriving Pauli traces beyond second order.  Only worth pursuing if Tier 2
is inconclusive.

## Outputs

- `scripts/q15_cd_qaoa_tier1.jl` — Tier 1 implementation
- `scripts/q15_cd_qaoa_tier2.jl` — Tier 2 implementation
- `scripts/q15_plots.py` — comparison figure
- `results/q15-cd-qaoa-tier1.csv` — `D,p,T_star,ctilde,gamma,beta`
- `results/q15-cd-qaoa-tier2.csv` — `D,p,T_star,ctilde,lambda_nodes,gamma,beta,converged`
- `figures/q15-cd-qaoa-comparison.png` — $\tilde c$ vs $p$ at each $D$:
  warm-start, Tier 1, Tier 2 (and E1's matched-magnitude linear-adi as a
  fourth curve, for completeness)
- Update [learning/21-qaoa-vs-trotterised-adiabatic.md](learning/21-qaoa-vs-trotterised-adiabatic.md)
  with results
- Journal Entry 34

## Tests

`julia --project=. -e 'using Pkg; Pkg.test()'` must remain at 1741/1741.
No `src/` changes expected; if α(λ) helper goes into `src/`, add tests.

## Interpretation Matrix

Outcomes of Tier 2 vs warm-start (call the gap $\Delta_D(p) =
\tilde c^{\text{warm}} - \tilde c^{\text{Tier2}}$):

| Outcome | Interpretation |
|---------|----------------|
| $\Delta_D(p) \to 0$ as $p$ grows, all $D$ | W–L fully vindicated; large-$p$ QAOA on D-regular MaxCut is CD-QAOA. |
| $\Delta_D(p)$ small at low $D$, large at high $D$ | W–L's construction works in the low-density regime where AGP is meaningful; fails where the MaxCut problem becomes "harder." |
| $\Delta_D(p)$ doesn't shrink with $p$ | Direct refutation of CD-QAOA on this problem.  Optimal QAOA does something genuinely different. |
| $\Delta_D(p)$ shrinks but $\tilde c^{\text{Tier2}}$ never catches up | Consistent with W–L's own caveat that "globally optimal protocols may be hard to find." |

The result feeds directly into the MaxCut paper's "QAOA-vs-adiabatic"
section.
