# Transfer Contraction for k-Body Constraints

> **Purpose:** Resolve the blocker identified in P1.3 — how do k-1 child branches combine at a constraint node?
> **Sources:** Basso et al. 2021, §8 (arXiv:2110.14206), Eqs. (8.7)–(8.8), (8.13)–(8.15)
> **Read after:** `05-tensor-derivation.md`, `P1.3 implementation notes`

---

## The Problem

The P1.3 spec assumed that at a constraint node with k-1 identical child branches, the contribution is just the branch tensor raised entrywise to the power (k-1):

$$T^{k-1}_\sigma = (T_\sigma)^{k-1}$$

This is **wrong for k > 2**. The developer agent correctly identified this. Here's why, and what the correct formula is.

## Why Entrywise Exponentiation Fails at Constraint Nodes

At a **variable node**, the D-1 child branches are independent — each connects to a *different* constraint, so there is no coupling between the children's hyperindices. Their contributions multiply independently, entry by entry. Entrywise `.^(D-1)` is correct.

At a **constraint node** with k variables, the k-body problem gate $\exp(-i\gamma Z_1 Z_2 \cdots Z_k / 2)$ couples all k variables' hyperindices simultaneously. When k-1 child branches meet at a constraint node, their hyperindices are entangled through this gate. You cannot factor the joint contribution into a product of individual contributions.

## The Correct Formula (Basso 2021, Eq. 8.7)

Basso et al. give the correct iterative formula. Let $H_D^{(m)}(\mathbf{a})$ be the branch tensor after $m$ contraction steps, where $\mathbf{a}$ is a $(2p+1)$-component bit string (their convention). Starting with $H_D^{(0)}(\mathbf{a}) = 1$, each step is:

$$H_D^{(m)}(\mathbf{a}) = \left[ \sum_{\mathbf{b}^1, \ldots, \mathbf{b}^{q-1}} \cos\!\left(\frac{1}{\sqrt{D}} \boldsymbol{\Gamma} \cdot (\mathbf{a}\,\mathbf{b}^1 \mathbf{b}^2 \cdots \mathbf{b}^{q-1})\right) \prod_{i=1}^{q-1} f(\mathbf{b}^i) H_D^{(m-1)}(\mathbf{b}^i) \right]^D$$

Here:
- $q$ is our $k$ (constraint arity)
- $D$ is the number of branching hyperedges per variable (our $D-1$ in the non-root context)
- $\boldsymbol{\Gamma}$ encodes the QAOA angles
- $f(\mathbf{b})$ is a product of mixer matrix entries
- The sum over $\mathbf{b}^1, \ldots, \mathbf{b}^{q-1}$ ranges over all $(2p+1)$-bit strings for each of the $q-1$ child variables

## Key Observation: The Cost is $O(p \cdot 4^{pq})$, NOT $O(p \cdot 4^p)$

The sum over $\mathbf{b}^1, \ldots, \mathbf{b}^{q-1}$ has $4^{p(q-1)}$ terms (each $\mathbf{b}^i$ ranges over $4^p$ values). Combined with the $4^p$ entries of $\mathbf{a}$, each iteration step costs $O(4^{pq})$.

For $q = k = 2$ (MaxCut): cost is $O(4^{2p}) = O(16^p)$ per step... but the cosine factorises into a product of 2-body terms, which reduces to $O(4^p)$. This is the Farhi 2025 trick.

For $q = k = 3$ (our target): cost is $O(4^{3p})$ per step naively. At $p = 10$ that's $4^{30} \approx 10^{18}$ — **completely infeasible**.

## But Wait: The Large-D Limit Gives $O(p^2 \cdot 4^p)$

Basso 2021 Eq. (8.9)–(8.10) gives a compact iteration in the $D \to \infty$ limit:

$$G_{j,k}^{(m)} = \sum_{\mathbf{a}} f(\mathbf{a}) a_j a_k \exp\!\left(-\frac{1}{2} \sum_{j',k'} \left(G_{j',k'}^{(m-1)}\right)^{q-1} \Gamma_{j'} \Gamma_{k'} a_{j'} a_{k'}\right)$$

And the final result:

$$\nu_p^{[q]}(\gamma, \beta) = \frac{i}{\sqrt{2q}} \sum_j \Gamma_j \left(G_{0,j}^{(p)}\right)^q$$

Note the $q$-th power of the matrix element — that's the only place $q$ enters. This iteration has cost $O(p^2 \cdot 4^p)$ and memory $O(p^2)$, **regardless of q**!

But this is the **large-D limit** with $O(1/D)$ corrections — exactly the approximation we're trying to avoid at $D = 4$.

## The Middle Ground: Exact Finite-D at Moderate Cost?

The core question is whether the $O(4^{pq})$ cost at finite D can be reduced. Let me examine the structure of Eq. (8.7) more carefully.

The sum over $\mathbf{b}^1, \ldots, \mathbf{b}^{q-1}$ at fixed $\mathbf{a}$ involves:

$$\cos\!\left(\frac{1}{\sqrt{D}} \boldsymbol{\Gamma} \cdot (\mathbf{a}\,\mathbf{b}^1 \cdots \mathbf{b}^{q-1})\right) \prod_{i=1}^{q-1} f(\mathbf{b}^i) H_D^{(m-1)}(\mathbf{b}^i)$$

The cosine is a function of the **joint parity** of $\mathbf{a}, \mathbf{b}^1, \ldots, \mathbf{b}^{q-1}$ at each round position. The inner product $\boldsymbol{\Gamma} \cdot (\mathbf{a}\,\mathbf{b}^1 \cdots \mathbf{b}^{q-1})$ is:

$$\sum_{\ell} \Gamma_\ell \cdot a_\ell \cdot b^1_\ell \cdot b^2_\ell \cdots b^{q-1}_\ell$$

where $a_\ell, b^i_\ell \in \{-1, +1\}$. This is a sum over rounds, and at each round the contribution depends on the **product** $a_\ell \prod_i b^i_\ell$.

### Factoring over rounds

Since the cosine is $\cos(\sum_\ell x_\ell)$ where $x_\ell = \Gamma_\ell a_\ell \prod_i b^i_\ell$, this does NOT factorise over rounds. The exponential form $e^{i \sum_\ell x_\ell} = \prod_\ell e^{ix_\ell}$ does factorise, but the real part (cosine) mixes all rounds through cross-terms.

### Factoring over child branches

The product $\prod_i f(\mathbf{b}^i) H_D^{(m-1)}(\mathbf{b}^i)$ DOES factorise over branches. And if the cosine could be expressed in terms of per-branch quantities, the sum would factorise too.

But $\prod_i b^i_\ell$ couples all branches at each round. This is the fundamental obstruction.

### Parity decomposition trick

However, $\prod_{i=1}^{q-1} b^i_\ell$ only depends on the **parity** of the $b^i_\ell$ values. Define:

$$s_\ell = b^1_\ell \cdot b^2_\ell \cdots b^{q-1}_\ell \in \{-1, +1\}$$

There are $2^{2p+1}$ possible parity vectors $\mathbf{s}$. For each fixed $\mathbf{s}$, the number of $(\mathbf{b}^1, \ldots, \mathbf{b}^{q-1})$ configurations that produce that parity can be computed from the individual branch tensors.

Specifically, define the **parity-projected branch tensor**:

$$\tilde{H}^{(m)}_\pm(\ell) = \sum_{\mathbf{b}: b_\ell = \pm 1} f(\mathbf{b}) H_D^{(m-1)}(\mathbf{b})$$

Then the number of (q-1)-tuples with $s_\ell = +1$ at position $\ell$ involves $q-1$ copies having an even number of $-1$'s at that position... This gets combinatorially complex.

## Assessment: Where This Leaves Us

### What's feasible

1. **Large-D iteration** ($O(p^2 \cdot 4^p)$): straightforward to implement, gives $O(1/D)$-approximate results. At $D = 4$ the error is ~25%, which is significant but could provide useful bounds and starting points for angle optimisation.

2. **Exact brute-force** ($O(2^n)$ where $n \sim 6^p$): what we already have. Capped at p=1 for (k=3, D=4) — p=2 requires 129 qubits (3 + 18 + 108), far beyond the 22-qubit guard. For k=2, D=3: feasible up to p≈4 (30 qubits).

3. **Exact finite-D Basso iteration** ($O(p \cdot 4^{3p})$): Basso Eq. (8.7) directly. For k=3:
   - p=5: $4^{15} \approx 10^9$ — feasible (minutes)
   - p=7: $4^{21} \approx 4 \times 10^{12}$ — hard (cluster, hours)
   - p=10: $4^{30} \approx 10^{18}$ — infeasible

4. **Hybrid approach**: use the exact finite-D iteration at moderate p, then extrapolate or use the large-D iteration for higher p.

### What's NOT feasible

The original P1.3 spec's claim of $O(4^p)$ cost at any k — this only holds for k=2. For k=3 the cost is at minimum $O(4^{3p})$ at finite D, or $O(4^p)$ only in the large-D limit.

### Recommendations

1. **Implement the exact finite-D Basso iteration** ($O(4^{3p})$ for k=3). This gives exact results and is feasible up to p ≈ 5–7 depending on compute budget. That may be enough to see whether QAOA is competitive with SA.

2. **Also implement the large-D iteration** ($O(4^p)$) as a comparison point. The $O(1/D)$ error at $D=4$ needs to be quantified empirically — compare against exact results at the p values where both are feasible.

3. **Look for a parity decomposition** that reduces $O(4^{3p})$ to something like $O(4^{2p})$. The parity structure of the k-body gate should allow partial factoring. This is a research problem worth investigating but not blocking.

4. **Update the P1.3 spec** to reflect the true cost and implement the Basso Eq. (8.7) iteration directly.

---

## Worked Example: k=3, D=4, p=1

At p=1, the hyperindex has 2p+1 = 3 components (in Basso's convention) or 2 bits (in our convention). Let me verify against the brute-force simulator.

The brute-force simulator gives:
- `qaoa_expectation(TreeParams(3, 4, 1), QAOAAngles([0.0], [0.0])) = 0.5` (zero-angle baseline)
- `parity_expectation(TreeParams(3, 4, 1), QAOAAngles([0.0], [0.0])) = 0.0`

At non-zero angles (e.g. γ=0.3, β=0.5):
- These can be computed by the brute-force simulator and should match the Basso finite-D formula

**TODO for developer agent**: Implement Eq. (8.7) and verify numerical agreement with the brute-force simulator at p=1 and p=2 (where the brute-force is feasible).

---

## Summary

| Approach | Cost | Exactness | Feasible p (k=3,D=4) |
|----------|------|-----------|----------------------|
| Brute-force state vector | $O(2^{6^p})$ | Exact | p=1 only (p=2 needs 129 qubits) |
| Basso finite-D, naive (Eq. 8.7) | $O(p \cdot 4^{3p})$ | Exact | p ≤ 5–7 |
| **Basso finite-D + WHT** | **$O(p^2 \cdot 4^p)$** | **Exact** | **p ≤ 15–17** |
| Basso large-D iteration (Eq. 8.9) | $O(p^2 \cdot 4^p)$ | $O(1/D)$ approximate | p ≤ 15+ |
| Original P1.3 spec (wrong) | $O(p \cdot 4^p)$ | N/A | N/A |

---

## Update (22 March 2026): WHT Factorisation Resolves Open Question 1

The parity decomposition research question posed in this document has been
**fully resolved**. The constraint fold is a convolution on $\mathbb{Z}_2^{2p+1}$,
and the Walsh-Hadamard transform diagonalises it:

$$\hat{S} = \hat{\kappa} \cdot \hat{g}^{k-1}$$

This reduces the exact finite-D constraint fold from $O(4^{kp})$ to
$O(p \cdot 4^p)$ per step, giving $O(p^2 \cdot 4^p)$ overall — the **same
asymptotic cost as the large-D approximation, but exact at any finite D**.

The key insight: the sum over $(k-1)$ child configurations is a convolution over
child *parities*, not a factorisation over *rounds*. The cosine kernel need not
decompose into per-round factors — it just needs to be a well-defined function
on $\mathbb{Z}_2^{2p+1}$, which it trivially is.

**Numerically verified** at p=1,2,3 for D=3,4 to machine precision ($< 10^{-14}$).

Full derivation and verification: see
[15-wht-factorisation-discovery.md](15-wht-factorisation-discovery.md).

This makes the project's primary deliverable — exact QAOA performance at (k=3, D=4)
up to p=15+ — fully feasible. The recommendations in this document (items 2–4)
are superseded: we do not need the large-D approximation or hybrid approaches.
