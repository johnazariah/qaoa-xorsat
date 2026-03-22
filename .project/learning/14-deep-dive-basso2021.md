# Deep Dive: Basso et al. 2021 — Max-q-XORSAT Generalisation

> **Source**: arXiv:2110.14206, full text at `papers/text/basso2021-qaoa-high-depth.txt`
> **Date**: 22 March 2026
> **Purpose**: Detailed findings from reading the full paper text, focused on
> the k-XORSAT generalisation (§8) and its relevance to our P1.3 implementation.

---

## 1. The Max-q-XORSAT Cost Function (Eq. 8.1)

$$C_J^{\text{XOR}}(z) = \sum_{(i_1,\ldots,i_q) \in E}
\frac{1 + J_{i_1\cdots i_q}\, z_{i_1} z_{i_2} \cdots z_{i_q}}{2}$$

where $J_{i_1\cdots i_q} \in \{+1,-1\}$. Satisfied iff
$z_{i_1}\cdots z_{i_q} = J_{i_1\cdots i_q}$.

**J-independence** (§8.1): On hypertrees the coupling signs $J$ can be gauged
away (no cycles → local bit flips absorb them). Performance is independent
of $J$.

## 2. CRITICAL Convention: Degree

**Basso's $D$ = our $D - 1$.** The paper uses $(D+1)$-regular hypergraphs
(vertex degree $D+1$, light cone is a $D$-ary hypertree). For our $D=4$
(degree 4), use **Basso $D = 3$**.

## 3. Two Distinct Iterations

### A. Finite-$D$ Iteration (§8.2) — EXACT at any $D$

Branch tensor $H_D^{(m)}(\mathbf{a})$ maps $\{-1,+1\}^{2p+1} \to \mathbb{C}$.
Initial: $H_D^{(0)} = 1$.

**Recurrence (Eq. 8.7):**

$$H_D^{(m)}(\mathbf{a}) = \left[\sum_{\mathbf{b}^1,\ldots,\mathbf{b}^{q-1}}
\cos\!\left(\frac{\boldsymbol{\Gamma} \cdot
(\mathbf{a}\,\mathbf{b}^1\cdots\mathbf{b}^{q-1})}{\sqrt{D}}\right)
\prod_{i=1}^{q-1} f(\mathbf{b}^i)\,H_D^{(m-1)}(\mathbf{b}^i)\right]^D$$

Where:
- $\mathbf{a}, \mathbf{b}^i \in \{-1,+1\}^{2p+1}$
- $\boldsymbol{\Gamma} = (\gamma_1,\ldots,\gamma_p, 0, -\gamma_p,\ldots,-\gamma_1)$
- $f(\mathbf{a})$ = product of mixer matrix elements (bra-ket sandwich)
- $\mathbf{a}\,\mathbf{b}^1\cdots\mathbf{b}^{q-1}$ = **entry-wise** product

**Final formula (Eq. 8.8):**

$$\nu_p^{[q]}(D,\gamma,\beta) = i\sqrt{\frac{D}{2q}}
\sum_{\mathbf{a}^1,\ldots,\mathbf{a}^q}
\sin\!\left(\frac{\boldsymbol{\Gamma}\cdot(\mathbf{a}^1\cdots\mathbf{a}^q)}{\sqrt{D}}\right)
\prod_{j=1}^{q} a_0^j\,f(\mathbf{a}^j)\,H_D^{(p)}(\mathbf{a}^j)$$

**Satisfaction fraction (Eq. 8.4):**

$$\text{fraction} = \frac{1}{2} + \nu_p^{[q]}(D,\gamma,\beta)\sqrt{\frac{q}{2D}}$$

**Cost**: $O(p \cdot 4^{pq})$. For $q=3$: **$O(p \cdot 64^p)$**.
Memory: $O(4^p)$ (branch tensor).

### B. $D\to\infty$ Iteration (§8.3) — Approximate at finite $D$

Uses correlation matrix $G^{(m)} \in \mathbb{C}^{(2p+1)\times(2p+1)}$.

**Recurrence (Eq. 8.9):**

$$G_{j,k}^{(m)} = \sum_{\mathbf{a}} f(\mathbf{a})\,a_j\,a_k
\exp\!\left(-\frac{1}{2}\sum_{j',k'}
\bigl(G_{j',k'}^{(m-1)}\bigr)^{q-1}\Gamma_{j'}\Gamma_{k'}\,a_{j'}a_{k'}\right)$$

**Final (Eq. 8.10):**

$$\nu_p^{[q]}(\gamma,\beta) = \frac{i}{\sqrt{2q}}\sum_j
\Gamma_j\,\bigl(G_{0,j}^{(p)}\bigr)^q$$

Only differences from MaxCut ($q=2$):
1. Exponent: $G \to G^{q-1}$
2. Final: $G^2 \to G^q$
3. Prefactor: $i/2 \to i/\sqrt{2q}$

**Cost**: $O(p^2 \cdot 4^p)$ — **independent of $q$**.

---

## 4. What $O(1/D)$ Means at $D=4$

The $D\to\infty$ formula comes from Taylor-expanding:

$$\cos(x/\sqrt{D})^D \approx (1 - x^2/(2D))^D \to e^{-x^2/2}$$

Neglected terms are $O(1/D)$ in the exponent. At our $D=4$ (Basso $D=3$),
the error is $O(1/3) \approx 33\%$ in the exponent coefficient. Numerical
impact: potentially several percent error in the satisfaction fraction.

**This applies ONLY to the $D\to\infty$ iteration.** The finite-$D$ iteration
(§8.2) is exact. The issue is cost, not accuracy.

---

## 5. The Constraint Fold Structure for $k=3$

The inner sum in Eq. 8.7:

$$\sum_{\mathbf{b}^1, \mathbf{b}^2}
\cos\!\left(\frac{\boldsymbol{\Gamma}\cdot(\mathbf{a}\,\mathbf{b}^1\mathbf{b}^2)}{\sqrt{D}}\right)
g(\mathbf{b}^1)\,g(\mathbf{b}^2)$$

where $g(\mathbf{b}) = f(\mathbf{b})\,H_D^{(m-1)}(\mathbf{b})$.

The entry-wise product $b^1_r b^2_r$ inside the cosine **couples the two child
branches at each round**. This is why the sum doesn't factorise: $\cos(\text{sum})
\neq \prod \cos(\text{terms})$.

Naive cost: $O(4^{2p})$ per parent value, $O(4^{3p})$ total.

**Open question (P1.3 spec OQ1):** Can this be reduced by:
- Grouping by the product $c = \mathbf{b}^1 \odot \mathbf{b}^2$?
- Exploiting the parity structure at each round?
- Walsh-Hadamard-type transforms?

> **Note (22 March 2026):** A research agent proposed that WHT factorises this
> to $O(p^2 \cdot 4^p)$. **Proof completed** in `learning/15-wht-research-spec.md`.
> The inner sum factors into two convolutions on $\mathbb{Z}_2^{2p+1}$ via the
> Walsh-Hadamard transform. **Awaiting numerical validation** — see §11 of the spec.

---

## 6. Table 1 — $\bar\nu_p$ Values (MaxCut, $D\to\infty$)

| $p$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|-----|---|---|---|---|---|---|---|---|---|
| $\bar\nu_p$ | 0.3033 | 0.4075 | 0.4726 | 0.5157 | 0.5476 | 0.5721 | 0.5915 | 0.6073 | 0.6203 |

| $p$ | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|-----|----|----|----|----|----|----|----|----|----|----|----|
| $\bar\nu_p$ | 0.6314 | 0.6408 | 0.6490 | 0.6561 | 0.6623 | 0.6679 | 0.6729 | 0.6773 | 0.6813 | 0.6848 | 0.6879 |

QAOA exceeds the classical threshold $2/\pi \approx 0.6366$ at $p=11$.

---

## 7. Performance for $q \geq 3$ ($D\to\infty$)

From Fig. 6, approximate values (read from plot — not tabulated):

| $q$ | $\bar\nu_{14}^{[q]} / \Pi_q$ | Notes |
|-----|------|-------|
| 3 | ~0.77 | No OGP barrier (odd $q$) |
| 4 | ~0.60 | |
| 5 | ~0.47 | |
| 6 | ~0.38 | |

Raw numerical values available in their GitHub repo:
`github.com/benjaminvillalonga/large-girth-maxcut-qaoa`

**Caveat**: these are $D\to\infty$ numbers. Our finite-$D=4$ values will differ.

---

## 8. Corrections to Our Explainer (File 02)

1. **Missing finite-$D$ iteration.** §3.1 and §8.2 give an exact iteration at
   any $D$. The explainer portrays Basso as $D\to\infty$ only.
2. **Two separate formulas.** Not "one formula with an approximation" but two
   distinct iterations with different costs and accuracy.
3. **Cost at $k \geq 3$**: finite-$D$ costs $O(p \cdot 4^{pq})$, NOT $O(4^p)$.
4. **Degree convention**: paper's $D$ = our $D-1$. Not mentioned in explainer.
5. **Table values**: explainer quotes unverified "fraction of Parisi value";
   paper tabulates raw $\bar\nu_p$ values (listed above).

---

## 9. Answers to PLAN.md Open Questions

**OQ1: "Does the Basso formula extend to exact finite-$D$?"**
**Yes** — §8.2 gives the exact finite-$D$ iteration. No approximation involved.
Cost is the obstacle: $O(p \cdot 64^p)$ at $k=3$.

**OQ2: "What is the precise contraction cost for (k=3, D=4)?"**
Finite-$D$ (Basso §8.2): $O(p \cdot 64^p)$.
$D\to\infty$ (Basso §8.3): $O(p^2 \cdot 4^p)$ but with $O(1/D)$ error.
Whether an intermediate option exists is Open Question 1 in P1.3.
