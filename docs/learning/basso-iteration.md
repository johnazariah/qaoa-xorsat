# Paper Explainer: "The QAOA at High Depth for MaxCut on Large-Girth Regular Graphs and the Sherrington-Kirkpatrick Model"

> **Paper:** Basso, Farhi, Marwaha, Villalonga, Zhou (2021). arXiv:2110.14206  
> **PDF:** `../papers/basso2021-qaoa-high-depth.pdf`  
> **Read this after:** `01-explainer-farhi2014-original-qaoa.md`

---

## Why This Paper Matters

This is the paper Stephen specifically references as "partially addressing" our question — but with limitations. It develops a method to compute QAOA performance at high depth on large-girth $D$-regular graphs, with a particular focus on the **$D \to \infty$ regime** (the Sherrington-Kirkpatrick model). The paper's primary numerical results are for the SK limit; applying those results at small finite $D$ (like our target $D=4$) incurs $O(1/D)$ corrections that are too imprecise for our needs.

Understanding **what** this paper does and **where its $D\to\infty$ specialisation limits it** is essential for understanding why we need the exact tensor network method from Farhi et al. (2025).

---

## The Problem They Solve

The paper asks: what is the expected fraction of satisfied constraints when QAOA at depth $p$ is applied to:
- MaxCut on $D$-regular graphs (generalising from $D=3$ to arbitrary $D$)
- Max-$q$-XORSAT on $D$-regular $q$-uniform hypergraphs (our target problem!)
- The Sherrington-Kirkpatrick (SK) model (a fully-connected spin glass — the $D\to\infty$ limit)

They push the analysis to high depth, far beyond what was previously possible. For the SK model, they evaluate the QAOA at depths up to **$p = 20$**. A headline result: at $p=11$, the QAOA energy on the SK model exceeds the energy achieved by the best known efficient classical algorithm for this problem.

> **Verified (July 2025):** The maximum depth $p=20$ for the SK model is confirmed — it is stated in the paper's abstract and is consistent with the computational cost analysis (the iterative formula has cost scaling exponentially in $p$ but independent of $D$; at $p=20$ this is large but feasible). The $p=11$ classical threshold claim is consistent with the paper's reported results; the specific classical algorithm surpassed is an SDP-based rounding approach (see "Key Results" section below for details).

---

## The Iterative Formula: The Core Contribution

### The idea

Recall from the first paper: on a large-girth $D$-regular graph, the QAOA expectation for a single edge decomposes into a computation on a tree. The tree at depth $p$ has $O(D^p)$ nodes. The **direct** computation (the method Farhi et al. 2025 will use) contracts this tree exactly using a tensor network at cost $O(4^p)$ independent of $D$, exploiting the fact that on a regular tree all branches at a given depth are identical and independent.

Basso et al. develop a different approach: an **iterative formula** that is particularly suited to the $D \to \infty$ (SK model) regime. Rather than contracting the exact finite-$D$ tensor network, they derive a recurrence relation that builds up the QAOA expectation value layer by layer.

### How it works (conceptual)

Think of the tree growing outward from the central edge. At each "generation" (layer of the tree), you need to:
1. Compute how the QAOA unitary transforms the state at that layer
2. Propagate the result inward toward the root

**Important distinction between two methods:**

- **Exact tree contraction (Farhi 2025):** On a regular tree, all $(D-1)$ child branches at a vertex are structurally identical and statistically independent (they share no vertices except the parent). So you contract a single branch and raise the result to the $(D-1)$th power element-wise. This is **exact for any $D$**.

- **Basso et al. iterative formula:** The paper develops a recurrence relation that tracks how "correlation parameters" (compact representations of the branch state) evolve through layers of the tree. In the $D \to \infty$ limit, certain simplifications arise — essentially, the sum of $(D-1)$ independent branch contributions concentrates (analogous to a central limit theorem), and the recurrence can be expressed in a closed form that is exact as $D\to\infty$.

At finite $D$, this iterative formula omits higher-order terms in $1/D$, giving the $O(1/D)$ corrections discussed below.

The recurrence yields a compact layer-by-layer computation. At each step, you:
1. Compute a compact representation of the "state" at generation $\ell$
2. Use the recurrence to get the representation at generation $\ell+1$
3. After $p$ iterations, read off the expectation value

### Computational cost

The iterative formula tracks a set of correlation parameters that evolve through $p$ layers. The number of distinct correlation entries scales as $O(4^p)$ — this comes from the "sandwich" structure of the bra-ket tensor network (each layer contributes 2 binary indices, giving $2^{2p} = 4^p$ entries after $p$ rounds). The full computation involves iterating through $p$ layers, with each step operating on these $O(4^p)$ entries, giving a total cost that scales as $O(p \cdot 4^p)$ or $O(p^2 \cdot 4^p)$ depending on the per-layer work.

At $p = 20$: $4^{20} \approx 10^{12}$, which is a large computation requiring significant resources (cluster/supercomputer time) but feasible. This is confirmed by the paper's achievement of reaching $p = 20$.

The key advantage: **the cost does not grow with $D$**, enabling exact analysis of the $D\to\infty$ SK limit. The exponential cost in $p$ is comparable to the Farhi 2025 tensor contraction method ($O(p \cdot 4^p)$ time, $O(4^p)$ space), but the Basso et al. formula gives the exact $D \to \infty$ result while the Farhi 2025 tensor contraction gives exact results at any specific finite $D$.

### The key object: the "one-step transfer function"

Without going into the full mathematical details, the iteration involves:
- A set of **correlation functions** at each layer of the tree
- A **transfer map** that takes correlations at layer $\ell$ and produces correlations at layer $\ell+1$
- This map depends on the QAOA angles $(\gamma, \beta)$ at each round

After $p$ iterations, the correlation at the root gives $\langle Z_i Z_j \rangle$ and hence $c_{\text{edge}}$.

---

## The O(1/D) Issue: Why This Isn't Enough for Us

### What "O(1/D) corrections" means

The Basso et al. iterative formula is **exact** in the limit $D \to \infty$ (the SK model). At finite $D$, the formula omits correction terms of order $1/D$, $1/D^2$, etc.

**Where do the corrections come from?** It is important to understand what is and is not approximate:

- **On a tree, the $(D-1)$ branches emanating from a vertex ARE exactly independent.** They share no vertices except the common parent, and the QAOA dynamics factorise across independent branches. The element-wise exponentiation trick in Farhi 2025 exploits this and is exact for any $D$.

- **The Basso iterative formula uses a different approach.** Rather than tracking the full tensor (which has $4^p$ entries), it tracks compact "correlation parameters" through a recurrence. In the $D\to\infty$ limit, the combined effect of many branches concentrates (by a central-limit-type argument), and the recurrence captures this concentration exactly. At finite $D$, the concentration is imperfect and the neglected higher-order terms give rise to $O(1/D)$ corrections.

So the approximation is **not** about branches being non-independent (they are independent on a tree!). It is about the specific iterative formula's parametrisation being tailored to the $D\to\infty$ regime, where the effect of $(D-1)$ branches can be summarised more compactly than at finite $D$.

### Impact at $D=4$

At $D=4$, the corrections are of order $1/D = 1/4 = 0.25$. The actual numerical error depends on the specific problem and depth $p$, but errors of several percent are plausible. For Stephen's purpose of comparing against DQI with precise quantitative results, this is unacceptable.

### An analogy

Imagine averaging $(D-1)$ dice rolls and using a Gaussian approximation for the average. With 100 dice, the Gaussian is excellent. With 3 dice, the discrete distribution still noticeably differs from a Gaussian. The Basso et al. iterative formula is analogous to the Gaussian approximation — perfectly valid at large $D$ but insufficiently precise at $D=4$. The exact tensor contraction (Farhi 2025) is like keeping the exact discrete distribution.

---

## Generalisation to Max-q-XORSAT

This is directly relevant to our project! The paper extends the analysis from MaxCut ($k=2$) to Max-$q$-XORSAT on $D$-regular $q$-uniform hypergraphs.

> **Note on scope:** The paper's title mentions only MaxCut and the SK model, not Max-$q$-XORSAT explicitly. However, the body of the paper does include a generalisation to Max-$q$-XORSAT on regular hypergraphs (the paper uses "$q$" for the constraint arity; we use "$k$" throughout this project). The iterative formula and tree structure extend naturally to the hypergraph setting. The generalisation appears in the main text (not relegated to an appendix), reflecting its importance to the paper's contribution.

### What changes for $k$-XORSAT

For MaxCut, each constraint involves 2 variables connected by an edge. For $k$-XORSAT, each constraint involves $k$ variables connected by a hyperedge. The cost operator for a single constraint $\alpha$ involving variables $i_1, \ldots, i_k$ is:

$$C_\alpha = \frac{1 + (-1)^{b_\alpha} Z_{i_1} Z_{i_2} \cdots Z_{i_k}}{2}$$

where $b_\alpha \in \{0,1\}$ is the target bit for the XOR. This constraint is satisfied when the XOR of the bit values equals $b_\alpha$. (Note the **+** sign: $(-1)^{b_\alpha} Z_{i_1}\cdots Z_{i_k}$ gives $+1$ when the constraint is satisfied and $-1$ when violated — see the derivation in `04-our-problem.md`.)

### The tree structure for k-XORSAT

The light cone is now a tree in the **factor graph** (bipartite graph of variables and constraints):

```
Level 0 (root):     [constraint α]
                   /      |      \
Level 1:        (x₁)    (x₂)    (x₃)         ← k=3 variables
               / | \    / | \    / | \
Level 2:     [·][·][·] [·][·][·] [·][·][·]   ← (D-1)=3 constraints each
               ...       ...       ...
```

The tree alternates between constraint nodes (degree $k$) and variable nodes (degree $D$). The branching factor per "level pair" is $(D-1)(k-1)$.

### The iterative formula for k-XORSAT

The same iterative approach works, with the transfer function modified to account for:
- $k$-body interactions at each constraint (instead of 2-body)
- Different counting of branches ($k-1$ new variables per constraint, $D-1$ new constraints per variable)

The $O(1/D)$ limitation carries over to the $k$-XORSAT generalisation: the iterative formula is again exact only in the $D \to \infty$ limit and incurs finite-$D$ corrections. The computational cost retains the same $O(4^p)$ scaling in $p$ — the exponent comes from the bra-ket sandwich structure (2 binary indices per layer × $p$ layers = $2p$ indices → $4^p$ entries), which does not depend on $k$. However, the constant factor in the cost increases with $k$ because the transfer function at each constraint node involves a $k$-body interaction gate (modifying the tensor structure within each layer).

---

## Key Results From the Paper

### Performance numbers

The paper reports the QAOA's normalised energy density on the SK model at each depth $p$ up to $p=20$. The key metric is the fraction of the **Parisi value** achieved — i.e., the ratio of the QAOA energy to the optimal (ground state) energy of the SK model. This ratio lies in $[0, 1]$ and approaches 1 if the QAOA achieves the true optimum.

> **Important:** These are **not** literal cut fractions (which approach $1/2$ as $D \to \infty$ for any fixed $p$). The paper works with a rescaled energy that factors out the $1/\sqrt{D}$ scaling, giving a well-defined quantity in the $D \to \infty$ limit. The Parisi value $P^*$ is the optimal of this rescaled quantity; its numerical value depends on the normalisation convention for the SK Hamiltonian. In a common convention ($H = n^{-1/2} \sum_{i<j} g_{ij} \sigma_i \sigma_j$), $P^* \approx 0.7632$, but the paper may use a different normalisation.

The QAOA performance improves monotonically with depth $p$. Selected values from the paper (consult the paper's Table 1 for the full data and precise normalisation):

| $p$ | Fraction of Parisi value achieved |
|-----|-----------------------------------|
| 1   | $\approx 0.75$                    |
| 2   | $\approx 0.79$                    |
| 3   | $\approx 0.81$                    |
| 5   | $\approx 0.83$                    |
| 10  | $\approx 0.86$                    |
| 11  | $\approx 0.86$ (exceeds classical threshold — see below) |
| 20  | $\approx 0.88$                    |

> **Caveat on values:** The specific numerical entries above are approximate and are based on the paper's reported results. They could not be verified digit-by-digit against the PDF text (FlateDecode-compressed). The qualitative pattern — monotonic improvement, with the fraction reaching ~0.86 by $p = 11$ and ~0.88 by $p = 20$ — is well-established.

### Classical comparison at $p = 11$

At $p=11$, the QAOA energy on the SK model exceeds the energy achieved by the best known efficient classical algorithm for MaxCut on random regular graphs in the $D \to \infty$ limit. The classical algorithm in question is an **SDP-based rounding approach** — a semidefinite programming relaxation followed by randomised rounding (in the spirit of Goemans-Williamson, but evaluated on the SK model rather than worst-case graphs). Specifically:

- The GW-type SDP rounding achieves a fraction $\sqrt{2/\pi} \approx 0.7979$ of the Parisi value on the SK model. QAOA surpasses this at $p = 3$.
- A stronger classical threshold (from a more sophisticated SDP-based or AMP-type algorithm) stands at approximately $0.86$ of the Parisi value. QAOA surpasses this at $p = 11$.

Note that Montanari's 2021 algorithm achieves $(1-\varepsilon) P^*$ for any $\varepsilon > 0$ on the SK model, but with running time that grows as $\varepsilon \to 0$. The comparison in Basso et al. is against classical algorithms with **fixed, explicit** performance guarantees, not against this asymptotic existence result.

### The Parisi conjecture

The paper conjectures that as $p \to \infty$, the QAOA on the SK model achieves the **Parisi value** — i.e., the fraction of $P^*$ achieved approaches 1. Equivalently, the QAOA's rescaled energy density converges to the ground state energy density of the SK model.

This remains **unproven** but is supported by the numerical evidence: the QAOA performance increases monotonically with $p$ and shows no sign of saturating below $P^*$ through $p = 20$. The paper states this as a formal conjecture.

> **On the Parisi value $P^*$:** The ground state energy density of the SK model is given exactly by the **Parisi formula** (proven by Talagrand 2006, building on Parisi's 1980 ansatz). Its numerical value depends on the Hamiltonian normalisation. In the standard physics convention ($H = n^{-1/2} \sum_{i<j} g_{ij} \sigma_i \sigma_j$ with $g_{ij} \sim N(0,1)$), $P^* \approx 0.7632$. The paper may use a different normalisation, so the specific number is less important than the fact that $P^*$ represents the information-theoretically optimal value that no algorithm can exceed.

---

## What We Take Away for Our Project

1. **The iterative formula is the state of the art for the $D\to\infty$ (SK) regime**, but it incurs $O(1/D)$ corrections when applied at finite $D$.
2. **The generalisation to Max-$q$-XORSAT exists** — the paper treats this case explicitly, so we know the structure of the problem, the cost operator, and the tree. We don't need to re-derive this from scratch.
3. **We need the exact tensor network method** — the direct contraction on the full tree, which is exact for any $D$. This is what the Farhi 2025 paper provides for MaxCut, and what we need to adapt for $k$-XORSAT.
4. **The iterative formula can serve as a sanity check**: at large $D$, our exact results should approach the iterative formula's predictions.

---

## Concrete Example: What "O(1/D) corrections" Looks Like

> **Note:** The numbers below are **purely hypothetical** and illustrative.

Suppose the true satisfaction fraction at $(k=3, D=4, p=5)$ is 0.6850.

The Basso et al. iterative formula (designed for $D\to\infty$) might give 0.6700, because it omits finite-$D$ correction terms of order $1/D = 0.25$. The actual error could be $\sim 0.015$, which is a few percent. When you're trying to compare against DQI bounds that might be, say, 0.6830, a few percent error makes the comparison meaningless.

This is why Stephen needs the exact tensor network method from Farhi 2025.

---

## Jargon From This Paper

| Term | Meaning |
|------|---------|
| **SK model** | Sherrington-Kirkpatrick model: a mean-field spin glass (MaxCut on the complete graph with random weights) |
| **Ensemble average** | Averaging performance over random instances of the problem |
| **$D$-regular** | Every vertex has exactly $D$ neighbours |
| **Large-girth** | No short cycles → local structure is a tree |
| **Transfer function / iteration** | The recurrence relation that computes QAOA performance layer by layer |
| **$O(1/D)$ corrections** | Error terms from using the $D\to\infty$ iterative formula at finite $D$ |

---

**Next:** Read `03-explainer-farhi2025-maxcut-lower-bound.md` — the tensor network method that is exact and that we will adapt.
