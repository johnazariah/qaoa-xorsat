# Paper Explainer: "The QAOA at High Depth for MaxCut on Large-Girth Regular Graphs and the Sherrington-Kirkpatrick Model"

> **Paper:** Basso, Farhi, Marwaha, Villalonga, Zhou (2021). arXiv:2110.14206  
> **PDF:** `../papers/basso2021-qaoa-high-depth.pdf`  
> **Read this after:** `01-explainer-farhi2014-original-qaoa.md`

---

## Why This Paper Matters

This is the paper Stephen specifically references as "partially addressing" our question — but with limitations. It develops a clever iterative formula that dramatically reduces the cost of computing QAOA performance, allowing analysis up to $p=20$. But its method is **accurate only up to $O(1/D)$ corrections**, which makes it unsuitable for precise results at small $D$ (like our target $D=4$).

Understanding **what** this paper does and **why** its approximation breaks down is essential for understanding why we need the more expensive direct method.

---

## The Problem They Solve

The paper asks: what is the expected fraction of satisfied constraints when QAOA at depth $p$ is applied to:
- MaxCut on $D$-regular graphs (generalising from $D=3$ to arbitrary $D$)
- Max-$q$-XORSAT on $D$-regular $q$-uniform hypergraphs (our target problem!)
- The Sherrington-Kirkpatrick (SK) model (a fully-connected spin glass)

They push the analysis to $p=20$, far beyond what was previously possible. At $p=11$, they show QAOA beats all known rigorous classical guarantees on random $D$-regular graphs as $D \to \infty$.

---

## The Iterative Formula: The Core Contribution

### The idea

Recall from the first paper: on a large-girth $D$-regular graph, the QAOA expectation for a single edge decomposes into a computation on a tree. The tree at depth $p$ has $O(D^p)$ nodes, and the direct computation costs something like $O(4^p)$ in time and space.

Basso et al. observe that by working in the **large-$D$ limit**, significant simplifications arise. Instead of computing on the full tree, they develop an **iterative formula** that builds up the answer layer by layer.

### How it works (conceptual)

Think of the tree growing outward from the central edge. At each "generation" (layer of the tree), you need to:
1. Compute how the QAOA unitary transforms the state at that layer
2. Propagate the result inward toward the root

In the naive method, each tree branch doubles the state space. But Basso et al. note that in the limit of large $D$:
- The contribution of each branch becomes approximately independent
- The combined effect of $(D-1)$ identical branches can be approximated by a function of a single branch's contribution raised to the $(D-1)$th power

This yields a recurrence relation. At each step, you:
1. Compute a compact representation of the "state" at generation $\ell$
2. Use the recurrence to get the representation at generation $\ell+1$
3. After $p$ iterations, read off the expectation value

The computational cost is $O(p^2 \cdot 4^p)$ — still exponential in $p$, but with a much smaller prefactor than the direct method, and crucially the constant factor doesn't grow with $D$.

### The key object: the "one-step transfer function"

Without going into the full mathematical details, the iteration involves:
- A set of **correlation functions** at each layer of the tree
- A **transfer matrix** or map that takes correlations at layer $\ell$ and produces correlations at layer $\ell+1$
- This map depends on the QAOA angles $(\gamma, \beta)$ at each round

After $p$ iterations, the correlation at the root gives $\langle Z_i Z_j \rangle$ and hence $c_{\text{edge}}$.

---

## The O(1/D) Issue: Why This Isn't Enough for Us

### What "O(1/D) corrections" means

The iterative formula is **exact** in the limit $D \to \infty$. At finite $D$, there are correction terms of order $1/D$, $1/D^2$, etc., that the formula neglects.

Concretely, the formula assumes that the $(D-1)$ branches emanating from a vertex are **statistically independent** of each other. In reality, they're not quite independent — they share the common vertex, and the QAOA dynamics through that vertex create correlations between branches.

These correlations are suppressed by factors of $1/(D-1)$ because each branch contributes one of $(D-1)$ terms. So at large $D$ (say $D=100$), the error is tiny. But at $D=4$:

$$\frac{1}{D-1} = \frac{1}{3} \approx 33\%$$

A 33% relative error in the corrections means the iterative formula could be off by a significant amount at $D=4$. For Stephen's purpose of comparing against DQI with precise quantitative results, this is unacceptable.

### An analogy

Imagine estimating the temperature of a room by averaging independent thermometer readings. If you have 100 thermometers, small correlations between them barely affect your estimate. But if you have only 4, correlations between readings could significantly bias your average. The Basso et al. formula is like assuming the thermometers are independent — fine for 100 of them, problematic for 4.

---

## Generalisation to Max-q-XORSAT

This is directly relevant to our project! The paper extends the analysis from MaxCut ($k=2$) to Max-$q$-XORSAT on $D$-regular $q$-uniform hypergraphs.

### What changes for $k$-XORSAT

For MaxCut, each constraint involves 2 variables connected by an edge. For $k$-XORSAT, each constraint involves $k$ variables connected by a hyperedge. The cost operator for a single constraint $\alpha$ involving variables $i_1, \ldots, i_k$ is:

$$C_\alpha = \frac{1 - (-1)^{b_\alpha} Z_{i_1} Z_{i_2} \cdots Z_{i_k}}{2}$$

where $b_\alpha \in \{0,1\}$ is the target bit for the XOR. This constraint is satisfied when the XOR of the bit values equals $b_\alpha$.

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

The cost remains $O(p^2 \cdot 4^p)$, and the $O(1/D)$ limitation remains.

---

## Key Results From the Paper

### Performance numbers

At optimal parameters, the QAOA achieves these cut fractions on $D$-regular MaxCut as $D \to \infty$:

| $p$ | Cut fraction (large $D$) |
|-----|-------------------------|
| 1   | 0.7500                  |
| 2   | 0.7925                  |
| 3   | 0.8132                  |
| 5   | 0.8347                  |
| 10  | 0.8572                  |
| 11  | 0.8594                  |
| 20  | 0.8756                  |

At $p=11$, the QAOA beats the best known classical algorithms (without unproven conjectures) in the large-$D$ limit.

### The Parisi conjecture

They conjecture that as $p \to \infty$, the QAOA achieves the Parisi value of the SK model (~0.7632 for the SK energy, corresponding to a graph cut fraction that approaches the information-theoretic limit). This remains unproven but the numerical evidence is suggestive.

---

## What We Take Away for Our Project

1. **The iterative formula is the state of the art for large $p$**, but it's approximate at small $D$.
2. **The generalisation to Max-$q$-XORSAT exists** — we know the structure of the problem, the cost operator, and the tree. We don't need to re-derive this from scratch.
3. **We need an exact method** — the direct computation on the full tree, without the $O(1/D)$ approximation. This is what the Farhi 2025 paper provides for MaxCut, and what we need to adapt for $k$-XORSAT.
4. **The iterative formula can serve as a sanity check**: at large $D$, our exact results should approach the iterative formula's predictions.

---

## Concrete Example: What "O(1/D) corrections" Looks Like

Suppose the true cut fraction at $(k=3, D=4, p=5)$ is 0.6850 (hypothetical).

The Basso et al. formula might give 0.6700, because it neglects terms of order $1/D = 1/4 = 0.25$. The actual error could be $\sim 0.6850 - 0.6700 = 0.015$, which is a few percent. When you're trying to compare against DQI bounds that might be, say, 0.6830, a few percent error makes the comparison meaningless.

This is why Stephen needs the exact method.

---

## Jargon From This Paper

| Term | Meaning |
|------|---------|
| **SK model** | Sherrington-Kirkpatrick model: a mean-field spin glass (MaxCut on the complete graph with random weights) |
| **Ensemble average** | Averaging performance over random instances of the problem |
| **$D$-regular** | Every vertex has exactly $D$ neighbours |
| **Large-girth** | No short cycles → local structure is a tree |
| **Transfer function / iteration** | The recurrence relation that computes QAOA performance layer by layer |
| **$O(1/D)$ corrections** | Error terms from the independence approximation, proportional to $1/D$ |

---

**Next:** Read `03-explainer-farhi2025-maxcut-lower-bound.md` — the tensor network method that is exact and that we will adapt.
