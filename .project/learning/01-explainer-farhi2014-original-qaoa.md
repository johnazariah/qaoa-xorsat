# Paper Explainer: "A Quantum Approximate Optimization Algorithm"

> **Paper:** Farhi, Goldstone, Gutmann (2014). arXiv:1411.4028  
> **PDF:** `../papers/farhi2014-original-qaoa.pdf`  
> **Read this after:** `00-foundations.md`

---

## Why This Paper Matters

This is the paper that launched an entire field. Edward Farhi, Jeffrey Goldstone, and Sam Gutmann introduced the Quantum Approximate Optimization Algorithm (QAOA) in 2014. It's one of the most cited papers in quantum computing.

For our project, this paper is important because:
1. It defines the QAOA framework that everything else builds on
2. It introduces the **direct calculation method** on tree-like neighbourhoods — the very approach we want to use
3. Its concrete results (e.g., the $p=1$ analysis on 3-regular MaxCut yielding 0.6924) are benchmarks we can validate against

---

## The Big Idea in Plain English

Imagine you have a hard optimisation problem — say, you want to split people at a party into two groups so that the maximum number of friends are in different groups (this is MaxCut on a social network graph).

Classical computers can approximate this, but there are limits. Farhi et al. ask: **can a quantum computer do better?**

They propose a specific quantum algorithm — QAOA — with a "depth" parameter $p$. As you increase $p$:
- The quantum circuit gets deeper (more layers of gates)
- The quality of the solution improves
- But the cost of both running the quantum circuit and classically analysing it increases

The remarkable result: even at $p=1$ (the shallowest possible circuit), QAOA on 3-regular graphs guarantees finding a cut that is at least **69.24%** of the optimal — strictly better than a random guess (50%) and competitive with some classical algorithms.

---

## The QAOA Circuit: Step by Step

### Starting state

Begin with $n$ qubits (one per vertex of the graph), all in the uniform superposition:

$$|s\rangle = |+\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{x \in \{0,1\}^n} |x\rangle$$

This means the quantum computer is simultaneously considering ALL $2^n$ possible partitions of the graph.

### The two unitaries

QAOA alternates between two operations:

**1. The "problem" unitary $U(C, \gamma) = e^{-i\gamma C}$:**

$C$ is the cost function (e.g., MaxCut). This operator applies a phase to each computational basis state proportional to its cost:

$$e^{-i\gamma C}|x\rangle = e^{-i\gamma \cdot C(x)}|x\rangle$$

For MaxCut, $C = \sum_{(j,k) \in E} \frac{1-Z_jZ_k}{2}$, so:

$$U(C,\gamma) = \prod_{(j,k) \in E} e^{-i\gamma(1-Z_jZ_k)/2}$$

Each edge contributes a two-qubit gate. If qubits $j$ and $k$ are different (edge is cut), the phase is $e^{-i\gamma/2} \cdot e^{+i\gamma/2} = 1$... Actually let me be careful. The gate $e^{-i\gamma Z_j Z_k / 2}$ applies:
- Phase $e^{-i\gamma/2}$ when $Z_jZ_k = +1$ (same bit values → edge NOT cut)
- Phase $e^{+i\gamma/2}$ when $Z_jZ_k = -1$ (different bit values → edge IS cut)

**Physical intuition:** This unitary "encodes the problem" into the phases of the quantum state. Good solutions (high cut) get different phases from bad solutions.

**2. The "mixer" unitary $U(B, \beta) = e^{-i\beta B}$ where $B = \sum_j X_j$:**

$$U(B,\beta) = \prod_j e^{-i\beta X_j}$$

Each single-qubit gate rotates qubit $j$ around the X-axis by angle $2\beta$:

$$e^{-i\beta X} = \begin{pmatrix} \cos\beta & -i\sin\beta \\ -i\sin\beta & \cos\beta \end{pmatrix}$$

**Physical intuition:** The mixer "explores" — it allows amplitude to flow between different bit strings, mixing good and bad solutions. Without it, the problem unitary alone would just change phases (and measurement probabilities wouldn't change).

### Full QAOA state at depth $p$

The QAOA applies $p$ alternating rounds:

$$|\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle = U(B,\beta_p) U(C,\gamma_p) \cdots U(B,\beta_1) U(C,\gamma_1) |s\rangle$$

The $2p$ angles $(\gamma_1, \ldots, \gamma_p, \beta_1, \ldots, \beta_p)$ are free parameters that we optimise to maximise the expected cost:

$$F_p(\boldsymbol{\gamma}, \boldsymbol{\beta}) = \langle\boldsymbol{\gamma}, \boldsymbol{\beta}| C |\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle$$

---

## The Light-Cone Insight (The Key Idea for Our Project)

Here's the crucial observation that makes everything tractable:

Consider one edge $(u,v)$ of the graph. Its contribution to the total cost is:

$$c_{uv} = \frac{1 - \langle\boldsymbol{\gamma}, \boldsymbol{\beta}| Z_u Z_v |\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle}{2}$$

Now, to compute $\langle Z_u Z_v \rangle$, we need to propagate the operator $Z_u Z_v$ backward through the circuit (Heisenberg picture). After propagating through one round:
- $U(B,\beta)$ only mixes each qubit locally → $Z_u$ becomes a function of qubit $u$ only
- $U(C,\gamma)$ entangles qubit $u$ with its **neighbours** in the graph → the operator now involves qubits within distance 1 of $u$

After $p$ rounds, the operator $Z_u Z_v$ has been "spread out" to involve all qubits within distance $p$ of edge $(u,v)$. This region is called the **light cone** of the operator.

**On a graph with large girth ($g \geq 2p+2$):** The light cone is a **tree** — there are no cycles in the neighbourhood. This means:
1. All edges in a regular graph have **identical** tree neighbourhoods → they all contribute the same $c_{\text{edge}}$
2. The total cost is just $|E| \cdot c_{\text{edge}}$ 
3. We only need to compute $c_{\text{edge}}$ for ONE tree neighbourhood

This is why the problem reduces from "simulate a quantum circuit on $n$ qubits" (exponentially hard) to "compute an expectation value on a tree with $O(D^p)$ qubits" (still exponential in $p$, but independent of the graph size $n$).

---

## The p=1 Calculation on 3-Regular Graphs

The paper works out the $p=1$ case in detail. Here's the structure:

For a 3-regular graph at $p=1$, the light cone of edge $(u,v)$ is:

```
    w₁  w₂      w₃  w₄
     \  /        \  /
      u --------- v
```

Vertices $u$ and $v$ are the endpoints of the central edge, and each has 2 additional neighbours ($w_1, w_2$ for $u$; $w_3, w_4$ for $v$). Total: 6 qubits.

The calculation proceeds:
1. Start all 6 qubits in $|+\rangle$
2. Apply $U(C, \gamma_1)$: phase gates on every edge in the tree
3. Apply $U(B, \beta_1)$: X-rotation on every qubit
4. Compute $\langle Z_u Z_v \rangle$

Since this involves only 6 qubits ($2^6 = 64$ amplitudes), it's easily computed analytically. The result is a trigonometric expression in $\gamma_1$ and $\beta_1$.

Optimising over $\gamma_1$ and $\beta_1$ yields the optimal per-edge cut fraction:

$$\tilde{c}_{\text{edge}}(p=1) \approx 0.6924 \text{ (for 3-regular)}$$

This means QAOA at $p=1$ cuts approximately **69.24%** of edges on a large-girth 3-regular graph. This is also the **approximation ratio guarantee**: since for any 3-regular graph we have $\text{MaxCut} \leq |E|$, the QAOA achieves at least a fraction $0.6924$ of the maximum cut. The bound is tight for bipartite 3-regular graphs (where $\text{MaxCut} = |E|$), while for non-bipartite graphs (where $\text{MaxCut} < |E|$) the actual approximation ratio is even better.

For comparison, the paper also analyses a warm-up problem called the **"Ring of Disagrees"** (Section IV) — MaxCut on a cycle (2-regular graph) — where the $p=1$ optimal cut fraction is exactly $3/4 = 0.75$. This 3/4 value applies to cycles, not to 3-regular graphs.

---

## How This Connects to Higher $p$

At $p=1$: the tree has 6 qubits (for 3-regular) → trivial computation.

At $p=2$: the tree grows — each leaf of the $p=1$ tree sprouts 2 more neighbours → about 14 qubits → still easy ($2^{14}$ amplitudes).

At $p=3$: about 30 qubits → $2^{30} \approx 10^9$ amplitudes → feasible but large.

At $p=10$: about $2^{12} - 2 = 4094$ qubits → $2^{4094}$ amplitudes → completely impossible by brute force!

**But we don't need brute force.** The tree structure allows us to contract from leaves to root, which is the subject of the later papers. The original paper does direct computation at small $p$, and the later papers develop efficient contraction methods.

---

## What the Paper DOESN'T Do (and Why We Need the Later Papers)

1. **Doesn't go to large $p$:** The direct computation in this paper is limited to small $p$ on small trees.
2. **Doesn't address k-XORSAT:** Only considers MaxCut ($k=2$).
3. **Doesn't give a practical code implementation:** The calculations are done analytically or with small numerical computations.
4. **Doesn't consider specific small $(k,D)$:** Focuses on 3-regular graphs (so $D=3$, $k=2$).

The Basso et al. 2021 paper addresses (1), (2), and partially (4) but introduces $O(1/D)$ approximations. The Farhi et al. 2025 paper addresses (1) and (3) with exact tensor network methods, but only for MaxCut. We need to combine these approaches for our specific target of Max-3-XORSAT on 4-regular hypergraphs.

---

## Key Equations to Remember

| Equation | Meaning |
|----------|---------|
| $\|s\rangle = \|+\rangle^{\otimes n}$ | Starting state: uniform superposition |
| $\|\boldsymbol{\gamma},\boldsymbol{\beta}\rangle = \prod_{l=1}^{p} U(B,\beta_l) U(C,\gamma_l) \|s\rangle$ | QAOA state at depth $p$ |
| $F_p = \langle\boldsymbol{\gamma},\boldsymbol{\beta}\| C \|\boldsymbol{\gamma},\boldsymbol{\beta}\rangle$ | Expected cost (what we maximise) |
| $c_{\text{edge}} = \frac{1-\langle Z_iZ_j\rangle}{2}$ | Per-edge cost contribution for MaxCut |
| $M_g \geq \tilde{c}_{\text{edge}}(p)$ for $g \geq 2p+2$ | QAOA provides lower bound on worst-case cut fraction |

---

## Jargon Glossary

| Term | Meaning |
|------|---------|
| **QAOA depth $p$** | Number of alternating (problem, mixer) rounds in the circuit |
| **Angles $(\boldsymbol{\gamma}, \boldsymbol{\beta})$** | The $2p$ free parameters we optimise |
| **Light cone** | The set of qubits that influence a particular measurement outcome after $p$ rounds |
| **Cut fraction** | Fraction of edges crossing the partition (absolute performance) |
| **Approximation ratio** | Fraction of the optimal cut achieved (relative performance) |
| **Large girth** | Graph has no short cycles, so local neighbourhoods are trees |

---

**Next:** Read `02-explainer-basso2021-high-depth.md` to see how the analysis scales to $p=20$.
