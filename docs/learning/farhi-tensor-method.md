# Paper Explainer: "Lower Bounding the MaxCut of High Girth 3-Regular Graphs Using the QAOA"

> **Paper:** Farhi, Gutmann, Ranard, Villalonga (2025). arXiv:2503.12789  
> **PDF:** `../papers/farhi2025-maxcut-lower-bound.pdf`  
> **Read this after:** `02-explainer-basso2021-high-depth.md`

---

## Why This Is the Most Important Paper for Our Project

This paper contains **the exact method we want to adapt**. While it applies the method to MaxCut on 3-regular graphs (i.e., $k=2, D=3$), the approach generalises to $k$-XORSAT on $D$-regular hypergraphs. Stephen's email specifically points to this paper as the template for our computation.

Key facts:
- They compute **exact** QAOA performance, with no $O(1/D)$ approximations
- They push to $p=17$ on 3-regular MaxCut using clever tensor network contraction
- Their code is written in C++ with OpenMP parallelisation and the Eigen/LBFGS++ libraries
- The computational cost scales as $O(p \cdot 4^p)$ time, $O(4^p)$ space, **independent of $D$**

---

## What They Achieve

They compute $\tilde{c}_{\text{edge}}(p)$ — the optimal QAOA cut fraction — for MaxCut on 3-regular graphs of girth $g \geq 2p+2$, up to $p=17$. Selected results:

| $p$ | $\tilde{c}_{\text{edge}}(p)$ | Required girth $g$ |
|-----|------------------------------|---------------------|
| 1   | 0.6924                       | $\geq 4$            |
| 2   | 0.7559                       | $\geq 6$            |
| 3   | 0.7923                       | $\geq 8$            |
| 4   | 0.8168                       | $\geq 10$           |
| 5   | 0.8363                       | $\geq 12$           |
| 6   | 0.8498                       | $\geq 14$           |
| 7   | 0.8597                       | $\geq 16$           |
| 8   | 0.8673                       | $\geq 18$           |
| 9   | 0.8734                       | $\geq 20$           |
| 10  | 0.8784                       | $\geq 22$           |
| 11  | 0.8825                       | $\geq 24$           |
| 12  | 0.8859                       | $\geq 26$           |
| 13  | 0.8888                       | $\geq 28$           |
| 14  | 0.8913                       | $\geq 30$           |
| 15  | 0.8935                       | $\geq 32$           |
| 16  | 0.8954                       | $\geq 34$           |
| 17  | 0.8971                       | $\geq 36$           |

> **Verified (2026-03-21):** All values confirmed against Table 1 of the paper
> (extracted via `pdftotext`). The previous version had wrong values at p=1
> (was 0.7500, actually 0.6924) and other depths. Now includes the complete
> table for p=1 through p=17.

The result at $p \geq 7$ **improves on all previously known lower bounds** for $M_g$ (the worst-case max cut fraction on 3-regular graphs of girth $\geq g$) when $g \geq 16$.

The asymptotic target is $\lim_{g\to\infty} M_g \geq 0.912$ (from Refs. [5] and [6] in the paper). Their results approach but don't yet reach this — Figure 4 plots $\tilde{c}_{\text{edge}}$ vs $1/p$ and shows the values trending toward 0.912 from below. The upper bound on the expected cut fraction of large random 3-regular graphs is 0.9239 [16]. Whether QAOA at large $p$ exceeds 0.912 remains an open question.

---

## The Method: Tensor Network Contraction on Trees

### Step 1: Set up the tree

For MaxCut on a $D$-regular graph at QAOA depth $p$, the light cone of the central edge $(i,j)$ is a tree:

```
         ●      ●              ●      ●
          \    /                \    /
    ●──────●──(i)────────(j)──●──────●
          /    \                /    \
         ●      ●              ●      ●
         ...   ...             ...   ...
```

For $D=3$, each internal vertex has 2 child branches (since 1 edge goes to the parent). The tree has depth $p$ from each endpoint of the central edge.

Total qubits at $D=3$: $2 \cdot \frac{2^{p+1}-1}{2-1} = 2(2^{p+1}-1)$. At $p=17$: 524,286 qubits.

Obviously we can't store a $2^{524286}$-dimensional state vector. The key is that we never need to.

### Step 2: Set up the tensor network

The computation of $\langle \boldsymbol{\gamma}, \boldsymbol{\beta} | Z_i Z_j | \boldsymbol{\gamma}, \boldsymbol{\beta} \rangle$ can be expressed as a tensor network. Think of it as a "sandwich": the bra $\langle \boldsymbol{\gamma}, \boldsymbol{\beta}|$, the observable $Z_i Z_j$, and the ket $|\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle$.

Each qubit $q$ has gates acting on it through $p$ rounds of $(U(C,\gamma_l), U(B,\beta_l))$. These contribute tensors:

- **Initial state tensor** ($|+\rangle$): a vector $\frac{1}{\sqrt{2}}(1, 1)$
- **Problem gate tensor** ($e^{i\gamma Z_q Z_{q'}}$): a diagonal 2-index tensor with entries $e^{\pm i\gamma}$
- **Mixer gate tensor** ($e^{-i\beta X_q}$): a $2\times 2$ matrix $\begin{pmatrix}\cos\beta & -i\sin\beta \\ -i\sin\beta & \cos\beta\end{pmatrix}$
- **Observable tensor** ($Z_i Z_j$ at the central edge): a diagonal 2-index tensor  

The "sandwich" $\langle\psi|O|\psi\rangle$ doubles the network (one copy for bra, one for ket), but since the problem gates are diagonal, many indices can be merged using **hyperindices** — this is a key simplification.

### Step 3: The contraction trick — element-wise exponentiation

**This is the big insight.** On a $D$-regular tree, every branch at a given depth is identical. So instead of contracting every branch independently, you:

1. **Contract a single branch** from its leaves inward. This produces a tensor $T$ with some indices.
2. **Raise each entry of $T$ to the power $(D-1)$.** Since there are $(D-1)$ identical branches at each vertex, and they are independent, their contributions multiply element-wise:

$$(T^{D-1})_{a_1, a_2, \ldots} = (T_{a_1, a_2, \ldots})^{D-1}$$

3. **Continue contracting** to the next layer inward.
4. **Repeat** until you reach the central edge.

This is expressed in the paper's Eq. (14) and Fig. 5(b).

### Why this works

Consider a vertex $v$ in the tree with $(D-1)$ child branches (besides the edge going toward the root). After contracting one child branch, you get a tensor $T$ that describes how that branch affects vertex $v$. Because:
- All $(D-1)$ branches are structurally identical (regular tree)
- The branches are independent (no connections between them — it's a tree!)
- The combined effect is a product of their individual effects

...the combined effect is just $T$ raised to the $(D-1)$th power, entry by entry.

### Resulting complexity

At each step of the contraction, you work with tensors indexed by $2p$ binary indices (for the $2p$ layers of the "sandwich" — $p$ bra and $p$ ket). Each tensor therefore has $2^{2p} = 4^p$ entries.

The contraction from leaf to root takes $O(p)$ steps, with each step involving $O(4^p)$ operations. So:

$$\text{Total cost} = O(p \cdot 4^p) \text{ time, } O(4^p) \text{ space}$$

The paper quotes $O(2^{2p})$ which is the same as $O(4^p)$.

**Critical point: the cost is independent of $D$!** The degree $D$ only enters through the exponentiation step, which replaces $T$ by $T^{D-1}$ element-wise. This is just an $O(4^p)$ operation regardless of $D$.

### What this means numerically

| $p$ | $4^p$ | Feasibility |
|-----|-------|-------------|
| 5   | 1,024 | Trivial |
| 10  | ~$10^6$ | Easy |
| 15  | ~$10^9$ | Feasible (minutes) |
| 17  | ~$1.7 \times 10^{10}$ | Feasible (hours) |
| 20  | ~$10^{12}$ | Challenging (days on cluster) |
| 25  | ~$10^{15}$ | Very expensive |
| 30  | ~$10^{18}$ | Heroic (exascale) |

The paper pushes to $p=17$ for $D=3$. With good code on a modern cluster, $p=20$ or beyond should be reachable.

---

## The Optimisation Over Angles

At each $p$, we need to find the angles $(\boldsymbol{\gamma}, \boldsymbol{\beta})$ that maximise $c_{\text{edge}}(\boldsymbol{\gamma}, \boldsymbol{\beta})$. This is a $2p$-dimensional optimisation problem.

The paper uses **L-BFGS** (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) — a quasi-Newton gradient-based method. Key details:

- Each function evaluation costs $O(4^p)$
- Gradients can be computed by finite differences or (better) by automatic differentiation through the tensor contraction
- Multiple random restarts are used to avoid local minima
- The optimised parameters $\tilde{\boldsymbol{\gamma}}, \tilde{\boldsymbol{\beta}}$ show smooth curves when plotted vs. $j/p$ (see their Fig. 3), suggesting that extrapolation from smaller $p$ could provide good initial guesses

**Important:** Even if the optimisation doesn't find the global optimum perfectly, any $c_{\text{edge}}(\boldsymbol{\gamma}, \boldsymbol{\beta})$ is a **valid lower bound** on $M_g$. We're computing a lower bound, not an exact optimum. We just want it to be as tight as possible.

---

## Their Implementation

The paper describes a practical implementation:

| Component | Choice |
|-----------|--------|
| Language | **C++** |
| Parallelism | **OpenMP** (shared-memory parallelism) |
| Linear algebra | **Eigen** (C++ template library for vectors/matrices) |
| Optimisation | **LBFGS++** (C++ implementation of L-BFGS) |

This is a high-performance setup suitable for pushing to large $p$. For our project, we could follow the same approach or use alternatives (Julia + multithreading, Python + JAX, etc.).

---

## Adapting This Method for k-XORSAT

This is what we need to do. The key differences when moving from MaxCut ($k=2$) to Max-$k$-XORSAT:

### 1. Hyperedges replace edges

The "central object" is now a **hyperedge** connecting $k$ variable nodes, not an edge connecting 2. The root of the tree is a $k$-body factor.

### 2. Tree structure changes

For MaxCut on $D$-regular graphs:
- Tree rooted at an edge with 2 endpoints
- Each endpoint has $(D-1)$ other edges → branches

For $k$-XORSAT on $D$-regular $k$-uniform hypergraphs:
- Tree rooted at a hyperedge with $k$ variable nodes
- Each variable node has $(D-1)$ other hyperedges
- Each hyperedge has $(k-1)$ other variable nodes
- Branching factor per level pair: $(D-1)(k-1)$

For our target $(k=3, D=4)$: branching factor = $3 \times 2 = 6$ per level pair.

### 3. Problem unitary changes

For MaxCut: $U(C,\gamma) = \prod_{(i,j)} e^{-i\gamma Z_iZ_j/2}$

For $k$-XORSAT: $U(C,\gamma) = \prod_{\alpha} e^{-i\gamma Z_{i_1}Z_{i_2}\cdots Z_{i_k}/2}$

The gate is now a $k$-body diagonal gate instead of 2-body. This changes the tensor structure.

### 4. Tensor indices

For MaxCut, the problem gate tensor has 2 indices (one per endpoint of the edge). For $k$-XORSAT, it has $k$ indices (one per variable in the hyperedge). This makes the tensors larger.

The contraction cost should still be $O(4^p)$-ish, but with a larger constant factor due to the $k$-body interaction. The element-wise exponentiation trick still works because the tree is still a regular tree with identical branches.

### 5. What stays the same

- The mixer unitary is still single-qubit X rotations → unchanged
- The contraction is still from leaves to root → same algorithm
- The element-wise exponentiation trick still works → cost still independent of $D$
- The angle optimisation is still $2p$-dimensional L-BFGS → same approach

---

## Connection to Stephen's Request

Stephen's email says:

> "I would like to compare QAOA against DQI at specific small (k,D), particularly k=3, D=4."

and:

> "There is a different and more direct way to calculate the performance of QAOA, which essentially takes advantage of the limited light cone of local observables..."

He's pointing directly at this paper's method. He wants us to:
1. Take the tensor network contraction method from this paper
2. Generalise it from MaxCut (2-body) to 3-XORSAT (3-body)
3. Implement it for $(k=3, D=4)$  
4. Push $p$ as high as possible
5. Compare the results against his DQI numbers

The paper's C++ implementation shows it's feasible to push to $p=17$ for MaxCut. For 3-XORSAT, the larger tensors (3-body → $k=3$ indices instead of 2) may limit us to smaller $p$, but this needs to be carefully estimated.

---

## Key Technical Details to Understand Before Reading the Paper

### Tensor definitions (Section 5.2, Eq. 13)

The paper defines four basic tensors:

| Tensor | Description | Entries |
|--------|-------------|---------|
| Initial state | $\|+\rangle$ preparation | $\frac{1}{\sqrt{2}}$ for all inputs |
| Problem gate | $e^{i\gamma Z_iZ_j}$ diagonal gate | $e^{i\gamma}$ if $a=b$, $e^{-i\gamma}$ if $a \neq b$ |
| Mixer gate | $e^{-i\beta X}$ rotation | $\cos\beta$ if $a=b$, $-i\sin\beta$ if $a \neq b$ |
| Observable | $Z_iZ_j$ measurement | $+1$ if $a=b=0$, $-1$ if $a=b=1$, $0$ otherwise |

> **Convention note:** The paper defines the cost function as $C = \sum_{(j,k)\in E} \frac{1-Z_jZ_k}{2}$
> (Eq. 4) and the cost unitary as $U(C,\gamma) = e^{-i\gamma C}$, following the standard
> Farhi 2014 convention. The per-edge gate is therefore $e^{i\gamma Z_jZ_k/2}$ (up to a
> global phase from the constant $1/2$ term). The tensor entries $e^{\pm i\gamma}$ shown
> in the explainer use a rescaled $\gamma$ that absorbs the factor of 2 for notational
> convenience. Both parametrisations are equivalent.

### The "sandwich" structure

To compute $\langle \boldsymbol{\gamma},\boldsymbol{\beta}|Z_iZ_j|\boldsymbol{\gamma},\boldsymbol{\beta}\rangle$, you need both the ket (forward evolution) and bra (backward evolution). In the tensor network, each qubit carries **two** indices at each time step: one from the ket and one from the bra. This is why you have $2p$ indices and tensors of size $4^p$ rather than $2^p$.

### Hyperindices

Because the problem gates are **diagonal** ($Z_iZ_j$ or $Z_{i_1}\cdots Z_{i_k}$), the bra and ket indices can be combined into a single "hyperindex" at each time step. This halves the number of effective indices for problem gates, which is a significant optimisation.

---

## Reading Guide for the Paper

**Must-read sections:**
- Section 2 (Review of QAOA) — concise refresher, establishes notation
- Section 3 (Pre-computing parameters) — the light-cone decomposition and why girth matters
- Section 5.2 (Methods) — the tensor network contraction, element-wise exponentiation, and complexity analysis. **This is the technical core we need to adapt.**

**Good to understand:**
- Section 5.1 (Results) — the numerical values and optimised parameters
- Section 6 (MaxCut comparisons) — context for the state of the art

**Can skim:**
- Section 4 (Time-complexity of running QAOA on quantum hardware) — relevant if actually running on a quantum computer, not for our classical computation
- Section 7 (Maximum Independent Set) — interesting extension but not our focus
- Section 8 (Conclusions) — brief summary

---

**Next:** Read `04-our-problem.md` to see how all this connects to our specific task.
