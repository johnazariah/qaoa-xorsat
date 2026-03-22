# Deep Dive: Farhi et al. 2025 — Exact Tensor Contraction for MaxCut

> **Source**: arXiv:2503.12789, full text extracted to `papers/text/farhi2025-maxcut-lower-bound.txt`
> **Date**: 22 March 2026
> **Purpose**: Detailed findings from reading the full extracted paper text,
> cross-referenced against our explainer (file 03) and tensor derivation (file 05).

---

## 1. The Exact Contraction Algorithm (step by step)

**Setup.** For MaxCut on a $D$-regular graph of girth $g \geq 2p+2$, the light
cone of any edge $(i,j)$ is a tree. Every edge has an isomorphic tree
neighbourhood, so the expected cut contribution per edge is:

$$c_{\text{edge}}(\gamma,\beta) = \frac{1 - \langle \gamma,\beta | Z_i Z_j | \gamma,\beta \rangle}{2}$$

**Hyperindex representation.** Each qubit carries a $2p$-bit "hyperindex"
$\sigma \in \{0,1\}^{2p}$ encoding both ket and bra indices across $p$ rounds.
The problem gate $e^{i\gamma Z_i Z_j}$ is diagonal, so ket and bra indices merge
into a single hyperindex, reducing the tensor from $2^{2p} \times 2^{2p}$ to
$2^{2p} = 4^p$ entries.

**Tensor primitives** (Eq. 13):

| Tensor | Role | Entries ($a,b \in \{0,1\}$) |
|--------|------|-------------------------------------|
| Initial state ($\mid+\rangle$) | Leaf boundary | $1/\sqrt{2}$ for all $a$ |
| Problem gate ($e^{i\gamma Z_iZ_j}$) | Edge, diagonal | $e^{i\gamma}$ if $a=b$; $e^{-i\gamma}$ if $a \neq b$ |
| Mixer gate ($e^{-i\beta X}$) | Per-qubit, 2×2 | $\cos\beta$ if $a=b$; $-i\sin\beta$ if $a \neq b$ |
| Observable ($Z_iZ_j$) | Root edge | $+1$ if $a{=}b{=}0$; $-1$ if $a{=}b{=}1$; $0$ otherwise |

> **Convention note:** The problem gate uses a rescaled $\gamma$ where entries are
> $e^{\pm i\gamma}$, not $e^{\pm i\gamma/2}$. The per-edge unitary is
> $e^{i\gamma Z_jZ_k/2}$; the tensor notation absorbs the factor of 2.

**Contraction algorithm** (Fig. 5(b)):

1. **Start at leaves.** Each leaf contributes the initial state tensor. In the
   bra-ket hyperindex picture → $4^p$-entry vector.

2. **Contract one branch inward.** Contract leaf tensor with problem gate and
   mixer tensors → produces "branch tensor" $T$ ($4^p$ entries, indexed by
   parent qubit's hyperindex).

3. **Element-wise exponentiation** (Eq. 14). At each vertex with $(d-1)$
   identical child branches: $T \leftarrow T^{(d-1)}$ entry-by-entry. Valid
   because branches are identical and independent (tree, no loops).

4. **Repeat** inward through each level: contract, exponentiate, contract, ...

5. **Root edge.** Two branch tensors (from the two endpoints) contracted with
   problem gate tensors (all $p$ rounds) and observable tensor → scalar
   $\langle Z_iZ_j \rangle$.

6. **Result:** $c_{\text{edge}} = (1 - \langle Z_iZ_j \rangle) / 2$.

**Complexity:**
- Time: $O(4^p)$ (paper's statement; $O(p \cdot 4^p)$ accounting for $p$ levels)
- Space: $O(4^p)$
- **Independent of $d$** — degree enters only as the exponent in element-wise power

The paper states this is "quadratically better" than Basso et al. 2022 and
Wybo-Leib 2024 (which cost $O(16^p)$), and independent of $d$ unlike
Wurtz-Lykov 2021 (which scales exponentially in $d$).

---

## 2. Table 1 — All Reported Cut Fractions

Verified character-by-character against extracted text:

| $p$ | $\tilde{c}_{\text{edge}}(p)$ | Girth $g \geq 2p+2$ |
|-----|------|------|
| 1 | 0.6924 | 4 |
| 2 | 0.7559 | 6 |
| 3 | 0.7923 | 8 |
| 4 | 0.8168 | 10 |
| 5 | 0.8363 | 12 |
| 6 | 0.8498 | 14 |
| 7 | 0.8597 | 16 |
| 8 | 0.8673 | 18 |
| 9 | 0.8734 | 20 |
| 10 | 0.8784 | 22 |
| 11 | 0.8825 | 24 |
| 12 | 0.8859 | 26 |
| 13 | 0.8888 | 28 |
| 14 | 0.8913 | 30 |
| 15 | 0.8935 | 32 |
| 16 | 0.8954 | 34 |
| 17 | 0.8971 | 36 |

Benchmarks:
- $p \geq 7$: exceeds all previously known lower bounds on $M_g$
- $p \geq 15$: exceeds Thompson-Parekh-Marwaha $g \to \infty$ bound of 0.8918
- Asymptotic target: $\lim_{g\to\infty} M_g \geq 0.912$ (not yet reached)
- Upper bound for large random 3-regular: 0.9239

---

## 3. Gaps in Our Explainer (File 03)

The explainer is accurate overall. Gaps identified:

**(a) "Quadratically better" comparison not mentioned.** Previous methods cost
$O(16^p)$; Farhi's costs $O(4^p)$. Should be noted.

**(b) Gradient method unspecified.** The explainer says "(better) by automatic
differentiation" — the paper doesn't specify gradient computation method. Only
mentions L-BFGS via LBFGS++ library. The "(better)" is editorial speculation.

**(c) Root edge handling vague.** The explainer should spell out the final
contraction: two branch tensors + problem gates + observable → scalar.

**(d) Observable sign convention.** Paper computes $\langle Z_iZ_j\rangle$ then
$(1-\langle Z_iZ_j\rangle)/2$ for MaxCut. Our code uses $(1 + Z_1 Z_2)/2$ for
XORSAT. These differ by sign — XORSAT satisfaction at $k=2$ = "edge NOT cut".
Correct but should be documented explicitly.

**(e) Quantum alternative remark.** Paper notes (§5.2): a quantum computer with
$2^{p+2}-2$ qubits could evaluate the contraction, scaling more favourably than
classical at $d \leq 4$. Not mentioned in explainer. Interesting context.

---

## 4. What Changes for k=3

**Tree structure:** Bipartite (alternating variable/constraint levels) instead of
homogeneous. Branching factors alternate: $(D-1)$ at variable levels, $(k-1)$ at
constraint levels.

**Problem gate:** 3-body diagonal ($Z_1 Z_2 Z_3$) instead of 2-body. Per-round
tensor has $4^3 = 64$ entries instead of $4^2 = 16$.

**Constraint node contraction — the key difficulty:** At $k=2$, each constraint
has 1 child variable (trivial). At $k=3$, each constraint has 2 children coupled
through the 3-body gate. Must sum over all $(4^p)^2$ child hyperindex pairs.
Naive cost: $O(4^{3p})$ per constraint node. Whether this can be reduced by
exploiting the diagonal structure or parity factorisation is an **open question**
(P1.3 spec Open Question 1).

**Mixer:** Unchanged (single-qubit $e^{-i\beta X}$).

**Observable:** Becomes 3-qubit diagonal: $(1 + Z_1 Z_2 Z_3)/2$.

**Angle optimisation:** Same framework (L-BFGS with restarts). Cost per eval
increases due to constraint contraction.

**Girth requirement:** Same — girth $\geq 2p+2$ in the factor graph.

---

## 5. Implementation Details

| Component | Choice |
|-----------|--------|
| Language | C++ |
| Parallelism | OpenMP (shared-memory) |
| Linear algebra | Eigen |
| Optimisation | LBFGS++ |
| Max depth | p=17 ($4^{17} \approx 1.7 \times 10^{10}$ entries, ~270 GB) |
| Accuracy | "good to at least four digits" |
| Previous record | p=11 (Wurtz-Lykov, at $d=3$ only) |

The paper notes: "Our numerical techniques are stretched to the limit at $p$ of 17."
