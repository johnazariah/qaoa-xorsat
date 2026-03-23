# QAOA Performance on Max-k-XORSAT: Project Brief

*John Azariah — Centre for Quantum Software and Information, UTS*
*Prepared for Dr. Stephen Jordan, 23 March 2026*

---

## Objective

Compute the exact QAOA satisfaction fraction $\tilde{c}(p)$ for Max-k-XORSAT on
D-regular k-uniform hypergraphs at (k=3, D=4) for each circuit depth p, filling
the QAOA column in the comparison table:

| Algorithm | Fraction at (k=3, D=4) |
|-----------|----------------------|
| Random | 0.500 |
| DQI+BP | 0.871 |
| Prange | 0.875 |
| Regev+FGUM | 0.892 |
| **SA** | **0.937** |
| **QAOA(p)** | **this work** |

---

## Phase 1 — Research & Mathematical Foundation

### Method

We adapt the exact tensor-network contraction of Farhi et al. (arXiv:2503.12789)
from MaxCut (k=2) to k-XORSAT (k≥3), using the finite-D iteration of Basso
et al. (arXiv:2110.14206, §8.2).

On a D-regular k-uniform hypergraph of girth > 2p, the QAOA expectation value
for a single constraint depends only on its tree-shaped light cone. By
regularity, every constraint's tree is isomorphic — one evaluation suffices.

### The fold

The computation is a structural fold on the light-cone tree. A branch tensor
(vector of $4^p$ complex entries) is initialised at the leaves and transformed
inward one level at a time — alternating variable-node and constraint-node
operations — until it reaches the root, where the observable yields the
satisfaction fraction.

At **variable nodes**, D-1 identical child branches combine by element-wise
exponentiation. At **constraint nodes** (k≥3), k-1 children interact through
the k-body problem gate.

![Light-cone tree at p=1](../diagrams/a1-light-cone.png)

### The k≥3 constraint fold and WHT acceleration

For k=2 (MaxCut), the constraint fold is trivial (one child). For k=3, naïve
evaluation costs $O(4^{kp}) = O(64^p)$, limiting exact reach to p≈7.

We identified that the constraint fold sum is a double convolution on
$\mathbb{Z}_2^{2p+1}$. The Walsh-Hadamard transform diagonalises both
convolutions:

$$\hat{S} = \hat{\kappa} \cdot \hat{g}^{k-1}$$

reducing cost to $O(p \cdot 4^p)$ — the same as MaxCut. This factorisation is
verified numerically at p=1,2,3 for D=3,4 to machine precision ($\sim 10^{-15}$).
It does not appear explicitly in the literature; Basso et al. quote $O(4^{kp})$
and the Villalonga reference code handles only the $D \to \infty$ regime.

![Variable fold vs constraint fold](../diagrams/a3-fold-comparison.png)

---

## Phase 2 — Implementation & Validation

### Architecture

The implementation (Julia, ~1200 lines) has three tiers:

| Tier | Method | Cost | Reach |
|------|--------|------|-------|
| 1 | Brute-force statevector | $O(2^n)$ | p=1 (k=3,D=4) |
| 2 | Basso finite-D + WHT | $O(p^2 \cdot 4^p)$ | p≈15-17 |
| 3 | Basso D→∞ (approximate) | $O(p^2 \cdot 4^p)$ | p≈17 (O(1/D) error) |

Each tier validates against the one below it.

### Validation chain (621 tests)

| Layer | What | How |
|-------|------|-----|
| Golden values | MaxCut p=1 optimum = 0.6924 | Farhi 2014 analytical formula |
| Brute-force oracle | Independent statevector simulators | Explicit 2ⁿ-amplitude simulation |
| Basso ↔ oracle | Cross-validated at overlapping (k,D,p) | Machine-precision agreement |
| WHT ↔ naive | Constraint fold equivalence at p=1,2,3 | Machine-precision agreement |

### MaxCut validation against Farhi 2025 Table 1

| p | Our result | Published | Match |
|---|-----------|-----------|-------|
| 1 | 0.6925 | 0.6924 | Yes |
| 2 | 0.7559 | 0.7559 | Yes |
| 3 | 0.7924 | 0.7923 | Yes |
| 4 | 0.8169 | 0.8168 | Yes |
| 5 | 0.8364 | 0.8363 | Yes |

---

## Phase 3 — Experimental Results

### Preliminary QAOA results for (k=3, D=4)

Optimisation: L-BFGS with 8 restarts, 100 max iterations, warm-started from
previous depth. Run on M4 Max Mac Studio (64 GB).

| p | $\tilde{c}(p)$ | Wall time |
|---|----------------|-----------|
| 1 | 0.6761 | 0.6 s |
| 2 | 0.7391 | 0.04 s |
| 3 | 0.7771 | 0.5 s |
| 4 | 0.8022 | 10 s |
| 5 | 0.8205 | 108 s |

The curve is monotonically increasing but with diminishing returns — consistent
with the MaxCut pattern. At p=5, QAOA is below DQI+BP (0.871) and well below
SA (0.937). Extrapolation is premature, but the trajectory suggests significant
further depth is needed.

### Computational scaling

| p | Branch tensor | Eval time | Optimisation |
|---|--------------|-----------|-------------|
| 5 | 8 KB | 1.5 ms | 108 s (53K evals) |
| 10 | 8 MB | ~1 s | hours |
| 15 | 16 GB | est. minutes | days |

---

## Resources

| Machine | RAM | Max p | Cost |
|---------|-----|-------|------|
| M4 Max Mac Studio | 64 GB | p=14 | on hand |
| Dual Xeon workstation | 128 GB | p=15 | on hand |
| Azure E96-24ds v6 | 768 GB | p=17 | ~$8/hr |
| Azure FX96-48ms v2 | 1.8 TB | p=18 | ~$11/hr |

---

## Questions for Discussion

1. **Is p≈10-15 sufficient?** The diminishing-returns pattern suggests the
   trend will be clear before p=17. What precision does the comparison require?

2. **The WHT factorisation.** The reduction from $O(4^{kp})$ to $O(p \cdot 4^p)$
   for exact finite-D evaluation at arbitrary k appears novel. Is this worth
   documenting independently?

3. **Other (k,D) values.** The code is parameterised. Extending to all 15 rows
   of the comparison table is straightforward once (3,4) is complete.
