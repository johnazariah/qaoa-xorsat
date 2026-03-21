# Work Plan: QAOA on D-Regular Max-k-XORSAT

## Phase 0 — Learning & Context Building

**Goal:** Build enough understanding to hold a meaningful conversation with Dr. Stephen Jordan and to understand why each design decision in our code matters.

### 0a. Foundational Concepts Primer
- [x] Write `learning/00-foundations.md` — prerequisite concepts:
  - Qubits, superposition, measurement, expectation values
  - Unitary operators and quantum circuits
  - Combinatorial optimisation (MaxCut, SAT, XORSAT)
  - Graphs, hypergraphs, regularity, girth
  - Approximation ratios vs. cut fractions
  - What "variational quantum algorithms" means

### 0b. Paper Explainers (read BEFORE the papers themselves)
- [x] Write `learning/01-explainer-farhi2014-original-qaoa.md`
  - What problem QAOA solves and why it matters
  - The QAOA circuit: what each layer does physically
  - The p=1 calculation on 3-regular MaxCut (the 0.6924 result)
  - The "light cone" idea and why it makes classical analysis possible
- [x] Write `learning/02-explainer-basso2021-high-depth.md`
  - How Basso et al. scale QAOA analysis to p=20
  - The iterative formula and its derivation from tree structure
  - What "O(1/D) corrections" means and why it limits precision at small D
  - The generalisation to Max-q-XORSAT
  - Why this paper is relevant but insufficient for our goal
- [x] Write `learning/03-explainer-farhi2025-maxcut-lower-bound.md`
  - The tensor network contraction method (our target approach)
  - The key trick: element-wise exponentiation of branch tensors
  - Why cost is O(4^p) and independent of D
  - How they pushed to p=17 and what computational resources it required
  - Why this is the method we want to adapt for k-XORSAT

### 0c. Connecting the Dots
- [ ] Write `learning/04-our-problem.md` — how everything connects to our specific task:
  - What is Max-k-XORSAT vs. MaxCut
  - What changes when k>2 (hypergraphs instead of graphs)
  - What is DQI and why Stephen wants the comparison
  - The specific challenge at (k=3, D=4)
  - What "fraction of constraints satisfied" means precisely

### 0d. Papers (PDFs for reference)
- [x] Downloaded to `papers/`:
  - `farhi2014-original-qaoa.pdf` (arXiv:1411.4028)
  - `basso2021-qaoa-high-depth.pdf` (arXiv:2110.14206)
  - `farhi2025-maxcut-lower-bound.pdf` (arXiv:2503.12789)

---

## Phase 1 — Mathematical Foundation

- [ ] Derive the QAOA expectation value formula for a single k-XORSAT constraint on a D-regular k-uniform hypergraph, at depth p, via the light-cone / local-tree method.
  - For MaxCut (k=2), the light cone at depth p is a tree of depth p rooted at the edge. Each vertex at the boundary is in the uniform superposition state.
  - For k-XORSAT, the light cone is a tree of depth p rooted at a **hyperedge** connecting k variable nodes, each of which has D-1 other neighbouring hyperedges, each of which connects to k-1 other variable nodes, etc., out to depth p.
  - The tree structure alternates: hyperedge → variable → hyperedge → variable → …
  - At depth p the tree has a computable (but exponentially growing) number of leaves.
- [ ] Characterise the tree structure precisely for (k=3, D=4) and determine tree sizes for p=1,2,…,12+.
- [ ] Write down the QAOA unitary decomposition on the tree and the resulting expectation value as a function of (γ₁,…,γₚ, β₁,…,βₚ).

## Phase 2 — Literature Deep Dive & Existing Code

- [ ] Read the full PDF of arXiv:2110.14206 carefully, especially:
  - Section on generalisation to Max-q-XORSAT
  - Their iterative formula and its derivation
  - Their code (check if a GitHub repo exists)
- [ ] Read arXiv:2503.12789 for the direct tree-enumeration method details.
- [ ] Read arXiv:1411.4028 Sections 3-5 for the original direct calculation on 3-regular MaxCut.
- [ ] Search for existing open-source QAOA tree-evaluation code:
  - QAOAKit (https://github.com/QAOAKit)
  - Basso et al. code
  - Farhi/Villalonga code
  - Any Julia/Python/C++ implementations

## Phase 3 — Implementation

- [ ] Choose language/stack. Candidates:
  - **Julia** — good numerics, easy parallelism, fast prototyping
  - **Python + NumPy/JAX** — JAX for auto-diff of angles + GPU, but may be slow for tree enumeration
  - **C++/Rust** — for maximum performance on the exponential tree computation
  - **Hybrid** — Julia or Python driver with C/Rust core for the inner loop
- [ ] Implement the core computation:
  1. Build the (k,D,p) tree structure (factor graph: variable nodes ↔ hyperedge nodes)
  2. Compute QAOA state on this tree for given (γ,β) angles
  3. Evaluate expected fraction of satisfied constraints for the root hyperedge
- [ ] Validate against known results:
  - k=2 (MaxCut) on 3-regular: p=1 should give ≥0.6924 (Farhi et al. 2014)
  - Compare with Basso et al. iterative formula at large D (should nearly agree)
- [ ] Optimise:
  - Exploit symmetries of the tree to reduce state-space dimension
  - Memory-efficient representation (the tree state lives in a 2^(#leaves) Hilbert space)
  - Parallelise over angles during optimisation

## Phase 4 — Computation & Optimisation

- [ ] For (k=3, D=4), compute optimal QAOA performance at each depth p=1,2,3,…,p_max
  - Optimise over 2p angles (γ₁,…,γₚ, β₁,…,βₚ)
  - Use gradient-based optimisation (L-BFGS or similar) with multiple random restarts
  - Record optimal angles and achieved fraction
- [ ] Determine p_max achievable on available hardware:
  - Estimate memory and time vs. p for (k=3, D=4)
  - The tree at depth p has O((D-1)^p · (k-1)^p) = O(3^p · 2^p) = O(6^p) leaves → Hilbert space ~2^(6^p) — this is **extremely** expensive
  - **Critical**: investigate whether clever contraction / tensor-network methods can reduce this
- [ ] If direct brute-force is infeasible beyond small p, explore:
  - Tensor network contraction on the tree
  - Belief-propagation style approximations (still exact on trees!)
  - Symmetry reductions

## Phase 5 — Comparison with DQI

- [ ] Obtain Stephen's DQI numbers for (k=3, D=4)
- [ ] Produce comparison table/plot: fraction satisfied vs. p for QAOA alongside DQI bound
- [ ] Analyse: at what p (if any) does QAOA surpass DQI?

## Phase 6 — Write-Up

- [ ] Document methodology and results
- [ ] Prepare figures
- [ ] Draft short note or contribute to Stephen's paper

---

## Complexity Estimates

For a single k-XORSAT constraint on a D-regular hypergraph at QAOA depth p:

| Layer | # nodes at layer ℓ |
|-------|-------------------|
| 0 (root hyperedge) | 1 hyperedge, k variables |
| 1 | k(D-1) new hyperedges, each with k-1 new variables |
| 2 | k(D-1)(k-1)(D-1) new hyperedges, … |
| … | branching factor (D-1)(k-1) per layer pair |

Total variable nodes in the tree ≈ k · ((D-1)(k-1))^p / ((D-1)(k-1) - 1)

For k=3, D=4: branching = 3×2 = 6. Leaves at depth p ≈ 3·6^p.

| p | ~# variables | ~Hilbert dim | Feasibility |
|---|-------------|-------------|-------------|
| 1 | ~21 | 2²¹ ≈ 2M | trivial |
| 2 | ~129 | 2¹²⁹ | intractable naively |

**Key insight**: We do NOT need to store the full state vector. The QAOA unitaries are products of diagonal (problem) and transversal (mixer) gates. On a tree, the computation can be done via a **contraction from leaves to root**, analogous to belief propagation. This is the approach used in the referenced papers and is the key to making this tractable.

The cost is then determined by the bond dimension at each cut of the tree, which for a path from root to leaf of length p is 2^(width at that cut). With careful scheduling, this can be done in time/space polynomial in 2^p times lower-order factors — matching the O(p² · 4^p) scaling quoted by Basso et al.

## Open Questions

1. Does the Basso et al. iterative formula extend straightforwardly to exact (finite-D) computation, or is the large-D limit baked in?
2. What is the precise tensor-network contraction cost for the direct method at (k=3, D=4)?
3. Are there existing implementations we can build on, or do we need to write from scratch?
4. What p values does Stephen need for a meaningful DQI comparison?
