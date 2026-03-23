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
- [x] Write `learning/04-our-problem.md` — how everything connects to our specific task:
  - What is Max-k-XORSAT vs. MaxCut
  - What changes when k>2 (hypergraphs instead of graphs)
  - What is DQI (Decoded Quantum Interferometry) and why Stephen wants the comparison
  - The known landscape at (k=3, D=4): DQI semicircle law, OGP barrier, SA/AMP baselines
  - The specific challenge at (k=3, D=4)
  - What "fraction of constraints satisfied" means precisely

### 0d. Papers (PDFs for reference)
- [x] Downloaded to `papers/`:
  - `farhi2014-original-qaoa.pdf` (arXiv:1411.4028) — Original QAOA
  - `basso2021-qaoa-high-depth.pdf` (arXiv:2110.14206) — QAOA at high depth, iterative formula
  - `farhi2025-maxcut-lower-bound.pdf` (arXiv:2503.12789) — Exact tensor network method
  - `jordan2024-dqi-nature.pdf` (arXiv:2408.08292) — Original DQI paper (Nature 2025)
  - `2509.14509-dqi-requires-structure.pdf` — DQI blocked by OGP on random instances
  - `2509.19966-no-advantage-maxcut.pdf` — No DQI advantage for MaxCut
  - `2510.10967-optimized-dqi-circuits.pdf` — Optimised DQI circuits (Jordan co-author)
  - `2603.04540-tight-inapproximability.pdf` — Tight limits of DQI on max-LINSAT

---

## Phase 1 — Mathematical Foundation

- [x] Derive the QAOA expectation value formula for a single k-XORSAT constraint on a D-regular k-uniform hypergraph, at depth p, via the light-cone / local-tree method.
  - For MaxCut (k=2), the light cone at depth p is a tree of depth p rooted at the edge. Each vertex at the boundary is in the uniform superposition state.
  - For k-XORSAT, the light cone is a tree of depth p rooted at a **hyperedge** connecting k variable nodes, each of which has D-1 other neighbouring hyperedges, each of which connects to k-1 other variable nodes, etc., out to depth p.
  - The tree structure alternates: hyperedge → variable → hyperedge → variable → …
  - At depth p the tree has a computable (but exponentially growing) number of leaves.
  - Landed as the exact Tier 1 light-cone reference in `src/qaoa.jl` and the exact Tier 2 finite-D branch-transfer evaluator in `src/basso_finite_d.jl`.
- [x] Characterise the tree structure precisely for (k=3, D=4) and determine tree sizes for p=1,2,…,12+.
  - `src/tree.jl` now exposes `TreeParams`, shell counts, total node counts, and leaf counts for arbitrary `(k, D, p)`.
- [x] Write down the QAOA unitary decomposition on the tree and the resulting expectation value as a function of (γ₁,…,γₚ, β₁,…,βₚ).
  - The physical round convention, raw tensor decomposition, and exact finite-D root observable are now encoded across `src/tensors.jl`, `src/qaoa.jl`, and `src/basso_finite_d.jl`.
- [x] Implement the raw tensor-network primitives for the light-cone sandwich (Spec P1.2):
  - `src/tensors.jl` now defines `QAOAAngles`, hyperindex utilities, and raw leaf/mixer/problem/observable tensors
  - `learning/05-tensor-derivation.md` records the hyperindex convention and contraction-ordering notes needed for P1.3

## Phase 2 — Literature Deep Dive & Existing Code

- [x] Read the full PDF of arXiv:2110.14206 carefully, especially:
  - Section on generalisation to Max-q-XORSAT → implemented as Tier 2 evaluator
  - Their iterative formula and its derivation → Eq. 8.7 implemented in `src/basso_finite_d.jl`
  - Their code → `github.com/benjaminvillalonga/large-girth-maxcut-qaoa` identified; MaxCut transfer ported to `src/maxcut_transfer.jl`
- [x] Read arXiv:2503.12789 for the direct tree-enumeration method details.
  - Identified that the O(4^p) trick does not extend to k>2 (constraint node multilinearity)
  - Led to the WHT factorisation discovery (`learning/15-wht-factorisation-discovery.md`)
- [x] Read arXiv:1411.4028 Sections 3-5 for the original direct calculation on 3-regular MaxCut.
  - p=1 analytical formula validated in `test/test_qaoa.jl`
  - Optimal angles (γ*=atan(1/√2), β*=π/8) confirmed to machine precision
- [x] Search for existing open-source QAOA tree-evaluation code:
  - Basso/Villalonga: `large-girth-maxcut-qaoa` (Python/C++) — MaxCut only, ported
  - QAOAKit: checked, not relevant (gate-level simulation, not tree contraction)
  - No existing Julia or k>2 implementations found — our code is novel

## Phase 3 — Implementation

- [x] Choose language/stack. Candidates:
  - **Julia** — good numerics, easy parallelism, fast prototyping
  - **Python + NumPy/JAX** — JAX for auto-diff of angles + GPU, but may be slow for tree enumeration
  - **C++/Rust** — for maximum performance on the exponential tree computation
  - **Hybrid** — Julia or Python driver with C/Rust core for the inner loop
- [x] Implement the core computation:
  1. Build the (k,D,p) tree structure (factor graph: variable nodes ↔ hyperedge nodes)
  2. Compute QAOA state on this tree for given (γ,β) angles
  3. Evaluate expected fraction of satisfied constraints for the root hyperedge
  - The repo now has both a guarded exact light-cone reference path and an exact finite-D Tier 2 evaluator wired through the public API.
- [x] Validate against known results:
  - k=2 (MaxCut) on 3-regular: p=1 should give ≥0.6924 (Farhi et al. 2014)
  - Compare with Basso et al. iterative formula at large D (should nearly agree)
  - MaxCut `p=1` optimum and `p=2` exact-statevector comparisons are covered in `test/test_qaoa.jl`.
  - Small exact finite-D overlap checks for `k=3` are covered in `test/test_qaoa.jl` and `test/test_basso_finite_d.jl`.
- [ ] Optimise:
  - [x] Exploit symmetries of the tree to reduce state-space dimension
  - [x] Memory-efficient representation (the tree state lives in a 2^(#leaves) Hilbert space)
  - Parallelise over angles during optimisation

## Phase 4 — Computation & Optimisation

- [ ] For (k=3, D=4), compute optimal QAOA performance at each depth p=1,2,3,…,p_max
  - [x] Optimise over 2p angles (γ₁,…,γₚ, β₁,…,βₚ)
  - [x] Use gradient-based optimisation (L-BFGS or similar) with multiple random restarts
  - [x] Record optimal angles and achieved fraction
  - Clean exploratory and reproduction-grade runs now exist through `p=5`, with the XORSAT sweep converging cleanly after the `g_abstol` adjustment, but the full `p=1..p_max` program remains incomplete.
- [ ] Determine p_max achievable on available hardware:
  - Estimate memory and time vs. p for (k=3, D=4)
  - The tree at depth p has O((D-1)^p · (k-1)^p) = O(3^p · 2^p) = O(6^p) leaves → Hilbert space ~2^(6^p) — this is **extremely** expensive
  - **Critical**: investigate whether clever contraction / tensor-network methods can reduce this
- [ ] If direct brute-force is infeasible beyond small p, explore:
  - Tensor network contraction on the tree
  - Belief-propagation style approximations (still exact on trees!)
  - Symmetry reductions

## Phase 5 — Comparison with DQI

- [x] Obtain Stephen's DQI numbers for (k=3, D=4)
  - Data recorded in `learning/04-our-problem.md`: DQI+BP=0.87065, Prange=0.875, Regev+FGUM=0.89187, SA=0.9366
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

**Key insight**: We do NOT need to store the full state vector once we leave the guarded reference regime. The QAOA unitaries are products of diagonal (problem) and transversal (mixer) gates, and the exact finite-D production path now contracts from leaves to root via branch tensors and Walsh-Hadamard/XOR-convolution structure.

The current exact evaluator keeps the Tier 1 statevector path only as a small-tree oracle and uses the Tier 2 finite-D transfer contraction for real work. The working target complexity is `O(p² · 4^p)` overall for fixed `(k, D)` rather than naive explicit-tree evolution.

## Open Questions

1. The exact finite-D recurrence is now implemented; the remaining question is empirical throughput and optimisation cost at `(k=3, D=4)` for increasing `p`.
2. What `p_max` is practical on available hardware once full angle optimisation and multiple restarts are included?
3. Are there existing optimisation / scheduling implementations we can build on, or do we need to write that layer from scratch?
4. What p values does Stephen need for a meaningful DQI comparison?
