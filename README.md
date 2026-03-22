# QAOA Performance on D-Regular Max-k-XORSAT

Numerically calculating the fraction of constraints satisfiable by QAOA on D-regular max-k-XORSAT, with a focus on exact results at small (k, D) — particularly k=3, D=4 — for comparison against DQI (Decoded Quantum Interferometry) performance.

## Motivation

Collaboration with Dr. Stephen Jordan. The goal is to produce precise, quantitative QAOA performance numbers (not large-D asymptotics) to compare against DQI results at specific small (k, D) values.

## Problem Summary

**Max-k-XORSAT on D-regular hypergraphs**: Given a D-regular k-uniform hypergraph, each hyperedge defines an XOR constraint on k bits. The objective is to maximise the fraction of satisfied constraints.

**QAOA at depth p**: The Quantum Approximate Optimization Algorithm applies p rounds of alternating problem/mixer unitaries parameterised by angles (γ₁,…,γₚ, β₁,…,βₚ). Performance improves with p but computational cost of classical simulation scales exponentially in p.

**Key target**: k=3, D=4, pushed to the largest feasible p.

## Approaches

### 1. Direct / Exact Light-Cone Method (our focus)

- Exploits the fact that on locally tree-like graphs the expectation value of a single constraint under QAOA at depth p depends only on a finite neighbourhood (the "light cone" of radius p around that constraint).
- Exact — no O(1/D) corrections.
- Used in the original QAOA paper (Farhi, Goldstone, Gutmann 2014) and in the 2025 Farhi–Gutmann–Ranard–Villalonga paper on MaxCut.
- Cost: exponential in p, but feasible for moderate p with optimised code + cluster compute.

### 2. Finite-D Branch-Transfer Method

- Basso, Farhi, Marwaha, Villalonga, Zhou (2021) — arXiv:2110.14206.
- In this repo, the branch-transfer / WHT contraction is now implemented in the physical finite-D convention, not just the large-D normalization.
- Cost O(p² · 4ᵖ) from the exact branch/root XOR convolutions.
- Validated against the exact light-cone oracle on current anchor cases including `(k=3, D=2, p=1)` and MaxCut `(k=2, D=3, p=1,2)`.
- The public `parity_expectation` and `qaoa_expectation` API now routes through this exact finite-D evaluator.

## Key References

1. **Farhi, Goldstone, Gutmann (2014)** — "A Quantum Approximate Optimization Algorithm" — [arXiv:1411.4028](https://arxiv.org/abs/1411.4028)
   - Original QAOA paper. Direct calculation for MaxCut on 3-regular graphs at small p.

2. **Basso, Farhi, Marwaha, Villalonga, Zhou (2021)** — "The QAOA at High Depth for MaxCut on Large-Girth Regular Graphs and the SK Model" — [arXiv:2110.14206](https://arxiv.org/abs/2110.14206)
   - Iterative formula for D-regular MaxCut & Max-q-XORSAT. O(p² 4ᵖ) cost. Accurate to O(1/D). Pushed to p=20.

3. **Farhi, Gutmann, Ranard, Villalonga (2025)** — "Lower bounding the MaxCut of high girth 3-regular graphs using the QAOA" — [arXiv:2503.12789](https://arxiv.org/abs/2503.12789)
   - Direct exact method for MaxCut on 3-regular graphs. Pushed to p≥7 showing improvements over classical bounds at girth ≥16.

4. **Jordan, Shutty, Wootters, Zalcman, Schmidhuber, King, Isakov, Khattar, Babbush (2024/2025)** — "Optimization by Decoded Quantum Interferometry" — [arXiv:2408.08292](https://arxiv.org/abs/2408.08292), Nature 646:831-836 (2025)
   - Introduces DQI: uses quantum Fourier transform to reduce optimisation to decoding. Achieves superpolynomial speedup for structured problems. The comparison target for our QAOA results.

## Plan

> See [.project/PLAN.md](.project/PLAN.md) for the detailed work plan, and [.project/journal.md](.project/journal.md) for the development journal.
