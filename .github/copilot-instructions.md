# QAOA-XORSAT — Copilot Instructions

## Project Overview

This repo computes the exact performance of QAOA (Quantum Approximate Optimization Algorithm) on D-regular Max-k-XORSAT, using tensor network contraction on the light-cone tree. The primary target is (k=3, D=4) for comparison against DQI (Decoded Quantum Interferometry) results from Dr. Stephen Jordan.

## Language & Style

- **Julia** is the language for all computational code.
- Write **idiomatic Julia**: small composable functions, multiple dispatch, clean pipelines.
- Prefer `|>` pipelines and broadcasting (`.`) over explicit loops where natural.
- Use descriptive names: `contract_branch`, `evaluate_edge_expectation`, not `f1`, `calc`.
- Types are for dispatch and clarity, not Java-style class hierarchies.
- Document public API with docstrings; internal functions need only be self-explanatory.

## Architecture Principles

- **Design first, code on request.** Default mode is design/discussion. Only write code when explicitly asked.
- Parameterise by `(k, D, p)` — never hardcode for a single case.
- Separate concerns: tree construction, tensor building, contraction, angle optimisation.
- Validate against known results: MaxCut (k=2, D=3) at p=1 should give c̃_edge ≈ 0.7500.

## Key References

- `papers/` — PDFs of all reference papers
- `learning/` — Explainers and foundational context (read 00-foundations.md first)
- `PLAN.md` — Full project work plan

## Domain Context

- QAOA depth `p` determines circuit depth and tree size
- The light-cone tree for (k, D) has branching factor (D-1)(k-1)
- Tensor contraction from leaves to root via element-wise exponentiation: O(4^p) cost
- Angle optimisation is 2p-dimensional (γ₁…γₚ, β₁…βₚ), use L-BFGS with multiple restarts
