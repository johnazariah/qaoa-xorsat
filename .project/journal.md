# Project Journal

## Entry 1 — Project Inception (21 March 2026)

### Context

John (first-year PhD candidate, quantum computing) received an email from **Dr. Stephen Jordan** (lead author of the DQI paper published in Nature 646:831-836, 2025). John is acknowledged in that paper and has a direct working relationship with Stephen.

Stephen wants to **numerically calculate the fraction of constraints satisfiable by QAOA on D-regular max-k-XORSAT**, particularly at **(k=3, D=4)**, for comparison against DQI and other algorithms. The key challenge: existing QAOA analysis methods (Basso et al. 2021) have O(1/D) errors, which are too large at D=4. The exact tensor-network method from Farhi et al. 2025 (arXiv:2503.12789) works for MaxCut (k=2) and needs to be **generalised to k-XORSAT (k≥3)**.

### What We've Done

1. **Created this repo** (`johnazariah/qaoa-xorsat`, private) with Julia project scaffolding and devcontainer.

2. **Downloaded 8 reference papers** to `.project/papers/`:
   - 3 QAOA papers: Farhi 2014 (original), Basso 2021 (high-depth iterative), Farhi 2025 (exact tensor network)
   - 5 DQI papers: Jordan 2024 (original Nature paper), plus follow-ups on structure requirements, MaxCut limitations, optimised circuits, and inapproximability

3. **Wrote extensive learning materials** in `.project/learning/`:
   - `00-foundations.md` — Qubits, gates, MaxCut, XORSAT, graphs, tensor networks
   - `01-explainer-farhi2014-original-qaoa.md` — Original QAOA paper explainer
   - `02-explainer-basso2021-high-depth.md` — Iterative high-depth method and O(1/D) limitation
   - `03-explainer-farhi2025-maxcut-lower-bound.md` — **The exact tensor network method we're adapting**
   - `04-our-problem.md` — Full synthesis: DQI mechanism, comparison data, what we compute

4. **Recorded Stephen's actual comparison data** — a table of satisfaction fractions across 15 (k,D) values for Prange, Simulated Annealing, DQI+BP, and Regev+FGUM. This table is in `04-our-problem.md`. Key finding: at (k=3, D=4), SA leads at **0.9366** — that's the bar QAOA needs to clear.

5. **Set up infrastructure:**
   - Devcontainer: Julia 1.11.4 on Bookworm + gh + az + LaTeX
   - `.github/copilot-instructions.md` with project context and Julia style guide
   - Julia project scaffolding: `src/QaoaXorsat.jl`, `test/runtests.jl`, `Project.toml`

### Key Design Decisions

- **Julia** as the implementation language. Idiomatic style: small composable functions, multiple dispatch, pipelines. C++ port only if profiling demands it.
- **Design mode by default.** Never write code unless explicitly asked.
- **Parameterise by (k, D, p)** — the code handles all 15 (k,D) pairs in Stephen's table, not just (3,4).
- **Validate against MaxCut (k=2, D=3)**: p=1 should give c̃_edge ≈ 0.7500. Farhi 2025 has results up to p=17 to validate against.

### What Comes Next (Phase 1 — Mathematical Foundation)

Before writing any code, we need to work through:

1. **The tensor network structure for k-XORSAT.** For MaxCut (k=2), the Farhi 2025 paper contracts a tensor network on a binary tree rooted at an edge, with cost O(4^p) independent of D. For k=3, the root is a hyperedge connecting 3 variable nodes, each branching into (D-1) further hyperedges. The tensor structure changes because the problem gate is now a 3-body diagonal gate instead of 2-body.

2. **Contraction cost analysis.** The key question: what is the exact computational cost for (k=3, D=4)? It's somewhere between O(4^p) and O(8^p) depending on contraction order. This determines our feasible p_max.

3. **The contraction algorithm.** The Farhi 2025 trick: contract a single branch from leaves to root, raise tensor entries to the (D-1)th power (element-wise exponentiation), continue inward. For k>2, the tree alternates variable nodes (degree D) and constraint nodes (degree k), so the contraction has two distinct step types.

4. **Angle optimisation strategy.** 2p parameters, expensive function evaluations. L-BFGS with multiple restarts, seeded from smaller-p solutions.

### The Comparison Landscape

For (k=3, D=4), the algorithms to beat:

| Algorithm | Fraction | Notes |
|-----------|----------|-------|
| Random | 0.5 | Trivial |
| DQI+BP | 0.87065 | Weak here — (k=3,D=4) is in DQI's unfavourable regime |
| Prange | 0.875 | Trivial DQI baseline |
| Regev+FGUM | 0.89187 | Quantum-inspired |
| **SA** | **0.9366** | **The real target** |
| QAOA (exact) | ??? | **Our computation** |

DQI is never the best at any (k,D) in Stephen's table — always beaten by SA or Regev+FGUM. The question is whether QAOA can beat SA.

### Critical Files

| File | Purpose |
|------|---------|
| `.project/PLAN.md` | Full 7-phase work plan |
| `.project/learning/04-our-problem.md` | Most complete synthesis — Stephen's data, DQI details, cost operators |
| `.project/learning/03-explainer-farhi2025-maxcut-lower-bound.md` | The method we're adapting |
| `.github/copilot-instructions.md` | Style guide and project context |
| `src/QaoaXorsat.jl` | Module stub — exports commented out until implemented |
| `test/runtests.jl` | Test stub with validation target |

### User Preferences (Critical!)

- **Design mode by default.** Never write code unless explicitly asked.
- **Idiomatic Julia:** tiny composable functions, multiple dispatch, pipelines. Think F# style but in Julia.
- **Infrastructure first.** The user likes "housework before the party."
- **No hallucinating.** If you don't know something, say so. The DQI definition was initially wrong (guessed "Dissipative Quantum Information" — it's "Decoded Quantum Interferometry").
