# Session Handoff — 22-23 March 2026

> **For**: Any agent continuing work on QAOA-XORSAT
> **Context**: Two-day coaching + research + design session. Read this before
> doing anything.

---

## What happened this session

### Coaching (22 March)

John was coached from first principles on the entire QAOA-XORSAT project.
Key pedagogical breakthroughs:
- **QAOA as twist-and-mix**: problem phase spray-paints invisible phase labels,
  mixer causes interference (clock hands analogy)
- **The fold**: the entire tensor contraction is a catamorphism on the
  light-cone tree — carry a 4^p-entry branch tensor from leaves to root
- **WHT factorisation**: the k≥3 constraint fold is a double convolution on
  Z₂^{2p+1}, diagonalised by Walsh-Hadamard transform — same trick as FFT
  for polynomial multiplication

### Research (22 March)

- Extracted all 8 paper PDFs to plain text (`.project/papers/text/`)
- Deep dives on Farhi 2025 and Basso 2021 → learning files 13, 14
- Cloned Villalonga reference code (`/tmp/large-girth-maxcut-qaoa/`) — D→∞ only
- WHT factorisation: numerically verified at p=1,2,3 for D=3,4 to ~10⁻¹⁵

### Design & specs (22-23 March)

- Created `.project/specs/property-tests.md` — 5 property-based tests
- Created `.project/specs/generic-fold-engine.md` — CostAlgebra abstraction
  (committed as `cc5ea0a`)
- Created `.project/handoff-testing-ci.md` — testing/CI handoff
- Created `.project/experiment-initial-sweep.md` — experiment runner prompt
- Created `.github/workflows/optimize.yml` and `reproduce.yml`
- Restructured `.github/workflows/ci.yml` into 6 validation layers

### Documentation (22-23 March)

- Updated learning files: 00 (GF(2), hypergraphs), 10 (fold framing),
  13 (Farhi deep dive), 14 (Basso deep dive), 15 (WHT discovery)
- Created 00-maths-roadmap.md (John's learning roadmap)
- Created 16-visual-walkthrough.md with Mermaid diagrams (may need recreation)
- Rewrote study notes for Stephen → `.project/briefings/study-notes-for-stephen.md`
  (terser, 3-phase structure, includes preliminary k=3 D=4 results)
- Journal Entry 12 (coaching session), Entry 13 (WHT), plan updates

### Experimental results (23 March)

Runs completed and stored in `.project/results/optimization/`:
- MaxCut k=2, D=3, p=1-5: validated against Farhi 2025 Table 1 (all match)
- k=3, D=4, p=1-5: preliminary results (0.676, 0.739, 0.777, 0.802, 0.820)

---

## Current state

### Branches

```
main (HEAD at cc5ea0a) — everything merged, clean working tree
  └── .worktree/phase4-optimization — may be stale (was merged to main)
```

### Test suite

621 tests across 11 testsets, all passing. Run with:
```bash
cd /workspace && julia --project=. -e 'using Pkg; Pkg.test()'
```

### Key files

| File | What |
|------|------|
| `.project/PLAN.md` | Master work plan (updated with hardware, WHT, Villalonga) |
| `.project/WORKBOOK.md` | Self-contained project walkthrough |
| `.project/briefings/study-notes-for-stephen.md` | Briefing doc for Stephen Jordan |
| `.project/specs/generic-fold-engine.md` | **NEW** — CostAlgebra abstraction spec |
| `.project/specs/property-tests.md` | Property test spec (not yet implemented) |
| `.project/specs/P1.3-contraction.md` | Contraction spec (OQ1 resolved) |
| `.project/handoff-testing-ci.md` | Testing/CI handoff for developer agent |
| `.project/experiment-initial-sweep.md` | Experiment runner prompt |
| `.project/learning/15-wht-factorisation-discovery.md` | WHT result writeup |
| `.project/results/optimization/index.csv` | All experimental results |
| `.github/workflows/ci.yml` | 6-layer CI pipeline |
| `.github/workflows/optimize.yml` | On-demand optimisation workflow |
| `.github/workflows/reproduce.yml` | Reproducibility workflow |

---

## What to do next (priority order)

### 1. Property tests (spec ready, not implemented)

Read `.project/specs/property-tests.md`. Create `test/test_properties.jl` with
5 properties. Include in `test/runtests.jl`. ~60 lines of code.

### 2. Push to higher p

Run k=3, D=4 at p=6-8 using the experiment prompt at
`.project/experiment-initial-sweep.md`. Current results stop at p=5 (0.820).
Need to see the trend.

### 3. Generic fold engine refactor

Read `.project/specs/generic-fold-engine.md`. Phase 1 (extract, non-breaking)
can be done immediately. This is the big architectural win — turning a one-off
calculator into a reusable engine.

### 4. Self-hosted runner setup

John has a 48GB machine to set up as a GitHub Actions self-hosted runner.
Setup guide was discussed (Ubuntu 24.04 LTS Server, Julia, runner service).
Tag: `testbed-48gb`. The workflow files already support it.

---

## Critical conventions

- **Basso's D = our D-1**. The paper uses (D+1)-regular graphs. Our D=4 means
  Basso D=3. Getting this wrong produces wrong but plausible-looking numbers.
- **clause_sign**: +1 for XORSAT (even parity), -1 for MaxCut (odd parity).
  `even + odd = 1` is a testable invariant.
- **WHT verification**: confirmed at p=1,2,3 to machine precision. The
  convolution theorem is exact — no approximation involved.
- **The D→∞ iteration has O(1/D) error at D=4** (~25% in the exponent). Do NOT
  use it for the comparison table. Only Tier 2 (exact finite-D + WHT) gives
  trustworthy numbers.

---

## John's preferences

- **Design first, code on request.** Default mode is discussion/spec.
- **Idiomatic Julia**: small composable functions, multiple dispatch, pipelines.
- **Fold framing** over tensor network jargon — John thinks in catamorphisms.
- **No hallucinating**: if you don't know, say so. He caught a false WHT claim
  early in the session and corrected it — verify claims against source material.
- **Writing style**: read `johnazariah.github.io` — structured, precise,
  technically confident, conversational where it helps.

---

## Memory files

- `/memories/john-learning-style.md` — how John learns, key moments
- `/memories/session/workbook-and-coaching.md` — session-specific notes
  (includes retracted WHT claim history and correction)
