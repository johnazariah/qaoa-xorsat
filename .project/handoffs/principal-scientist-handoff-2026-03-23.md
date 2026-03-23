# Principal Scientist Handoff — 23 March 2026

> Written by the principal scientist agent at the end of the devcontainer session.
> The next agent should read this FIRST, then `.project/PLAN.md`, then the specs.

---

## What This Project Does

Computes exact QAOA performance on Max-3-XORSAT at D=4 regular hypergraphs,
for comparison against DQI (Dr. Stephen Jordan's algorithm, Nature 2025). The
deliverable is a table of satisfaction fractions at each QAOA depth p.

## Where We Are

### Completed

| Phase | Status | Key Output |
|-------|--------|------------|
| 0 — Learning | ✅ | 17 learning docs, 8 papers, 10+ explainers |
| 1 — Math Foundation | ✅ | Tree (P1.1), tensors (P1.2), contraction (P1.3) |
| 2 — Literature | ✅ | Basso §8 implemented, Farhi 2025/2014 validated |
| 3 — Implementation | ✅ | Full exact evaluator, 704 tests, 100% coverage |
| 4 — Optimization | 🔧 p=1-5 done | L-BFGS multistart, warm-start chain |
| 5 — DQI Comparison | 🔜 | Stephen's data recorded, awaiting higher-p results |

### XORSAT Results So Far (k=3, D=4)

| p | c̃(p) | Gap to SA (0.9366) |
|---|-------|--------------------|
| 1 | 0.6761 | 0.2606 |
| 2 | 0.7391 | 0.1975 |
| 3 | 0.7771 | 0.1595 |
| 4 | 0.8022 | 0.1344 |
| 5 | 0.8205 | 0.1161 |

Gain per step decays ~0.73×. Projected plateau ~0.87 (near DQI+BP=0.8707).
Need p=6-12 to determine if QAOA clears DQI+BP or flattens below it.

## Critical Technical Details

### The Evaluator Stack

- **Tier 1** (`src/qaoa.jl`): Brute-force state vector, ≤22 qubits. Reference oracle.
- **Tier 2** (`src/basso_finite_d.jl`): Exact finite-D Basso iteration + WHT acceleration.
  Cost O(p²·4^p), independent of k. This is the production path.
- `qaoa_expectation()` routes to `basso_expectation()` (Tier 2).

### The WHT Discovery

The naive Basso finite-D iteration costs O(4^{kp}). We discovered that the
constraint fold is a convolution on Z₂^{2p+1}, which the Walsh-Hadamard transform
diagonalises: Ŝ = κ̂ · ĝ^{k-1}. This reduces cost to O(p²·4^p) for any k.
Numerically verified to machine precision. Documented in
`learning/15-wht-factorisation-discovery.md`. This appears to be a novel result.

### Convention Traps (WILL bite the next agent)

1. **clause_sign**: `-1` for MaxCut (k=2), `+1` for XORSAT (k≥3)
2. **Basso D vs our D**: Basso's D = `params.D - 1` (branching degree, not total degree)
3. **Phase scale**: Must be `0.5` (physical γ/2 convention), NOT `1/√D` (Basso's paper convention). The branch iteration and root fold both use 0.5. This was the hardest bug to find — documented in `learning/17-root-interface-derivation.md` §10.
4. **Root fold**: k-fold XOR convolution via WHT, NOT a per-leg factorization. The sin kernel couples all k root legs.

### Optimizer Issues (solved)

- **g_abstol**: Set to `1e-6`, not Optim.jl default `1e-8`. The default is below the finite-difference noise floor, preventing convergence at p≥5.
- **canonicalize_angles**: Removed from the objective function. Was creating discontinuities at periodic boundaries that confused L-BFGS.
- **Warm-start**: `optimize_depth_sequence` seeds each p+1 from p's converged angles via `extend_angles` (repeats last angle pair).

## Immediate Next Steps

### 1. Migrate to M4 Mac (user is doing this)

```bash
bash scripts/setup-m4.sh
julia --project=. -t 10 scripts/optimize_qaoa.jl 3 4 1 12 4 200 1234 true
```

Setup script at `scripts/setup-m4.sh`. Checklist at `.project/checklist-m4-setup.md`.

### 2. Run p=6-12

Per-evaluation benchmarks (devcontainer, expect 2-4× faster on native M4):

| p | Eval time | Full opt (200 evals) |
|---|-----------|---------------------|
| 8 | 0.4s | ~80s |
| 10 | 7.3s | ~24 min |
| 12 | 128s | ~7 hours |

With 10-thread parallelism across restarts, divide by ~4-8×.

### 3. Produce comparison table (Phase 5)

Stephen's data is in `learning/04-our-problem.md`:
- DQI+BP: 0.87065
- Prange: 0.875
- Regev+FGUM: 0.89187
- SA: 0.9366

### 4. Performance improvements (if needed for higher p)

- **Thread-parallel restarts**: Not yet implemented. Easy ~10 line change in `optimization.jl` restart loop.
- **ForwardDiff.jl**: Add `autodiff=:forward` to `Optim.optimize` call for exact gradients. Eliminates FD noise and ~2p redundant evaluations per gradient step.
- **Metal.jl GPU**: Only worth it at p≥13. WHTs and element-wise ops on `MtlArray`.

## Key Files

| File | Purpose |
|------|---------|
| `.project/PLAN.md` | Master plan with phase status |
| `.project/specs/P1.3-contraction.md` | Revised spec (three-tier, WHT, convention mapping) |
| `src/basso_finite_d.jl` | Core evaluator (510 lines) |
| `src/optimization.jl` | L-BFGS optimizer |
| `src/qaoa.jl` | Public API + Tier 1 oracle |
| `src/wht.jl` | Walsh-Hadamard utilities |
| `scripts/optimize_qaoa.jl` | CLI experiment runner |
| `scripts/setup-m4.sh` | M4 Mac setup |
| `.project/results/optimization/index.csv` | All results |
| `.project/learning/04-our-problem.md` | Stephen's comparison data |
| `.project/learning/12-transfer-contraction-k-body.md` | Cost analysis + WHT update |
| `.project/learning/15-wht-factorisation-discovery.md` | WHT proof |
| `.project/learning/17-root-interface-derivation.md` | Root formula + convention fix |
| `.project/testing-register.md` | 704 tests documented |

## User Working Style

- **PI / principal scientist dynamic**: user is the PI, agent is the scientist. Developer agents are the implementers.
- **Design first**: specs in `.project/specs/` before code. Developer agent implements from specs.
- **Pushback expected**: the user will challenge claims. Verify before asserting. The WHT episode is the model — I wrongly dismissed it, user pushed back, it turned out to be real.
- **Daily reports**: `.project/reports/YYYY-MM-DD.md` via `start-work` skill.
- **Testing register**: `.project/testing-register.md` updated via `/update-testing-register` prompt.
- **No large-D approximations**: the whole point is exact finite-D results. Tier 3 (D→∞) is off the table as a deliverable.

## Repo State

- Branch: `main` at `d242ed1`
- Working tree: **clean**
- Tests: **704 passing**
- Coverage: **100%**
- No active worktrees
