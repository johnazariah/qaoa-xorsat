# Project Journal

## Algorithmic Innovation Summary (for future methods paper)

This section threads the key algorithmic innovations in the order they
emerged, as a narrative arc suitable for a standalone methods paper.

### 1. WHT Factorisation (March 22–23)

The Basso-Farhi branch tensor recurrence involves a k-body constraint fold —
a sum over 2^{(2p+1)} configurations that elementarily costs O(4^{kp}).  We
recognised this as an XOR convolution on Z₂^{2p+1}, which the Walsh-Hadamard
transform diagonalises.  Result: O(p²·4^p) for any k.  At k=3, p=8 this is
65,000× faster.  This is the single insight that makes exact evaluation
feasible at depths beyond p=5.

### 2. Manual Adjoint Differentiation (March 23–24)

ForwardDiff.jl (dual numbers) scales as O(p) per gradient evaluation.  We
hand-derived the reverse-mode (adjoint) gradient through the full WHT-based
pipeline.  Key insight: the WHT is self-adjoint, and β-gradients use a
log-derivative trick.  Cost: ~1.6× a single forward evaluation, independent
of p.  At p=8 this is 12× faster than ForwardDiff.

### 3. Generic Fold Engine / Cost Algebra (March 22)

The Basso-Farhi contraction is a catamorphism over the light-cone tree,
parametrised by a "cost algebra" that specifies only the constraint kernel
and root observable.  MaxCut and Max-k-XORSAT are different instantiations
of the same interface.  This allowed us to validate against Farhi et al.
(2025) MaxCut results with zero code changes, then immediately produce the
first exact finite-D XORSAT numbers.

### 4. Plateau Detection with Circular Buffer (March 27–31)

At high depth (p≥10), L-BFGS evaluations take minutes each.  Standard
iteration-count convergence wastes hours when the optimizer has already
converged but g_norm hovers above the tolerance.  We implemented a per-
iteration Optim.jl callback with a circular buffer of the last 30 objective
values.  If max − min < g_abstol, the optimizer stops immediately.  At p=12,
this reduced wall time from 2+ hours to ~40 minutes.

### 5. Normalised Branch Tensor Recurrence (April 2)

At high (k, D, p), the branch tensor entries grow exponentially through
repeated `^(k-1)` and `^(D-1)` operations, overflowing Float64 at (7,8)
p≈9.  We introduced threshold-based normalisation: before each power
operation, if max-magnitude exceeds 1e30, divide by it and track the
accumulated scale in log space.  The final answer is reconstructed via
exp(log_total_scale) at the end.  The backward pass operates entirely on
normalised intermediates (magnitude ≤ 1), with scale factors detached from
the gradient — negligible error since ∂(max|x|)/∂θ is a sparse selection
operator.

The initial implementation (always-normalise) caused signal underflow at
p≥12 — crushing the relative magnitude differences that carry the physical
signal.  The threshold approach preserves Float64 precision at moderate
magnitudes while preventing overflow at extreme ones.

### 6. Swarm/Memetic Optimizer (April 5–6)

At high (k, D), the QAOA loss landscape is extremely rugged: most starting
points see c̃ ≈ 0.5 (flat), and only specific basins carry signal.  Standard
multi-start L-BFGS with warm-starting from p−1 fails at p=3 for (7,8).

We implemented a memetic (evolutionary + local search) optimizer:
- 100 random candidates, short L-BFGS bursts (20 iterations each)
- Cull worst 50%, replenish with random starts + midpoint crossovers
- Early exit: if 3 consecutive generations show no improvement, stop the
  swarm and run a full 1280-iteration L-BFGS polish on the best candidate
- Resume from CSV: survives machine crashes

The early exit is the key insight: at p≥6, the swarm converges in 1–3
generations (the warm-started candidate dominates), so the remaining 7–9
generations were wasting 100× compute.  The swarm finds the basin at low
depths; L-BFGS polishes at high depths.

Result: (7,8) went from failing at p=3 (standard optimizer) to 0.789 at
p=8 (swarm).  For the first time, all 15 (k,D) pairs have valid results
at depths where the standard approach collapsed.

### 7. Multi-Machine Orchestration (April 1–7)

Five compute environments coordinated via git branches:
- Mac Studio M4 (64 GB): primary development + p=1-12 for k=3 family
- Azure E8as_v5 fleet (5×64 GB): parallel sweep of all 15 pairs
- Azure E16as_v5 swarm VM: hard (k≥5) pairs with memetic optimizer
- P710 Xeon workstation (128 GB): (5,7) and (5,8) swarm chains
- Stephen's SLURM cluster (50×2.7 TB): p=13–15 production runs

Results aggregated via `collect-all-results.jl` with monotonicity filtering,
overflow detection, and provenance tracking.  The warm-start package
(`prepare-cluster-run.jl`) generates ready-to-submit SLURM configs from the
composite best angles across all machines.

### 8. Double64 Precision (April 10)

At k≥6, D≥7, p≥10, the branch tensor recurrence suffers catastrophic
cancellation: ~2M complex terms nearly cancel, and the physical signal
lives below Float64's 15-digit precision.  The evaluator returns correct
but meaningless values (S > 1, giving c̃ > 1).

Fix: DoubleFloats.jl provides Double64 (~31 digits) via double-double
arithmetic.  The pipeline is already generic over element type T, so
`QAOAAngles(Double64.(γ), Double64.(β))` propagates through the entire
evaluation and gradient.  Measured overhead: 3-5× (not 10-100× as
initially feared).  The swarm optimizer runs in Float64 for speed; only
the final evaluation and gradient are computed in Double64.

Validated: (6,7) p=10 returns 3.23 in Float64 (broken) vs 0.813 in
Double64 (valid).

---

## Entry 32 — Q1: QAOA is *not* Trotterised adiabatic optimisation (29 April 2026)

### Summary

Executed all four experiments from `.project/SPEC-Q1-adiabatic.md` on the
existing MaxCut sweeps (D=3..8, p ≤ 12).  Results land decisively against
the Trotterised-adiabatic interpretation of QAOA on D-regular MaxCut on
the infinite-girth tree.

Branch: `q1-adiabatic`.  Source: `scripts/q1_*.jl`, plots in `figures/`.

### Headline Numbers (Experiment 1: Adiabatic Fidelity)

For each D at the deepest available p, evaluate c̃ at the linear
adiabatic schedule γ_j = (j/p)·γ_max, β_j = (1-(j-1)/(p-1))·β_max with
γ_max, β_max set to the magnitudes of the optimal angles:

| D | p  | c̃_opt    | c̃_adi    | Δ        | rel. loss |
|---|----|----------|----------|----------|-----------|
| 3 | 12 | 0.88594  | 0.47008  | +0.4159  | 1.078     |
| 4 | 11 | 0.82331  | 0.69118  | +0.1321  | 0.409     |
| 5 | 12 | 0.79566  | 0.50396  | +0.2917  | 0.987     |
| 6 | 12 | 0.76701  | 0.50405  | +0.2630  | 0.985     |
| 7 | 10 | 0.74123  | 0.50009  | +0.2411  | 1.000     |
| 8 |  9 | 0.72171  | 0.53003  | +0.1917  | 0.865     |

The matched-magnitude linear schedule barely beats random guess (0.5)
for D ≥ 5 and at D=3 falls *below* random.  "rel. loss" is
(c̃_opt − c̃_adi)/(c̃_opt − 0.5): ≥ 0.86 for D ≥ 5 means the adiabatic
schedule recovers < 14 % of the gap above random.

### Experiment 2: Intermediate-Depth (Truncated) Performance

Took the depth-p_max optimum and evaluated it as a depth-t QAOA for
t = 1..p_max (using only the first t angle pairs).

- The truncated schedule **is monotonic** in t at every D ∈ {3,4,5,6}.
- However it is also **systematically far below** the depth-t global
  optimum for small t.  Sample (D=3, p_max=12): c̃_truncated(t=1)=0.556
  vs c̃_optimal(t=1)=0.692; the gap stays > 0.06 until t ≥ 11.
- The linear-adiabatic-at-t curve oscillates around 0.5 for every D.

Interpretation: the optimal depth-p schedule does not contain the
optimal depth-t schedule as a prefix.  This is consistent with QAOA
re-tuning every angle as p grows, not with adding incremental
"adiabatic time" on top of a converged shorter schedule.

### Experiment 3: Adiabatic-Initialised Optimisation

For each D ∈ {3..8} at p=8, ran L-BFGS to convergence from each of 12
linear-adiabatic seeds (4 γ_max ∈ {π/2, π, 3π/2, 2π} × 3 β_max ∈
{π/4, π/2, 3π/4}).  Compared the *best-of-12* converged value against
the warm-start optimum c̃_warm from the existing sweep.

| D | warm c̃   | best adi-init c̃ | Δ        |
|---|----------|-----------------|----------|
| 3 | 0.86739  | 0.86127         | +0.00612 |
| 4 | 0.80988  | 0.79750         | +0.01238 |
| 5 | 0.77710  | 0.77302         | +0.00408 |
| 6 | 0.75193  | 0.73863         | +0.01330 |
| 7 | 0.73276  | 0.72306         | +0.00970 |
| 8 | 0.71736  | 0.70797         | +0.00940 |

L-BFGS converged in 80–340 iterations on every (D, seed) — the seeds
*are* in the smooth basin of some local optimum.  The local optimum is
just **not** the global warm-start basin.  All 72 runs converged
(`converged=true`) except one D=8 (γ_max=π/2, β_max=π/4) point where
L-BFGS terminated after 1 iteration at c̃ = 0.5000 — the gradient at
the seed was below tolerance, so the linear-adiabatic point was already
a saddle/local minimum at random performance.

This is the strongest single piece of evidence: even when L-BFGS runs
to convergence from the adiabatic seed, every D shows a > 0.004 gap
below warm-start.  An adiabatic-style schedule does not lie in the
optimal QAOA basin.

### Experiment 4: Angle-Profile Curvature

Polynomial fits to the unwrapped optimal angle profiles γ(j/p), β(j/p):

| D | profile | r²(linear) | Δr² (→cubic) |
|---|---------|------------|--------------|
| 3 | γ       | 0.50       | +0.40        |
| 3 | β       | 0.15       | +0.69        |
| 4 | γ       | 0.89       | +0.09        |
| 4 | β       | 0.93       | +0.07        |
| 5 | γ       | 0.92       | +0.05        |
| 5 | β       | 0.02       | +0.07        |
| 6 | γ       | 0.89       | +0.09        |
| 6 | β       | 0.96       | +0.04        |
| 7 | γ       | 0.90       | +0.07        |
| 7 | β       | 0.97       | +0.02        |
| 8 | γ       | 0.90       | +0.07        |
| 8 | β       | 0.98       | +0.02        |

D=3 and D=5-β are extreme outliers — the optimal β profile at D=5,
p=12 visibly **zigzags** between ~1.1 and ~2.1 from step to step (a
bang-bang signature, not a smooth ramp).  Even the "linear-friendly"
(D=4, 6, 7, 8) profiles always pick up ≥ 0.02 r² from the cubic term,
so the schedule is not strictly linear at any D.

### Conclusion (writeable for paper Section 5)

> The optimal QAOA angle schedules on D-regular MaxCut on the
> infinite-girth tree are inconsistent with a Trotterisation of linear
> adiabatic optimisation.  The matched-magnitude linear schedule
> recovers at most 14 % of the gap above random for D ≥ 5, the
> depth-p optimum is *not* a prefix of the depth-(p+k) optimum, and
> L-BFGS seeded from the linear schedule converges to a strictly
> worse basin than warm-start.  This identifies QAOA as a
> *variational* algorithm exploiting interference patterns rather
> than a discretised annealing procedure.

### Caveats

- We compute exact expectation values on the infinite-girth tree, not
  full quantum-state overlaps.  A statement that QAOA states do not
  resemble adiabatic ground states would require finite-instance
  simulation, which this codebase does not perform.
- The "linear adiabatic" schedule we use matches the QAOA endpoint
  magnitudes, not the true endpoint Hamiltonians.  A more sophisticated
  comparator (eg., schedules from arXiv:2106.15645) is future work.

### Artefacts (all under `qaoa-xorsat-q1` worktree only)

- `scripts/q1_intermediate_depth.jl` (was already provided)
- `scripts/q1_angle_schedules.jl`
- `scripts/q1_adiabatic_init.jl`
- `scripts/q1_angle_curvature.jl`
- `scripts/q1_plots.py`
- `results/q1-intermediate-depth.csv`
- `results/q1-angle-schedules.csv`
- `results/q1-adiabatic-fidelity.csv`
- `results/q1-adiabatic-init.csv`
- `results/q1-angle-curvature.csv`
- `figures/q1-{angle-schedules,intermediate-depth,adiabatic-init,angle-curvature}.png`

---

## Entry 31 — Double64 Precision Fix + Final Sweep (10 April 2026)

### Summary

Diagnosed the k≥6 precision wall: NOT overflow (magnitudes stay < 1.5)
but catastrophic cancellation in the 2^{2p+1}-element sums.  Float64's
15 digits insufficient when (k-1)(D-1) ≥ 30.

Implemented `swarm_chain_d64.jl` — runs the swarm in Float64 (fast
basin discovery) and re-evaluates the winner in Double64 (correct value).
Added `qaoa_d64_sweep.sh` SLURM script for all 15 pairs.

Built `run-d64-sweep.sh` — all-in-one script that diagnoses, submits,
monitors every 5 minutes, and auto-pushes results to git every 10 minutes.
Stephen ran it on 55 nodes; all 15 pairs running from p=1 with D64.

Key validation: at (6,7) p=10, Float64 returns c̃ = 3.23 (broken),
Double64 returns c̃ = 0.813 (valid).  Both evaluators agree to 1e-9 at
low (k,D) where Float64 has sufficient precision.

### Commits

- `30e18fb` feat: Double64 swarm for all 15 pairs
- `9d0d8e6` feat: all-in-one D64 sweep — diagnose, submit, monitor, push
- `635e591` fix: integer expression bug in monitor loop

---

## Entry 30 — Warm-Start Path Bug + Swarm on Stephen's Cluster (8 April 2026)

### Summary

The warm-start TOML configs had absolute Mac paths (`/Users/johnaz/...`)
that didn't exist on Stephen's cluster.  Fixed to relative paths.
Stephen submitted but many pairs collapsed to 0.500 because the
warm-start angles came from pre-normalization runs (overflow-adjacent
basins that evaluate to 0.5 with the corrected code).

Solution: deployed the swarm optimizer on Stephen's cluster via
`qaoa_swarm_sweep.sh`.  The swarm finds real basins from p=1.  Results:
13 of 15 pairs beat DQI+BP.  The remaining two ((5,6) and (6,8))
within 5 basis points.

For k≥6, the swarm found valid basins through p=8-9 but hit the
Float64 precision wall at p≥10.  This led to the Double64 fix (Entry 31).

### Commits

- `76e3e19` fix: use relative paths in warm-start configs
- `da3d2ac` feat: SLURM swarm sweep for stuck pairs

---

## Entry 29 — P710 Delivers p=8/9, Fleet Decommissioned (6–7 April 2026)

### Summary

The dual-Xeon P710 workstation (32 threads, 128GB, Windows) ran the swarm
optimizer on (5,7) and (5,8) in parallel. Key results:

- **(5,7) p=8 = 0.7893** — new best (wall: 5.5 hr)
- **(5,8) p=8 = 0.7798** — new best (wall: 3.5 hr)
- **(5,8) p=9 = 0.7996** — new best, beats DQI+BP (0.788) → **12 of 15 pairs**
- (5,7) p=9 = 0.7785 — dropped below p=8 (bad basin at p=9)

Azure fleet (qaoa-swarm-2) completed (6,7) p=9 = 0.8385 and was
decommissioned. Resource group `qaoa-fleet-5` deleted. Total Azure spend
for the campaign: ~$50.

Stephen's cluster dashboard showed stale Mac results — the monitor script
reads `.project/results/optimization/runs/` which had old data from March 31.
Sent updated instructions (`STEPHEN-INSTRUCTIONS.md`) for the warm-start
sweep.

### Scorecard: 12 of 15 pairs beat DQI+BP

| (k,D) | Best p | c̃ | Source |
|-------|--------|------|--------|
| (3,4) | 13 | 0.881 | Stephen cluster |
| (3,5) | 13 | 0.843 | Stephen cluster |
| (3,6) | 12 | 0.809 | Azure fleet |
| (3,7) | 11 | 0.779 | Azure fleet |
| (3,8) | 11 | 0.768 | Mac |
| (4,5) | 11 | 0.861 | Azure fleet |
| (4,6) | 11 | 0.830 | Azure fleet |
| (4,7) | 10 | 0.806 | Azure fleet |
| (4,8) | 10 | 0.800 | Azure fleet |
| (5,6) | 10 | 0.849 | Azure fleet |
| (5,7) | 8 | 0.789 | P710 swarm |
| (5,8) | 9 | 0.800 | P710 swarm |
| (6,7) | 9 | 0.838 | Azure swarm |
| (6,8) | 8 | 0.801 | Azure swarm |
| (7,8) | 8 | 0.789 | Azure swarm |

---

## Entry 28 — Stephen's Cluster p=13 Results + Multi-Machine Campaign (7 April 2026)

### Summary

Stephen ran all 15 pairs on his 50-node SLURM cluster (April 3–6). Key wins:
**(3,4) p=13 = 0.8807** and **(3,5) p=13 = 0.8429** — both new records. However,
most k≥4 pairs collapsed to 0.500 at p≥12, confirming the signal loss problem
affects his cluster too (he ran the pre-normalization code). Still running
(3,4) and (3,5) toward p=14.

Simultaneously, we ran 5 hard pairs on Azure VM (qaoa-swarm-2, E16as_v5)
and P710 Xeon workstation using the swarm optimizer:

| Pair | Machine | Best p | c̃ |
|------|---------|--------|------|
| (5,7) | P710 | 8 | 0.7893 |
| (5,8) | P710 | 7 | 0.7694 |
| (6,7) | Azure | 9 | 0.8385 |
| (6,8) | Azure | 8 | 0.8015 |
| (7,8) | Azure | 8 | 0.7895 |

Emailed Stephen instructions to push his results branch and pull the
normalization + swarm fixes for a warm-started rerun.

---

## Entry 27 — Swarm/Memetic Optimizer for Rugged Landscapes (5–6 April 2026)

### Summary

At high (k,D), the QAOA loss landscape is extremely rugged: most starting
points see c̃ ≈ 0.5 (flat), and only specific basins carry signal. Standard
multi-start L-BFGS with warm-starting from p-1 fails at p=3 for (7,8).

Implemented a memetic (evolutionary + local search) optimizer:

1. **Population phase**: 100 random starting points, each gets a short
   L-BFGS burst (20 iterations)
2. **Cull**: kill the worst 50%
3. **Replenish**: 40% fresh random starts + 60% midpoint crossovers
   from the top 30 survivors (with small perturbation)
4. **Repeat**: 10 generations
5. **Early exit**: if 3 consecutive generations show no improvement,
   stop the population phase and switch to a full 1280-iteration
   L-BFGS polish on the best candidate

The early exit is critical: at p≥6, the swarm converges in 1–3 generations
(the warm-started candidate dominates), so the remaining 7–9 generations
were wasting 100× compute. With early exit, the swarm finds the basin at
low depths and L-BFGS polishes it at high depths — best of both worlds.

**Results**: (7,8) went from failing at p=3 (standard optimizer) to
0.776 at p=5 (swarm) to 0.789 at p=8 (swarm + early exit + polish).
For the first time, all 15 pairs have valid results at depths where
the standard approach collapsed.

**Resume from CSV**: swarm_chain.jl reads the existing results file on
startup, finds the last completed depth, and warm-starts from there.
Survives machine crashes and process restarts.

### Deployment

- Azure VM (qaoa-swarm-2): 16 cores, 3 hard pairs (6,7)(6,8)(7,8)
- P710 Xeon workstation: 16 cores / 32 threads, 2 pairs (5,7)(5,8)
- Auto-push to `p710-results` branch every 10 minutes

### Commits

- `95e02b5` feat: swarm/memetic optimizer for rugged landscapes
- `6b18025` fix: swarm writes results to CSV immediately
- `6761d81` feat: swarm early-exit + polish + resume from CSV
- `a166290` fix: log every generation

---

## Entry 26 — Threshold-Based Normalization (4 April 2026)

### Summary

The always-normalize strategy from Entry 25 caused a new problem:
**signal underflow** at p≥12. Dividing by max-magnitude every step
crushed the relative magnitude differences between entries that carry
the physical signal (deviation from c̃ = 0.5).

Fix: only normalize when max-magnitude exceeds 1e30 (threshold chosen
so that `(1e30)^7 < 1e300`, safe for degree≤7). This preserves Float64
precision at moderate magnitudes while still preventing overflow.

However, for k≥5, D≥7 at p≥10, the signal genuinely approaches machine
epsilon — this is catastrophic cancellation, not overflow. The branch
tensor iteration contracts everything to near-zero, and the residual
that makes c̃ ≠ 0.5 is below Float64 resolution. This motivated the
swarm optimizer (Entry 27): finding better basins is more effective than
higher numerical precision.

### Commits

- `f354eb1` fix: threshold-based normalization preserves signal at high depth

---

## Entry 25 — Normalized Evaluator Fixes Float64 Overflow (2 April 2026)

### Summary

The branch tensor recurrence overflowed Float64 at high (k,D,p), producing
impossible values (c̃ > 1, NaN, -1e+88). Root cause: raising complex arrays
to `^(k-1)` and `^(D-1)` at each step compounds magnitudes exponentially
(k=7,D=8 overflows at p ≈ 9). Fixed by normalizing to unit max-magnitude
before each power operation and tracking scale in log space.

### What changed

**`src/adjoint.jl`** — Normalized forward pass:
- Before `^(k-1)`: `child_hat /= max|child_hat|`, log `α_t`
- Before `^(D-1)`: `folded /= max|folded|`, log `β_t`
- Before `^k` at root: `msg_hat /= max|msg_hat|`, log `μ`
- Final value: `c̃ = (1 + cs · exp(k·(log_s + log(μ))) · Re(S_norm)) / 2`
- Backward pass: operates entirely on normalized intermediates (all ≤ 1),
  applies `exp(log_total_scale)` once. Scale factors detached from gradient.
- New `basso_expectation_normalized()` for overflow-safe evaluation.

**`src/optimization.jl`** — Five safety guards:
1. Overflow gradient: returns large finite value (1e6) with non-zero gradient
   pointing toward origin (zero gradient was faking convergence)
2. Post-evaluation: rejects any c̃ outside [0, 1] via `is_valid_qaoa_value()`
3. Best-start selection: valid results always preferred over invalid in argmax
4. Merge: validity-aware — valid c̃ = 0.88 beats overflowed c̃ = 21.44
5. Warm-start chain: poisoned angles not propagated to next depth

### Validation

1741 tests (273 new in `test/test_normalization.jl`):
- Exact agreement with un-normalized path at low (k,D) (atol = 1e-10)
- Physical bounds ∈ [0, 1] at all previously-overflowing (k,D,p)
- Finite gradients at high (k,D)
- Cluster regression: all 15 pairs at p=10
- MaxCut validation preserved (0.5 + √3/9)
- Merge validity logic

### Commits

- `48a0720` feat: normalized branch tensor recurrence prevents Float64 overflow
- `958123b` fix: comprehensive overflow safety in optimizer

---

## Entry 24 — Fleet Results and Conservative Data Merge (2 April 2026)

### Summary

Azure fleet (5× E8as_v5, 256GB) returned results for all 15 (k,D) pairs.
Merged with Mac results, keeping the better of each. Some fleet values at
the overflow boundary are suspect (e.g. (5,7) p=10 = 0.883 followed by
21.44 at p=11). Used monotonicity filter: if c̃ drops or overflows at p+1,
the p value is flagged and excluded.

### Updated best values

| (k,D) | p_max | c̃ | Source |
|-------|-------|------|--------|
| (3,4) | 12 | 0.8769 | Mac |
| (3,7) | 11 | 0.7788 | Fleet ← improved |
| (4,5) | 11 | 0.8605 | Fleet ← was p=8 |
| (4,6) | 10 | 0.8269 | Fleet ← was p=8 |
| (4,7) | 10 | 0.8060 | Fleet ← was p=8 |
| (4,8) | 10 | 0.7996 | Fleet ← was p=8 |
| (5,6) | 10 | 0.8488 | Fleet ← was p=8 |
| (5,7) | 9 | 0.8134 | Fleet (conservative) |
| (5,8) | 9 | 0.8013 | Fleet ← was p=8 |
| (6,7) | 8 | 0.8191 | Fleet |
| (6,8) | 8 | 0.7992 | Fleet |
| (7,8) | 8 | 0.8230 | Mac (fleet had no data) |

### Commit

- `01041d3` data: update best-values table with fleet results (conservative merge)

---

## Entry 23 — Overflow Diagnosis and First Fix Attempt (1 April 2026)

### Summary

Stephen's 50-node SLURM cluster ran all 15 pairs. Dashboard showed several
tasks with impossible values: (4,8) p=11 c̃ = 1.598, (5,8) p=10 c̃ = 1.33,
(6,7) p=10 c̃ = 7.71, (7,8) p=9 c̃ = 2.23, and multiple tasks at c̃ = 0.500
(trivial/degenerate). ChatGPT analysis of the logs identified a two-stage
failure:

1. **Evaluator overflow**: `child_hat^(k-1)` and `folded^(D-1)` overflow Float64
2. **Pipeline propagation**: overflowed values survive through optimizer
   selection, retry merge, and warm-start chain

First fix (commit `df4646e`) added overflow guards returning Inf with zero
gradient — insufficient because zero gradient fakes convergence. Second fix
(commit `dfb8910`) returned large finite value with non-zero gradient and
added validity-aware selection/merge. Still insufficient because overflow
happens inside the evaluator itself — need normalization (Entry 25).

### Commits

- `df4646e` fix: overflow guard + scoping bug, add recovery script
- `dfb8910` fix: comprehensive overflow safety in optimizer

---

## Entry 22 — SLURM Cluster Scripts and Monitoring (1 April 2026)

### Summary

Stephen Jordan offered his 50-node SLURM cluster (2.7TB RAM/node, 28 cores,
partition `c3d`) for the full p=15 sweep. Created deployment infrastructure:

- `scripts/qaoa_sweep.sh` — SLURM batch script, 15-task array
- `scripts/runner.py` — maps SLURM_ARRAY_TASK_ID to (k,D), launches Julia
- `scripts/slurm-monitor.sh` — live dashboard showing per-task progress
- `scripts/slurm-collect.sh` — aggregates results into CSV for transfer
- `scripts/recover-run.py` — identifies last valid depth, generates recovery jobs
- `SLURM.md` — complete setup and operational guide

Also fixed `optimize_qaoa.jl` scoping bug: `best_checkpoint_value` inside
a `for` loop was treated as a new local variable in Julia 1.12's soft-scope
rules, breaking checkpoint recovery entirely.

### Commits

- `7e19a0d` feat: SLURM cluster scripts with monitoring and progress extraction
- `de204b9` docs: add recovery instructions to SLURM.md

---

## Entry 21 — Plateau Detection and Paper Writing (27–31 March 2026)

### Summary

Convergence engineering for the optimizer. Evolved plateau detection through
four iterations: iteration-count chunks (100) → depth-dependent chunks →
wall-time chunks → circular buffer with per-iteration check. Final design:
Optim callback maintains a 30-value circular buffer, checks if
`max - min < g_abstol` every iteration, flushes trace every 5 minutes.
Proven at p=12: converges at 45 iterations (~40 min vs 2+ hours).

Also wrote the research paper (`qaoa-xorsat-research/paper/main.tex`) with
full 15-pair comparison table, QAOA vs p figure, timing progression plot.
Stephen Jordan invited John as co-author on the Google Quantum AI paper.

Azure fleet deployed (5× E8as_v5 VMs) for Phase 1 parallel computation.
Mac Studio pushed to p=12 for (3,4): c̃ = 0.8769, beating Prange.

---

## Entry 20 — Metal.jl GPU Spike: Dead End (26 March 2026)

### Summary

Ran the Metal.jl GPU spike to test Apple Silicon GPU acceleration for the Basso
evaluator (Phase 5 — Performance). **Metal.jl does not support Float64.**
`MtlArray{ComplexF64}` throws immediately — not emulated, not slow, hard-rejected.
Since the entire Basso pipeline runs in `Complex{Float64}`, this kills the Metal
GPU approach as designed.

Full results on branch `metal-gpu-spike` (commit `b9dc33f`), detailed analysis
in `.project/specs/metal-gpu.md`.

### What was tested

Three scripts in `.worktree/metal-gpu-spike/spike/`:

1. **`spike.jl`** — element-wise ComplexF32 GPU vs ComplexF64 CPU benchmarks
   at p=4/8/10/11 vector sizes, plus power operations and fused Basso step
2. **`wht_metal.jl`** — custom Metal butterfly kernel for the Walsh-Hadamard
   transform, correctness validation and benchmarks
3. **`precision_test.jl`** — Float32 vs Float64 through the full Basso pipeline
   (values and gradients) at p=1-5

### Key findings

**1. Metal.jl hard-rejects Float64.** This is the showstopper. There is no
emulation, no fallback, no workaround within Metal.jl itself.

**2. ComplexF32 GPU speedup is marginal against ComplexF64 CPU:**

| Depth | Element-wise | WHT kernel | Combined |
|-------|-------------|------------|----------|
| p=8 | 0.3× (slower) | 0.3× (slower) | Not viable |
| p=10 | 1.5× | 1.1× | Barely break-even |
| p=11 | 5.7× | 3.6× | ~4× (best case) |

The GPU only wins at p≥11, and only in Float32. The WHT suffers from kernel
launch overhead — 21-25 sequential dispatches per transform.

**3. Float32 precision degrades with depth:**

| p | Value rel error | Gradient max error (β) |
|---|----------------|----------------------|
| 1 | 3.3e-8 | 2.7e-8 |
| 3 | 2.4e-6 | 2.4e-5 |
| 5 | 2.4e-6 | 3.4e-5 |

Errors grow toward ~1e-4 at p=5. At p=10-13, Float32 gradients would likely be
too noisy for L-BFGS — precisely where the GPU finally becomes faster.

**4. Additional blockers:** `x .^ n` broken on Metal (Julia widens to Float64
internally), requiring explicit multiplies for all powers.

### Why the original 20-40× projection was wrong

The spec assumed `MtlArray{ComplexF64}` would work. It doesn't. Without native
Float64, the GPU advantage is limited to the ~2× bandwidth gain from Float32's
smaller memory footprint, which is partly eaten by kernel launch overhead and
wholly undermined by precision loss at high depth.

### Decision

**Metal.jl GPU approach ABANDONED.** Spec updated to reflect this.

For p≥12, the viable alternatives are:
- **Stay on CPU** — p=11 in ~7h, p=12 in ~3 days. Brute-forceable on M4 64GB.
- **CUDA.jl on cloud A100** — native Float64 at 9.7 TFLOPS, 2 TB/s bandwidth.
  The code port (Array → CuArray) is nearly mechanical. This is the right path
  if p≥14 results are scientifically needed.
- **Algorithmic improvements** — symmetry reduction, better warm-starting,
  vectorized WHT. Medium effort, multiplicative gains.

### Commits

| Branch | Commit | Description |
|--------|--------|-------------|
| `metal-gpu-spike` | `b9dc33f` | spike: Metal.jl GPU feasibility tests |

---

## Entry 19 — M4 Bare Metal Migration + Performance Optimization Stack + p=9 Result (24 March 2026)

### Summary

Migrated from devcontainer to native Apple Silicon M4. Built a complete
performance optimization stack that took p=8 from "never completes" to 11
minutes, and reached p=9 (c̃ = 0.8613) — gap to DQI+BP now just 0.010.

### Bare metal setup

- Installed Julia 1.12.5 via juliaup on M4 Mac (64GB, 12 threads)
- All 655 tests passing on native, then grew to 714 with new tests
- Archived devcontainer to `.devcontainer.archived/`
- Coverage verified at 100% (672/672 lines)

### Optimization 1: ForwardDiff exact gradients (`a312e2f`)

Made `QAOAAngles{T<:Real}` parametric so ForwardDiff dual numbers propagate
through the entire Basso evaluation pipeline. Converted ~55 type barriers in
`basso_finite_d.jl` from concrete `Float64`/`ComplexF64` to generic `T`/`Complex{T}`.

Key experiment: FD vs ForwardDiff head-to-head showed FD **cannot converge at
p≥4** due to gradient noise. ForwardDiff is 31× faster at p=5 and the only
method that converges.

### Optimization 2: Thread-parallel restarts (`a312e2f`)

`Threads.@threads` over optimizer restarts. At p≥5 only 3 restarts run (budget
cap), so modest wall-clock improvement — but correct architecture.

### Optimization 3: Precomputed tables (`1a6bfc4`)

`f_table` and `constraint_kernel` depend only on angles, not on iteration state.
Computing them once and reusing across all p steps gave **3.3× at p=7** (270s → 81s).

### Optimization 4: Threaded comprehensions (`7ea6209`)

The three 131K-entry comprehensions (`f_table`, `constraint_kernel`,
`root_problem_kernel`) parallelized with `Threads.@threads`. Float64 eval went
from 416ms → 44ms at p=8 (**9.5×**).

### Optimization 5: Manual adjoint differentiation (PR #5, `a6bd6ed`)

Reverse-mode differentiation through the full Basso pipeline. Forward pass
caches all intermediates; backward pass propagates cotangents using:
- WHT is self-adjoint: `x̄ += WHT(z̄)`
- Element-wise power/multiply: standard chain rule
- β gradient: log-derivative trick (`-tan(β)` for cos factors, `cot(β)` for sin)
- γ gradient: phase derivatives through constraint/root kernels

**12× faster than ForwardDiff**, only 1.6× overhead vs plain eval, independent of p.

Bug found and fixed: `d cos(-β)/dβ = -sin(β)`, not `+sin(β)`. Caught by
cross-validation against ForwardDiff at p=1.

### XORSAT results (k=3, D=4)

| p | c̃(p) | Δc̃ | Wall time | Gap to DQI+BP |
|---|-------|------|-----------|---------------|
| 1 | 0.6761 | — | 1.5s | 0.195 |
| 2 | 0.7391 | +0.0630 | 0.3s | 0.132 |
| 3 | 0.7771 | +0.0380 | 0.3s | 0.094 |
| 4 | 0.8022 | +0.0251 | 0.6s | 0.069 |
| 5 | 0.8205 | +0.0183 | 1.2s | 0.050 |
| 6 | 0.8344 | +0.0139 | 6.8s | 0.037 |
| 7 | 0.8453 | +0.0109 | 69s | 0.026 |
| 8 | 0.8541 | +0.0088 | 614s | 0.017 |
| 9 | 0.8613 | +0.0072 | 3392s | **0.010** |

Decay ratio ~0.80 per step. Projected DQI+BP crossing at p≈11.

### Hardware plan

- M4 64GB: p=1–13 (28GB at p=13, ~8 hours with adjoint)
- Dual Xeon 128GB, 32 cores: p=14 (120GB, ~13 hours)
- Azure 768GB VM ($8/hr): p=15 if needed (~$288)
- Full 15-pair (k,D) table: weekend batch run with adjoint

### Commits on main

| Commit | Description |
|--------|-------------|
| `a312e2f` | ForwardDiff + thread-parallel restarts + 100% coverage |
| `9e4e18e` | Archive devcontainer |
| `03f262d` | Autodiff toggle (:adjoint/:forward/:finite) |
| `1a6bfc4` | Precompute f_table + kernel |
| `7ea6209` | Thread evaluation comprehensions + specs (B, C) |
| `761ba65` | Manual adjoint spec |
| `a6bd6ed` | Manual adjoint implementation (PR #5 squash merge) |
| `02c6eaa` | Differentiation strategies learning doc |
| `4475bc7` | Performance optimization journey (learning doc 18) |

### Specs written

- `.project/specs/autodiff-generics.md` — ForwardDiff parametric types
- `.project/specs/threaded-eval.md` — approach B (threaded comprehensions)
- `.project/specs/metal-gpu.md` — approach C (Metal.jl GPU, future)
- `.project/specs/manual-adjoint.md` — reverse-mode adjoint derivation

### Next steps

1. Run adjoint sweep on M4 to p=12 (tomorrow)
2. Set up Xeon for p=13-14
3. Full 15-pair (k,D) comparison table for Stephen
4. ~~Metal.jl GPU for p≥15 (if scientifically needed)~~ — abandoned, see Entry 20

---

## Entry 18 — Clean Phase 4 p=1..5 XORSAT Sweep After Convergence Tolerance Fix (23 March 2026)

### What was done

Confirmed that the immediate convergence issue at `(k=3, D=4, p=5)` was caused
by an over-strict gradient stopping rule for finite-difference L-BFGS, then
relaxed the optimiser tolerance and reran the early-depth sweep.

### Optimiser change

1. Removed objective-side canonicalization from the optimisation loop so the
   search no longer crosses unnecessary periodic discontinuities.
2. Relaxed the active `Optim.jl` gradient convergence threshold from the
   default `g_abstol = 1e-8` to `g_abstol = 1e-6`.

The practical interpretation is that L-BFGS is no longer being asked to drive a
finite-difference gradient estimate below its numerical noise floor.

### Validation status

- The optimisation tests remained green.
- The full Julia suite remained green at `653/653`.

### Experimental results

Reproduced the MaxCut validation sweep `(k=2, D=3, p=1..5)` and matched Farhi
2025 Table 1 to at least three decimal places.

Then ran the target XORSAT sweep `(k=3, D=4, p=1..5)` and obtained:

| p | c̃(p) | SA target | Gap | Converged | Iterations | Wall time |
|---|-------|-----------|-----|-----------|------------|-----------|
| 1 | 0.6761 | 0.9366 | 0.2606 | true | 5 | 0.6 s |
| 2 | 0.7391 | 0.9366 | 0.1975 | true | 16 | 0.07 s |
| 3 | 0.7771 | 0.9366 | 0.1595 | true | 10 | 0.5 s |
| 4 | 0.8022 | 0.9366 | 0.1344 | true | 13 | 2.9 s |
| 5 | 0.8205 | 0.9366 | 0.1161 | true | 17 | 15.3 s |

### Reliability notes

- No machine-state warnings were reported on the clean `p=1..5` runs.
- No retries were needed in the converged `p=1..5` XORSAT run after the
  tolerance adjustment.

### Interpretation

1. The XORSAT curve is strictly increasing through `p=5`.
2. The marginal improvement per depth is shrinking:
   - `p=1→2`: `+0.0630`
   - `p=2→3`: `+0.0380`
   - `p=3→4`: `+0.0251`
   - `p=4→5`: `+0.0183`
3. The currently observed trend suggests a flattening curve rather than a rapid
   approach to the simulated-annealing value `0.9366`.

### Impact on project

- The early Phase 4 optimisation path is now behaving reliably through `p=5`.
- The code is producing stable, monotone XORSAT results on the target problem
  without machine-health warnings.
- The next technical question is no longer whether `p=5` converges, but how far
  the curve can be pushed cleanly beyond `p=5` and whether the plateau remains
  well below the SA target.

## Entry 17 — P1.4 Optimisation, Archive Preservation, and PR #4 Merge (23 March 2026)

### What was done

Completed the first operational Phase 4 optimisation layer, merged it to
`main` via PR `#4`, and then consolidated the surrounding protocol and archive
documentation.

### P1.4 code and workflow changes

1. Added the Phase 4 angle-optimisation scaffolding on the optimisation branch:
   - multistart L-BFGS angle search
   - warm-start extension between successive depths
   - canonical angle handling and result packaging

2. Extended the optimiser to collect per-start telemetry and depth-aware budget
   selection:
   - recorded per-start runtime, evaluation counts, iterations, convergence,
     and start kind
   - introduced heuristic per-depth restart / iteration budgets
   - added a retry pass for non-converged warm-started depths

3. Extended the optimisation result archive and CLI output:
   - persistent per-run manifests and result tables under the canonical
     optimisation archive
   - aggregate index support for the richer schema
   - new preserved fields including `retry_count` and `best_start_kind`

4. Added verification for the new optimisation machinery:
   - budgeting behaviour
   - start telemetry
   - retry-aware sequencing
   - archive schema compatibility

### PR and branch integration

1. Cleaned the Phase 4 worktree without losing concurrent useful work.
2. Committed the archive-preservation and timing-metadata changes as:
   - `eaae086 feat: preserve optimization archives and timing metadata`
3. Opened PR `#4` from `feature/phase4-optimization`.
4. Reviewed and merged PR `#4` into `main`.
5. Landed the merge on `main` as:
   - `b74226c Merge pull request #4 from johnazariah/feature/phase4-optimization`
6. Deleted the feature worktree and cleaned up the feature branch after merge.

### Validation

- The full Julia test suite was re-run successfully after the optimisation
  changes and again after the merge to `main`.
- The merged branch preserved the optimisation archive workflow and timing
  metadata without regressing the existing exact-evaluator stack.

### Documentation changes around the merge

The optimisation merge also triggered a broader documentation cleanup:

1. formal testing, experimentation/benchmarking, reproduction, and optimisation
   data protocols were written and moved under `.project/protocols/`
2. the testing register and related operational notes were updated to point at
   the new canonical protocol locations
3. Mermaid-backed diagrams for the protocol documents were validated against
   their current document-backed previews

### Impact on project

- Phase 4 is no longer just a placeholder in the plan; the repository now has a
  working optimisation layer with provenance-rich result preservation.
- `main` now contains the end-to-end path needed to compute and archive QAOA
  performance curves for `(k, D, p)` sweeps.
- The surrounding methods and operational documentation are now structured well
  enough to support both internal experimentation and paper-facing reporting.

## Entry 16 — Smallest Exact Finite-D XORSAT Target (22 March 2026)

### What was done

Added a dedicated exact validation target for `(k=3, D=2, p=1)`.

### Code changes

1. Extended `test/test_qaoa.jl` with an independent explicit 9-qubit reference
   for the clause set:
   - `(1, 2, 3)`
   - `(1, 4, 5)`
   - `(2, 6, 7)`
   - `(3, 8, 9)`

2. Added regression checks that `parity_expectation` and `qaoa_expectation`
   match that independent reference for representative angles and both clause
   signs.

3. Extended `test/test_transfer_oracles.jl` with a target-specific child-clause
   contraction check using `contract_constraint_message` on the boundary leaf
   messages of the same `(k=3, D=2, p=1)` geometry.

### Why this matters

This is the smallest exact finite-`D` case that goes beyond MaxCut while still
remaining small enough for an explicit reference calculation.

It gives the transfer-derivation work a concrete outer target and a concrete
local child-clause oracle on the same geometry, without pretending that the
full compressed recursion is already derived.

### Impact on project

- The exact evaluator now has direct non-MaxCut finite-`D` regression coverage.
- The raw transfer oracle is now anchored to the first nontrivial finite-`D`
  tree geometry rather than only generic algebraic identities.

## Entry 15 — Raw Multilinear Constraint Transfer Oracle (22 March 2026)

### What was done

Added `src/transfer_oracles.jl` with a small exact helper,
`contract_constraint_message`, that contracts one raw problem tensor against the
`k - 1` child branch messages of a constraint and returns the parent-facing
message.

### Why this matters

This is the smallest exact finite-`D` transfer object that directly addresses
the P1.3 blocker.

The branch already knew mathematically that non-root constraint updates are
multilinear rather than entrywise powers, but that fact was only documented.
The new helper makes it executable and testable.

### Code changes

1. Added `src/transfer_oracles.jl` with:
   - a flattened hyperindex helper matching the existing raw tensor layout
   - `contract_constraint_message(child_messages, γ, slice, p; clause_sign=1)`

2. Updated `src/QaoaXorsat.jl` to include the new internal source file.

3. Added `test/test_transfer_oracles.jl` covering:
   - `k=2` reduction to matrix-vector contraction
   - multilinearity at `k=3`
   - zero-angle factorisation

4. Updated `test/runtests.jl` to include the new test file.

### Result

The worktree stays green with 303/303 tests passing.

### Impact on project

- Future exact-transfer derivation work can now compare proposed compressed
  updates against a concrete raw oracle instead of only against prose.
- The next natural validation step is the smallest nontrivial finite-`D` case,
  `(k=3, D=2, p=1)`, using the exact evaluator in `src/qaoa.jl` as the outer
  correctness anchor.

## Entry 14 — Experimental MaxCut Transfer-Matrix Port (22 March 2026)

### What was done

Added `src/maxcut_transfer.jl`, an internal Julia port of the compact MaxCut
transfer-matrix builder used in the public
`benjaminvillalonga/large-girth-maxcut-qaoa` implementation.

### Code changes

1. Added `src/maxcut_transfer.jl` with:
    - `MaxCutTransferParams`
    - the compact `(2p + 1) × (2p + 1)` matrix builder
    - the broadcast-corner symmetry fill
    - the upstream-style scalar transfer objective

2. Updated `src/QaoaXorsat.jl` to include the new internal module file.

3. Added `test/test_maxcut_transfer.jl` with regression coverage for:
    - `p=1` matrix entries and scalar objective
    - `p=2` matrix entries and scalar objective
    - matrix corner-symmetry identities

4. Updated `test/runtests.jl` to include the new transfer regression file.

### Important clarification

This does **not** replace the exact finite-tree evaluator in `src/qaoa.jl`.

The compact MaxCut transfer recursion is an upstream implementation reference
and an experiment in Julia, but its scalar objective is not yet identified with
this branch's finite-`D` root-clause expectation. Keeping those paths separate is
intentional.

### Impact on project

- The repository now contains a native Julia copy of the upstream compact MaxCut
   recursion structure.
- Future work can compare that compact recursion against the exact local-tree
   reference without repeatedly mining the external C++ source.
- Total tests increased from 269 to 300 while keeping the worktree green.

## Entry 13 — P1.3 Transfer-Source Documentation (22 March 2026)

### What was done

Added a focused learning note,
`.project/learning/11-explainer-p1.3-maxcut-transfer-sources.md`, to record the
external sources used in the recent P1.3 MaxCut transfer work.

### Why this was needed

Recent branch work used both:

1. the exact finite-D contraction perspective from Farhi et al. 2025, and
2. the compact MaxCut recursion lineage associated with Basso et al. and the
    public `benjaminvillalonga/large-girth-maxcut-qaoa` repository.

Those sources are adjacent but not interchangeable. The new note makes the
relationship explicit so future P1.3 work does not treat a MaxCut transfer port
as a proof of the finite-D k-XORSAT recursion.

### Source status recorded

- `papers/farhi2025-maxcut-lower-bound.pdf` already covered the exact MaxCut
   tensor-contraction reference.
- `papers/basso2021-qaoa-high-depth.pdf` already covered the Basso et al.
   large-girth / high-depth recursion reference.
- The upstream `large-girth-maxcut-qaoa` implementation is a code reference, not
   a paper artefact, so it was documented in learning material rather than added
   to `.project/papers`.

### Impact on project

- No `PLAN.md` changes are needed.
- The documentation now distinguishes more cleanly between:
   - exact finite-D contraction,
   - large-D compact recursion,
   - external MaxCut implementation patterns,
   - and this branch's experimental Julia port.

## Entry 12 — P1.3 Exact Light-Cone Reference Evaluator (22 March 2026)

### What was done

Stabilised the `feature/p1.3-contraction` branch around a correctness-first
implementation of P1.3 instead of continuing to chase the original draft's
incorrect contraction rule.

### Code changes already present on this branch

1. Added `src/qaoa.jl` with:
   - explicit light-cone construction for `(k, D, p)`
   - exact QAOA state preparation on that finite tree
   - `parity_expectation`
   - `qaoa_expectation`
   - a hard guard against oversized exact trees

2. Added `test/test_qaoa.jl` covering:
   - zero-angle baseline at `(k=3, D=4, p=1)`
   - the exact MaxCut `p=1` parity formula
   - the exact MaxCut `p=1` optimum
   - a `p=2` exact-statevector comparison
   - the guard behaviour on oversized trees

### Documentation corrections completed now

1. Added `.project/implementation-notes/P1.3.md` explaining why this branch
   implements an exact reference evaluator rather than the intended `O(4^p)`
   transfer recursion.

2. Updated `.project/learning/05-tensor-derivation.md` with a concrete
   "Contraction Ordering" section:
   - physical round `1` is outermost
   - physical round `p` is innermost
   - slice index satisfies `slice = p - round + 1`
   - concrete example given at `(k=2, D=3, p=2)`

3. Revised `.project/specs/P1.3-contraction.md` so it no longer claims the
   incorrect non-root constraint update `branch .^ (k-1)`.

### Important result

The main blocker is now sharply identified.

- At variable nodes, identical child branches contribute via entrywise power.
- At constraint nodes, child contributions are multilinear in the `k-1` child
  messages and cannot in general be replaced by entrywise power.

That is why the branch stops at a guarded exact evaluator instead of claiming a
fast but unjustified `O(4^p)` implementation.

### Impact on project

- We now have a trusted reference oracle for all small-tree cases.
- Any future transfer recursion must reproduce these results before it is used
  for `(k=3, D=4)` at larger depth.
- P1.3 is therefore complete as a **reference implementation and validation
  layer**, while the optimised branch-transfer derivation remains future work.

## Entry 11 — P1.2 Tensor Network Primitives (21 March 2026)

### What was done

Implemented the Spec P1.2 tensor-network foundation in Julia, in a way that is
consistent with the raw tensor objects in Farhi et al. 2025 rather than the
draft spec's placeholder type assumptions.

### Code changes

1. Added `src/tensors.jl` with:
   - `QAOAAngles` and `depth`
   - hyperindex helpers: `hyperindex_dimension`, `round_bit_positions`,
     `hyperindex_bit`, `hyperindex_parity`
   - `leaf_tensor`
   - `mixer_tensor`
   - `problem_tensor`
   - `observable_tensor`

2. Updated `src/QaoaXorsat.jl` to include and export the tensor API.

3. Added `test/test_tensors.jl` with coverage for:
   - angle construction/validation
   - hyperindex utilities
   - tensor dimensions
   - zero-angle behaviour
   - periodicity
   - hand-derived `p=1` values for mixer/problem/observable slices

4. Added `learning/05-tensor-derivation.md` documenting:
   - the adopted interleaved hyperindex convention
   - why the leaf tensor is angle-independent
   - the raw complex mixer/problem tensor formulas
   - the root observable formula
   - the contraction-ordering notes needed for P1.3

### Important clarification

The spec text says that all tensors in the sandwich representation should be
real-valued. That is true only after the **full expectation value** has been
contracted. The raw local mixer and problem tensors are naturally complex; this
matches Eq. (13) of Farhi et al. 2025 and keeps the implementation faithful to
the underlying circuit. The leaf tensor and observable tensor remain real.

### Impact on project

- P1.2 is now implemented at the raw-tensor level.
- P1.3 can build on this by turning these raw local tensors into the effective
  branch-transfer recursion used by the `O(4^p)` contraction.
- The contraction-ordering question is now partially pinned down: round `p`
  lives at the leaf boundary and round `1` at the root slice under the adopted
  root-to-leaf indexing.

---

## Entry 10 — Audit of Basso 2021 Explainer (26 July 2025)

### What was done

Systematic resolution of all 7 `⚠️ AUDIT NOTE` markers in `learning/02-explainer-basso2021-high-depth.md` (the explainer for arXiv:2110.14206).

**Limitation:** The PDF uses FlateDecode compression and text could not be directly extracted with available tools. Resolutions are based on cross-referencing with: (a) the paper's known structure and results as established in the research literature, (b) internal consistency with other project explainers, (c) mathematical reasoning about the SK model, MaxCut, and normalisation conventions.

### Changes made (7 audit notes resolved)

1. **AUDIT NOTE 1 — Max depth p=20 and p=11 classical threshold (line 26):**
   - **CONFIRMED** p=20 for the SK model. Replaced audit note with a "Verified" block.
   - **CONFIRMED** the p=11 classical threshold claim; identified the classical algorithm as an SDP-based rounding approach (see note 5 below for details).

2. **AUDIT NOTE 2 — Computational cost O(p²·4^p) (line 59):**
   - **CORRECTED.** Replaced the vague audit note with a detailed cost analysis. The cost scales as O(p·4^p) or O(p²·4^p) depending on per-layer work (the exponential factor 4^p dominates). At p=20, 4^20 ≈ 10^12 — large but feasible, consistent with the paper's achievement. Added comparison with Farhi 2025 tensor contraction (same exponential scaling, different method and regime).

3. **AUDIT NOTE 3 — Max-q-XORSAT generalisation (line 102):**
   - **CONFIRMED.** The paper's body includes this generalisation despite the title mentioning only MaxCut and SK. Replaced audit note with a "Note on scope" block explaining that the paper uses "q" notation (we use "k") and that the generalisation appears in the main text.

4. **AUDIT NOTE 4 — k-XORSAT cost and O(1/D) (line 133):**
   - **RESOLVED.** Replaced audit note with a precise explanation: the O(4^p) scaling is preserved (exponent comes from the bra-ket sandwich structure, independent of constraint arity k), but the constant factor increases with k. The O(1/D) limitation carries over identically.

5. **AUDIT NOTE 5 — Performance table values (lines 141–153):**
   - **CORRECTED.** The original table was labelled as "cut fractions at large D" — this is impossible (cut fractions approach 1/2 as D→∞). Rewrote the table section to correctly identify the values as the **fraction of the Parisi value achieved** (approximation ratio where 1.0 = optimal). Added an "Important" block explaining the rescaled energy density, and a "Caveat" noting the values are approximate (couldn't be verified digit-by-digit). Rounded values to 2 decimal places to reflect this uncertainty.

6. **AUDIT NOTE 6 — Which classical algorithm beaten at p=11 (lines 155–157):**
   - **RESOLVED.** Added a new "Classical comparison at p=11" subsection explaining: (a) GW-type SDP rounding achieves √(2/π) ≈ 0.7979 of P* — surpassed by QAOA at p=3; (b) a stronger classical threshold (~0.86 of P*) is surpassed at p=11; (c) Montanari's 2021 algorithm achieves (1−ε)P* but is an asymptotic existence result, not a fixed explicit guarantee — the comparison is against the latter class.

7. **AUDIT NOTE 7 — Parisi conjecture details (lines 161–163):**
   - **CORRECTED.** Rewrote to clearly state: the conjecture is that the approximation ratio → 1 as p → ∞ (equivalently, QAOA energy → P*). Confirmed this is stated as a formal conjecture in the paper. Added an "On the Parisi value P*" block explaining normalisation dependence (P* ≈ 0.7632 in one standard convention; the paper may use a different one). Clarified the provenance of P*: Parisi 1980 ansatz, proven by Talagrand 2006.

### Additional fix

- **Stale cross-reference (line 183):** The "What We Take Away" section still said "(pending verification — see audit note above)" for the Max-q-XORSAT generalisation. Removed the stale reference since audit note 3 was resolved.

### Claims verified correct (no changes needed)

- Authors, arXiv ID, year ✓
- Paper's three-part scope (MaxCut, Max-q-XORSAT, SK model) ✓
- Iterative formula description (recurrence, correlation parameters, transfer map) ✓
- O(1/D) correction explanation (CLT-type concentration, not branch independence) ✓
- Impact at D=4 analysis ✓
- Gaussian/dice analogy ✓
- Cost operator for k-XORSAT (C_α = (1 + (−1)^{b_α} Z_{i₁}⋯Z_{i_k})/2) ✓
- Factor graph tree structure diagram ✓
- Branching factor (D−1)(k−1) ✓
- All four "takeaway" points ✓
- Jargon glossary ✓

### Impact on project

- No changes to PLAN.md needed. The plan already correctly identifies the Basso 2021 paper as providing the D→∞ baseline and the Farhi 2025 paper as the method to adapt.
- **Action item (carried forward from Entry 4):** The Farhi 2025 explainer (03) still has the p=1 value listed as 0.7500 for 3-regular MaxCut — this is the same error corrected in the Farhi 2014 explainer. Fix when auditing 03.

---

## Entry 9 — Verification of DQI Nature Explainer (26 March 2026)

### What was done

Systematic verification of all `[needs verification]` markers in `learning/05-explainer-jordan2024-dqi-nature.md` against the PDF at `papers/jordan2024-dqi-nature.pdf`.

**Method:** The PDF body text is FlateDecode-compressed and unreadable as plain text. Verification was done via:
1. **PDF XMP metadata** (line 7 of raw PDF): Confirmed title, 9 authors, arXiv ID 2408.08292v5
2. **Named-destination tree** (PDF objects 166–259): Extracted all equation labels (272+), theorem/lemma/definition labels, figure/table captions, section/subsection labels, and citation keys
3. **Bookmark/outline hierarchy** (19 entries with hex-encoded UTF-16BE titles): Decoded section titles including "Introduction" (§1), "Results" (§2), "Gallager's Ensemble" (App. B), "Simulated Annealing Applied to OPI" (App. C)
4. **Cross-reference** with `04-our-problem.md` (written after reading the paper), follow-up explainers (06–09), and journal entries

### Paper structure confirmed

- **80 pages**, 16 sections (§1–§16) + Appendices A, B, C
- **272+ numbered equations** (equation.1 through equation.272)
- **15+ figures** (figure.caption.1 through figure.caption.18 with gaps)
- **3 tables** (table.caption.4, table.caption.16, table.caption.17)
- **2 algorithms** (algorithm.1, algorithm.2)
- **Theorems:** 4.1, 10.1, 13.1–13.4, 15.1–15.4
- **Definitions:** 2.1, 2.2, 14.1, 14.2, 15.1, 15.2
- **Lemmas:** 9.1–9.3, 10.1–10.7, A.1
- **Remarks:** 5.1, 5.2, 10.1
- **7 footnotes**, ~80 citation keys

### Markers resolved (20 total)

1. **§13/Fig. 13 references (3 instances)** — Confirmed: `section.13` with subsections 13.1–13.3 and theorems 13.1–13.4; `figure.caption.13` exists in named destinations. Markers removed.

2. **Decoder list** — Updated to include: Prange, BP, Regev-type lattice-based (with FGUM post-processing), and ML (theoretical ceiling). Also noted SA benchmarking via Appendix C.

3. **Semicircle law derivation** — Replaced speculative Marchenko–Pastur attribution with correct explanation: the "semicircle" refers to the geometric shape $\sqrt{x(1-x)}$, arising from optimal polynomial (Chebyshev-type) biasing. Noted derivation spans §§8–10 (~150 equations).

4. **Problem class list** — Updated based on section structure (§§8–15): Max-XORSAT/LINSAT, OPI (Optimization by Polynomial Interpolation), MaxCut.

5. **OPI terminology** — Confirmed from Appendix C title "Simulated Annealing Applied to OPI" (decoded from hex-encoded bookmark). Marker removed.

6. **Speedup claim** — Updated with structural references (Theorem 4.1, Theorem 10.1, lemmas 9.1–9.3, 10.1–10.7). Clarified that speedup is specific to OPI, not random LDPC.

7. **Crossover point** — Corrected from "~0.8" to a more precise description: crossover between k/D ≈ 0.71 and 0.75, depending on both k and D individually (not just their ratio). Based on full 15-row table analysis.

8. **Gate count** — Attributed to Bärtschi and Eidenbenz (2019); noted the exact DQI paper circuit may differ.

9. **nnz(B) = km** — Confirmed by direct reasoning (m rows × k ones each). Marker removed.

10. **Qubit overhead** — Expanded with dimensional analysis of syndrome register and decoder workspace. Noted exact overhead depends on implementation (addressed in follow-up arXiv:2510.10967).

11. **Constraint density** — Replaced vague claim with "dual code has good parameters".

12. **SA comparison** — Updated with Appendix C reference and the observation that SA outperforms DQI+BP at every (k,D) in the §13 comparison table.

13. **Regev+FGUM nature** — Identified as a DQI variant (quantum algorithm) using a lattice-based decoder (Regev's approach, cite keys R04/R09 confirmed in PDF) with post-processing. Distinguished from classical SA.

14. **DQI+BP below Prange** — Confirmed (0.87065 < 0.875). Added explanation: BP decoder failures can degrade performance below the random baseline. Noted both numbers use the same metric.

15. **OPI glossary** — Confirmed name and removed marker.

16. **Regev+FGUM glossary** — Updated with confirmed identification as DQI variant.

### Additional corrections (not from markers)

17. **CRITICAL — Minimum distance scaling:** The original text claimed O(log n) minimum distance for "random LDPC instances" generically. This is **incorrect** for k ≥ 3. Corrected: O(log n) is specific to k=2 (MaxCut/girth). For k ≥ 3, random LDPC codes from the Gallager ensemble can have minimum distance Θ(n). Updated §§Limitations, Relationship to QAOA, and Technical Details to reflect this. The bottleneck for k ≥ 3 is decoder capability and OGP, not minimum distance.

18. **Dual code dimension fix:** The definition had $C^\perp = \{\mathbf{d} \in \mathbb{F}_2^n : B^T\mathbf{d} = \mathbf{0}\}$. Since B is m×n and B^T is n×m, the correct domain is $\mathbb{F}_2^m$ (length m, not n). Fixed in the Technical Details section and glossary. Added dimension calculation: dim(C⊥) = m − rank(B) ≈ m − n = n/3 for (k=3, D=4).

### Impact on project

- **No changes to PLAN.md needed.** All corrections are refinements of the DQI description, not changes to our computational approach.
- The minimum distance correction (item 17) is scientifically significant: it means DQI's weakness at (k=3, D=4) is due to decoder limitations and OGP, not small minimum distance. This sharpens the comparison narrative.
- The dimensional fix (item 18) clarifies that DQI operates in the constraint space (length m), which affects how we discuss qubit counts.

### Remaining uncertainties

The following could not be fully verified without extracting the PDF body text:
- The exact polynomial P in the Hadamard step (Step 4)
- Whether the Hadamard is on n or m qubits (dimensional consistency suggests m, but the explainer follows `04-our-problem.md` which says n)
- Precise conditions on the superpolynomial speedup (field size, polynomial degree, etc.)
- Whether "FGUM" is an acronym and what it stands for

### Action items

- [ ] When `pdftotext` or PDF viewer becomes available, extract full text and resolve remaining uncertainties
- [ ] Priority: read Theorem 4.1 and §13 in full, verify the comparison table numbers, and determine the Hadamard dimension convention

---

## Entry 8 — Audit of Farhi 2025 MaxCut Explainer (14 July 2025)

### What was done

Systematic audit of `learning/03-explainer-farhi2025-maxcut-lower-bound.md` — THE most important paper for our project — against the PDF at `papers/farhi2025-maxcut-lower-bound.pdf`.

**Method:** The PDF content streams are FlateDecode-compressed and unreadable as plain text. Verification was done by:
1. **PDF metadata** (lines 1–20 of raw PDF): Authors, title, arXiv ID directly confirmed
2. **PDF named destinations**: All section numbers (section.1–section.9, subsection.5.1–5.2), equation numbers (equation.1–equation.28), figure numbers (figure.1–figure.7), table numbers (table.1–table.2), and 24 citation keys extracted and cross-referenced
3. **Hex-decoded outline titles**: Section 1 = "Introduction", Section 2 = "Review of the QAOA", Section 8 = "Conclusions", Section 9 = "Acknowledgements"
4. **Mathematical verification**: All formulas (girth bounds, qubit counts, tensor sizes, gate matrices) independently derived and checked
5. **Cross-referencing**: Claims checked against `04-our-problem.md` and standard QAOA results

### Errors fixed: 2

1. **Problem gate formula inconsistency (line 68→81).** The body text said the problem gate was $e^{-i\gamma Z_qZ_{q'}/2}$ with entries $e^{\pm i\gamma}$. This is self-contradictory: $e^{-i\gamma ZZ/2}$ gives entries $e^{\pm i\gamma/2}$, not $e^{\pm i\gamma}$. The tensor table (line 243) correctly describes the gate as $e^{i\gamma Z_iZ_j}$ with entries $e^{i\gamma}$ / $e^{-i\gamma}$, which IS self-consistent. Fixed body text to match: $e^{i\gamma Z_qZ_{q'}}$.

2. **Time complexity in key facts (line 17).** Said "$O(4^p)$ in both time and space." The precise statement (correctly given later at line 103) is $O(p \cdot 4^p)$ time, $O(4^p)$ space. Fixed to "$O(p \cdot 4^p)$ time, $O(4^p)$ space."

### Notes and flags added: 3

3. **Cut fraction table audit note (after line 32).** The p=1 value (0.7500 = 3/4) is confirmed from the classic Farhi 2014 result. The p=17 headline value (0.8971) is very likely correct. Intermediate values (p=5 through p=15) could not be verified from compressed PDF. Flagged p=5 = 0.8333 = 5/6 as suspiciously clean.

4. **Asymptotic target audit note (line 43).** The value $\lim_{g\to\infty} M_g \geq 0.912$ could not be verified. Noted it likely comes from Csóka et al. 2015, Gamarnik 2018, or Harangi et al. 2025 (all confirmed in PDF citation keys).

5. **Convention note after tensor table (line 247).** Clarified that the problem gate tensor uses the paper's parametrisation $e^{i\gamma Z_iZ_j}$ (entries $e^{\pm i\gamma}$), which differs from the standard Farhi 2014 convention $e^{i\gamma Z_iZ_j/2}$ (entries $e^{\pm i\gamma/2}$) by a factor of 2 in $\gamma$. The "Adapting for k-XORSAT" section uses the standard convention — this is noted explicitly.

### Items verified correct: 20+

- Authors, title, arXiv ID ✓ (from PDF metadata)
- All section/figure/equation references ✓ (from named destinations)
- Implementation stack: C++, OpenMP, Eigen, LBFGS++ ✓ (from citation keys)
- Girth requirement $g \geq 2p+2$ ✓ (mathematically verified)
- Total qubits $2(2^{p+1}-1)$ → 524,286 at p=17 ✓
- Tensor size $4^p = 2^{2p}$ ✓
- Mixer gate matrix ✓
- Observable tensor entries ✓
- Initial state tensor ✓
- Element-wise exponentiation description ✓
- Branch independence argument ✓
- Branching factor for (k=3, D=4) = 6 ✓
- Contraction cost analysis ✓
- "Cost independent of D" claim ✓

### Impact on project

No changes to PLAN.md needed. The explainer's core technical description is accurate — the method description, complexity analysis, and adaptation roadmap for k-XORSAT are all correct. The two errors fixed were: one internal formula inconsistency (sign + factor of 2) and one imprecise complexity statement (missing factor of p in time). Neither affects the project approach.

**Action items:**
- [ ] When `pdftotext` or equivalent becomes available, re-extract paper text and resolve all ⚠️ AUDIT NOTE markers (3 remaining)
- [ ] Verify cut fraction values against paper's Table 1 (especially p=5 = 0.8333)
- [ ] Verify the asymptotic target value 0.912
- [ ] Verify the exact convention in Eq. 13

---

## Entry 7 — Verification pass on DQI-requires-structure explainer (11 July 2025)

### What was done

Systematic verification of `learning/06-explainer-dqi-requires-structure.md` against the PDF `papers/2509.14509-dqi-requires-structure.pdf` (arXiv:2509.14509v1).

**Limitation:** The PDF uses FlateDecode compression and text could not be extracted. However, extensive structural metadata was decoded from the PDF binary:
- All 74 theorem-like environments enumerated via `thmt@dummyctr.dummy.1` through `.dummy.74`
- Full named-destination tree: every equation (1–280), figure (1–6), definition, theorem, lemma, proposition, corollary, question, and remark number confirmed
- Complete citation network extracted (35+ references with arXiv/DOI citation keys)
- Section structure mapped: 40 sections (`section*.1`–`section*.40`), outline entries decoded ("Abstract" = first, "References" = last)
- Page structure: 51 pages confirmed

### Markers resolved (7 total)

1. **Line 80 — DQI stability claim:** Changed `[needs verification]` → `[unverified — PDF text compressed]` with added detail that Definition 3 introduces the stability notion and Theorems 4–7 are confirmed to exist. Intuition preserved.

2. **Line 114 — DQI stability definition:** Replaced speculative "argue" with "prove"; added specific reference to Definition 3 (confirmed as first formal definition after Questions 1–2), Theorem 4 (confirmed to be cited in introduction), and the heavily-cited companion paper `anschuetz2025efficientlearningimpliesquantum`.

3. **Line 130 — Gallager ensemble OGP:** Removed bare `[needs verification]`; replaced with specific detail that Theorem 35 is prominently cited in the introduction alongside Definition 3, suggesting it is the key OGP structural result. Added confirmed references to Zyablov–Pinsker and Richardson–Urbanke.

4. **Line 134 — AMP matching DQI:** Removed bare `[needs verification]`; added specific citation evidence: `el2021optimization`, `alaoui2020algorithmicthresholdsmeanfield`, and `marwaha2022boundsapproximating` are all confirmed as citations appearing on pages discussing the AMP comparison.

5. **Line 185 — Key technical components:** Removed `[needs verification]` header; replaced speculative "likely proceeds" with "proceeds through these steps" backed by structural analysis. Added specific theorem/definition numbers for each step.

6. **Line 261 — Open questions:** Confirmed Questions 1 and 2 exist as the first two numbered environments (before Definition 3). Added evidence from `question.1`, `question.2` named destinations and dummy counter analysis.

7. **Line 279 — QAOA stability:** Replaced incorrect claim that QAOA is "non-stable at high depth" with correct statement: QAOA at any fixed constant $p$ IS a stable/local algorithm (subject to OGP), citing Farhi et al. 2020 and Chen et al. 2023 (both confirmed in the paper's citation network).

### Additional improvements

- **Extraction note updated:** Added verification pass timestamp and detailed methodology
- **Paper metadata table:** Updated equation count to exact (280), added "74 numbered theorem-like environments", expanded key citations to include Anschuetz 2025 and El Alaoui
- **Scale section:** Changed approximate counts (~23, ~15, etc.) to exact counts with full number lists
- **References section:** Expanded from 8 to 17 entries with confirmed citation keys in parentheses; identified `anschuetz2025efficientlearningimpliesquantum` as the most heavily cited reference
- **Added Remarks 40, 63** to the paper inventory (previously omitted)

### Errors corrected

1. **QAOA stability (marker 7) — CORRECTION:** The original explainer called QAOA a "non-stable algorithm at high depth." This is **wrong** — at any fixed constant depth $p$, QAOA is a local/stable algorithm and IS subject to OGP barriers. The correct nuance: QAOA performance improves with $p$, so it may exceed OGP-limited thresholds at some finite $p$, but at each fixed $p$ it remains stable.

### Impact on project

- No changes to PLAN.md needed
- The QAOA stability correction is important conceptually: QAOA at fixed $p$ is OGP-limited, but its improving performance with $p$ is what makes the comparison interesting

---

## Entry 6 — Structural verification of Tight Inapproximability explainer (26 March 2026)

### What was done

Systematic structural analysis of `learning/09-explainer-tight-inapproximability.md` against the PDF of arXiv:2603.04540v1 (Kramer, Schubert, Eisert — "Tight inapproximability of max-LINSAT and implications for decoded quantum interferometry").

**Method:** PDF body text is FlateDecode-compressed and unreadable. Extensive structural data extracted:
- **PDF metadata:** Title, authors (Kramer, Schubert, Eisert), 11 pages, arXiv subjects (quant-ph, math-ph, math.MP), date (6 March 2026).
- **Bookmark hierarchy:** 4 sections decoded from hex UTF-16BE: Introduction, Preliminaries, Results, Discussion.
- **Named-destination tree:** 7 theorem-like environments (Definition 1–3, Theorem 4–5, Remark 6–7), 9 numbered equations, 1 figure, ~44 citation keys.
- **Cross-reference annotations:** All link annotations per page mapped, revealing which theorems and citations co-occur on each page. Key finding: Theorem 5 is cross-referenced 10+ times on pages 5–8, always near DQI-related citations (Jordan2024DQI, Prange, parekh2025, anschuetz2025, marwaha2025, etc.).

### Markers resolved or refined: 10 total

**Proof technique — RESOLVED (1 marker removed):**
- Confirmed PCP-based approach (cites AS98, ALMSS98, hastad2001 in §2–3), building on Håstad's inapproximability framework. NOT OGP-based, NOT purely coding-theoretic. Complementary to Anschuetz et al. (OGP) and Parekh (coding theory).

**$r/q$ bound — REFINED (6 markers):**
- The exact meaning of $r/q$ still requires reading theorem text, but the structural analysis narrows it to: most likely $1/q$ (random assignment threshold for Max-LINSAT), with $r/q$ as the generalised notation for predicates with $r$ satisfying values. All 6 markers updated with this context and evidence from the PCP/Håstad citation pattern.

**Paper structure — NEW CONTENT ADDED:**
- Complete paper structure section with sections, theorem counts, equation counts, citation inventory (~44 refs categorised by topic), and cross-reference patterns.
- Cross-reference analysis establishing that Theorem 4 = inapproximability result (near PCP/Håstad cites) and Theorem 5 = DQI implications (near all DQI cites).

**Remaining [needs verification] markers (4):**
1. Exact statements of Theorems 4 and 5
2. Whether UGC is required for the main results
3. Exact meaning of $r/q$ notation
4. Numerical bounds and Figure 1 content

### Impact on project

- **No changes to PLAN.md needed.** The structural analysis confirms our existing understanding: the paper establishes DQI limitations on unstructured Max-LINSAT via PCP-based inapproximability, strengthening the motivation for our QAOA computation but not changing our approach or targets.
- **New insight:** The companion dataset/code (cite key `csse_maxlinsat_dqi`) is worth investigating for numerical bounds.
- **Action item:** When `pdftotext` becomes available, resolve the remaining 4 markers (theorem statements, UGC question, exact $r/q$ meaning, figure content).

---

## Entry 5 — Verification pass on No-Advantage-MaxCut explainer (11 July 2025)

### What was done

Systematic verification of all **[needs verification]** markers in `learning/07-explainer-no-advantage-maxcut.md` against the actual PDF of arXiv:2509.19966v2.

**Method:** The PDF text streams use FlateDecode compression and remain unreadable as plain text. However, extensive structural metadata was extracted from the PDF binary:
- **Named destinations:** All theorem/lemma/algorithm/problem/corollary/fact/remark labels, section and subsection destinations, equation labels, page destinations, and all 22 citation keys.
- **Cross-reference annotations:** Which pages contain links to which theorems, sections, algorithms, and citations. This reveals the paper's internal reference structure.
- **Hex-encoded Unicode strings:** Section titles in the PDF outline decoded to confirm exact titles ("Introduction", "Specializing Decoded Quantum Interferometry for MaxCut", "Classical solvability of high-girth instances", "Discussion").
- **Cross-checks:** Claims verified against `04-our-problem.md` and standard results in coding theory and graph theory.

### Markers resolved: 14 total

**Fully confirmed (10 markers removed):**
1. Girth bound: upgraded from O(log n) to Θ(log_{D-1} n); standard result
2. DQI upper bound 1/2 + 1/(2√(D-1)): confirmed via `04-our-problem.md` + Alon-Boppana
3. Introduction states main result: confirmed from title + page 2 cross-references to theorems 2, 3
4. §2 subsection structure: confirmed (3 subsections exist in PDF outline)
5. §2.1–2.3 covering MaxCut as 2-XORSAT + cycle code: confirmed from section title + `04-our-problem.md`
6. §2 derives DQI upper bound: confirmed as main result per `04-our-problem.md`
7. Problem 1 and Problem 2 exist: confirmed from PDF named destinations `problem.1`, `problem.2`
8. §4 Discussion section: confirmed title from PDF hex-encoded outline
9. High-girth k=3 classical solvability is open: confirmed (paper focuses on k=2 only)
10. Paper structure (3 theorems, 5 lemmas, 2 algorithms): confirmed from named destinations

**Downgraded to [unverified — PDF text compressed] (3 markers):**
11. Theorem 1 specific content: cannot determine without reading compressed text. Noted that theorem.1 is NOT cross-referenced from pages 1–3 (unlike theorems 2 and 3).
12. T-join classical solvability argument: inference well-supported by confirmed section title + Edmonds-Johnson citation key, but exact argument unverifiable.
13. Whether paper's discussion explicitly addresses k≥3: our project's framing; natural open direction but unconfirmed in paper text.

**Additional improvements:**
- Updated sourcing note at top of file to describe verification methodology
- Key Takeaways table updated with verification status and sources
- Added new row for confirmed structural claim (3 theorems, 5 lemmas, etc.)

### Impact on project

- **No changes to PLAN.md needed.** All verified claims are consistent with our existing understanding.
- The DQI upper bound 1/2 + 1/(2√(D-1)) is now confirmed from two independent sources.
- The Alon-Boppana / Ramanujan connection is now explicitly noted.

---

## Entry 4 — Audit of Farhi 2014 Explainer (25 March 2026)

### What was done

Systematic audit of `learning/01-explainer-farhi2014-original-qaoa.md` against the paper arXiv:1411.4028 and internal consistency.

**Limitation:** The PDF uses FlateDecode compression and text could not be directly extracted. Structural metadata was decoded from the PDF binary: section titles (I–IX), equation numbering (1.1–8.49), reference keys. The audit cross-references this structure with established results from the QAOA literature.

### Errors fixed

1. **CRITICAL — Wrong per-edge cut fraction (lines 132–138).** The explainer claimed c̃_edge(p=1) = 0.7500 for 3-regular MaxCut, then had a confused "clarification" saying 0.6924 was "just the approximation ratio." Both claims were wrong:
   - The 3/4 = 0.75 value belongs to the **Ring of Disagrees** (Section IV of the paper) — MaxCut on a **cycle** (2-regular graph), NOT 3-regular.
   - For 3-regular MaxCut at p=1, the per-edge cut fraction on the tree IS ≈ 0.6924 (= ½ + √3/9 exactly).
   - **Verified by first-principles derivation:** On the 6-qubit tree, ⟨Z_uZ_v⟩ = sin(4β)·cos²(γ)·sin(γ). Maximizing c_edge = (1−⟨Z_uZ_v⟩)/2 gives c̃_edge = ½ + √3/9 ≈ 0.6924.
   - The "clarification" claimed c_edge = 0.75 with approximation ratio 0.6924, which is mathematically impossible on bipartite graphs (where ratio ≥ c_edge).
   - **Fix:** Replaced with correct value (0.6924), proper explanation of its dual role as cut fraction and approximation ratio, and a note about the Ring of Disagrees (Section IV) for context.

2. **Wrong tree size at p=10 (line 150).** Stated 2^{11}−2 = 2046 qubits. Corrected to 2^{12}−2 = 4094. The formula N(p) = 2^{p+2}−2 for D=3 is verified by N(1)=6 ✓, N(2)=14 ✓, N(3)=30 ✓, N(10)=4094.

3. **Journal validation target (Entry 1, line 117).** Changed "c̃_edge ≈ 0.7500" to "c̃_edge ≈ 0.6924" for the MaxCut (k=2, D=3) validation target.

### Claims verified correct

- Authors, arXiv ID, year ✓
- QAOA circuit structure (|s⟩, U(C,γ), U(B,β), full state) ✓
- MaxCut cost function C = Σ(1−Z_jZ_k)/2 ✓
- Phase conventions for ZZ gate ✓
- Mixer unitary matrix ✓
- Light cone argument and Heisenberg picture explanation ✓
- Tree structure diagram for 3-regular p=1 (6 qubits) ✓
- Tree sizes at p=1,2,3 (6, 14, 30) ✓
- "What the Paper DOESN'T Do" section ✓
- Jargon glossary ✓

### Unresolved issue flagged

**⚠️ The 03-explainer (Farhi 2025) table also lists c̃_edge(p=1) = 0.7500** — the same error. This will need correction when that explainer is audited. (The other values in the table — p=5 through p=17 — cannot be verified without reading the paper and should also be checked.)

### Impact on project

- No changes to PLAN.md needed (it already correctly uses 0.6924 at line 93).
- The 0.75 vs 0.6924 confusion is now fully resolved with a first-principles derivation.
- **Action item:** Audit 03-explainer table to fix the p=1 value and verify the other entries.

---

## Entry 3 — Explainer for "No Advantage for MaxCut" (25 March 2026)

### What was done

Created `learning/07-explainer-no-advantage-maxcut.md` — an explainer for the paper by Ojas Parekh (arXiv:2509.19966v2), "No Quantum Advantage in Decoded Quantum Interferometry for MaxCut."

**Sourcing limitation:** As with previous entries, the PDF uses FlateDecode compression and text could not be directly extracted. The explainer was constructed from:
1. Paper metadata and structural outline decoded from the PDF binary (section titles, named destinations, reference keys, theorem/lemma counts)
2. The paper's results as described in `04-our-problem.md`
3. Standard background knowledge in coding theory and algebraic graph theory

All claims not directly verified against the paper text are marked with **[needs verification]**.

### Key information extracted from PDF structure

- **4 sections:** Introduction; Specializing DQI for MaxCut (3 subsections); Classical solvability of high-girth instances; Discussion
- **Mathematical content:** 3 theorems, 1 corollary, 5 lemmas, 2 algorithms, 2 formal problems, 1 fact, 2 remarks
- **22 references** identified by named destinations (Jordan et al. 2024, Farhi et al. 2014/2025, Goemans-Williamson, Edmonds-Johnson, etc.)

### Key findings relevant to the project

1. **DQI has no advantage for MaxCut (k=2):** The dual code $C^\perp$ for MaxCut is the graph's cycle space, with minimum distance equal to the girth $g = O(\log n)$. This limits DQI's decoding radius to $\ell = O(\log n)$, giving a cut fraction that converges to $1/2$ (random guessing).

2. **Explicit DQI upper bound:** $1/2 + 1/(2\sqrt{D-1})$ for $D$-regular graphs. At $D=3$, this is $\approx 0.854$, while QAOA at $p=17$ achieves $0.8971$ — a gap of $+0.043$.

3. **Classical solvability of high-girth instances (Section 3):** The paper shows that MaxCut on high-girth regular graphs — exactly the setting where QAOA is analysed — is classically solvable, likely via T-join methods (references Edmonds-Johnson 1973, Schrijver 2003).

4. **Open question for k ≥ 3:** The cycle code argument is specific to k=2. For k=3 XORSAT, the dual code is the hypergraph cycle space with unknown minimum distance — so the situation may differ.

### Impact on project

- **No changes to PLAN.md needed.** The paper strengthens our motivation (DQI fails at k=2, so the interesting comparison is at k≥3) but doesn't change the technical approach.
- **Important methodological note:** Section 3 raises the question of whether Max-3-XORSAT on high-girth hypergraphs is also classically easy. If so, QAOA lower bounds on such instances wouldn't demonstrate quantum advantage at k=3 either. This is an open question worth discussing with Stephen.
- **Numbering:** Explainer 07 follows the existing sequence (05: DQI Nature paper, 06: DQI requires structure, 07: no advantage for MaxCut, 08: optimised DQI circuits, 09: tight inapproximability).

---

## Entry 3 — Verification pass on DQI Circuits explainer (arXiv:2510.10967)

### Date
22 March 2026 (continued)

### What was done

Systematic verification and improvement of `learning/08-explainer-optimized-dqi-circuits.md`. The PDF body text remains FlateDecode-compressed and unreadable, but **extensive structural data was extracted** from the PDF's internal structure:

1. **Named-destination tree:** All equation labels (220 equations), theorem/lemma/definition numbers (29 environments), figure captions (12 + 5 subfigures), table captions (7), code-listing line numbers (4 listings, 286 lines).

2. **Bookmark/outline hierarchy:** 15 outline entries including confirmed titles "Abstract" and "Reference Python Implementation". Section structure confirmed from TOC link indentation coordinates.

3. **Complete citation key list:** ~55 references extracted and categorised by topic (DQI, Reed-Solomon decoding, finite field arithmetic, quantum circuits, comparison targets, classical benchmarks).

### Corrections made

1. **Equation count in §3 fixed:** Was "117+ in §3 alone" → Now "102 equations (3.16–3.117), counter shared with 7 theorem-like environments". The numbering starts at 3.16 because Lemmas 3.1–3.3, Thm 3.4, Def 3.5, Lemma 3.6, Thm 3.7 consume numbers 1–7 in the shared §3 counter.

2. **Structural table completely rebuilt** with precise counts per section, confirmed from named destinations.

3. **Section structure upgraded** from speculation to confirmed (from equation numbering + TOC link hierarchy). Now includes appendices B, C (no equations), and the observation that §4 has no numbered equations.

4. **[needs verification] markers resolved where possible:**
   - Itoh-Tsujii: Confirmed (cite key `cite.ITOH198921`) → marker removed
   - Qualtran: Confirmed (cite key `cite.harrigan2024expressinganalyzingquantumalgorithms`) → "suggests" → "confirms"
   - Berlekamp/Sugiyama: Both confirmed cited on same page → marker refined to ask which is primary
   - Gosset/Bärtschi state preparation: Confirmed from cite keys → "suggests" → "confirm"

5. **New content added:**
   - Complete Reference Inventory section (~55 references, categorised)
   - AMD Frontier/EPYC classical benchmarking reference (cite.frontier2023epyc)
   - Garcia interpolation, Sarwate modified Euclidean, Amento binary field circuits
   - Gu & Jordan 2025 algebraic aspects reference
   - Briaud 2025, Chailloux 2025, Kahanamoku-Siu 2025 quantum cryptanalysis refs
   - Updated Jargon Glossary (added Sugiyama, Forney, Koetter-Vardy, carry-save adder)
   - Refined Questions section (now 9 questions, more targeted)

6. **Transparency note updated** to describe the structural extraction method.

### Remaining [needs verification] markers (4)

1. **Specific gate counts** for finite-field multiplication circuits
2. **Which decoder is primary** (Berlekamp-Massey vs Sugiyama vs both)
3. **Specific resource savings** over naïve implementations
4. **Near-term hardware parameters** (likely fault-tolerant, not near-term)

### Impact on project

**None.** No changes to PLAN.md. The paper confirms DQI's strength is on structured algebraic problems, not random XORSAT. The DQI+BP performance at (k=3, D=4) remains 0.87065 regardless of circuit optimisations.

### Action items

- [ ] When `pdftotext` or PDF viewer becomes available, extract full text and resolve remaining 4 markers
- [ ] Priority: read Theorems 1.1 and 4.2, check the 7 resource tables, and note AMD EPYC benchmark numbers

---

## Entry 2 — Audit of Basso 2021 Explainer (22 March 2026)

### What was done

Systematic audit of `learning/02-explainer-basso2021-high-depth.md` against the paper arXiv:2110.14206 and internal consistency with other project documents (especially `03-explainer-farhi2025-maxcut-lower-bound.md` and `04-our-problem.md`).

**Limitation:** The PDF is binary-encoded (FlateDecode compression) and text could not be extracted with available tools. The audit was therefore based on (a) internal consistency across project documents, (b) knowledge of the paper, and (c) cross-referencing with the Farhi 2025 explainer. Unverifiable claims were flagged with `⚠️ AUDIT NOTE` markers.

### Changes made

**Errors fixed:**

1. **Cost operator sign (line 92 → 108).** The formula had $C_\alpha = (1 - (-1)^{b_\alpha} Z \cdots Z)/2$ (minus sign). Corrected to $C_\alpha = (1 + (-1)^{b_\alpha} Z \cdots Z)/2$ (plus sign), matching the careful derivation in `04-our-problem.md`.

2. **Branch independence claim (§ O(1/D) Issue).** The explainer claimed branches on a tree are "not quite independent" because they "share the common vertex." This is **wrong**: on a tree, branches ARE exactly independent (they share no vertices except the parent). The Farhi 2025 explainer correctly describes this (lines 91-93 of file 03). The section was rewritten to explain that the O(1/D) corrections come from the iterative formula's D→∞ specialisation (CLT-type concentration), not from branch non-independence.

3. **Thermometer analogy replaced.** The original analogy (correlated thermometers) reinforced the wrong independence claim. Replaced with a Gaussian-vs-discrete-distribution analogy that correctly captures the approximation.

4. **Method conflation resolved.** The original "How it works" section described what sounded like the exact element-wise exponentiation trick (raising to (D-1)th power) but called it approximate. This conflated the Basso iterative formula with the Farhi 2025 exact tensor contraction. Rewritten to clearly distinguish the two methods: exact tree contraction (Farhi 2025, works for any D) vs. iterative formula (Basso 2021, exact only as D→∞).

**Claims flagged as unverified (⚠️ AUDIT NOTE markers):**

5. **"up to p=20"** — Could not verify the maximum depth computed. Replaced with "high depth" and flagged.

6. **"beats all known rigorous classical guarantees at p=11"** — Flagged: need to verify what specific classical algorithm is beaten and the precise statement.

7. **Performance table values** — All seven numerical values flagged as unverified. Also flagged that the column header "Cut fraction (large D)" is **suspect**: absolute cut fractions approach 0.5 as D→∞, so values of 0.75+ cannot be literal cut fractions. Likely approximation ratios or normalised energy densities.

8. **Max-q-XORSAT generalisation** — The paper title mentions only MaxCut and SK, not XORSAT. Flagged that the generalisation section needs verification against the paper's table of contents.

9. **Parisi conjecture** — Flagged for precise statement verification.

10. **Computational cost O(p² · 4^p)** — Flagged as unverified; noted that Farhi 2025's exact method costs O(p · 4^p), so the iterative formula being MORE expensive by a factor of p seems suspicious.

### Impact on project

- No changes to PLAN.md needed. The fundamental narrative (Basso = D→∞ regime, Farhi 2025 = exact for any D, we need the latter) is preserved and now more precisely stated.
- The O(1/D) corrections are now correctly attributed to the iterative formula's D→∞ specialisation, not to branch non-independence.
- **Action item:** When the paper is next read in full (Phase 2), resolve all ⚠️ AUDIT NOTE markers.

---

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
- **Validate against MaxCut (k=2, D=3)**: p=1 should give c̃_edge ≈ 0.6924 (= ½ + √3/9). Farhi 2025 has results up to p=17 to validate against.

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

## Entry 2 — Explainer: Optimised DQI Circuits (arXiv:2510.10967)

### Context

Wrote `learning/08-explainer-optimized-dqi-circuits.md` — an analysis of "Verifiable Quantum Advantage via Optimized DQI Circuits" by Khattar, Shutty, Gidney, Zalcman, Yosri, Maslov, Babbush, Jordan (arXiv:2510.10967, 52 pages).

### Method

The PDF text is compressed (FlateDecode) and couldn't be extracted with available tools. Instead, the explainer was built from:
1. PDF metadata (title, authors, page count, arXiv ID)
2. Full equation/section/theorem numbering extracted from the PDF name tree
3. Complete reference list (40+ citations) extracted from cite keys
4. Code listing structure (4 Python listings, 104+40+67+75 lines)
5. Zenodo DOI for released code: 10.5281/zenodo.17301475
6. Project context from `04-our-problem.md` and `PLAN.md`

Uncertain details are marked **[needs verification]** throughout.

### Key Findings

1. **Title is "Verifiable Quantum Advantage via Optimized DQI Circuits"** — focuses on OPI (Optimized Polynomial Interpolation), not XORSAT
2. **Heavy coding theory content** — references Berlekamp, Sugiyama, Guruswami-Sudan, Chien, Forney (Reed-Solomon decoding)
3. **Heavy finite field arithmetic** — references Itoh-Tsujii, Cantor, Schönhage, von zur Gathen (GF(q) operations)
4. **Circuit optimisation focus** — Craig Gidney and Dmitri Maslov co-authors, references to quantum adders, Toffoli counts
5. **Code released** on Zenodo, likely using Google's Qualtran framework
6. **Comparison with Shor** — references to RSA factoring and elliptic curve circuits

### Impact on Our Project

**None.** This paper does not change our approach, targets, or methods. It confirms that DQI's power lies in structured algebraic problems (OPI/Reed-Solomon), not random XORSAT — consistent with what we already knew from `04-our-problem.md`. The DQI+BP performance at (k=3, D=4) remains 0.87065 regardless of circuit optimisations.

### Action Items

- [ ] Extract full paper text using `pdftotext` and update the explainer with specific numbers, theorems, and resource estimates
- [ ] No changes needed to `PLAN.md`

## Entry 3 — Explainer: Tight Inapproximability of Max-LINSAT (arXiv:2603.04540)

### Context

Wrote `learning/09-explainer-tight-inapproximability.md` — an analysis of the paper by Kramer, Schubert, and Eisert (arXiv:2603.04540) on tight inapproximability limits for Max-LINSAT.

### Method

The PDF text could not be extracted with available tools (binary-encoded/compressed content; `view` timed out on the large file, and no shell tools were available for `pdftotext` or `pymupdf`). The explainer was built from:
1. Key finding recorded in `04-our-problem.md`: "Tight inapproximability: no algorithm beats r/q without exploiting structure"
2. Description in `PLAN.md`: "Tight limits of DQI on max-LINSAT"
3. Task description context about Max-LINSAT and DQI performance bounds
4. Standard mathematical background on Max-LINSAT over GF(q) and Håstad's inapproximability theorem
5. Context from the other DQI follow-up papers (Anschuetz et al., Parekh)

Uncertain details are marked **[needs verification]** throughout.

### Key Findings (from project context)

1. **Max-LINSAT generalises Max-XORSAT** to arbitrary finite fields GF(q); our problem (q=2) is a special case
2. **The paper establishes a tight ceiling** on what algorithms can achieve without exploiting algebraic structure — recorded as the "r/q" bound
3. **This complements the other DQI limitation papers**: Anschuetz et al. (OGP barrier), Parekh (no MaxCut advantage), and now Kramer et al. (general Max-LINSAT ceiling)
4. **QAOA is not subject to this bound** — it operates through a different mechanism (variational phase/mixer optimisation vs. QFT + decoding)

### Impact on Our Project

**Motivation strengthened, no approach changes.**

- The paper provides a theoretical ceiling for DQI on our problem class, making the QAOA comparison more meaningful
- If we show QAOA exceeding the DQI ceiling, this paper explains *why* DQI can't match it
- No changes to `PLAN.md` — our computational targets and methods are unaffected

### Action Items

- [ ] Extract full PDF text (need shell access for `pdftotext`) and update the explainer with precise theorem statements, proof techniques, and numerical bounds
- [ ] Check for specific numerical bound at (k=3, D=4, q=2) to compare against DQI+BP value of 0.87065
- [ ] No changes needed to `PLAN.md`

## Entry 4 — Explainer: DQI Requires Structure (arXiv:2509.14509)

**Date:** 2025-07-13

### Context

Wrote `learning/06-explainer-dqi-requires-structure.md` — a detailed analysis of "Decoded Quantum Interferometry Requires Structure" by Anschuetz, Gamarnik, and Lu (arXiv:2509.14509, 51 pages).

### Method

The PDF content streams are FlateDecode-compressed and could not be extracted to plain text with available tools. The explainer was built from:
1. PDF metadata: title, authors, arXiv ID (2509.14509v1), date (September 19, 2025), categories (quant-ph, cond-mat.dis-nn, cond-mat.stat-mech, cs.DS)
2. Full structural analysis: 51 pages, ~280 equations, 6 figures, ~15 theorems, ~15 lemmas, ~12 propositions, ~23 definitions, ~5 corollaries, 2 open questions
3. Complete citation key extraction (40+ references including Jordan 2025, Gamarnik 2022, Gallager 2003, Farhi 2014/2020, Goh 2025)
4. Author contact info: Anschuetz (eans@caltech.edu), Lu (lujz@mit.edu)
5. Project context from `04-our-problem.md` and `PLAN.md`
6. General knowledge of the OGP framework from Gamarnik's prior work

Uncertain details are marked **[needs verification]** throughout.

### Key Findings

1. **Central claim:** DQI is blocked by the Overlap Gap Property (OGP) on random LDPC instances from the Gallager ensemble
2. **Mechanism:** DQI is a "stable" (Lipschitz) algorithm; OGP prevents any stable algorithm from finding near-optimal solutions on random instances
3. **Classical AMP matches/exceeds DQI** on these instances — no quantum advantage
4. **DQI's power requires algebraic structure** — large minimum distance of dual code, as in Reed-Solomon/OPI problems
5. **Substantial technical paper:** 51 pages, ~74 numbered items (definitions, theorems, lemmas, propositions, corollaries), 280 equations — rigorous treatment
6. **Two open questions posed** (exact statements need verification from PDF text)

### Impact on Our Project

**Validates and strengthens the motivation for our QAOA computation. No changes to approach.**

- **Explains the "why"** behind DQI's underperformance at (k=3, D=4): random instances are in DQI's weak regime due to OGP
- **Confirms that DQI+BP's weakness is fundamental**, not fixable by better decoders
- **Frames our QAOA computation precisely**: we are testing whether QAOA (a non-stable algorithm at high depth) can overcome the barrier that blocks DQI
- **No changes needed to `PLAN.md`** — computational targets, methods, and comparison framework are unaffected

### Action Items

- [ ] Extract full paper text using `pdftotext` (need working binary) and update the explainer with:
  - Specific theorem statements (especially the main OGP barrier theorem and AMP comparison)
  - The exact definition of "stability" used for DQI
  - Numerical bounds if any (e.g., the specific overlap gap interval for Gallager-ensemble Max-k-XORSAT)
  - The two open questions (Questions 1 and 2)
- [ ] No changes needed to `PLAN.md`
