# Post-WHT Innovations: From p=10 to p=15

This document synthesises the algorithmic innovations discovered after
the Walsh-Hadamard factorisation (learning doc 15), covering the
period March 27 – April 10, 2026.

## 1. Plateau Detection (March 27–31)

**Problem**: At p≥10, each L-BFGS iteration takes minutes. Standard
convergence criteria (g_abstol) can't distinguish "converged but noisy"
from "still improving", wasting hours on iterations that don't improve
the objective.

**Solution**: Per-iteration Optim.jl callback maintaining a circular
buffer of the last 30 objective values. If max − min < g_abstol, the
optimizer stops immediately regardless of the gradient norm. A separate
5-minute timer flushes the trace to disk for monitoring.

**Impact**: p=12 wall time reduced from 2+ hours to ~40 minutes.

**Key insight**: The optimizer has converged when the *value* plateaus,
even if the gradient hasn't reached the tolerance. The gradient noise
floor at high p (~1e-4 at p=12) makes tight gradient tolerances
unreachable.

## 2. Threshold-Based Normalization (April 2–4)

**Problem**: At high (k,D), the branch tensor entries grow exponentially
through `^(k-1)` and `^(D-1)` operations, overflowing Float64 at
(7,8) around p≈9.

**First attempt** (April 2): Always-normalize — divide by max magnitude
before each power, track scale in log space. Prevents overflow but
causes *signal underflow* at p≥12: crushing the relative magnitude
differences that carry the physical signal (deviation from c̃ = 0.5).

**Final solution** (April 4): Threshold-based — only normalize when
max-magnitude exceeds 1e30. Preserves Float64 precision at moderate
magnitudes while preventing overflow at extreme ones.

**In the backward pass**: Scale factors are detached from the gradient
(treated as constants). Negligible error because ∂(max|x|)/∂θ is a
sparse selection operator contributing O(1/N) to the sum.

## 3. Safety Guards in the Optimizer (April 2)

Five-point fix for the two-stage failure where overflow propagates
through the optimization pipeline:

1. **Overflow gradient**: Returns 1e6 (large but finite) with non-zero
   gradient pointing toward the origin. Zero gradient was faking
   convergence.

2. **Post-evaluation validation**: Re-evaluates final angles and
   rejects c̃ outside [0, 1] via `is_valid_qaoa_value()`.

3. **Best-start selection**: `argmax` prefers valid values over invalid,
   so c̃ = 21.44 can never beat c̃ = 0.88.

4. **Merge logic**: Validity-aware — retry results can't keep an
   overflowed primary over a valid secondary.

5. **Warm-start chain**: `optimize_depth_sequence` refuses to propagate
   poisoned angles to the next depth.

## 4. Swarm/Memetic Optimizer (April 5–6)

**Problem**: At high (k,D), the QAOA loss landscape is extremely rugged.
Most starting points see c̃ ≈ 0.5 (flat landscape). Standard multi-start
L-BFGS with warm-starting from p−1 fails at p=3 for (7,8).

**Solution**: Population-based search combining evolutionary exploration
with L-BFGS polishing:

- 100 random candidates, 20-iteration L-BFGS bursts
- Cull worst 50%, replenish with 40% random + 60% midpoint crossovers
- 10 generations (or early exit after 3 stagnant generations)
- Full 1280-iteration L-BFGS polish on winner

**Key insight**: The swarm is only needed at low depths (p=1-5) where
the basin must be discovered. At p≥6, the warm-started candidate
dominates every generation — the swarm converges in 1-3 gens and the
early exit kicks in, switching to pure L-BFGS. This makes the swarm
~100× more efficient than running all 10 generations.

**Result**: (7,8) went from failing at p=3 to c̃ = 0.819 at p=8.

**Provenance**: The evolutionary approach was inspired by BRKGA
(Biased Random-Key Genetic Algorithm), learned from Dr. Helmut
Katzgraber during a 2021 collaboration on F# scientific computing.

## 5. Double64 Precision (April 10)

**Problem**: At k≥6, D≥7, p≥10, the branch tensor entries have
magnitudes near 1.0 throughout — no overflow, no normalization triggers.
But the sum of ~2^{2p+1} nearly-cancelling complex terms loses the
physical signal below Float64's 15-digit precision. The evaluator
returns S > 1 (giving c̃ > 1), which is unphysical.

The (k-1)(D-1) exponent at each step determines the rate of precision
loss:
- k=3, D=4: (k-1)(D-1) = 6 → precision sufficient through p=13+
- k=5, D=8: (k-1)(D-1) = 28 → marginal at p=9 (4e-4 error)
- k=7, D=8: (k-1)(D-1) = 42 → insufficient at p≥9

**Solution**: DoubleFloats.jl provides Double64 — double-double
arithmetic with ~31 digits of precision. The pipeline is already
generic over element type T via `QAOAAngles{T}`, so the change is
transparent. The swarm runs in Float64 (fast); only the final
evaluation and gradient use Double64.

**Overhead**: 3-5× (addition ~2×, multiplication ~4×; WHT is
add-heavy so closer to 3×).

**Validation**: (6,7) p=10 returns c̃ = 3.23 in Float64 vs c̃ = 0.813
in Double64. At low (k,D), both agree to 1e-9.

## Architecture: How the Innovations Stack

Each innovation was forced by a concrete wall the previous approach hit:

```
WHT factorisation (p≤5 → p=12)
  └→ Plateau detection (2h → 40min at p=12)
      └→ Normalized recurrence (overflow at p=9 for k≥6)
          └→ Safety guards (overflowed values poisoning optimizer)
              └→ Swarm optimizer (flat landscape at k≥6)
                  └→ Double64 (cancellation at p≥10 for k≥6)
```

No innovation was designed speculatively — each was a direct response
to a failure mode observed in production. The code was clear enough at
each stage to diagnose the next wall, which led to the next fix.
