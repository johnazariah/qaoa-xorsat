# Q3: Are the Universal Curves Globally Optimal?

## Motivation

Basso et al. and Farhi et al. compute QAOA performance using angle
optimisation that starts from a warm-start chain: the optimal angles at
depth p−1 are extended to depth p. This reliably finds a local maximum,
but nobody has established whether it is the *global* maximum.

Stephen Jordan asks: "Are the universal curves really the globally
optimal choice of rotation angles or are they just locally optimal?
If they are only locally optimal, how much better is the global optimum?"

## What We Already Have

- **Swarm/memetic optimiser** (`swarm_optimize`): population-based search
  with random initialisation, crossover, and L-BFGS polish. Already built
  and tested for XORSAT rugged landscapes.
- **Exact evaluator** at any angles for MaxCut at D=3..8, p up to 13.
- **Warm-start trajectory** data: optimal angles at every p from 1..12.

## Experimental Design

### Phase 1: Landscape Probing (cheap)

For each (D, p) with existing data:

1. **Multi-start random search**: Run `optimize_angles` with
   `restarts=50`, `initial_guess_kind=:random` (no warm-start).
   Compare best-found against the warm-start chain value.
   
2. **Grid perturbation**: Take the warm-start optimum, perturb each
   angle by ±δ for δ ∈ {0.1, 0.3, 0.5, 1.0} radians, re-optimise.
   Does L-BFGS return to the same basin or find a better one?

3. **Symmetry-sector search**: Exploit the known symmetries
   (γ → γ+π for odd k, β → β+π) to systematically search each
   symmetry sector rather than relying on random sampling.

Target: D=3..6, p=6..12 (p≤6 is too cheap to matter, p≥13 is
expensive but informative).

### Phase 2: Swarm Search (moderate cost)

For selected (D, p) pairs where Phase 1 shows interesting structure:

1. Run `swarm_optimize` with population=100, generations=20,
   burst_iters=50. Compare against warm-start.

2. Record the *distribution* of local optima found: how many distinct
   basins, what are their values, how far apart are the angle vectors?

3. Plot basin depth vs. angular distance from warm-start optimum.

### Phase 3: High-Depth Confirmation (expensive)

At p=12 and p=13 (where each eval takes minutes):

1. Run swarm with population=20, generations=5 with no warm-start.
2. If it finds the same optimum (within 1e-6), declare global
   optimality at that depth.
3. If it finds something better, that's the headline result.

## Expected Outcomes

**Hypothesis A (likely for MaxCut)**: The warm-start chain finds the
global optimum at every depth for D=3..6. Evidence: the angle
trajectories are smooth, the gain ratios are monotonic, and the
landscape appears single-basin from the angle plots.

**Hypothesis B (possible for D≥7)**: Multiple basins exist at high D.
We already see angle trajectory jumps at D≥7 in the XORSAT data. If
MaxCut D=7,8 shows the same, the warm-start chain may miss better
basins.

**Either outcome is publishable**:
- A: "We confirm global optimality of the universal curves through p=13"
- B: "We find that the universal curves are suboptimal at D≥7; the true
  global optimum exceeds the warm-start trajectory by X basis points"

## Resource Estimate

- Phase 1 (D=3..6, p=6..12): ~50 restarts × 7 depths × 4 D values ×
  ~10s/eval = ~40 hours. Can parallelise across D.
- Phase 2 (selected pairs): ~10 hours per (D, p) pair.
- Phase 3 (p=12,13): ~2 hours per pair with checkpointing.

Total: ~60-80 hours of compute, spread over a few days.

## Output Artefacts

1. Table: warm-start c̃ vs. best-found c̃ at each (D, p)
2. Histogram of basin depths at selected (D, p)
3. Angular distance between warm-start and swarm-found optima
4. Statement: "globally optimal through p=X" or "suboptimal at D≥Y"

## Integration with MaxCut Paper

This directly extends Section 4.2 (Convergence Analysis) and Section
4.3 (Angle Profiles). It either strengthens the paper ("our values are
provably optimal") or upgrades it substantially ("we discover that the
standard warm-start approach misses better basins").
