# Q1: Is QAOA a Trotterised Adiabatic Optimisation?

## Agent Instructions

This spec is for autonomous execution. The worktree is at
`/Users/johnaz/PhD/qaoa-xorsat-q1` on branch `q1-adiabatic`.

**Codebase**: Julia package `QaoaXorsat` in `src/`. Run with
`julia --project=. -t 16` from the worktree root.

**Key functions**:
- `basso_expectation_normalized(TreeParams(k, D, p), QAOAAngles(γ, β); clause_sign=-1)` → Float64
- `basso_expectation_and_gradient(...)` → (value, γ_grad, β_grad)
- `optimize_angles(TreeParams(k,D,p); clause_sign, initial_guesses, restarts, ...)` → result

**Existing data**: `results/maxcut-k2-d{3..8}-sweep.csv` — columns:
`k,D,p,ctilde,wall_seconds,gamma_semicolon_separated,beta_semicolon_separated`

**Starter script**: `scripts/q1_intermediate_depth.jl` — runs Experiment 2
(intermediate-depth truncated performance). Execute it first.

**Workflow**:
1. Read the two papers linked below
2. Run `scripts/q1_intermediate_depth.jl` to get Experiment 2 data
3. Implement Experiments 1, 3, 4 as additional scripts in `scripts/`
4. Write results to `results/q1-*.csv`
5. Generate plots (Julia or Python) in `figures/`
6. Commit results on the `q1-adiabatic` branch

**Tests**: `julia --project=. -e 'using Pkg; Pkg.test()'` — 1741 tests must pass.
Do not modify `src/` unless you need a new analysis function.

## Motivation

A common claim in the QAOA literature is that the optimal angle
schedules γ(t), β(t) can be interpreted as large-timestep
Trotterisations of an adiabatic path from the mixer Hamiltonian to the
problem Hamiltonian. Papers regularly assume this and use it to
initialise QAOA angles from adiabatic schedules.

Stephen Jordan (and Eddie Farhi) are skeptical. The question: do the
intermediate QAOA states actually resemble the ground states of the
associated adiabatic Hamiltonian at the corresponding interpolation
parameter?

## Key References to Read First

1. **https://arxiv.org/abs/2604.24580** — Recent paper using the
   Trotterisation assumption. Example of the claim in practice.

2. **https://arxiv.org/abs/2106.15645** — More sophisticated treatment
   of the QAOA-adiabatic relationship. Read critically.

3. **Basso et al. (2021)** — Source of the "universal curves" angle
   schedules that are implicitly assumed to come from an adiabatic path.

## What We Can Compute

We have exact QAOA evaluations at finite D on the infinite-girth tree.
This is not the same as simulating the full quantum state on a finite
graph, but we can compute:

1. **The exact optimal angle schedules** γ*(p), β*(p) for p=1..13 at
   D=3..8.

2. **The QAOA satisfaction fraction** at any angles — including
   angles derived from an adiabatic schedule.

3. **Partial expectations** at intermediate depths: evaluate the
   depth-t circuit (t < p) with the first t angles from the depth-p
   optimum. This gives "QAOA at intermediate time".

## Experimental Design

### Experiment 1: Angle Schedule Comparison

Compare the optimal QAOA angles against the linear adiabatic schedule:
- Adiabatic: γ_j = (j/p) · γ_max, β_j = (1 - j/p) · β_max
- Actual QAOA optimal: γ*(p), β*(p)

Plot both on the same axes. If QAOA is Trotterised adiabatic, the
shapes should match (up to rescaling). If they diverge, QAOA is doing
something qualitatively different.

Compute the "adiabatic fidelity": evaluate c̃ at the linear adiabatic
angles vs. the optimal angles. How much performance is lost?

### Experiment 2: Intermediate-Depth Performance

For the depth-p optimal angles, evaluate the QAOA objective at
intermediate depths t = 1, 2, ..., p (using only the first t angle
pairs). Plot c̃(t) for the "truncated" schedule.

If QAOA is adiabatic-like, this curve should be monotonically
increasing (the system is "annealing" toward the optimum). If it
dips or oscillates, QAOA is doing something non-adiabatic —
temporarily worsening the objective to set up interference patterns
that pay off at later rounds.

### Experiment 3: Adiabatic-Initialised QAOA

Use the linear adiabatic schedule as initialisation for the optimiser:
- Set γ_j = (j/p) · γ_max, β_j = (1 - j/p) · β_max for various γ_max, β_max
- Optimise with L-BFGS from this starting point
- Compare the resulting c̃ and angles against warm-start initialisation

If the adiabatic initialisation lands in the same basin as warm-start,
the connection is real. If it lands in a different (worse) basin, the
adiabatic picture is misleading.

### Experiment 4: Angle Profile Curvature

The adiabatic schedule predicts linear γ(t/p) and linear β(t/p) in
the continuum limit. Our exact angle profiles (already computed) show
the actual shape. Fit γ_j vs j/p to polynomials and measure the
curvature. Large curvature = non-adiabatic character.

For D=3, the γ profile is concave-up (accelerating phase kicks). For
a linear adiabatic schedule it would be linear. The β profile is
concave-down. Quantify the deviation.

## What We Cannot Compute

Our evaluator computes the *expectation value* on the infinite-girth
tree, not the full quantum state. We cannot directly compute:
- State overlaps with adiabatic ground states
- Entanglement entropy at intermediate times
- Energy gap of the interpolating Hamiltonian

These would require finite-instance simulation (tensor network or
exact diagonalisation on small graphs), which is a different
computational problem. We should be clear about this limitation.

## Expected Outcomes

**Hypothesis (likely)**: The optimal QAOA angles are NOT a
Trotterisation of adiabatic optimisation. Evidence:
- The angle profiles are visibly non-linear (we already see this)
- The intermediate-depth performance likely oscillates
- The adiabatic initialisation likely lands in a worse basin

**If confirmed**: This is a significant result. It means QAOA should
be understood as a *variational* algorithm that exploits quantum
interference patterns, not as a discretised annealing procedure.
The practical implication: initialising QAOA from adiabatic schedules
(as many papers do) may be suboptimal.

## Resource Estimate

All experiments are cheap (p ≤ 12, existing infrastructure):
- Experiment 1: minutes (angle comparison, a few evaluations)
- Experiment 2: ~1 hour (p evaluations at each depth, D=3..6)
- Experiment 3: ~2 hours (multiple adiabatic starts per D)
- Experiment 4: minutes (analysis of existing data)

Total: < 1 day of compute.

## Output Artefacts

1. Figure: optimal QAOA angles vs. linear adiabatic schedule
2. Figure: intermediate-depth c̃(t) curve — monotonic or oscillating?
3. Table: adiabatic-initialised c̃ vs. warm-start c̃
4. Polynomial fits and curvature metrics for angle profiles
5. Statement: "QAOA is / is not consistent with Trotterised adiabatic
   optimisation on D-regular MaxCut"

## Integration with MaxCut Paper

This could be a new Section 5: "Relationship to Adiabatic
Optimisation". It directly addresses a question that Stephen
identifies as a "big question in QAOA" and would significantly
elevate the paper from "here are some numbers" to "here is a
qualitative insight about the nature of QAOA".
