# Agent Handoff — 23 March 2026

## Current repo state

- Branch: `main`
- HEAD: `d242ed1 chore: preserve session state before M4 migration`
- Working tree: clean
- Dev environment: Debian 12 devcontainer under `/workspace`

This handoff is for the next agent picking up after the devcontainer session.

## What landed in this session

### 1. Optimizer convergence fix

- Removed objective-side angle canonicalization from the optimization loop in
  `src/optimization.jl`.
- Kept boundary canonicalization for inputs / stored outputs.
- Relaxed `Optim.Options(..., g_abstol=1.0e-6, ...)` to avoid finite-difference
  gradient-noise stalls.

Result:

- XORSAT `(k=3, D=4)` now converges cleanly through `p=5`.
- Clean run recorded `p=5 = 0.820480966927`.

### 2. Incremental optimization-result streaming

- Added `on_result` callback support to `optimize_depth_sequence`.
- Updated `scripts/optimize_qaoa.jl` to:
  - print the CSV header immediately
  - flush stdout after each completed depth
  - append each result immediately to the per-run archive and aggregate index
  - update the manifest `result_count` incrementally
- Added test coverage for callback delivery in `test/test_optimization.jl`.

This work was committed as:

- `7583ef1 feat: stream optimization results incrementally`

### 3. M4 migration setup

The repo now also contains M4-host bootstrap material:

- `.project/checklist-m4-setup.md`
- `scripts/setup-m4.sh`

Current HEAD `d242ed1` is the follow-up migration/session-state commit that now
contains those changes on `main`.

## Results and evidence worth trusting

### Confirmed good runs

- MaxCut `(k=2, D=3)` validation through `p=5` is clean.
- XORSAT `(k=3, D=4)` through `p=5` is clean after the tolerance fix.

Canonical successful XORSAT run:

- `.project/results/optimization/runs/20260323T205738-k3-d4-p1-5-r8-i100-s1234/`

Key value from that run:

- `p=5`: `0.820480966927`, `converged=true`, `iterations=17`,
  `retry_count=0`, `best_start_kind=warm`

### Known missing results

There are still no reliable archived results for `p=6` or `p=7` from the older
attempts. Earlier raw machine-capture directories for higher-depth runs existed
but were incomplete and had empty stdout/stderr.

## The main unfinished task

The open technical item is still:

- verify the `p=6..8` run path on the updated streaming code

This is the highest-value next step because the code now streams per-depth
output and preserves partial progress, but that behavior has not yet been
validated on a real higher-depth run after the streaming changes landed.

## Recommended next actions

1. Re-run a real higher-depth XORSAT sweep on the target path:

   `julia --project=. scripts/optimize_qaoa.jl 3 4 1 8 4 200 1234 true`

   If running on the M4 host, use threads:

   `julia --project=. -t 10 scripts/optimize_qaoa.jl 3 4 1 8 4 200 1234 true`

2. Confirm during the run that:

   - stdout emits one row per completed depth
   - `.project/results/optimization/runs/<run_id>/results.csv` grows during the run
   - `.project/results/optimization/index.csv` appends incrementally
   - manifest `result_count` increments as depths finish

3. If the run reaches `p=6` or beyond cleanly, update:

   - `.project/journal.md`
   - `.project/PLAN.md`
   - any comparison notes against the DQI numbers in `learning/04-our-problem.md`

4. If convergence degrades again at higher `p`, the next likely technical move
   is exact gradients / autodiff rather than further tightening finite-diff
   stopping criteria.

## Useful commands

### Full test suite

`julia --project=. -e 'using Pkg; Pkg.test()'`

### Optimization tests only

`julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_optimization"])'`

### M4 host setup

See:

- `.project/checklist-m4-setup.md`
- `scripts/setup-m4.sh`

## Important context from the user

- The user is focused on the target problem `(k=3, D=4)` and the exact QAOA
  performance curve versus depth.
- They care about convergence quality, reproducibility, and preserving useful
  partial results from long runs.
- They also care about clean documentation of what changed and which runs are
  trustworthy.

## If you need to orient quickly

Read these first:

1. `.project/PLAN.md`
2. `.project/journal.md`
3. `.project/checklist-m4-setup.md`
4. `src/optimization.jl`
5. `scripts/optimize_qaoa.jl`
6. `test/test_optimization.jl`

## Bottom line

The optimizer path is materially better than it was at the start of the day:

- clean convergence through `p=5`
- partial-progress streaming implemented
- M4 migration bootstrap prepared

The next agent should spend time on validating real `p=6..8` execution rather
than re-litigating the earlier `p=5` convergence fix.