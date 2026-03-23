# Experiment: Initial QAOA Performance Sweep

> **For**: Developer agent running on the local development box
> **Purpose**: Get rough QAOA satisfaction fractions for (k=3, D=4) at each
> depth p=1 through p=8, plus MaxCut validation at p=1-5.
> **Priority**: Speed over perfection — we want a trend line, not benchmark-grade final numbers.

---

## Context

Read `.project/protocols/testing-benchmarking-policy.md` first.
Read `.project/specs/property-tests.md` for the current verification context.
Read `.project/learning/15-wht-factorisation-discovery.md` for WHT background.

The code is on branch `main` in the workspace root at `/workspace`.
(The `feature/phase4-optimization` branch was merged via PR #4.)

This is an **experiment** run on the development box, not a benchmark run on the
dedicated 48 GB testbed. Treat timings as informative but not benchmark-grade.
Preserve the outputs so they can still be compared later, but do not present
them as canonical performance evidence.

---

## Task 1: Verify tests pass

```bash
cd /workspace
julia --project=. -e 'using Pkg; Pkg.test()'
```

All tests must pass before running experiments. Do not proceed if any fail.

If tests fail: STOP and report the failure instead of running the sweep.

---

## Task 2: MaxCut validation sweep (p=1-5)

Run MaxCut (k=2, D=3) to validate against Farhi 2025 Table 1.

```bash
cd /workspace
mkdir -p results
TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
COMMIT=$(git rev-parse --short HEAD)
RUN_DIR="results/${TIMESTAMP}-${COMMIT}-maxcut-k2-D3-p1-5.machine"
export QAOA_RUN_KIND=experiment
export QAOA_RUNNER_LABEL=devbox
export QAOA_RELIABILITY_DIR="$RUN_DIR"
bash scripts/testbed/run-with-machine-state.sh "$RUN_DIR" -- bash -lc '
  set -euo pipefail
  cd /workspace
  julia --project=. scripts/optimize_qaoa.jl 2 3 1 5 8 100 1234 true \
    | tee results/'"${TIMESTAMP}-${COMMIT}-maxcut-k2-D3-p1-5.csv"'
'
bash scripts/testbed/analyze-machine-state.sh "$RUN_DIR" \
  | tee "$RUN_DIR/machine-analysis.json"
```

Expected results (must match to 3 decimal places):

| p | Expected c̃ |
|---|------------|
| 1 | 0.6924 |
| 2 | 0.7559 |
| 3 | 0.7923 |
| 4 | 0.8168 |
| 5 | 0.8363 |

**If these don't match**: STOP. The code has a bug. Do not proceed to k=3.
**If they match**: Record the output and proceed.

---

## Task 3: Target sweep — k=3, D=4

Run each depth range with the current pragmatic exploratory budgets.
Save all output to `results/` with sortable filenames and preserve a machine
state snapshot directory for each run.

### Current pragmatic settings

| p range | Restarts | Maxiters | Operational note |
|--------|----------|----------|------------------|
| 1-5 | 8 | 100 | fast local exploration |
| 6-8 | 5 | 80 | moderate local exploration |
| 9-12 | 3 | 60 | warm-start only; prefer dedicated runner |
| 13+ | 2 | 50 | overnight; dedicated runner only |

### p=1-5 (fast)

```bash
cd /workspace
mkdir -p results
TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
COMMIT=$(git rev-parse --short HEAD)
RUN_DIR="results/${TIMESTAMP}-${COMMIT}-xorsat-k3-D4-p1-5.machine"
export QAOA_RUN_KIND=experiment
export QAOA_RUNNER_LABEL=devbox
export QAOA_RELIABILITY_DIR="$RUN_DIR"
bash scripts/testbed/run-with-machine-state.sh "$RUN_DIR" -- bash -lc '
  set -euo pipefail
  cd /workspace
  julia --project=. scripts/optimize_qaoa.jl 3 4 1 5 8 100 1234 true \
    | tee results/'"${TIMESTAMP}-${COMMIT}-xorsat-k3-D4-p1-5.csv"'
'
bash scripts/testbed/analyze-machine-state.sh "$RUN_DIR" \
  | tee "$RUN_DIR/machine-analysis.json"
```

### p=6-8 (moderate — may take 5-30 minutes)

```bash
cd /workspace
TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
COMMIT=$(git rev-parse --short HEAD)
RUN_DIR="results/${TIMESTAMP}-${COMMIT}-xorsat-k3-D4-p6-8.machine"
export QAOA_RUN_KIND=experiment
export QAOA_RUNNER_LABEL=devbox
export QAOA_RELIABILITY_DIR="$RUN_DIR"
bash scripts/testbed/run-with-machine-state.sh "$RUN_DIR" -- bash -lc '
  set -euo pipefail
  cd /workspace
  julia --project=. scripts/optimize_qaoa.jl 3 4 6 8 5 80 1234 true \
    | tee results/'"${TIMESTAMP}-${COMMIT}-xorsat-k3-D4-p6-8.csv"'
'
bash scripts/testbed/analyze-machine-state.sh "$RUN_DIR" \
  | tee "$RUN_DIR/machine-analysis.json"
```

**Note**: The optimize_qaoa.jl script warm-starts from previous depths within
a single run, but these are separate invocations. For p=6-8, the script will
do random restarts only (no warm-start from p=5). To get warm-starting, run
the full range in one go:

```bash
cd /workspace
TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
COMMIT=$(git rev-parse --short HEAD)
RUN_DIR="results/${TIMESTAMP}-${COMMIT}-xorsat-k3-D4-p1-8-warmstart.machine"
export QAOA_RUN_KIND=experiment
export QAOA_RUNNER_LABEL=devbox
export QAOA_RELIABILITY_DIR="$RUN_DIR"
bash scripts/testbed/run-with-machine-state.sh "$RUN_DIR" -- bash -lc '
  set -euo pipefail
  cd /workspace
  julia --project=. scripts/optimize_qaoa.jl 3 4 1 8 5 80 1234 true \
    | tee results/'"${TIMESTAMP}-${COMMIT}-xorsat-k3-D4-p1-8-warmstart.csv"'
'
bash scripts/testbed/analyze-machine-state.sh "$RUN_DIR" \
  | tee "$RUN_DIR/machine-analysis.json"
```

This is slower overall but gives better results at higher p because each
depth is seeded from the previous optimum.

Do **not** run `p >= 9` on the development box as part of this prompt. Those
runs are now testbed territory unless there is a specific reason to accept a
long exploratory local run.

---

## Task 4: Report results

Print a summary table:

```
p | c̃(p) for (k=3, D=4) | SA target (0.9366) | Gap | Reliability note
```

Key questions to answer:
1. Is c̃ increasing with p? (it should be — monotonicity)
2. What's the value at p=5? At p=8?
3. Is the curve flattening, or still climbing steeply?
4. At the current rate, what p would be needed to reach 0.9366?
5. Did any run show suspicious machine-state warnings that make the timing less trustworthy?

---

## Expected Timings on Development Box (M4 Max Mac Studio, 64GB)

| Range | Restarts | Maxiters | Single eval time | Est. total |
|-------|----------|----------|-----------------|------------|
| p=1-5 | 8 | 100 | μs → 1.5ms | < 2 min |
| p=6-8 | 5 | 80 | ~5ms → ~40ms | 5-30 min |
| p=1-8 (warmstart) | 5 | 80 | μs → ~40ms | 10-40 min |

These are rough — actual time depends on how quickly L-BFGS converges.
They are **not** benchmark-grade timing targets.

---

## What NOT to do

- Do NOT run p≥10 on the development box — it will take hours. That's for
  the dedicated runner.
- Do NOT treat development-box timings as benchmark evidence.
- Do NOT modify source code. This is an experiment run, not a development task.
- Do NOT commit results to main. Leave them in the `results/` directory.
  The proper pipeline (GitHub Actions) handles commits.

---

## Output

Report back:
1. Whether MaxCut validation passed (Task 2)
2. The full CSV output from Task 3
3. A summary table with the trend analysis (Task 4)
4. Wall-clock time for each run
5. Any errors or unexpected behaviour
6. The contents of each `machine-analysis.json` file, or a concise summary of any warnings
