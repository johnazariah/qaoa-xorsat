# SLURM Cluster Setup — QAOA-XORSAT

Quick guide for running on a SLURM cluster with high-memory nodes.

## What's new (April 10, 2026)

### Double64 precision for k≥6 pairs

At k≥6, D≥7, p≥10, Float64 (15 digits) suffers catastrophic cancellation
in the 2^{2p+1}-element sums. The evaluator returns correct but meaningless
values (c̃ > 1). This is NOT overflow — intermediates stay near magnitude 1.

Fix: `swarm_chain_d64.jl` uses DoubleFloats.jl (~31 digits). The swarm
runs in Float64 for speed; only the evaluation and gradient use Double64.
Overhead: 3-5×. Submit all 15 pairs via:

    sbatch scripts/qaoa_d64_sweep.sh

Or use the all-in-one script that also monitors and auto-pushes results:

    bash scripts/run-d64-sweep.sh 2>&1 | tee /tmp/qaoa-d64-run.log

### Swarm/memetic optimizer (April 5)

At high (k,D), standard L-BFGS with warm-starting fails — the landscape
is flat (c̃ ≈ 0.5) at most starting points. The swarm finds basins that
warm-starting misses: 100 random candidates, short L-BFGS bursts,
cull/crossover, early exit + polish. Submit via:

    sbatch scripts/qaoa_swarm_sweep.sh

### Normalized evaluator (April 2)

Before each power operation, vectors are divided by their max magnitude.
Scale factors are tracked in log space and applied once at the end:

```
child_hat_norm = child_hat / max|child_hat|     # safe to raise to ^(k-1)
folded_norm    = folded / max|folded|            # safe to raise to ^(D-1)
msg_hat_norm   = msg_hat / max|msg_hat|          # safe to raise to ^k at root
```

The physical answer is reconstructed in log space:
`c̃ = (1 + cs · exp(log_total_scale) · Re(S_normalized)) / 2`

The backward pass (gradient) operates entirely on normalized intermediates
(all magnitudes ≤ 1, cannot overflow), with the single `exp(log_scale)`
multiplier applied to the final gradients.

### 2. Safety guards (prevents bad values from propagating)

Even if some edge case still produces an invalid value:
- **Post-evaluation check**: any final c̃ outside [0, 1] is rejected
- **Best-start selection**: valid results always beat invalid ones in argmax
- **Merge logic**: validity-aware — a valid c̃ = 0.88 beats an overflowed c̃ = 21.44
- **Warm-start chain**: poisoned angles are not propagated to the next depth
- **Overflow gradient**: returns a non-zero gradient pointing away from the
  overflow region (zero gradient was faking convergence in previous versions)

### Validation

1741 tests pass (273 new in `test/test_normalization.jl`), structured to
cover the full failure surface — not just the symptom but every stage where
overflow could propagate:

| Test group | Count | What it verifies |
|---|---|---|
| Value matches un-normalized at low (k,D) | 60 | Normalization doesn't change the answer |
| Gradient matches ForwardDiff | 18 | Backward pass correctly incorporates scale |
| Value ∈ [0,1] at high (k,D) | 33 | No overflow at previously-broken cases |
| Gradient finite at high (k,D) | 7 | Backward pass doesn't overflow |
| Value+gradient consistent | 7 | Both paths return same value |
| MaxCut validation preserved | 2 | Known result still matches |
| Optimizer valid at high (k,D) | 3 | Full pipeline stays physical |
| Scale self-consistency | 4 | Log accumulation is correct |
| Cluster overflow regression (all 15 pairs at p=10) | 75 | The exact cases that broke the cluster |
| `is_valid_qaoa_value` | 11 | Utility function boundary conditions |
| Merge validity-aware | 3 | Overflow values can't beat valid ones |

The regression suite (75 tests) runs every (k,D) pair at p=10 — the depth
where most pairs overflowed on the cluster — and checks that both the value
and gradient are finite and physical.  The ForwardDiff cross-validation (18
tests) confirms that the normalized backward pass produces the same gradient
as an independent numerical differentiation through the un-normalized
evaluator.  The self-consistency tests verify that the log-scale accumulation
doesn't drift: at moderate (k,D) where both paths work, the normalized and
un-normalized evaluators agree to machine precision (atol = 1e-10).

## Running a fresh sweep

Pull the latest code and submit:

```bash
cd ~/qaoa-xorsat
git pull
sbatch scripts/qaoa_sweep.sh
```

This runs all 15 (k,D) pairs from p=1 through p=15 with warm-starting.
The normalized evaluator handles all depths without overflow.

## Recovering from a failed run

If a run overflows or crashes partway through, the recovery script identifies
what's salvageable and generates a fresh SLURM job that resumes from there:

```bash
# 1. Stop the broken jobs
scancel <JOB_ID>

# 2. Pull the latest code (includes overflow guard)
git pull

# 3. Run the recovery script — scans results, identifies last good p per (k,D)
python3 scripts/recover-run.py

# 4. Review the status table, then submit the generated recovery job
sbatch scripts/qaoa_recovery.sh
```

The recovery script:
- Reads each run's `results.csv` and finds the highest p with c̃ ∈ (0, 1)
- Generates TOML configs in `experiments/recovery/` with `resume_from`
  pointing at the previous run directory (copies valid results, warm-starts
  from last good angles)
- Writes `scripts/qaoa_recovery.sh` as a SLURM array job covering only
  the pairs that need re-running

## Prerequisites

- **Julia 1.10+** installed (`juliaup` recommended)
- **Python 3** (any version — only used as a dispatcher)
- **Git** access to the repo

## One-time setup (on a login node)

```bash
# Clone the repo
cd ~
git clone https://github.com/johnazariah/qaoa-xorsat.git
cd qaoa-xorsat

# Install Julia dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Smoke test
julia --project=. -e 'using QaoaXorsat; println("OK")'
```

## Submit all 15 pairs

```bash
# Default: p=1–15, all 15 (k,D) pairs, 28 threads
sbatch scripts/qaoa_sweep.sh

# Override max depth
sbatch --export=QAOA_P_MAX=12 scripts/qaoa_sweep.sh

# Only k=3 family (pairs 1–5)
sbatch --array=1-5 scripts/qaoa_sweep.sh

# Single pair (e.g. k=3, D=4 = task 1)
sbatch --array=1 scripts/qaoa_sweep.sh
```

## Task ID → (k,D) mapping

| ID | k | D |  | ID | k | D |  | ID | k | D |
|----|---|---|--|----|---|---|--|----|---|---|
| 1  | 3 | 4 |  | 6  | 4 | 5 |  | 10 | 5 | 6 |
| 2  | 3 | 5 |  | 7  | 4 | 6 |  | 11 | 5 | 7 |
| 3  | 3 | 6 |  | 8  | 4 | 7 |  | 12 | 5 | 8 |
| 4  | 3 | 7 |  | 9  | 4 | 8 |  | 13 | 6 | 7 |
| 5  | 3 | 8 |  |    |   |   |  | 14 | 6 | 8 |
|    |   |   |  |    |   |   |  | 15 | 7 | 8 |

## Memory requirements

| Depth p | RAM per node | Wall time (est.) |
|---------|-------------|-----------------|
| ≤ 12    | ~19 GB      | minutes–hours   |
| 13      | ~84 GB      | hours           |
| 14      | ~394 GB     | hours–day       |
| 15      | ~1.6 TB     | days            |

Your 2.7 TB nodes can handle p=15 comfortably.

## Monitor & extract progress

### Live dashboard

```bash
# One-shot status of all 15 tasks
bash scripts/slurm-monitor.sh

# Auto-refresh every 60 seconds
watch -n 60 bash scripts/slurm-monitor.sh

# Filter to a specific job ID
bash scripts/slurm-monitor.sh -j 12345
```

The dashboard shows:
- SLURM job states (running, pending, completed)
- Best depth and c̃ value achieved per (k,D) pair
- Active checkpoints (in-progress optimizations) with age
- Partial trace data (optimizer iteration count, current value, gradient norm)
- Latest log output tails

### Extract results (yoink progress)

```bash
# Aggregate all completed results + checkpoints into a timestamped CSV
bash scripts/slurm-collect.sh

# Print to stdout (for piping to scp, etc.)
bash scripts/slurm-collect.sh --stdout

# Only the best value per (k,D) pair
bash scripts/slurm-collect.sh --best --stdout

# Include in-progress checkpoint angles (for warm-starting elsewhere)
bash scripts/slurm-collect.sh --checkpoints

# Copy results off the cluster
scp cluster:~/qaoa-xorsat/results/cluster-progress-latest.csv .
```

### Raw SLURM commands

```bash
# Job status
squeue -u $USER

# Watch a running task's stdout
tail -f qaoa_<jobid>-<taskid>.out

# Cancel a specific task
scancel <jobid>_<taskid>

# Results are saved to:
#   .project/results/optimization/runs/<run_id>/results.csv
#   .project/results/optimization/runs/<run_id>/trace-p<N>.csv
#   .project/results/optimization/runs/<run_id>/checkpoint-p<N>.csv
```

### Transferring results home

```bash
# From your local machine — pull the latest aggregated CSV
scp cluster:~/qaoa-xorsat/results/cluster-progress-latest.csv results/

# Or rsync all run data
rsync -avz cluster:~/qaoa-xorsat/.project/results/optimization/ \
    .project/results/optimization/
```

## Files

- `scripts/qaoa_sweep.sh` — SLURM batch script (sbatch this)
- `scripts/runner.py` — maps array task ID to (k,D), calls Julia
- `scripts/optimize_qaoa.jl` — the actual optimizer
- `scripts/slurm-monitor.sh` — live dashboard for cluster progress
- `scripts/slurm-collect.sh` — extract/aggregate results into CSV
