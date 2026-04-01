# SLURM Cluster Setup — QAOA-XORSAT

Quick guide for running on a SLURM cluster with high-memory nodes.

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
