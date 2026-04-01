#!/usr/bin/env python3
"""
Recovery script for QAOA-XORSAT cluster runs.

Identifies which (k,D) pairs overflowed and generates TOML configs to
re-run them from the last known-good depth using resume_from.

Usage:
    python3 scripts/recover-run.py [RUNS_DIR]

Generates TOML config files in experiments/recovery/ and a SLURM submission script.
"""

import csv
import os
import sys
from pathlib import Path

# All 15 (k, D) pairs
PAIRS = [
    (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
    (4, 5), (4, 6), (4, 7), (4, 8),
    (5, 6), (5, 7), (5, 8),
    (6, 7), (6, 8),
    (7, 8),
]


def find_last_good_depth(results_csv_path):
    """
    Read a results.csv and find the highest p with a valid c̃ ∈ (0, 1).
    Returns (last_good_p, value, run_dir) or None if no valid results.
    """
    if not os.path.isfile(results_csv_path):
        return None

    with open(results_csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)

        # Find column indices
        try:
            p_col = header.index("p")
            val_col = header.index("value")
        except ValueError:
            return None

        last_good = None
        for row in reader:
            try:
                p = int(row[p_col])
                val = float(row[val_col])
            except (ValueError, IndexError):
                continue

            # Valid QAOA value must be in (0, 1)
            if 0.0 < val < 1.0:
                if last_good is None or p > last_good[0]:
                    last_good = (p, val)
            else:
                # Once we hit overflow, don't trust subsequent values
                break

    return last_good


def scan_runs(runs_dir):
    """
    Scan all run directories and find the best valid result per (k, D).
    Returns dict: (k, D) -> (last_good_p, value, run_dir_path)
    """
    results = {}

    if not os.path.isdir(runs_dir):
        print(f"Runs directory not found: {runs_dir}", file=sys.stderr)
        return results

    for run_name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, run_name)
        if not os.path.isdir(run_dir):
            continue

        results_csv = os.path.join(run_dir, "results.csv")
        if not os.path.isfile(results_csv):
            continue

        # Parse k, D from the run directory name
        import re
        m = re.search(r"k(\d+)-d(\d+)", run_name)
        if not m:
            continue
        k, D = int(m.group(1)), int(m.group(2))

        result = find_last_good_depth(results_csv)
        if result is None:
            continue

        last_good_p, val = result

        # Keep the best across multiple runs for the same (k, D)
        if (k, D) not in results or last_good_p > results[(k, D)][0]:
            results[(k, D)] = (last_good_p, val, run_dir)

    return results


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        repo_root, ".project", "results", "optimization", "runs"
    )

    p_max = int(os.environ.get("QAOA_P_MAX", "15"))

    print(f"Scanning: {runs_dir}")
    print(f"Target p_max: {p_max}")
    print()

    results = scan_runs(runs_dir)

    if not results:
        print("No valid results found. Nothing to recover from.")
        sys.exit(1)

    # Report status
    print(f"{'(k,D)':<8} {'Last good p':<12} {'c̃':<16} {'Status':<12} Run dir")
    print("-" * 90)

    needs_rerun = []
    complete = []

    for k, D in PAIRS:
        if (k, D) not in results:
            print(f"({k},{D})   {'---':<12} {'---':<16} {'NO DATA':<12}")
            needs_rerun.append((k, D, 1, None))  # start from scratch
            continue

        last_p, val, run_dir = results[(k, D)]
        run_name = os.path.basename(run_dir)

        if last_p >= p_max:
            print(f"({k},{D})   p={last_p:<10} {val:<16.12f} {'COMPLETE':<12} {run_name}")
            complete.append((k, D))
        else:
            print(f"({k},{D})   p={last_p:<10} {val:<16.12f} {'RERUN':<12} {run_name}")
            needs_rerun.append((k, D, last_p + 1, run_dir))

    print()
    print(f"Complete: {len(complete)}, Need rerun: {len(needs_rerun)}")

    if not needs_rerun:
        print("All pairs complete!")
        sys.exit(0)

    # Generate TOML configs for recovery
    recovery_dir = os.path.join(repo_root, "experiments", "recovery")
    os.makedirs(recovery_dir, exist_ok=True)

    task_configs = []
    for i, (k, D, p_min, prev_run_dir) in enumerate(needs_rerun, 1):
        config_name = f"k{k}-d{D}.toml"
        config_path = os.path.join(recovery_dir, config_name)

        lines = [
            f"k = {k}",
            f"D = {D}",
            f"p_min = {p_min}",
            f"p_max = {p_max}",
            f"restarts = 2",
            f"maxiters = 320",
            f"seed = 1234",
            f"preserve = true",
            f'autodiff = "adjoint"',
        ]
        if prev_run_dir:
            lines.append(f'resume_from = "{prev_run_dir}"')

        with open(config_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        task_configs.append((i, k, D, p_min, config_path))
        print(f"  Wrote {config_path}")

    # Generate SLURM script for recovery
    slurm_script = os.path.join(repo_root, "scripts", "qaoa_recovery.sh")
    n_tasks = len(task_configs)

    slurm_content = f"""#!/bin/bash
#
# SLURM recovery job for QAOA-XORSAT: re-run overflowed (k,D) pairs.
# Generated by recover-run.py — re-run this script to regenerate.
#
#SBATCH --job-name=qaoa-recovery
#SBATCH --array=1-{n_tasks}
#SBATCH --partition=c3d
#SBATCH --time=999:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=2700G
#SBATCH --output=qaoa-recovery_%A-%a.out
#SBATCH --error=qaoa-recovery_%A-%a.err

set -euo pipefail

THREADS="${{SLURM_CPUS_PER_TASK:-28}}"

# Task ID → TOML config mapping
declare -a CONFIGS=(
"""
    for i, k, D, p_min, config_path in task_configs:
        rel_path = os.path.relpath(config_path, repo_root)
        slurm_content += f'    "{rel_path}"  # {i}: k={k}, D={D}, resume from p={p_min}\n'

    slurm_content += f""")

TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
CONFIG="${{CONFIGS[$TASK_INDEX]}}"

echo "=== QAOA-XORSAT Recovery ==="
echo "Job ID:    ${{SLURM_JOB_ID:-local}}"
echo "Array ID:  $SLURM_ARRAY_TASK_ID"
echo "Config:    $CONFIG"
echo "Node:      $(hostname)"
echo "Started:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd ~/qaoa-xorsat

julia --project=. -t "$THREADS" scripts/optimize_qaoa.jl "$CONFIG"

echo ""
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"""

    with open(slurm_script, "w") as f:
        f.write(slurm_content)
    os.chmod(slurm_script, 0o755)
    print(f"\n  Wrote {slurm_script}")
    print(f"\n  Submit with: sbatch scripts/qaoa_recovery.sh")


if __name__ == "__main__":
    main()
