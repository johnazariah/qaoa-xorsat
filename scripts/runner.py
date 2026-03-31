#!/usr/bin/env python3
"""
SLURM array task dispatcher for QAOA-XORSAT.

Maps SLURM_ARRAY_TASK_ID (1–15) to (k, D) pairs from Jordan et al.
and launches the Julia optimizer.

Usage:
    srun python3 runner.py $SLURM_ARRAY_TASK_ID
    python3 runner.py 1          # runs (k=3, D=4)
    python3 runner.py 1 --p-max 12 --threads 28
"""

import os
import subprocess
import sys
from datetime import datetime, timezone

# All 15 (k, D) pairs from Jordan et al., Table 1
# Index 1–15 matches SLURM --array=1-15
PAIRS = [
    (3, 4),   #  1 — primary target
    (3, 5),   #  2
    (3, 6),   #  3
    (3, 7),   #  4
    (3, 8),   #  5
    (4, 5),   #  6
    (4, 6),   #  7
    (4, 7),   #  8
    (4, 8),   #  9
    (5, 6),   # 10
    (5, 7),   # 11
    (5, 8),   # 12
    (6, 7),   # 13
    (6, 8),   # 14
    (7, 8),   # 15
]


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} TASK_ID [--p-max N] [--threads N]", file=sys.stderr)
        print(f"  TASK_ID: 1–{len(PAIRS)} (SLURM_ARRAY_TASK_ID)", file=sys.stderr)
        print(f"\nPair table:", file=sys.stderr)
        for i, (k, d) in enumerate(PAIRS, 1):
            print(f"  {i:2d}: k={k}, D={d}", file=sys.stderr)
        sys.exit(1)

    task_id = int(sys.argv[1])
    if task_id < 1 or task_id > len(PAIRS):
        print(f"Error: TASK_ID must be 1–{len(PAIRS)}, got {task_id}", file=sys.stderr)
        sys.exit(1)

    # Parse optional arguments
    p_max = int(os.environ.get("QAOA_P_MAX", "15"))
    threads = int(os.environ.get("QAOA_THREADS", "28"))
    restarts = 2
    maxiters = 320
    seed = 1234

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--p-max" and i + 1 < len(sys.argv):
            p_max = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--threads" and i + 1 < len(sys.argv):
            threads = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--restarts" and i + 1 < len(sys.argv):
            restarts = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--maxiters" and i + 1 < len(sys.argv):
            maxiters = int(sys.argv[i + 1])
            i += 2
        else:
            print(f"Unknown argument: {sys.argv[i]}", file=sys.stderr)
            sys.exit(1)

    k, D = PAIRS[task_id - 1]

    # Build Julia command
    cmd = [
        "julia",
        f"--project=.",
        f"-t", str(threads),
        "scripts/optimize_qaoa.jl",
        str(k), str(D),
        "1", str(p_max),          # p_min=1 always (warm-start chain)
        str(restarts),
        str(maxiters),
        str(seed),
        "true",                   # preserve results
        "adjoint",                # manual adjoint differentiation
    ]

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[{ts}] Task {task_id}: k={k}, D={D}, p=1–{p_max}", file=sys.stderr)
    print(f"[{ts}] Command: {' '.join(cmd)}", file=sys.stderr)
    print(f"[{ts}] Threads: {threads}, RAM available: {_mem_gb():.0f} GB", file=sys.stderr)
    sys.stderr.flush()

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def _mem_gb():
    """Best-effort memory detection in GB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024 / 1024
    except FileNotFoundError:
        pass
    return 0.0


if __name__ == "__main__":
    main()
