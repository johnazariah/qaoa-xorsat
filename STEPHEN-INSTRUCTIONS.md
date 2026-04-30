# Cluster Instructions for Stephen — April 30, 2026

## TL;DR — Clean Run (3 Tiered Submissions)

**This is a clean run. Delete all old results before starting.**

```bash
git pull origin main

# Clean out ALL old results and logs
rm -rf results/slurm-*.csv results/*.csv results/*.machine logs/

# Fresh start
mkdir -p logs

# Tier 1: k=3 (5 jobs) → c3dssd — 1440GB RAM, 180 CPUs, local SSD for spillover
sbatch --partition=c3dssd --mem=1400G --cpus-per-task=176 --array=0-4  scripts/slurm_xorsat_all.sh

# Tier 2: k=4 (4 jobs) → n2 — 864GB RAM, 128 CPUs
sbatch --partition=n2     --mem=850G  --cpus-per-task=124 --array=5-8  scripts/slurm_xorsat_all.sh

# Tier 3: k≥5 (6 jobs) → c2 — 240GB RAM, 60 CPUs
sbatch --partition=c2     --mem=235G  --cpus-per-task=56  --array=9-14 scripts/slurm_xorsat_all.sh
```

This launches **15 jobs** across 3 partitions. Every (k,D) pair starts from p=1.

### Why these partitions?

This is **CPU-only** code — no GPU needed. Don't use a2ugpu machines.

| Tier | Queue | RAM | CPUs | $/hr | Why |
|------|-------|-----|------|------|-----|
| k=3 (p→16) | c3dssd | 1440GB | 180 | $11.68 | Local SSD for disk spillover at p≥16 |
| k=4 (p→14) | n2 | 864GB | 128 | $6.17 | Best value, ~120GB needed at p=14 |
| k≥5 (p→12-13) | c2 | 240GB | 60 | $3.13 | Cheapest, ~34GB needed at p=13 |

## What's new since last run

### CPU Gradient Checkpointing (Innovation 10)
- Stores only √p branch-tensor checkpoints instead of all p+1
- Enables p=13-15 on 1.5TB nodes in Float64
- p=16 uses disk spillover (checkpoints to NVMe)
- Verified: bit-identical gradients, 1779 tests pass

### Memory-Efficient Backward Pass
- Recomputes child_hat/folded one step at a time during backward
- Peak memory reduced from 3×segment to 1×segment + 2 vectors

### Gradient Plateau Detection
- Secondary exit: if gradient norm stuck < 100×g_abstol for 20 iterations with value range < 10×g_abstol, exit early
- Saves ~10 min per converged depth

### Swarm Optimizer at Low Depth
- p≤6: population 50, 5 generations
- p=7-9: population 30, p=10-12: population 15
- p≥13: single L-BFGS warm-start (too expensive for population search)

## What each node computes

| Array ID | (k,D) | Target p | Precision |
|----------|-------|----------|-----------|
| 0 | (3,4) | 16 | D64 from p=14 |
| 1 | (3,5) | 16 | D64 from p=14 |
| 2 | (3,6) | 16 | D64 from p=12 |
| 3 | (3,7) | 15 | D64 from p=12 |
| 4 | (3,8) | 15 | D64 from p=12 |
| 5 | (4,5) | 14 | D64 from p=12 |
| 6 | (4,6) | 14 | D64 from p=11 |
| 7 | (4,7) | 14 | D64 from p=11 |
| 8 | (4,8) | 14 | D64 from p=10 |
| 9 | (5,6) | 13 | D64 from p=10 |
| 10 | (5,7) | 13 | D64 from p=10 |
| 11 | (5,8) | 13 | D64 from p=10 |
| 12 | (6,7) | 12 | D64 from p=9 |
| 13 | (6,8) | 12 | D64 from p=9 |
| 14 | (7,8) | 12 | D64 from p=8 |

## Resources per tier

| Tier | Partition | Nodes needed | RAM/node | CPUs/node | Wall time |
|------|-----------|-------------|----------|-----------|-----------|
| k=3 | c3dssd | 5 | 1400G | 176 | 504h |
| k=4 | n2 | 4 | 850G | 124 | 504h |
| k≥5 | c2 | 6 | 235G | 56 | 504h |

## Monitoring

```bash
squeue -u $USER
tail -f logs/xorsat-0-*.out
grep "^  ✓" logs/xorsat-*.out
```

## If killed: jobs have --requeue, resume from CSV automatically.

## Results: results/slurm-xorsat-k{K}-d{D}.csv
