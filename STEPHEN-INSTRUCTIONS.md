# Cluster Instructions for Stephen — April 30, 2026

## TL;DR — Clean Run

**This is a clean run. Please delete all old results before starting.**

```bash
git pull origin main

# Clean out ALL old results and logs
rm -rf results/slurm-*.csv results/*.csv results/*.machine logs/

# Fresh start
mkdir -p logs
sbatch scripts/slurm_xorsat_all.sh
```

This launches **15 SLURM array jobs** (one per (k,D) pair), all running in parallel.
Every (k,D) pair starts from p=1 — no warm-starting from previous runs.

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

## Resources: 1400G RAM, 32 CPUs, 504h wall time per node

## Monitoring

```bash
squeue -u $USER
tail -f logs/xorsat-0-*.out
grep "^  ✓" logs/xorsat-*.out
```

## If killed: jobs have --requeue, resume from CSV automatically.

## Results: results/slurm-xorsat-k{K}-d{D}.csv
