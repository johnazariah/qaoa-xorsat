# Instructions for Stephen — April 8, 2026 (updated)

## What's working

Tasks 1, 2, 4, 6 — (3,4), (3,5), (3,7), (4,5) — are producing good results at p=14. **Don't touch these.**

## What's stuck

The other pairs show 0.500 because the warm-start angles came from pre-normalization runs. Those angles were in overflow-adjacent basins that evaluate to 0.500 with the corrected code. No amount of L-BFGS will fix them — they need new basins found from scratch.

## Fix: run the swarm optimizer on the stuck pairs

    cd ~/qaoa-xorsat
    git pull origin main
    sbatch scripts/qaoa_swarm_sweep.sh

This submits 10 SLURM tasks — one for each stuck pair:

    (3,8) (4,6) (4,7) (4,8) (5,6) (5,7) (5,8) (6,7) (6,8) (7,8)

Each task runs the swarm/memetic optimizer from p=1 through p=15:
- 100 random candidates per generation
- Short L-BFGS bursts, cull worst 50%, crossover from survivors
- Early exit when the population stops improving, then full L-BFGS polish
- No dependency on old angles — finds basins from scratch

Results appear immediately in `results/swarm-k{K}d{D}.csv` as each depth completes.

## Monitoring

    squeue -u $USER
    cat results/swarm-k7d8.csv    # check a specific pair

## When ready

    git checkout -b stephen-swarm-results
    git add -f results/swarm-*.csv
    git commit -m "Stephen: swarm results"
    git push origin stephen-swarm-results
