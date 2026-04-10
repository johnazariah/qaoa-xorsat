# Status Update for Stephen — April 11, 2026

## CRITICAL: Previous D64 results are garbage — must restart

We discovered overnight that the previous `swarm_chain_d64.jl` was
running the optimizer in Float64 and only re-evaluating the winner in
Double64. This means:

- The L-BFGS was following Float64 gradients (corrupted at high k,D,p)
- It converged to angles that minimize Float64 noise, not the real objective
- Re-evaluating garbage angles in D64 gives you the true value of garbage

**Evidence:**
- (6,7) p=9: F64 optimized → c̃=0.855, D64 re-eval → 0.814, pure D64 → TBD
- (7,8) p=9: F64 optimized → c̃=0.999, D64 re-eval → 0.629 (worse than p=7!)
- (6,8) p=9: F64 optimized → c̃=0.948, D64 re-eval → 0.798

The fix (now pushed to main): `eval_eltype=Double64` parameter that makes
the optimizer evaluate f and ∇f in Double64 throughout. The L-BFGS still
operates on Float64 parameter vectors (Optim.jl requirement), but every
function/gradient call promotes angles to D64 before evaluation.

## What happened to the cluster jobs

From the `.err` logs you pushed (thank you!):

**Three job IDs** (1323963, 1323978, 1323993) from what was likely a
single run of `run-d64-sweep.sh`:

There's a bug in `run-d64-sweep.sh`: if `sbatch --parsable` returns
anything unexpected (a warning, extra whitespace, etc.), the error
handling path runs `sbatch` a SECOND time to "show the error details"
— accidentally submitting a duplicate job. So:

- Job 1323963: First submission (from `sbatch --parsable`)
- Job 1323978: Second submission (from the error-handler's `sbatch`)
- Script exits with error
- Job 1323993: You likely ran `sbatch` manually after the script failed

Two overlapping array jobs on the same nodes causes resource contention.
Tasks 1=(3,4), 3=(3,6), 4=(3,7), 6=(4,5) got SIGTERM (signal 15) —
probably evicted by the scheduler to make room, or you manually
cancelled the stray jobs. Tasks 2=(3,5) and 5=(3,8) have no .err file
at all — they may not have been allocated nodes.

**All 15 pairs DID produce CSV data** through p=7-9. The bug has been
fixed: the error handler now prints a message instead of re-submitting.

**Bottom line:** Nothing is wrong with the compute code. The SLURM
submission script had a double-submit bug that caused chaos.

## Action: pull new code and restart from scratch

The code on `main` now has the pure D64 fix. **You must restart from
p=1** because all existing D64 results were optimised with corrupted
Float64 gradients.

Step 1 — kill everything:

    scancel -u $USER

Step 2 — pull and rebuild:

    cd ~/qaoa-xorsat
    git pull origin main
    julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

Step 3 — delete old (garbage) results:

    rm -f results/swarm-d64-k*.csv

Step 4 — submit directly (don't use run-d64-sweep.sh to avoid the scancel issue):

    sbatch scripts/qaoa_d64_sweep.sh

Step 5 — monitor (optional, in a separate terminal):

    watch -n 300 'for f in results/swarm-d64-k*.csv; do [ -f "$f" ] || continue; tail -1 "$f" | grep "^[0-9]"; done'

Step 6 — push results periodically:

    git add -f results/swarm-d64-k*.csv
    git commit -m "Stephen: pure D64 swarm results"
    git push origin HEAD:stephen-d64-results

**Important:** Use `git push origin HEAD:stephen-d64-results` (not
`git push origin stephen-d64-results`) — the latter fails with
"src refspec does not match" because there's no local branch with
that name; HEAD:remote-branch is the correct syntax.

## What's different in the new code

1. `src/optimization.jl`: Added `eval_eltype` keyword to `optimize_angles()`
   and `swarm_optimize()`. When set to `Double64`, all f/∇f evaluations
   promote angles to Double64 before calling the evaluator. Gradients
   are computed in D64 and converted back to Float64 for L-BFGS.

2. `scripts/swarm_chain_d64.jl`: Now passes `eval_eltype=Double64` to
   `swarm_optimize`. No more "optimize in F64, re-evaluate in D64" hack.

3. `scripts/run-d64-sweep.sh`: Added cleanup of old CSV files. But I
   recommend NOT using this script (see Step 4 above) — its `scancel -u`
   is what killed your previous runs.

## Expected timing

Pure D64 is ~3-5× slower than Float64 per evaluation. Rough estimates
per depth (wall time for one (k,D) pair with 28 threads, pop=100):

    p=1-5:  minutes
    p=6-8:  1-4 hours
    p=9:    5-20 hours (varies by k,D)
    p=10:   1-3 days
    p=11+:  days to week

With 55 nodes running all 15 pairs simultaneously, you should have
p=9 for all pairs within ~24 hours and p=10+ within a few days.

## Best values (what we can trust from Float64 runs)

These values from the PREVIOUS Float64 runs are still valid because
they're below the precision wall:

    (3,4) p=13 c̃=0.881   ← F64 reliable through p=13
    (3,5) p=13 c̃=0.843
    (3,6) p=11 c̃=0.807
    (3,7) p=11 c̃=0.779
    (3,8) p=11 c̃=0.768
    (4,5) p=11 c̃=0.861
    (4,6) p=10 c̃=0.827
    (4,7) p=10 c̃=0.806   ← from local runs, not swarm
    (4,8) p=9  c̃=0.779
    (5,6) p=9  c̃=0.838
    (5,7) p=9  c̃=0.815   ← from warm-start, not swarm
    (5,8) p=9  c̃=0.805
    (6,7) p=8  c̃=0.819   ← p=9 was garbage (F64 said 0.855)
    (6,8) p=8  c̃=0.802
    (7,8) p=7  c̃=0.800   ← p=8 was inflated (0.819 F64 vs 0.803 D64)

The D64 sweep should match or exceed these at the same depths,
then push higher.
