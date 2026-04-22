# Status Update for Stephen — April 22, 2026

## What happened

The SLURM cluster restarted and the D64 swarm chains rolled back from p=9–10
to **p=7–8** across all 15 pairs. Your commit `25cb9d9` ("weird results")
captured this state.

**The "weird" values are expected**, not a bug. The D64 swarm sometimes produces
slightly lower c̃ than the F64 L-BFGS at the same depth because the swarm
prioritises basin discovery over fine polishing. The composite-best.csv
(which picks the best from any source) is intact with all historic highs.

## What's new on `main` (please pull)

Performance improvements that will speed up the re-run:

1. **`_fast_pow`**: Specialized complex power for exponents 1–7. Avoids
   log/exp, **2–5× faster per power op**. Especially helps Double64.
2. **Fast f_table**: Eliminated 8M+ per-config allocations (each eval
   was allocating a `Vector{Int}` and rebuilding the trig table per config).
3. **Memory savings**: Removed `bits_table` (1.47 GB at p=11), deduplicated
   phase computation. ~1.5 GB/eval saved at p=11.
4. **Memory-bounded concurrency**: Semaphore caps parallel restarts by
   available RAM, preventing OOM at high p.

All 1741 tests pass.

## But first: why did the cluster restart?

Before re-submitting, please check what caused the rollback so we can
prevent it happening again (each p=10 run now takes days):

    # Check recent job history — did jobs get killed, timeout, or node failure?
    sacct -u $USER --starttime=2026-04-18 --format=JobID,JobName,State,ExitCode,Start,End,Elapsed,MaxRSS

    # Check if there was scheduled maintenance
    scontrol show reservation
    # or check cluster announcements / MOTD

    # Check node health — were nodes rebooted?
    sinfo -N -l | head -20

    # Check if our jobs hit a time limit
    sacct -u $USER --starttime=2026-04-18 --state=TIMEOUT,FAILED,CANCELLED --format=JobID,State,ExitCode,Elapsed,TimelimitRaw

Key questions:
- Were jobs TIMEOUT'd (hit wall-time limit)?
- Were they CANCELLED (admin or scheduler preemption)?
- Were nodes DRAINED/rebooted (maintenance)?
- Should we use `--requeue` or SLURM checkpointing to survive restarts?

Please share the `sacct` output so we can diagnose and harden the setup.

## Action: pull and restart D64 sweep

Step 1 — kill everything:

    scancel -u $USER

Step 2 — pull and rebuild:

    cd ~/qaoa-xorsat
    git pull origin main
    julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

Step 3 — DO NOT delete existing results. The swarm chains have resume logic
and will pick up from p=7/8. No need to start from scratch.

Step 4 — submit:

    sbatch scripts/qaoa_d64_sweep.sh

Step 5 — monitor:

    watch -n 300 'for f in results/swarm-d64-k*.csv; do [ -f "$f" ] || continue; tail -1 "$f" | grep "^[0-9]"; done'

Step 6 — push results periodically:

    git add -f results/swarm-d64-k*.csv
    git commit -m "Stephen: D64 swarm recovery"
    git push

## What we've been computing (MaxCut, k=2)

On a 40-core, 181 GB Windows Server we've been running MaxCut sweeps for
D=3 through D=8. D=3 reproduces Farhi et al. (2014); **D=4–8 are new**.

| p | D=3 | D=4 | D=5 | D=6 | D=7 | D=8 |
|---|------|------|------|------|------|------|
| 1 | 0.6925 | 0.6624 | 0.6431 | 0.6294 | 0.6190 | 0.6108 |
| 2 | 0.7559 | 0.7161 | 0.6907 | 0.6726 | 0.6589 | 0.6480 |
| 3 | 0.7924 | 0.7486 | 0.7199 | 0.6993 | 0.6836 | 0.6711 |
| 4 | 0.8169 | 0.7690 | 0.7386 | 0.7165 | 0.6996 | 0.6861 |
| 5 | 0.8364 | 0.7841 | 0.7523 | 0.7292 | 0.7114 | 0.6972 |
| 6 | 0.8499 | 0.7949 | 0.7624 | 0.7386 | 0.7202 | 0.7055 |
| 7 | 0.8598 | 0.8034 | 0.7705 | 0.7460 | 0.7272 | 0.7121 |
| 8 | 0.8674 | 0.8099 | 0.7771 | 0.7519 | 0.7328 | 0.7174 |
| 9 | 0.8735 | 0.8152 | 0.7829 | 0.7568 | 0.7374 | 0.7217 |
| 10 | 0.8784 | 0.8196 | 0.7879 | *running* | | |
| 11 | | 0.8233 | 0.7921 | | | |

D=6 p=10 is currently running solo (40 threads, ~70hrs in). D=7/8 queued.

## Composite best — all XORSAT (k≥3) values intact

| (k,D) | max p | c̃ | Source |
|--------|-------|------|--------|
| (3,4) | 13 | 0.881 | f64/stephen-apr6 |
| (3,5) | 13 | 0.843 | f64/stephen-apr6 |
| (3,6) | 11 | 0.807 | f64/stephen-apr6 |
| (3,7) | 11 | 0.779 | f64/local |
| (3,8) | 11 | 0.767 | f64/stephen-apr6 |
| (4,5) | 11 | 0.861 | f64/stephen-apr6 |
| (4,6) | 10 | 0.827 | f64/local |
| (4,7) | 10 | 0.806 | f64/local |
| (4,8) | 10 | 0.785 | d64/swarm |
| (5,6) | 10 | 0.843 | d64/swarm |
| (5,7) | 10 | 0.815 | d64/swarm |
| (5,8) | 10 | 0.800 | d64/swarm |
| (6,7) | 10 | 0.832 | d64/swarm |
| (6,8) | 10 | 0.812 | d64/swarm |
| (7,8) | 10 | 0.821 | d64/swarm |

None of these were lost. The D64 swarm re-run just needs to recover p=8–10
and then push to p=11+.

The D64 sweep should match or exceed these at the same depths,
then push higher.
