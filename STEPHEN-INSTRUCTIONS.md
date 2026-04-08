# Instructions for Stephen — April 8, 2026

The dashboard you ran is showing stale Mac results from March 31, not your cluster data. Ignore those numbers.

## Check if jobs are already running

If you already ran `sbatch scripts/qaoa_warmstart_sweep.sh` yesterday, the jobs may be in progress or finished:

    squeue -u $USER
    sacct -u $USER --starttime=2026-04-07 --format=JobID,JobName,State,Elapsed

If you see `qaoa-ws` jobs, they're the warm-start sweep — no need to resubmit.

## If jobs haven't been submitted yet

    cd ~/qaoa-xorsat
    git pull origin main
    sbatch scripts/qaoa_warmstart_sweep.sh

That submits a 15-task SLURM array. Each task warm-starts from the best angles we've found across all machines (including your p=13 results for (3,4) and (3,5)). The code has the threshold-based normalization fix so the 0.500 collapses won't happen again.

## What each task does

- Tasks 1–2: (3,4) and (3,5) resume from your p=13, run p=14–15
- Tasks 3–5: rest of k=3 family, resume from p=11–14
- Tasks 6–9: k=4 family, resume from p=11–13
- Tasks 10–15: k≥5 pairs, resume from p=9–11

## Monitoring

Once jobs are running:

    squeue -u $USER

Check for completed results:

    ls -t .project/results/optimization/runs/ | head -20
    cat .project/results/optimization/index.csv | awk -F',' 'NR>1 {printf "k=%s D=%s p=%s val=%s\n", $8, $9, $10, $15}' | sort -t= -k2,2n -k4,4n -k6,6n

Results appear in `.project/results/optimization/runs/` as each depth completes.

## Push results when ready

    git checkout -b stephen-apr8-results
    git add -A .project/results/
    git commit -m "Stephen: cluster results Apr 8"
    git push origin stephen-apr8-results
