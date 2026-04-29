# SLURM Startup - QAOA-XORSAT

The supported SLURM workflow is Stephen's Max-k-XORSAT cluster run. Local MaxCut runs do not use SLURM; use `scripts/start-maxcut-local.sh` or `scripts/start-maxcut-local.ps1` for those. Local MaxCut logs are written under `logs/maxcut-local/`.

## Stephen Cluster Run

On the SLURM login node:

```bash
cd ~/qaoa-xorsat
git pull --ff-only origin main
bash scripts/start-xorsat-slurm.sh
```

For a dry run:

```bash
bash scripts/start-xorsat-slurm.sh --dry-run
```

The startup wrapper submits:

```bash
sbatch scripts/qaoa_cluster_p16.sh
```

`qaoa_cluster_p16.sh` runs one SLURM array task per target `(k,D)` pair and calls `scripts/cluster_p16_chain.jl` with:

- Double64 evaluation arithmetic
- CPU gradient checkpointing
- per-task progress logs under `logs/cluster-p16/`
- checkpoint spill directories under `$TMPDIR` by default
- result CSVs under `results/cluster-p16-k{K}d{D}.csv`
- auto-push of result/log commits to `cluster-p16-results`

## Target Pairs

| Task | k | D | Target p |
|------|---|---|----------|
| 1 | 3 | 4 | 16 |
| 2 | 3 | 5 | 16 |
| 3 | 3 | 6 | 15 |
| 4 | 3 | 7 | 14 |
| 5 | 3 | 8 | 14 |
| 6 | 4 | 5 | 14 |
| 7 | 4 | 6 | 13 |
| 8 | 4 | 7 | 13 |
| 9 | 4 | 8 | 13 |

## Useful Overrides

```bash
QAOA_REPO=$HOME/qaoa-xorsat
QAOA_PUSH_BRANCH=cluster-p16-results
QAOA_POPULATION=100
QAOA_GENERATIONS=10
QAOA_BURST=20
QAOA_MAX_RAM_CHECKPOINTS=4
QAOA_SWARM_CONCURRENCY=1
```

Example:

```bash
QAOA_SWARM_CONCURRENCY=1 QAOA_MAX_RAM_CHECKPOINTS=4 bash scripts/start-xorsat-slurm.sh
```

## Legacy Names

The old startup names remain only as compatibility wrappers and route to the current p16 cluster workflow:

- `scripts/run-d64-sweep.sh`
- `scripts/qaoa_sweep.sh`
- `scripts/qaoa_d64_sweep.sh`
- `scripts/qaoa_swarm_sweep.sh`
- `scripts/qaoa_warmstart_sweep.sh`

Do not use broad user-wide cancellation as a startup step; cancel only specific job IDs.
