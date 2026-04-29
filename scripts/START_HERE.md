# Startup Scripts

Only two startup variants are supported.

## Local MaxCut

Use this on local machines, including the Windows workstation:

```powershell
.\scripts\start-maxcut-local.ps1 all auto 42
.\scripts\start-maxcut-local.ps1 8 12 42
```

```bash
bash scripts/start-maxcut-local.sh all auto 42
bash scripts/start-maxcut-local.sh 8 12 42
```

This runs `scripts/maxcut_sweep.jl` and writes `results/maxcut-k2-d*-sweep.csv`.
The sweep resumes from the last completed depth in each CSV. Console output is
also captured in timestamped logs under `logs/maxcut-local/`.

Compatibility wrappers:

- `setup-and-run-maxcut.sh`
- `setup-and-run-maxcut.ps1`
- `setup-p710.ps1`

## Stephen SLURM Max-k-XORSAT

Use this on Stephen's SLURM login node:

```bash
cd ~/qaoa-xorsat
git pull --ff-only origin main
bash scripts/start-xorsat-slurm.sh
```

This submits `scripts/qaoa_cluster_p16.sh`, which runs `scripts/cluster_p16_chain.jl`
with Double64 arithmetic, CPU checkpointing, progress logs, and auto-push.

Compatibility wrappers:

- `run-d64-sweep.sh`
- `qaoa_sweep.sh`
- `qaoa_d64_sweep.sh`
- `qaoa_swarm_sweep.sh`
- `qaoa_warmstart_sweep.sh`

Older cloud, batch, and experimental scripts are not startup entry points for the
current workflow.