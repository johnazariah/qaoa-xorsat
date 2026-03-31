# Deployment State — March 31, 2026

## Immediate Action: Deploy Phase 1

From Windows (Git Bash or WSL):
```bash
git clone https://github.com/johnazariah/qaoa-xorsat.git
cd qaoa-xorsat
bash infra/deploy-fleet.sh -s YOUR-SUBSCRIPTION-ID -p 12 --dry-run
bash infra/deploy-fleet.sh -s YOUR-SUBSCRIPTION-ID -p 12
```

## Phased Plan (budget: $500, deadline: April 13)

| Phase | Command | VMs | Cost | Time |
|-------|---------|-----|------|------|
| 1 (validate) | `deploy-fleet.sh -p 12` | 5× E8as_v5 (64GB) | ~$10 | ~2hr |
| 2 (production) | `deploy-fleet.sh -p 13` | 5× E32as_v5 (256GB) | ~$90 | ~12hr |
| 3 (high depth) | `deploy-fleet.sh -p 14` | 5× E64as_v5 (512GB) | ~$300 | ~72hr |

## What's Running

- **Google VM** (Stephen's): m4-ultramem-56, 28 cores, 1.5TB
  - Running all 15 pairs through p=14
  - Log: /home/stephenjordan_google_com/qaoa-xorsat/results/logs/cloud-20260328T030315-p14.log
  - Started: March 28

- **Mac Studio**: overnight p=11 sweep was 5 of 15 done (k=3 family complete)

## Confirmed Results

| (k,D) | Best p | c̃ | vs DQI+BP | vs Prange | vs Regev |
|--------|--------|------|-----------|-----------|----------|
| (3,4) | 12 | 0.8769 | +0.006 | +0.002 | -0.015 |
| (3,5) | 11 | 0.8352 | +0.019 | +0.035 | -0.001 |
| (3,6) | 11 | 0.8067 | +0.031 | +0.057 | +0.023 |
| (3,7) | 11 | 0.7765 | +0.030 | +0.063 | +0.017 |
| (3,8) | 11 | 0.7676 | +0.044 | +0.080 | +0.039 |
| Others | 8 | see results/qaoa-best-values.csv | | | |

## Key Scripts

| Script | Purpose |
|--------|---------|
| `infra/deploy-fleet.sh` | Deploy 5 VMs with pairs distributed |
| `infra/deploy-vm.sh` | Deploy single big VM |
| `infra/monitor.sh` | Monitor ACI containers |
| `scripts/cloud-run.sh` | One-command setup on any Linux VM |
| `scripts/run_parallel_table.jl` | Multi-pair parallel on big machines |
| `scripts/run_full_table.jl` | Sequential 15-pair sweep |
| `scripts/optimize_qaoa.jl` | Single (k,D) pair (CLI or TOML) |

## Memory Requirements

| p | Per pair | VM size |
|---|---------|---------|
| ≤12 | 19 GB | E8as_v5 (64GB) |
| 13 | 84 GB | E32as_v5 (256GB) |
| 14 | 394 GB | E64as_v5 (512GB) |
| 15 | 1.6 TB | M128s (2TB) — needs TVM exception |

## Context

- Stephen Jordan (Google Quantum AI) invited John as co-author
- Paper: "Optimization Using Locally-Quantum Decoders" (Shutty, Jordan et al.)
- arXiv target: week of April 13
- JPMorgan team also computing QAOA numbers — we need best values
- Our differentiator: open code, documented methodology, reproducible results
- No prior exact finite-D QAOA for k≥3 existed before this work
