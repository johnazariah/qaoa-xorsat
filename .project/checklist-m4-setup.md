# M4 Host Setup Checklist

## Prerequisites

- [ ] macOS on Apple Silicon M4 (verify: `uname -m` → `arm64`)
- [ ] Git installed (`git --version`)
- [ ] SSH key configured for GitHub (`ssh -T git@github.com`)
- [ ] Terminal open (not inside devcontainer)

## Automated Setup

```bash
# Run from the Mac host terminal (NOT inside VS Code devcontainer)
curl -fsSL https://raw.githubusercontent.com/johnazariah/qaoa-xorsat/main/scripts/setup-m4.sh | bash
```

Or if you already have the repo cloned:

```bash
cd ~/work/qaoa-xorsat
bash scripts/setup-m4.sh
```

## What the Script Does

1. **Reports system info** — CPU cores, memory, GPU cores
2. **Installs juliaup** (if not present) and ensures latest Julia
3. **Clones/updates the repo** to `~/work/qaoa-xorsat` (override with `QAOA_REPO_DIR`)
4. **Installs Julia dependencies** and precompiles
5. **Runs the full test suite** with all threads (`-t auto`)
6. **Benchmarks single evaluations** at p=5,8,10,12 to establish a baseline

## Manual Verification

After setup, verify:

```bash
cd ~/work/qaoa-xorsat

# Check threads
julia -e 'println("Threads: ", Sys.CPU_THREADS)'
# Expected: 10 (M4)

# Quick smoke test
julia --project=. -t 10 -e '
using QaoaXorsat
result = basso_expectation(TreeParams(3, 4, 1), QAOAAngles([0.3], [0.5]))
println("k=3, D=4, p=1: $result")
@assert 0.0 ≤ result ≤ 1.0
println("OK")
'
```

## Running Experiments

### Quick sweep (p=1–8, ~5 min on M4)

```bash
julia --project=. -t 10 scripts/optimize_qaoa.jl 3 4 1 8 4 200 1234 true
```

### Full sweep (p=1–12, estimate ~1–2 hours on M4)

```bash
julia --project=. -t 10 scripts/optimize_qaoa.jl 3 4 1 12 4 200 1234 true
```

### MaxCut validation (p=1–5, ~1 min)

```bash
julia --project=. -t 10 scripts/optimize_qaoa.jl 2 3 1 5 8 200 1234 true
```

### Key flags

| Flag | Meaning |
|------|---------|
| `-t 10` | Use 10 threads (all M4 performance cores) |
| `3 4` | k=3, D=4 (our XORSAT target) |
| `1 12` | p_min=1, p_max=12 |
| `4` | 4 restarts per depth (+ warm start from previous) |
| `200` | max L-BFGS iterations per start |
| `1234` | RNG seed (reproducibility) |
| `true` | preserve results to `.project/results/` |

### Monitoring

While running, in another terminal:

```bash
# Watch CPU usage
top -l 1 -n 5 -s 0

# Watch memory
vm_stat | head -10

# Watch GPU (if Metal.jl is used)
sudo powermetrics --samplers gpu_power -i 1000 -n 5
```

## Results Location

Results are saved to:
- `.project/results/optimization/index.csv` — aggregated CSV
- `.project/results/optimization/runs/<run_id>/` — per-run details

After a run, push results:

```bash
git add .project/results/
git commit -m "results: XORSAT k=3 D=4 p=1-12 on M4"
git push origin main
```

## Troubleshooting

### Julia not found after install
```bash
export PATH="$HOME/.juliaup/bin:$PATH"
```

### Tests fail with package errors
```bash
julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
```

### Out of memory at high p
At p=12, the branch tensor needs ~256 MB. At p=14, ~4 GB. Check available memory:
```bash
sysctl -n hw.memsize | awk '{printf "%.0f GB available\n", $1/1073741824}'
```

### Optimizer not converging
Check that `g_abstol` is set to 1e-6 (not the default 1e-8). See `src/optimization.jl`.
