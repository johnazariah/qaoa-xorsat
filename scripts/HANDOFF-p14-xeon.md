# Handoff: finish p=14 D=3 MaxCut on the Xeon

**Date opened:** 2026-05-26
**Branch:** `feature/charge-decomposition-v2`
**Status:** waiting for a host with > 96 GB RAM

## Why this exists

The goal: complete the MaxCut k=2, D=3 sweep that previously stopped at p=13.
On 2026-05-26 we tried to finish p=14 on a 64 GB M-series Mac using the
manual charge adjoint. The process was silently killed by macOS `jetsam`
after the forward eval and during the first adjoint gradient.

What we measured on the Mac:

| stage                              | value                  |
|------------------------------------|------------------------|
| forward eval at p=14               | 30.7 s                 |
| RSS after forward (single eval)    | 25.9 GB                |
| adjoint gradient at p=14           | killed before finishing |
| estimated peak during adjoint      | ~40-45 GB              |

Root cause: the cached forward in
[src/charge_manual_adjoint.jl](../src/charge_manual_adjoint.jl) keeps
`F_levels[1..p]` and `children[1..p]` resident during the backward pass.
At p=14 each level is a `4^14 = 268M`-element `ComplexF64` vector
(4.3 GB). With working scratch + Julia's GC headroom, peak exceeds
what macOS will give a single user-process before jetsam triggers
(typically around 45-55 GB on a 64 GB box once Code/browsers/Slack are
running).

128 GB Xeon resolves this directly — no code change required.

## What's in the repo

- [scripts/run_p14_d3_warm.jl](run_p14_d3_warm.jl) — the run script.
  Loads optimal p=13 angles from `results/maxcut-k2-d3-sweep.csv`,
  extends them to p=14 with `extend_angles`, times one forward and one
  adjoint gradient, then runs `optimize_angles` with
  `autodiff=:charge_adjoint`, `restarts=0`, `maxiters=200`.
- The p=13 starting point is on disk: the last `k=2,D=3,p=13` row of
  `results/maxcut-k2-d3-sweep.csv` (c̃₁₃ = 0.888870518623, recorded
  back in early May). Do not regenerate it.

## Run instructions for the Xeon

### 1. Verify environment

```bash
julia --version       # need 1.12.x
free -g               # need >= 96 GB available
nproc                 # cores; the algorithm scales modestly past ~8
```

If Julia isn't installed, use `juliaup`:

```bash
curl -fsSL https://install.julialang.org | sh
juliaup add 1.12
juliaup default 1.12
```

### 2. Sync the repo

Clone fresh, or pull. Make sure you are on `feature/charge-decomposition-v2`:

```bash
git clone git@github.com:johnazariah/qaoa-xorsat.git    # or pull
cd qaoa-xorsat
git checkout feature/charge-decomposition-v2
git pull --ff-only
```

### 3. Instantiate

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

This will take a few minutes the first time. Watch for any registry
warnings about `Double64` (the `DoubleFloats` package is a transitive
dep used by the Basso path, but the charge path uses `ComplexF64`
directly, so a missing `DoubleFloats` here is OK — it should resolve
cleanly).

### 4. Smoke test

Confirm the adjoint runs end-to-end at p=10 (~75 s, < 1 GB):

```bash
julia --project=. -e '
  using QaoaXorsat, Random
  params = QaoaXorsat.TreeParams(2, 3, 10)
  angles = QAOAAngles(fill(0.1, 10), fill(0.2, 10))
  r = QaoaXorsat.charge_expectation_and_gradient(params, angles; clause_sign=-1)
  println("ok: ", r[1])
'
```

Expect `ok: <some value near 0.7>` in about a minute. If this fails,
fix it before launching the real run.

### 5. Launch the real run (detached)

```bash
mkdir -p logs results
LOG=logs/p14-d3-$(date +%Y%m%dT%H%M%S).log

# Use tmux (preferred) so you can re-attach:
tmux new -d -s p14 \
  "julia --project=. -t auto scripts/run_p14_d3_warm.jl 2>&1 | tee $LOG"

# Or nohup if tmux is unavailable:
# nohup julia --project=. -t auto scripts/run_p14_d3_warm.jl > $LOG 2>&1 &
# echo $! > logs/p14.pid

tmux ls
echo "log: $LOG"
```

`-t auto` lets Julia pick threads; expect best results at ~8-16 threads
(the algorithm is level-sequential, so more threads only helps the
mode-product BLAS calls).

### 6. Monitor

```bash
tail -f logs/p14-d3-*.log
```

What you should see in order:

```
Loading saved p=13 angles... done.  p=13 c̃ = 0.888870518623
Extended to p=14  (γ length=14, β length=14)
[timing] single charge forward eval at p=14... XX.XXs
[memory] RSS after fwd: XX.XX GB
[timing] single charge adjoint gradient at p=14... XXX.XXs  (~5x fwd)
[memory] RSS after grad: XX.XX GB
============================================================
Optimizing p=14 (autodiff=:charge_adjoint, restarts=0)...
============================================================
  [   XXXs] evals=  1  c̃=0.XXXXXXXXXXXX  ||g||=X.XXe-XX  rss=XX.XXGB
  [   XXXs] evals=  2  ...
```

**Sanity check after the warm c̃ prints**:
the first reported `c̃` will be the *parity* `⟨Z⊗Z⟩` (around 0.749),
not the cost `(1 − ⟨Z⊗Z⟩)/2`. The optimizer's `c̃` (next block) is
the actual cost and should start near 0.8889 (matching p=13) and
climb toward 1.

**Memory check**: if RSS approaches 80 GB, kill the run and ping me —
that means something has regressed since the Mac measurement.

### 7. When it finishes

The script writes three artefacts:

- `results/maxcut-k2-p14-timing.csv` — appended summary row
- `results/maxcut-k2-d3-p14-angles.txt` — final γ, β
- `results/maxcut-k2-d3-p14-progress.csv` — per-evaluation trace

Also append a row to `results/maxcut-k2-d3-sweep.csv` so future
sweeps can pick up from p=14:

```bash
julia --project=. -e '
  using QaoaXorsat, DelimitedFiles
  # parse angles file -> append CSV row in the existing schema:
  #   k,D,p,ctilde,wall_seconds,gamma_semicolon_list,beta_semicolon_list
'
```

(If that one-liner is awkward, just append the row by hand from the
values in `maxcut-k2-d3-p14-angles.txt` and
`maxcut-k2-p14-timing.csv`.)

Commit the new result files:

```bash
git add results/maxcut-k2-p14-timing.csv \
        results/maxcut-k2-d3-p14-angles.txt \
        results/maxcut-k2-d3-p14-progress.csv
# optionally also the appended sweep row:
git add results/maxcut-k2-d3-sweep.csv
git commit -m "results(maxcut): add p=14 D=3 (charge_adjoint, Xeon)"
git push
```

## Success criteria

1. The run finishes (`converged = true`, or it bottoms out at
   `maxiters=200`).
2. The final c̃ at p=14 is **at least** c̃₁₃ = 0.888870518623, and
   ideally noticeably above it. If c̃₁₄ < c̃₁₃, the warm-start
   failed and we need to debug.
3. Peak RSS stayed under ~80 GB.

## If it dies again

- **OOM on the Xeon** (peak > available RAM): not expected, but if
  it happens, drop in `option (C)` — gradient checkpointing in
  [src/charge_manual_adjoint.jl](../src/charge_manual_adjoint.jl).
  The cache (`FwdCache` in that file) stores `F_levels[1..p]` and
  `children[1..p]`; switching to a `√p`-stride checkpoint scheme
  cuts the resident set roughly in half at ~1.5x backward cost.
- **NaN / garbage gradient**: likely a precision issue; try halving
  `t_max`-based normalization step sizes inside
  `_charge_branch_instrumented`, or re-run with the FD fallback
  (`autodiff=:charge`) at p=14 just to confirm the optimum.
- **Slow convergence (no improvement past evals=20)**: the warm
  start is local-minimum-bound; retry with two extra random
  restarts (`restarts=2`).

## Related files

- [src/optimization.jl](../src/optimization.jl) — `optimize_angles`,
  `extend_angles`, `AngleOptimizationResult`
- [src/charge.jl](../src/charge.jl) — forward evaluator
  (`charge_parity_expectation`, `charge_expectation`)
- [src/charge_manual_adjoint.jl](../src/charge_manual_adjoint.jl) —
  manual adjoint (`charge_expectation_and_gradient`, `_fwd_cached`,
  `_bwd_root!`, `_bwd_branch!`)
- [results/maxcut-k2-d3-sweep.csv](../results/maxcut-k2-d3-sweep.csv) —
  source of the p=13 warm-start row
