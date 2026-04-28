# Q1 Results — Is QAOA Trotterised Adiabatic Optimisation?

Branch: `q1-adiabatic`. Spec: `.project/SPEC-Q1-adiabatic.md`.

All four experiments execute against the existing
`results/maxcut-k2-d{3..8}-sweep.csv` warm-start data (no new long-running
optimisations needed for E1/E2/E4; E3 runs L-BFGS from adiabatic seeds).

## Reproduce

```bash
# from the qaoa-xorsat-q1 worktree
julia --project=. -t 16 scripts/q1_intermediate_depth.jl    # E2
julia --project=. -t 16 scripts/q1_angle_schedules.jl       # E1
julia --project=. -t 16 scripts/q1_angle_curvature.jl       # E4
julia --project=. -t 16 scripts/q1_adiabatic_init.jl        # E3 (~25 min)
python3 scripts/q1_plots.py                                 # figures
```

The Julia scripts only depend on the `QaoaXorsat` package already
declared in `Project.toml`. The plot script uses `matplotlib` + `numpy`
from the system `python3`.

## Output files

| File | Experiment | Contents |
|------|------------|----------|
| `results/q1-intermediate-depth.csv`    | E2 | per-(D, t) c̃ for truncated, optimal-at-t, and linear-adiabatic schedules |
| `results/q1-angle-schedules.csv`       | E1 | per-(D, j) optimal vs linear-adiabatic γ_j, β_j (unwrapped) |
| `results/q1-adiabatic-fidelity.csv`    | E1 | per-D c̃_opt, c̃_adi, Δ, relative loss |
| `results/q1-angle-curvature.csv`       | E4 | r², rmse, polynomial coefficients for deg=1..3 fits |
| `results/q1-adiabatic-init.csv`        | E3 | per-(D, γ_max, β_max) c̃(seed), c̃(adi-opt), c̃(warm), iterations |
| `figures/q1-angle-schedules.png`       | E1 | side-by-side γ and β schedules |
| `figures/q1-intermediate-depth.png`    | E2 | c̃ vs depth t for truncated/optimal/adiabatic |
| `figures/q1-adiabatic-init.png`        | E3 | best-of-grid c̃ after L-BFGS vs warm-start |
| `figures/q1-angle-curvature.png`       | E4 | r²(linear) and Δr²(linear→cubic) bars per D |

## Headline finding

**The optimal QAOA schedules are not consistent with a Trotterisation of
linear adiabatic optimisation on D-regular MaxCut on the infinite-girth
tree.** See `.project/journal.md` "Entry 32" for the full write-up,
tables, and caveats.
