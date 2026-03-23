# Optimization Data Protocol

## Goal

Preserve optimisation runs as immutable historical artefacts so we can:

- chart value vs. depth `p`
- chart runtime vs. depth `p`
- compare search-budget changes over time
- demonstrate improvements in the optimiser or evaluator

## Storage Layout

All optimisation artefacts live under:

`/.project/results/optimization/`

with two layers:

1. `index.csv`
2. `runs/<run_id>/`

## Append-Only Rule

Runs are append-only.

- Never overwrite an old run directory.
- Never rewrite historical rows in `index.csv`.
- New runs append new rows and create a new run directory.

If a run is invalid, mark it in downstream analysis rather than mutating the raw archive.

## Run Identity

Each invocation gets a unique `run_id` of the form:

`YYYYMMDDTHHMMSS-kK-dD-pPmin-Pmax-rR-iI-sSEED`

This captures the core budget knobs in the identifier itself.

## Per-Run Artefacts

Each run directory contains:

1. `manifest.json`
   - timestamp
   - git commit
   - git branch
   - git dirty flag
   - `(k, D)`
   - depth range
   - clause sign
   - restart budget
   - iteration budget
   - RNG seed

2. `results.csv`
   - one row per depth `p`
   - includes value and runtime fields for charting

## Global Index

`index.csv` is the charting-friendly aggregate table.

It stores one row per `(run_id, p)` with the columns:

- `run_id`
- `timestamp_utc`
- `git_commit`
- `git_branch`
- `git_dirty`
- `k`
- `D`
- `p`
- `clause_sign`
- `restarts`
- `maxiters`
- `seed`
- `value`
- `wall_time_seconds`
- `best_start_wall_time_seconds`
- `evaluations`
- `starts`
- `iterations`
- `converged`
- `retry_count`
- `best_start_kind`
- `gamma`
- `beta`

This file is the canonical source for plotting historical performance and runtime curves.

## Charting Guidance

For performance charts:

- plot `value` vs `p`
- group by `run_id` or by `(git_commit, restarts, maxiters)`

For timing charts:

- plot `wall_time_seconds` vs `p`
- use `best_start_wall_time_seconds` when you specifically want the winning local solve rather than full multistart cost
- compare across commits to show evaluator or optimiser improvements

For optimisation-quality charts:

- compare `value` against `evaluations`, `starts`, and `iterations`
- separate `converged=true` from capped runs
- use `retry_count` to identify depths that needed an extra warm-seed pass
- use `best_start_kind` to distinguish warm-start wins from random-start wins

## Interpretation Rule

Every stored value is a lower bound from the search budget that produced it.

- Higher values at the same `(k, D, p)` are better optimisation outcomes.
- Lower runtime at the same value indicates an efficiency improvement.
- Non-converged runs remain useful and should still be preserved.
- `git_dirty=true` means the run came from uncommitted local changes and should not be used as a clean commit-to-commit benchmark.

## Operational Rule

When running `scripts/optimize_qaoa.jl`, leave preservation enabled unless you are doing a disposable local probe.
