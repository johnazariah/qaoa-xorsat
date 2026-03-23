# Testing and Benchmarking Rollout Plan

## Goal

Implement the policy in `.project/protocols/testing-benchmarking-policy.md` with the
smallest sequence of changes that produces a stable and credible workflow.

## Phase 1 — Canonicalise Documentation

### Deliverables

- keep one durable policy document
- keep one durable rollout plan
- keep implementation-specific specs only where they belong

### Actions

1. Use `.project/protocols/testing-benchmarking-policy.md` as the canonical policy.
2. Keep `.project/specs/property-tests.md` as the implementation spec for
   property tests.
3. Keep `.project/testing-register.md` as the factual inventory of tests.
4. Remove transient handoff notes once their content has been extracted.

## Phase 2 — Complete Verification Baseline

### Deliverables

- `test/test_properties.jl`
- `test/runtests.jl` includes it
- full suite remains green

### Actions

1. Implement the five property-test families from `.project/specs/property-tests.md`.
2. Record the confirmed `β` periodicity convention in the tests or spec.
3. Optionally add guarded slow validation for Farhi 2025 table reproduction via
   an environment variable, but do not force it into ordinary CI.

## Phase 3 — Normalize Result Preservation

### Deliverables

- one canonical archive layout under `/.project/results/optimization/`
- one aggregate `index.csv`
- per-run immutable manifests and result tables

### Actions

1. Treat the append-only optimisation archive as canonical.
2. Extend run metadata to include `run_kind` and runner label if not already present.
3. Ensure local experimentation can preserve runs without pretending they are benchmark-grade.
4. Ensure benchmark-grade runs can be filtered cleanly from exploratory runs.

## Phase 4 — Provision The Dedicated Testbed

### Deliverables

- Ubuntu LTS server host
- repo-scoped self-hosted runner service
- runner labels: `self-hosted`, `linux`, `x64`, `testbed-48gb`
- stable Julia installation on `PATH`

### Actions

1. Provision the 48 GB machine as a dedicated runner host.
2. Disable suspend and lid-close sleep while on AC power.
3. Keep a small swap or `zram` safety buffer rather than disabling swap entirely.
4. Verify the machine can execute a representative optimisation run without throttling badly.

## Phase 5 — Harden GitHub Actions

### Deliverables

- fast correctness CI remains separate
- experiment workflow remains manual or controlled-schedule
- reproduction workflow remains deliberate
- benchmark-grade jobs are serialized

### Actions

1. Keep `.github/workflows/ci.yml` focused on correctness verification only.
2. Update `.github/workflows/optimize.yml` so it is explicitly an experiment workflow.
3. Update `.github/workflows/reproduce.yml` so it is explicitly a deliberate reproduction workflow.
4. Add workflow `concurrency` so the testbed runs one heavy job at a time.
5. Restrict testbed workflows to trusted triggers only.

## Phase 6 — Separate Generated Data From Source History

### Deliverables

- dedicated `results` branch for generated benchmark history
- source branches remain free of machine-generated result commits

### Actions

1. Change result-writing workflows to commit only to the `results` branch.
2. Ensure result-only pushes do not retrigger ordinary correctness CI.
3. Preserve append-only history there rather than on `main`.

## Phase 7 — Establish Benchmark Protocols

### Deliverables

- at least one named benchmark protocol
- one or more named experiment protocols
- one reproduction protocol

### Actions

1. Define a default experiment sweep, for example `(k=3, D=4, p=1..5)` with modest budgets.
2. Define a benchmark protocol with fixed budgets, seeds, and runner requirements.
3. Define a reproduction protocol that validates MaxCut first, then computes target XORSAT runs.
4. Document which charts are allowed to combine which run classes.

## Phase 8 — Reporting and Regression Tracking

### Deliverables

- chart-ready aggregate history
- simple summaries per workflow run
- clear regression questions that can actually be answered

### Actions

1. Use the aggregate index as the source for value-vs-`p` and runtime-vs-`p` plots.
2. Compare only benchmark-grade runs when discussing regressions.
3. Keep experiment runs visible, but off the main regression dashboard.

## Order of Execution

The recommended implementation order is:

1. Canonicalise docs
2. Finish property tests
3. Normalize preservation metadata
4. Provision the testbed runner
5. Harden workflows and concurrency
6. Move generated history onto the `results` branch
7. Run the first deliberate benchmark and reproduction cycles

## Done Criteria

This rollout is complete when:

1. The policy document is the only canonical operational source.
2. The property-test suite exists and passes.
3. The 48 GB machine is the canonical benchmark runner.
4. Heavy workflows are decoupled from ordinary CI.
5. Historical results are append-only, provenance-rich, and stored on the `results` branch.
6. We can answer, with evidence, whether a change improved or regressed runtime or optimisation quality.
