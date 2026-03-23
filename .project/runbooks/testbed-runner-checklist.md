# Checklist: Repo Readiness and Local GitHub Runner Setup

Use this checklist to get the repository into a consistent operational state and
bring the dedicated 48 GB machine online as the canonical benchmark runner.

---

## 1. Repository Readiness

### Policy and docs

- [ ] Confirm the canonical policy is present in `.project/protocols/testing-benchmarking-policy.md`
- [ ] Confirm the rollout plan is present in `.project/implementation-notes/testing-benchmarking-rollout.md`
- [ ] Confirm the local experiment prompt is updated in `.project/runbooks/initial-qaoa-performance-sweep.md`
- [ ] Confirm transient handoff notes are removed or no longer treated as authoritative

### Code and test state

- [ ] In `/workspace`, review `git status --short` and understand all untracked and modified files
- [ ] In `/workspace/.worktree/phase4-optimization`, review `git status --short` and understand all modified and generated files
- [ ] Run the full test suite in the optimization worktree:

```bash
cd /workspace/.worktree/phase4-optimization
julia --project=. -e 'using Pkg; Pkg.test()'
```

- [ ] Do not proceed to workflow rollout until tests pass

### Benchmark archive plumbing

- [ ] Confirm `scripts/optimize_qaoa.jl` preserves runs under `/.project/results/optimization/`
- [ ] Confirm run manifests include:
  - `run_kind`
  - `runner_label`
  - `git_commit`
  - `git_branch`
  - `git_dirty`
  - reliability artefact references
- [ ] Confirm the aggregate index rebuilds cleanly from canonical per-run directories after schema changes

---

## 2. Workflow Readiness

### Workflow files

- [ ] Review `.github/workflows/ci.yml`
- [ ] Review `.github/workflows/optimize.yml`
- [ ] Review `.github/workflows/reproduce.yml`
- [ ] Confirm ordinary CI is verification-only
- [ ] Confirm optimization and reproduction workflows are deliberate, not push-triggered CI

### Workflow safety

- [ ] Add or confirm `permissions: contents: write` only where result-writing is required
- [ ] Add or confirm `concurrency` for heavy testbed workflows
- [ ] Confirm self-hosted workflows do not run on untrusted PR code
- [ ] Confirm result-writing logic targets the intended branch strategy
- [ ] Confirm result-only pushes will not create CI noise on the main development path

### Workflow tool integration

- [ ] Confirm workflows call `scripts/testbed/run-with-machine-state.sh`
- [ ] Confirm workflows call `scripts/testbed/analyze-machine-state.sh`
- [ ] Confirm workflow metadata records the machine artefact directory
- [ ] Confirm `QAOA_RUN_KIND`, `QAOA_RUNNER_LABEL`, and `QAOA_RELIABILITY_DIR` are exported before benchmark-grade runs

---

## 3. Local Runner Machine Provisioning

### Base OS

- [ ] Install Ubuntu 24.04 LTS Server on the 48 GB machine
- [ ] Do not install a desktop environment
- [ ] Fully update the OS:

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### Base packages

- [ ] Install required packages:

```bash
sudo apt-get install -y git curl wget tar xz-utils ca-certificates
```

- [ ] Optionally install `lm-sensors` if available for richer thermal reporting:

```bash
sudo apt-get install -y lm-sensors
```

### Runner user

- [ ] Create a dedicated runner user:

```bash
sudo useradd --create-home --shell /bin/bash gha-runner
sudo passwd -l gha-runner
```

- [ ] Verify the home directory exists at `/home/gha-runner`

### Julia installation

- [ ] Install Julia under `/opt`
- [ ] Expose Julia on `PATH`
- [ ] Verify the installed version:

```bash
julia --version
```

Suggested install sequence:

```bash
cd /tmp
wget https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.7-linux-x86_64.tar.gz
sudo tar -C /opt -xzf julia-1.11.7-linux-x86_64.tar.gz
sudo ln -sfn /opt/julia-1.11.7 /opt/julia
echo 'export PATH=/opt/julia/bin:$PATH' | sudo tee /etc/profile.d/julia.sh
source /etc/profile.d/julia.sh
```

### Runner registration

- [ ] Log into GitHub and open repository Settings → Actions → Runners
- [ ] Add a new self-hosted runner for the repository
- [ ] Install it under `/home/gha-runner/actions-runner/`
- [ ] Register it with labels:
  - `self-hosted`
  - `linux`
  - `x64`
  - `testbed-48gb`
- [ ] Install it as a service with the provided `svc.sh`
- [ ] Start the service
- [ ] Reboot once and confirm the runner comes back online automatically

---

## 4. Laptop Host Hardening

### Power and sleep

- [ ] Keep the machine on AC power for benchmark-grade runs
- [ ] Disable suspend on lid close
- [ ] Disable automatic sleep while on AC power
- [ ] Avoid battery-powered benchmark runs unless explicitly marked experimental

### Background workload control

- [ ] Do not use the laptop interactively during benchmark-grade runs
- [ ] Disable unnecessary background applications or services
- [ ] Confirm no large package upgrades, indexing tasks, or backup jobs are running

### Memory and swap

- [ ] Keep a small swap or `zram` safety buffer
- [ ] Do not disable swap entirely unless you have a strong reason and accept harsher OOM failures

### Thermal sanity

- [ ] Verify cooling is adequate
- [ ] If available, run `sensors` once and confirm temperatures are readable
- [ ] If available, inspect CPU governor behaviour before treating results as benchmark-grade

---

## 5. Testbed Script Validation

Run these on the repository host where the scripts are available.

### Syntax checks

- [ ] Check shell syntax:

```bash
bash -n /workspace/scripts/testbed/capture-machine-state.sh \
        /workspace/scripts/testbed/run-with-machine-state.sh \
        /workspace/scripts/testbed/analyze-machine-state.sh
```

### Snapshot smoke test

- [ ] Verify snapshot generation:

```bash
bash /workspace/scripts/testbed/capture-machine-state.sh /tmp/qaoa-machine-state.json smoke
sed -n '1,40p' /tmp/qaoa-machine-state.json
```

### Wrapper smoke test

- [ ] Verify before/after capture and summary generation:

```bash
tmpdir=$(mktemp -d /tmp/qaoa-testbed-wrapper-XXXXXX)
bash /workspace/scripts/testbed/run-with-machine-state.sh "$tmpdir" -- bash -lc 'echo benchmark-smoke; sleep 1'
find "$tmpdir" -maxdepth 1 -type f | sort
sed -n '1,80p' "$tmpdir/run-summary.json"
```

### Analyzer smoke test

- [ ] Verify machine analysis generation:

```bash
bash /workspace/scripts/testbed/analyze-machine-state.sh "$tmpdir" | tee "$tmpdir/machine-analysis.json"
sed -n '1,80p' "$tmpdir/machine-analysis.json"
```

Expected artefacts:

- [ ] `machine-state-before.json`
- [ ] `machine-state-after.json`
- [ ] `stdout.txt`
- [ ] `stderr.txt`
- [ ] `run-summary.json`
- [ ] `machine-analysis.json`

---

## 6. End-to-End Repository Validation On The Runner

### Checkout and instantiate

- [ ] Clone the repository on the runner host or let Actions do the checkout
- [ ] Verify Julia can instantiate the project:

```bash
cd /path/to/repo
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

### Full tests

- [ ] Run the full test suite once on the dedicated machine:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

- [ ] Record whether the testbed produces any environment-specific failures

### Tiny preserved optimization test

- [ ] Run one tiny preserved optimization with reliability metadata:

```bash
tmpdir=$(mktemp -d /tmp/qaoa-reliability-XXXXXX)
echo '{}' > "$tmpdir/machine-analysis.json"
export QAOA_RUN_KIND=benchmark
export QAOA_RUNNER_LABEL=testbed-48gb
export QAOA_RELIABILITY_DIR="$tmpdir"
julia --project=. scripts/optimize_qaoa.jl 3 4 1 1 0 1 1234 true
```

- [ ] Inspect the latest run manifest and confirm it contains:
  - run kind
  - runner label
  - reliability artefact directory
  - reliability artefact filenames

---

## 7. GitHub Wiring Validation

### Repository settings

- [ ] In GitHub repo Settings → Actions → General, confirm workflow permissions are set appropriately for result-writing workflows
- [ ] Confirm the self-hosted runner appears online with the `testbed-48gb` label

### Workflow dry run

- [ ] Trigger `.github/workflows/optimize.yml` manually with a very small job
- [ ] Confirm the job lands on `testbed-48gb` when requested
- [ ] Confirm the workflow produces:
  - result CSV
  - machine snapshot directory
  - machine analysis JSON
- [ ] Confirm the workflow can push its intended results if result-writing is enabled

### Reproduction dry run

- [ ] Trigger `.github/workflows/reproduce.yml` manually with the smallest acceptable settings
- [ ] Confirm MaxCut validation runs first
- [ ] Confirm target-problem run follows only after validation succeeds

---

## 8. Operational Definition of Done

You are finished when all of the following are true:

- [ ] Repository policy, rollout, and experiment docs are aligned
- [ ] Full tests pass in the optimization worktree
- [ ] The self-hosted runner is online and survives reboot
- [ ] The runner is registered with `testbed-48gb`
- [ ] The machine-state scripts work end to end
- [ ] The optimization archive preserves run kind, runner label, and reliability artefacts
- [ ] A tiny workflow run succeeds on the dedicated runner
- [ ] You can trust the difference between:
  - exploratory dev-box runs
  - benchmark-grade dedicated-runner runs

---

## Suggested Execution Order

If you want the shortest sensible path, do the checklist in this order:

1. Repository readiness
2. Local runner machine provisioning
3. Laptop host hardening
4. Testbed script validation
5. End-to-end repository validation on the runner
6. GitHub wiring validation
