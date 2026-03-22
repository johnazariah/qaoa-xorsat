---
description: "Update the testing register with current test counts and descriptions. Run after adding or changing tests."
agent: agent
tools:
  - runTerminal
  - editFiles
  - readFile
  - textSearch
---

# Update Testing Register

Sync `.project/testing-register.md` with the actual test suite.

## Procedure

### Step 1 — Run Tests and Collect Counts

```bash
cd /workspace && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -5
```

Record the total runtime test count from the output.

### Step 2 — Count Static Tests Per File

```bash
grep -c '@test ' test/*.jl | grep -v ':0$'
```

### Step 3 — Enumerate Test Sets

```bash
grep -n '@testset' test/*.jl
```

### Step 4 — Compare Against Register

Read `.project/testing-register.md` and compare:
- Total runtime count in the summary table header
- Per-file static and runtime counts in the summary table
- Test set descriptions — are there new `@testset` blocks not yet documented?
- Are any documented test sets no longer present?

### Step 5 — Update the Register

For each change found:
1. Update the total count in the header
2. Update per-file counts in the summary table
3. Add descriptions for new test sets in the appropriate file section
4. Remove descriptions for deleted test sets
5. Update the "Last updated" date

### Step 6 — Verify Validation Targets

Check that the validation targets table at the bottom still matches reality:
- All listed test locations still exist
- Expected values haven't changed
- No new validation targets should be added (e.g. new published results)

## Rules

- Do NOT change test descriptions for tests that haven't changed
- Do NOT rewrite existing descriptions — only add/remove/update counts
- Keep descriptions in plain English, not code
- Parametric tests: describe what parameters are swept, note that runtime count > static count
