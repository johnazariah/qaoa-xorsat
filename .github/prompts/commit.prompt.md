---
description: "Prepare clean, logically grouped commits from the current working tree."
agent: agent
tools:
  - runTerminal
  - editFiles
  - readFile
  - textSearch
---

# Commit Workflow

You are preparing commits in the QAOA-XORSAT repository.

## Goal

Create clean, logically grouped commits from the current working tree.

## Mandatory Workflow

1. Run the test suite first:
   ```bash
   julia --project=. -e 'using Pkg; Pkg.test()'
   ```
2. If tests fail, stop and fix issues before any commit.
3. Group changes into related commit sets by concern (core code, tests, docs, learning, infrastructure).
4. For each group:
   - Stage only files for that group
   - Write an imperative commit message
   - Commit
5. Repeat until all intended changes are committed.
6. Confirm working tree is clean at the end.

## Commit Quality Rules

- Do not create one giant commit unless all changes are truly one concern.
- Keep commits reviewable and cohesive.
- Do not mix unrelated refactors with feature logic.
- Include test updates in the same commit as the behaviour they validate.
- Documentation updates (plan, journal, README) can be a separate commit.

## Message Convention

Use imperative mood and concise scope.

Examples:
- `Add tree construction for (k, D, p) factor graphs`
- `Validate MaxCut p=1 against known 0.7500 result`
- `Write explainer for Farhi 2025 tensor network method`
- `Update PLAN.md: complete Phase 1 tree characterisation`

## Co-authorship

If Copilot helped write the code, add the trailer:
```
Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
```

## Safety

- Never bypass test failures — fix the root cause.
- If changes are ambiguous, propose grouping options and choose the simplest coherent split.
