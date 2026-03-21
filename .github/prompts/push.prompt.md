---
description: "Push current branch and monitor CI until green."
agent: agent
tools:
  - runTerminal
  - readFile
---

# Push & CI Triage

You are handling a push workflow in the QAOA-XORSAT repository.

## Goal

Push safely, then monitor CI/CD and diagnose failures quickly.

## Mandatory Workflow

1. Run tests locally before pushing:
   ```bash
   julia --project=. -e 'using Pkg; Pkg.test()'
   ```
2. If tests fail, stop and fix before push.
3. Push current branch to origin:
   ```bash
   git push -u origin HEAD
   ```
4. Monitor CI using `gh`:
   ```bash
   gh run list --limit 5
   gh run watch <run-id> --exit-status
   ```
5. If CI fails, extract failure evidence:
   ```bash
   gh run view <run-id> --log-failed
   ```
6. Diagnose and fix.

## Output Contract

After push, report one of:
- **PASS**: CI passed, include run URL/ID.
- **FAIL**: CI failed, include failing step summary and a short fix plan.

## Guardrails

- Do not guess: base diagnosis on actual run logs.
- Prefer first failing step as primary signal.
- Keep summary short and actionable.
