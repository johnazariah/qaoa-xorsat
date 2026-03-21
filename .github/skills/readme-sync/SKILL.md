---
name: readme-sync
description: "Sync README.md with live codebase stats. Use when: README numbers are stale, after significant code changes, after adding papers or learning docs."
---

# README Sync

Update README.md with accurate statistics from the live codebase.

## When to Use

- After significant code changes
- After adding papers or learning materials
- When README numbers look stale
- As part of start-work or pre-commit workflows

## Procedure

### Step 1 — Collect Live Stats

```bash
cd "$(git rev-parse --show-toplevel)"

echo "=== Julia Source Lines ==="
find src -name "*.jl" | xargs wc -l | tail -1

echo "=== Julia Test Lines ==="
find test -name "*.jl" | xargs wc -l | tail -1

echo "=== Papers ==="
ls -1 .project/papers/*.pdf 2>/dev/null | wc -l

echo "=== Learning Docs ==="
ls -1 .project/learning/*.md 2>/dev/null | wc -l

echo "=== Plan Completion ==="
echo "Done: $(grep -c '\[x\]' .project/PLAN.md 2>/dev/null || echo 0)"
echo "Todo: $(grep -c '\[ \]' .project/PLAN.md 2>/dev/null || echo 0)"

echo "=== Git Commits ==="
git --no-pager log --oneline | wc -l

echo "=== Dependencies ==="
grep -c 'deps' Project.toml 2>/dev/null || echo "0"
```

### Step 2 — Update README.md

Compare collected stats with current README values. Update only values that have changed:

- Status/progress indicators
- Line counts
- Paper/document counts
- Any build badges or test results

### Step 3 — Verify

Read the updated README and confirm all numbers match live data. Do not change prose, structure, or non-numeric content unless specifically asked.
