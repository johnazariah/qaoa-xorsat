---
name: start-work
description: "Start-of-day bootstrap for QAOA-XORSAT. Use when: beginning a coding session, starting work for the day. Writes yesterday's audit report, syncs README stats, and surfaces the current project state and next steps from the plan."
---

# Start Work

Run once at the start of each coding session. Writes the audit report for **yesterday** (so the full day's work is captured), syncs README.md with live stats, and surfaces the project state.

## When to Use

- Beginning a coding session
- User says "start work", "morning", "let's go", "bootstrap"
- First interaction of the day

## Important Rules

- **Always audit yesterday.** Today is incomplete — never write a report for today.
- **If yesterday's report exists**, read it, confirm accuracy against live data, and update if stale. Do not overwrite correct content.
- **README sync uses live data** — reflects the current state, not yesterday's.

## Procedure

### Step 1 — Determine Yesterday's Date

Calculate yesterday's date. The report file is `.project/reports/YYYY-MM-DD.md`.

### Step 2 — Collect Live Stats

Run the [audit data script](./scripts/collect-audit-data.sh) with yesterday's date to gather:
- Git commits for yesterday
- Current Julia line counts (src/ and test/)
- Test health
- Paper and learning document counts
- Plan progress

### Step 3 — Write or Update Yesterday's Report

**If the report file does not exist**, create `.project/reports/YYYY-MM-DD.md` with:

```markdown
# Daily Audit — YYYY-MM-DD

## Executive Summary
2-3 sentence health check of the project.

## Section A — Daily Changes
- Commits (from git log for that date)
- New/changed features or implementations
- Test changes
- New learning materials or papers added
- Any issues introduced

## Section B — System Snapshot
- Test health (pass/fail + count)
- Julia source line count (src/)
- Julia test line count (test/)
- Papers downloaded (count)
- Learning documents written (count)
- Plan phase status (which phases have checked items)

## Section C — Plan Progress
- Items completed today (from git diff on PLAN.md)
- Next items to tackle
- Blockers or open questions
```

**If the report file already exists**, read it, compare with collected data, and update any stale sections. Report what was changed.

### Step 4 — Sync README.md

Collect live stats and update README.md:

```bash
# Julia source lines
find src -name "*.jl" | xargs wc -l | tail -1

# Julia test lines
find test -name "*.jl" | xargs wc -l | tail -1

# Paper count
ls -1 .project/papers/*.pdf 2>/dev/null | wc -l

# Learning doc count
ls -1 .project/learning/*.md 2>/dev/null | wc -l

# Plan completion
grep -c '\[x\]' .project/PLAN.md
grep -c '\[ \]' .project/PLAN.md
```

Update the README's status section with current numbers. Only update values that actually changed.

### Step 5 — Surface Plan Status

Read `.project/PLAN.md` and identify:
- The current phase (first phase with unchecked items)
- Completed items in current phase
- Next unchecked item(s) to work on
- Any blockers noted in Open Questions

### Step 6 — Cross-reference Journal

Read the last 2-3 entries in `.project/journal.md` for continuity context. Note any decisions or changes that affect today's work.

### Step 7 — Report to User

Summarise:
- Yesterday's commit count and key themes
- Current project health (tests, build)
- What was updated (report, README)
- Current phase and next steps from the plan
- Any blockers or decisions needed

Present the next steps as a suggested work queue, ordered by plan priority.
