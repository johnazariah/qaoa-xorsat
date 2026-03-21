#!/usr/bin/env bash
# Collect data for daily audit report — QAOA-XORSAT project
# Usage: bash .github/skills/start-work/scripts/collect-audit-data.sh [YYYY-MM-DD]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

DATE="${1:-$(date -d yesterday +%Y-%m-%d 2>/dev/null || date -v-1d +%Y-%m-%d)}"
echo "=== Audit Data for $DATE ==="

echo ""
echo "=== Git Commits ==="
git --no-pager log --since="$DATE 00:00:00" --until="$DATE 23:59:59" --oneline --no-decorate 2>/dev/null || echo "No commits found for $DATE"

echo ""
echo "=== Commit Count ==="
git --no-pager log --since="$DATE 00:00:00" --until="$DATE 23:59:59" --oneline --no-decorate 2>/dev/null | wc -l

echo ""
echo "=== Julia Source Lines (src/) ==="
find src -name '*.jl' 2>/dev/null | xargs wc -l 2>/dev/null | tail -1 || echo "0"

echo ""
echo "=== Julia Test Lines (test/) ==="
find test -name '*.jl' 2>/dev/null | xargs wc -l 2>/dev/null | tail -1 || echo "0"

echo ""
echo "=== Test Health ==="
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -5 || echo "Tests failed or not runnable"

echo ""
echo "=== Papers Downloaded ==="
ls -1 .project/papers/*.pdf 2>/dev/null | wc -l

echo ""
echo "=== Learning Documents ==="
ls -1 .project/learning/*.md 2>/dev/null | wc -l

echo ""
echo "=== Plan Progress ==="
echo "Completed items:"
grep -c '\[x\]' .project/PLAN.md 2>/dev/null || echo "0"
echo "Remaining items:"
grep -c '\[ \]' .project/PLAN.md 2>/dev/null || echo "0"

echo ""
echo "=== Recent Journal Entries ==="
tail -20 .project/journal.md 2>/dev/null || echo "No journal found"

echo ""
echo "Done. Use this data to write .project/reports/$DATE.md"
