#!/bin/bash
# Sync cluster state: commit results + logs, pull latest code
# Run from the repo root on the cluster.
#
# Usage: bash scripts/cluster-sync.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "=== Committing local results and logs ==="
git add -f results/ logs/ 2>/dev/null || true
if ! git diff --cached --quiet 2>/dev/null; then
    git commit -m "cluster snapshot: $(date -u '+%Y-%m-%d %H:%M UTC')"
    echo "  Committed."
else
    echo "  Nothing new to commit."
fi

echo "=== Pulling latest code ==="
git stash push -m "cluster-sync autostash" --include-untracked 2>/dev/null || true
git pull --rebase origin main
git stash pop 2>/dev/null || true

echo "=== Done ==="
git log --oneline -3
