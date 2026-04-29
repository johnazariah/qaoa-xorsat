#!/usr/bin/env bash
# Compatibility wrapper. The canonical Stephen cluster startup is:
#   scripts/start-xorsat-slurm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/start-xorsat-slurm.sh" "$@"
