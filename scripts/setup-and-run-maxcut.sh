#!/usr/bin/env bash
# Compatibility wrapper. Canonical local MaxCut startup lives in:
#   scripts/start-maxcut-local.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/start-maxcut-local.sh" "$@"
