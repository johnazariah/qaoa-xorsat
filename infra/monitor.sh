#!/usr/bin/env bash
#
# Monitor running QAOA ACI instances.
# Shows status, runtime, and latest log output for each container.
#
# Usage:
#   ./infra/monitor.sh [RESOURCE_GROUP]

set -euo pipefail

RG="${1:-qaoa-xorsat}"

echo "=== QAOA-XORSAT Container Status ==="
echo ""

az container list -g "$RG" --query "[].{Name:name, State:instanceView.state, Started:containers[0].instanceView.currentState.startTime}" -o table 2>/dev/null

echo ""
echo "=== Latest output per container ==="
echo ""

for name in $(az container list -g "$RG" --query "[].name" -o tsv 2>/dev/null); do
    state=$(az container show -g "$RG" -n "$name" --query "instanceView.state" -o tsv 2>/dev/null)
    echo "--- $name ($state) ---"
    az container logs -g "$RG" -n "$name" --tail 5 2>/dev/null || echo "  (no logs yet)"
    echo ""
done
