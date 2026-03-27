#!/usr/bin/env bash
#
# Submit Azure Batch jobs for all 15 (k,D) pairs.
#
# Prerequisites:
#   - Docker image pushed to ACR: qaoa-xorsat:latest
#   - Azure Batch pool with Docker support
#
# Usage:
#   ./scripts/batch-submit.sh [P_MAX] [--dry-run]
#
# Each (k,D) pair runs as an independent task. Results are written
# to an Azure Storage container mounted at /workspace/results.
#
# VM sizing guide:
#   p ≤ 12: Standard_E16as_v5  (16 vCPU, 128 GB)  ~$1.22/hr
#   p = 13: Standard_E32as_v5  (32 vCPU, 256 GB)  ~$2.44/hr
#   p = 14: Standard_E64as_v5  (64 vCPU, 512 GB)  ~$4.88/hr
#   p = 15: Standard_M128s     (128 vCPU, 2 TB)   ~$13.34/hr

set -euo pipefail

P_MAX="${1:-12}"
DRY_RUN="${2:-}"
IMAGE="${QAOA_IMAGE:-qaoa-xorsat:latest}"
POOL="${QAOA_POOL:-qaoa-pool}"

# All 15 (k,D) pairs from Jordan et al.
declare -a PAIRS=(
    "3,4" "3,5" "3,6" "3,7" "3,8"
    "4,5" "4,6" "4,7" "4,8"
    "5,6" "5,7" "5,8"
    "6,7" "6,8"
    "7,8"
)

echo "=== QAOA-XORSAT Azure Batch Submission ==="
echo "Image:  $IMAGE"
echo "Pool:   $POOL"
echo "P_MAX:  $P_MAX"
echo "Pairs:  ${#PAIRS[@]}"
echo ""

for pair in "${PAIRS[@]}"; do
    IFS=',' read -r K D <<< "$pair"
    TASK_ID="k${K}-d${D}-p1-${P_MAX}"

    CMD="julia --project=. -t auto scripts/optimize_qaoa.jl ${K} ${D} 1 ${P_MAX} 2 320 1234 true adjoint"

    if [[ "$DRY_RUN" == "--dry-run" ]]; then
        echo "  [dry-run] Task: $TASK_ID"
        echo "    CMD: $CMD"
    else
        echo "  Submitting: $TASK_ID"
        az batch task create \
            --job-id "qaoa-sweep-p${P_MAX}" \
            --task-id "$TASK_ID" \
            --image "$IMAGE" \
            --command-line "$CMD" \
            2>/dev/null || echo "    FAILED (is batch configured?)"
    fi
done

echo ""
echo "Monitor: az batch task list --job-id qaoa-sweep-p${P_MAX} -o table"
