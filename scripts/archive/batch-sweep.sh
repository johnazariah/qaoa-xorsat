#!/usr/bin/env bash
#
# Generate and run Azure Batch tasks for all (k,D) pairs in Stephen's table.
#
# Usage:
#   ./scripts/batch-sweep.sh [--dry-run]       # print commands without executing
#   ./scripts/batch-sweep.sh                   # run locally via Docker
#   ./scripts/batch-sweep.sh --azure           # submit to Azure Batch (requires az CLI)
#
# Environment:
#   QAOA_IMAGE     Docker image (default: qaoa-xorsat:latest)
#   QAOA_P_MAX     Maximum depth (default: 12)
#   QAOA_RESTARTS  Restarts per depth (default: 5)
#   QAOA_MAXITERS  Max L-BFGS iterations (default: 80)
#   QAOA_SEED      Random seed (default: 1234)

set -euo pipefail

IMAGE="${QAOA_IMAGE:-qaoa-xorsat:latest}"
P_MAX="${QAOA_P_MAX:-12}"
RESTARTS="${QAOA_RESTARTS:-5}"
MAXITERS="${QAOA_MAXITERS:-80}"
SEED="${QAOA_SEED:-1234}"
MODE="${1:---local}"

# Stephen's table: all (k,D) pairs
declare -a PAIRS=(
    "3,4"
    "3,5"
    "3,6"
    "3,7"
    "3,8"
    "4,5"
    "4,6"
    "4,7"
    "4,8"
    "5,6"
    "5,7"
    "5,8"
    "6,7"
    "6,8"
    "7,8"
)

for pair in "${PAIRS[@]}"; do
    IFS=',' read -r K D <<< "$pair"
    TASK_ID="k${K}-d${D}-p1-${P_MAX}"
    OUTPUT_DIR="results/batch/${TASK_ID}"

    case "$MODE" in
        --dry-run)
            echo "Task: $TASK_ID — K=$K D=$D P=1-$P_MAX"
            echo "  docker run --rm -v \$(pwd)/results:/workspace/results $IMAGE $K $D 1 $P_MAX $RESTARTS $MAXITERS $SEED true adjoint"
            echo ""
            ;;
        --local)
            echo "=== Starting $TASK_ID ==="
            mkdir -p "$OUTPUT_DIR"
            docker run --rm \
                -v "$(pwd)/results:/workspace/results" \
                -e QAOA_RUN_KIND=experiment \
                -e QAOA_RUNNER_LABEL=docker-local \
                "$IMAGE" \
                "$K" "$D" 1 "$P_MAX" "$RESTARTS" "$MAXITERS" "$SEED" true adjoint \
                | tee "$OUTPUT_DIR/stdout.csv"
            echo "=== Finished $TASK_ID ==="
            ;;
        --azure)
            echo "Submitting Azure Batch task: $TASK_ID"
            # Azure Batch task submission — adapt pool/job names to your setup
            az batch task create \
                --job-id qaoa-sweep \
                --task-id "$TASK_ID" \
                --image "$IMAGE" \
                --command-line "julia --project=. -t auto scripts/optimize_qaoa.jl $K $D 1 $P_MAX $RESTARTS $MAXITERS $SEED true adjoint" \
                --environment-settings "QAOA_RUN_KIND=experiment" "QAOA_RUNNER_LABEL=azure-batch" \
                2>/dev/null || echo "  (failed — check az batch config)"
            ;;
        *)
            echo "Unknown mode: $MODE" >&2
            echo "Usage: $0 [--dry-run|--local|--azure]" >&2
            exit 1
            ;;
    esac
done

echo ""
echo "Total tasks: ${#PAIRS[@]}"
