#!/usr/bin/env bash
#
# Deploy QAOA-XORSAT to Azure Container Instances.
#
# This script:
#   1. Creates an ACR (if needed) and builds+pushes the Docker image
#   2. Deploys 15 ACI instances via Bicep (one per (k,D) pair)
#   3. Sets up monitoring commands
#
# Prerequisites:
#   - az CLI logged in (az login)
#   - Docker running
#
# Usage:
#   ./infra/deploy.sh [RESOURCE_GROUP] [LOCATION] [P_MAX]
#
# Examples:
#   ./infra/deploy.sh qaoa-xorsat australiaeast 13
#   ./infra/deploy.sh qaoa-xorsat eastus 14

set -euo pipefail

RG="${1:-qaoa-xorsat}"
LOCATION="${2:-australiaeast}"
P_MAX="${3:-13}"
ACR_NAME="qaoaxorsat$(echo $RG | tr -d '-')"
IMAGE_TAG="v$(date -u +%Y%m%d)"

# Memory sizing based on pMax
if [ "$P_MAX" -le 12 ]; then
    MEMORY_GB=32
    CPU_CORES=8
elif [ "$P_MAX" -le 13 ]; then
    MEMORY_GB=128
    CPU_CORES=16
elif [ "$P_MAX" -le 14 ]; then
    MEMORY_GB=256
    CPU_CORES=32
else
    echo "ERROR: p=$P_MAX needs 512GB+ — use a VM instead of ACI"
    exit 1
fi

echo "=== QAOA-XORSAT Azure Deployment ==="
echo "Resource Group: $RG"
echo "Location:       $LOCATION"
echo "P_MAX:          $P_MAX"
echo "Memory/task:    ${MEMORY_GB}GB"
echo "CPU/task:       ${CPU_CORES} cores"
echo "ACR:            $ACR_NAME"
echo "Image tag:      $IMAGE_TAG"
echo ""

# Step 1: Create resource group
echo "Creating resource group..."
az group create -n "$RG" -l "$LOCATION" -o none

# Step 2: Create ACR
echo "Creating container registry..."
az acr create -g "$RG" -n "$ACR_NAME" --sku Basic --admin-enabled true -o none 2>/dev/null || true

# Step 3: Build and push image
echo "Building and pushing Docker image..."
az acr build -r "$ACR_NAME" -t "qaoa-xorsat:$IMAGE_TAG" -t "qaoa-xorsat:latest" . --no-logs

# Step 4: Deploy via Bicep
echo "Deploying 15 ACI instances..."
az deployment group create \
    -g "$RG" \
    -f infra/main.bicep \
    -p acrName="$ACR_NAME" \
       pMax=$P_MAX \
       imageTag="$IMAGE_TAG" \
       cpuCores=$CPU_CORES \
       memoryGB=$MEMORY_GB \
    -o none

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Monitor all containers:"
echo "  az container list -g $RG -o table"
echo ""
echo "Check a specific container:"
echo "  az container logs -g $RG -n qaoa-k3-d4"
echo ""
echo "Stream logs live:"
echo "  az container logs -g $RG -n qaoa-k3-d4 --follow"
echo ""
echo "Download results from Azure Files:"
STORAGE_KEY=$(az storage account keys list -g "$RG" -n "$(az storage account list -g "$RG" --query '[0].name' -o tsv)" --query '[0].value' -o tsv 2>/dev/null || echo "???")
echo "  az storage file download-batch -d ./results-azure/ --source results --account-name $(az storage account list -g "$RG" --query '[0].name' -o tsv 2>/dev/null || echo '<storage>') --account-key $STORAGE_KEY"
echo ""
echo "Clean up when done:"
echo "  az group delete -n $RG --yes"
