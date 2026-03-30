#!/usr/bin/env bash
#
# Deploy a single high-memory Azure VM and run the QAOA sweep.
# Results are uploaded to Azure Blob Storage every 10 minutes.
#
# This script is self-contained — run it from any machine with az CLI.
# It creates all resources, starts the computation, and sets up monitoring.
#
# Usage:
#   ./infra/deploy-vm.sh [OPTIONS]
#
# Options:
#   -s, --subscription ID    Azure subscription ID (required)
#   -g, --resource-group RG  Resource group name (default: qaoa-xorsat)
#   -l, --location LOC       Azure region (default: australiaeast)
#   -p, --pmax N             Maximum depth (default: 13)
#   -v, --vm-size SIZE       VM size (default: auto-detected from pmax)
#   --dry-run                Print commands without executing
#
# Examples:
#   ./infra/deploy-vm.sh -s 12345-abcde -p 13
#   ./infra/deploy-vm.sh -s 12345-abcde -p 14 -v Standard_E64as_v5 -l eastus
#   ./infra/deploy-vm.sh -s 12345-abcde -p 12 --dry-run

set -euo pipefail

# ── Parse arguments ───────────────────────────────────────────────────────────
SUBSCRIPTION=""
RG="qaoa-xorsat"
LOCATION="australiaeast"
P_MAX=13
VM_SIZE=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--subscription) SUBSCRIPTION="$2"; shift 2 ;;
        -g|--resource-group) RG="$2"; shift 2 ;;
        -l|--location) LOCATION="$2"; shift 2 ;;
        -p|--pmax) P_MAX="$2"; shift 2 ;;
        -v|--vm-size) VM_SIZE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$SUBSCRIPTION" ]; then
    echo "ERROR: --subscription is required"
    echo "Usage: ./infra/deploy-vm.sh -s <subscription-id> [-p 13] [-l australiaeast]"
    exit 1
fi

# Auto-detect VM size from pmax
if [ -z "$VM_SIZE" ]; then
    case $P_MAX in
        8|9|10|11) VM_SIZE="Standard_E8as_v5"  ;; # 64GB
        12)        VM_SIZE="Standard_E16as_v5"  ;; # 128GB
        13)        VM_SIZE="Standard_E32as_v5"  ;; # 256GB
        14)        VM_SIZE="Standard_E64as_v5"  ;; # 512GB
        15)        VM_SIZE="Standard_M128s"     ;; # 2TB
        *)         echo "ERROR: unsupported pmax=$P_MAX"; exit 1 ;;
    esac
fi

STORAGE_NAME="qaoa$(echo ${RG}${LOCATION} | md5sum | head -c 10)"
CONTAINER_NAME="results"
VM_NAME="qaoa-compute"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%S)

echo "=== QAOA-XORSAT Azure VM Deployment ==="
echo "Subscription: $SUBSCRIPTION"
echo "Resource Group: $RG"
echo "Location:       $LOCATION"
echo "VM Size:        $VM_SIZE"
echo "P_MAX:          $P_MAX"
echo "Storage:        $STORAGE_NAME"
echo "Timestamp:      $TIMESTAMP"
echo ""

run_cmd() {
    if $DRY_RUN; then
        echo "  [dry-run] $*"
    else
        "$@"
    fi
}

# ── Step 1: Set subscription ──────────────────────────────────────────────────
echo "Setting subscription..."
run_cmd az account set -s "$SUBSCRIPTION"

# ── Step 2: Create resource group ─────────────────────────────────────────────
echo "Creating resource group..."
run_cmd az group create -n "$RG" -l "$LOCATION" -o none

# ── Step 3: Create storage account + blob container ───────────────────────────
echo "Creating storage account..."
run_cmd az storage account create \
    -g "$RG" -n "$STORAGE_NAME" -l "$LOCATION" \
    --sku Standard_LRS --kind StorageV2 \
    --min-tls-version TLS1_2 -o none 2>/dev/null || true

echo "Creating blob container..."
STORAGE_KEY=$(az storage account keys list -g "$RG" -n "$STORAGE_NAME" --query '[0].value' -o tsv 2>/dev/null || echo "")
if [ -n "$STORAGE_KEY" ] && ! $DRY_RUN; then
    az storage container create -n "$CONTAINER_NAME" \
        --account-name "$STORAGE_NAME" --account-key "$STORAGE_KEY" -o none 2>/dev/null || true
fi

# ── Step 4: Create the cloud-init script ──────────────────────────────────────
# This runs on the VM at first boot: installs Julia, clones repo, starts sweep,
# and sets up periodic blob upload.
cat > /tmp/qaoa-cloud-init.sh << CLOUD_INIT
#!/usr/bin/env bash
set -euo pipefail
exec > /var/log/qaoa-setup.log 2>&1

echo "=== QAOA Setup starting at \$(date -u) ==="

# Install Julia
curl -fsSL https://install.julialang.org | sh -s -- --yes
export PATH="\$HOME/.juliaup/bin:\$PATH"

# Install Azure CLI (for blob uploads)
curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# Clone repo
cd /home/azureuser
git clone https://github.com/johnazariah/qaoa-xorsat.git
cd qaoa-xorsat

# Install Julia dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Create results directory
mkdir -p results/logs

# ── Periodic blob upload (every 10 minutes) ───────────────────────────────
cat > results/logs/blob-sync.sh << 'BLOBSYNC'
#!/usr/bin/env bash
STORAGE_NAME="${STORAGE_NAME}"
STORAGE_KEY="${STORAGE_KEY}"
CONTAINER="${CONTAINER_NAME}"
RUN_ID="${TIMESTAMP}"

while true; do
    sleep 600
    echo "\$(date -u): syncing to blob..."

    # Upload the summary CSV
    if [ -f .project/results/full-table-summary.csv ]; then
        az storage blob upload -f .project/results/full-table-summary.csv \
            -c "\$CONTAINER" -n "\${RUN_ID}/full-table-summary.csv" \
            --account-name "\$STORAGE_NAME" --account-key "\$STORAGE_KEY" \
            --overwrite 2>/dev/null || true
    fi

    # Upload the log tail
    tail -200 results/logs/cloud-*.log > /tmp/latest-progress.txt 2>/dev/null
    az storage blob upload -f /tmp/latest-progress.txt \
        -c "\$CONTAINER" -n "\${RUN_ID}/latest-progress.txt" \
        --account-name "\$STORAGE_NAME" --account-key "\$STORAGE_KEY" \
        --overwrite 2>/dev/null || true

    # Upload individual p=N results as they land
    for f in .project/results/optimization/runs/*/results.csv; do
        if [ -f "\$f" ]; then
            blobname="\${RUN_ID}/\$(basename \$(dirname \$f))/results.csv"
            az storage blob upload -f "\$f" -c "\$CONTAINER" -n "\$blobname" \
                --account-name "\$STORAGE_NAME" --account-key "\$STORAGE_KEY" \
                --overwrite 2>/dev/null || true
        fi
    done

    echo "\$(date -u): sync complete"
done
BLOBSYNC
chmod +x results/logs/blob-sync.sh

# Start blob sync in background
nohup bash results/logs/blob-sync.sh > results/logs/blob-sync.log 2>&1 &
echo "Blob sync started (PID=\$!)"

# ── Run the computation ───────────────────────────────────────────────────
THREADS=\$(nproc)
LOGFILE="results/logs/cloud-${TIMESTAMP}-p${P_MAX}.log"

echo "Starting sweep: 15 pairs, p=1..${P_MAX}, \${THREADS} threads"
echo "Log: \$LOGFILE"

nohup julia --project=. -t "\$THREADS" scripts/run_full_table.jl ${P_MAX} > "\$LOGFILE" 2>&1 &
JULIA_PID=\$!
echo "Julia PID=\$JULIA_PID"

echo "=== Setup complete at \$(date -u) ==="
CLOUD_INIT

# ── Step 5: Create the VM ─────────────────────────────────────────────────────
echo "Creating VM ($VM_SIZE)..."
run_cmd az vm create \
    -g "$RG" -n "$VM_NAME" -l "$LOCATION" \
    --size "$VM_SIZE" \
    --image "Canonical:ubuntu-24_04-lts:server:latest" \
    --admin-username azureuser \
    --generate-ssh-keys \
    --custom-data /tmp/qaoa-cloud-init.sh \
    --os-disk-size-gb 64 \
    --priority Spot \
    --eviction-policy Deallocate \
    --max-price -1 \
    -o none

# Get the public IP
if ! $DRY_RUN; then
    VM_IP=$(az vm show -g "$RG" -n "$VM_NAME" -d --query publicIps -o tsv)
    echo ""
    echo "=== Deployment Complete ==="
    echo ""
    echo "VM IP: $VM_IP"
    echo ""
    echo "SSH into the VM:"
    echo "  ssh azureuser@$VM_IP"
    echo ""
    echo "Check setup progress:"
    echo "  ssh azureuser@$VM_IP 'tail -20 /var/log/qaoa-setup.log'"
    echo ""
    echo "Check computation progress:"
    echo "  ssh azureuser@$VM_IP 'tail -20 ~/qaoa-xorsat/results/logs/cloud-*.log'"
    echo ""
    echo "Download latest results from blob:"
    echo "  az storage blob download-batch -d ./results-azure/ -s $CONTAINER_NAME \\"
    echo "    --account-name $STORAGE_NAME --account-key '$STORAGE_KEY' \\"
    echo "    --pattern '${TIMESTAMP}/*'"
    echo ""
    echo "Monitor blob uploads:"
    echo "  az storage blob list -c $CONTAINER_NAME --account-name $STORAGE_NAME \\"
    echo "    --account-key '$STORAGE_KEY' --prefix '${TIMESTAMP}/' -o table"
    echo ""
    echo "Clean up (stops billing!):"
    echo "  az group delete -n $RG --yes"
fi

rm -f /tmp/qaoa-cloud-init.sh
