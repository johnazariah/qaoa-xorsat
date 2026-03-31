#!/usr/bin/env bash
#
# Deploy a fleet of Azure VMs to run the QAOA sweep in parallel.
# Each VM runs a subset of (k,D) pairs independently.
# Results sync to Azure Blob every 10 minutes.
#
# Usage:
#   ./infra/deploy-fleet.sh -s <subscription-id> [OPTIONS]
#
# Options:
#   -s, --subscription ID    Azure subscription (required)
#   -g, --resource-group RG  Resource group (default: qaoa-fleet)
#   -l, --location LOC       Region (default: eastus2)
#   -p, --pmax N             Max depth (default: 12)
#   -n, --num-vms N          Number of VMs (default: 5)
#   -v, --vm-size SIZE       VM size (default: auto from pmax)
#   --dry-run                Print without executing
#
# Examples:
#   ./infra/deploy-fleet.sh -s 12345-abcde -p 12           # Phase 1: validate
#   ./infra/deploy-fleet.sh -s 12345-abcde -p 13           # Phase 2: production
#   ./infra/deploy-fleet.sh -s 12345-abcde -p 14 -n 5      # Phase 3: high depth

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SUBSCRIPTION=""
RG="qaoa-fleet"
LOCATION="eastus2"
P_MAX=12
NUM_VMS=5
VM_SIZE=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--subscription) SUBSCRIPTION="$2"; shift 2 ;;
        -g|--resource-group) RG="$2"; shift 2 ;;
        -l|--location) LOCATION="$2"; shift 2 ;;
        -p|--pmax) P_MAX="$2"; shift 2 ;;
        -n|--num-vms) NUM_VMS="$2"; shift 2 ;;
        -v|--vm-size) VM_SIZE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

[ -z "$SUBSCRIPTION" ] && { echo "ERROR: -s <subscription> required"; exit 1; }

# Auto-size VM
if [ -z "$VM_SIZE" ]; then
    case $P_MAX in
        8|9|10|11) VM_SIZE="Standard_E4as_v5"  ;; # 32GB
        12)        VM_SIZE="Standard_E8as_v5"   ;; # 64GB
        13)        VM_SIZE="Standard_E32as_v5"  ;; # 256GB
        14)        VM_SIZE="Standard_E64as_v5"  ;; # 512GB
        *)         echo "ERROR: p=$P_MAX — use deploy-vm.sh for single big VM"; exit 1 ;;
    esac
fi

# ── The 15 pairs, split across VMs ────────────────────────────────────────────
ALL_PAIRS=(
    "3,4" "3,5" "3,6" "3,7" "3,8"
    "4,5" "4,6" "4,7" "4,8"
    "5,6" "5,7" "5,8"
    "6,7" "6,8"
    "7,8"
)

# Distribute pairs round-robin across VMs
declare -a VM_PAIRS
for i in $(seq 0 $((NUM_VMS - 1))); do
    VM_PAIRS[$i]=""
done
for i in "${!ALL_PAIRS[@]}"; do
    vm_idx=$((i % NUM_VMS))
    if [ -z "${VM_PAIRS[$vm_idx]}" ]; then
        VM_PAIRS[$vm_idx]="${ALL_PAIRS[$i]}"
    else
        VM_PAIRS[$vm_idx]="${VM_PAIRS[$vm_idx]} ${ALL_PAIRS[$i]}"
    fi
done

# Storage
STORAGE_NAME="qaoa$(echo ${RG} | tr -d '-' | head -c 10)$(date +%d)"
CONTAINER_NAME="results"

echo "=== QAOA Fleet Deployment ==="
echo "Subscription: $SUBSCRIPTION"
echo "Resource Group: $RG"
echo "Location:       $LOCATION"
echo "VM Size:        $VM_SIZE"
echo "Num VMs:        $NUM_VMS"
echo "P_MAX:          $P_MAX"
echo "Storage:        $STORAGE_NAME"
echo ""
echo "Pair distribution:"
for i in $(seq 0 $((NUM_VMS - 1))); do
    echo "  VM-$i: ${VM_PAIRS[$i]}"
done
echo ""

run_cmd() {
    if $DRY_RUN; then echo "  [dry-run] $*"; else "$@"; fi
}

# ── Create infrastructure ─────────────────────────────────────────────────────
echo "Setting subscription..."
run_cmd az account set -s "$SUBSCRIPTION"

echo "Creating resource group..."
run_cmd az group create -n "$RG" -l "$LOCATION" -o none

echo "Creating storage account..."
run_cmd az storage account create -g "$RG" -n "$STORAGE_NAME" -l "$LOCATION" \
    --sku Standard_LRS --kind StorageV2 -o none 2>/dev/null || true

STORAGE_KEY=""
if ! $DRY_RUN; then
    STORAGE_KEY=$(az storage account keys list -g "$RG" -n "$STORAGE_NAME" --query '[0].value' -o tsv)
    az storage container create -n "$CONTAINER_NAME" \
        --account-name "$STORAGE_NAME" --account-key "$STORAGE_KEY" -o none 2>/dev/null || true
fi

# ── Deploy VMs ────────────────────────────────────────────────────────────────
TIMESTAMP=$(date -u +%Y%m%dT%H%M%S)

for i in $(seq 0 $((NUM_VMS - 1))); do
    VM_NAME="qaoa-vm-${i}"
    PAIRS="${VM_PAIRS[$i]}"
    
    # Build the cloud-init script for this VM
    INIT_FILE="/tmp/qaoa-init-${i}.sh"
    cat > "$INIT_FILE" << INIT_EOF
#!/usr/bin/env bash
set -euo pipefail
exec > /var/log/qaoa-setup.log 2>&1
echo "=== VM-${i} setup at \$(date -u) ==="
echo "Pairs: ${PAIRS}"

# Install Julia
curl -fsSL https://install.julialang.org | sh -s -- --yes
export PATH="\$HOME/.juliaup/bin:\$PATH"

# Install az CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | bash 2>/dev/null

# Clone and setup
cd /home/azureuser
git clone https://github.com/johnazariah/qaoa-xorsat.git
cd qaoa-xorsat
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
mkdir -p results/logs

# Blob sync loop
cat > /tmp/blob-sync.sh << 'BSYNC'
while true; do
    sleep 600
    cd /home/azureuser/qaoa-xorsat
    for f in .project/results/optimization/runs/*/results.csv; do
        [ -f "\$f" ] || continue
        bn="\$(basename \$(dirname \$f))/results.csv"
        az storage blob upload -f "\$f" -c "${CONTAINER_NAME}" \
            -n "${TIMESTAMP}/vm-${i}/\$bn" \
            --account-name "${STORAGE_NAME}" --account-key "${STORAGE_KEY}" \
            --overwrite 2>/dev/null || true
    done
    # Upload log snapshot
    tail -100 results/logs/pair-*.log > /tmp/vm-${i}-progress.txt 2>/dev/null || true
    az storage blob upload -f /tmp/vm-${i}-progress.txt -c "${CONTAINER_NAME}" \
        -n "${TIMESTAMP}/vm-${i}/progress.txt" \
        --account-name "${STORAGE_NAME}" --account-key "${STORAGE_KEY}" \
        --overwrite 2>/dev/null || true
done
BSYNC
nohup bash /tmp/blob-sync.sh &

# Run pairs sequentially within this VM
THREADS=\$(nproc)
for PAIR in ${PAIRS}; do
    K=\$(echo \$PAIR | cut -d, -f1)
    D=\$(echo \$PAIR | cut -d, -f2)
    LOG="results/logs/pair-k\${K}-d\${D}.log"
    echo "\$(date -u): Starting (k=\$K, D=\$D) p=1..${P_MAX}"
    julia --project=. -t \$THREADS scripts/optimize_qaoa.jl \$K \$D 1 ${P_MAX} 2 320 1234 true adjoint > "\$LOG" 2>&1
    echo "\$(date -u): Completed (k=\$K, D=\$D)"
done

echo "=== VM-${i} all pairs complete at \$(date -u) ==="
INIT_EOF

    echo "Creating $VM_NAME (pairs: $PAIRS)..."
    run_cmd az vm create \
        -g "$RG" -n "$VM_NAME" -l "$LOCATION" \
        --size "$VM_SIZE" \
        --image "Canonical:ubuntu-24_04-lts:server:latest" \
        --admin-username azureuser \
        --generate-ssh-keys \
        --custom-data "$INIT_FILE" \
        --os-disk-size-gb 64 \
        --priority Spot \
        --eviction-policy Deallocate \
        --max-price -1 \
        -o none
    
    rm -f "$INIT_FILE"
done

# ── Print monitoring commands ─────────────────────────────────────────────────
echo ""
echo "=== Fleet Deployed ==="
echo ""

if ! $DRY_RUN; then
    echo "VM IPs:"
    for i in $(seq 0 $((NUM_VMS - 1))); do
        IP=$(az vm show -g "$RG" -n "qaoa-vm-${i}" -d --query publicIps -o tsv 2>/dev/null || echo "pending")
        echo "  qaoa-vm-${i}: $IP (pairs: ${VM_PAIRS[$i]})"
    done
fi

echo ""
echo "Monitor setup (run ~5 min after deploy):"
for i in $(seq 0 $((NUM_VMS - 1))); do
    echo "  ssh azureuser@<vm-${i}-ip> 'tail -5 /var/log/qaoa-setup.log'"
done
echo ""
echo "Monitor progress:"
echo "  az storage blob list -c $CONTAINER_NAME --account-name $STORAGE_NAME --prefix '$TIMESTAMP/' -o table"
echo ""
echo "Download a VM's progress:"
echo "  az storage blob download -c $CONTAINER_NAME -n '$TIMESTAMP/vm-0/progress.txt' -f progress-vm0.txt --account-name $STORAGE_NAME"
echo ""
echo "Download all results:"
echo "  az storage blob download-batch -d ./results-fleet/ -s $CONTAINER_NAME --account-name $STORAGE_NAME --pattern '$TIMESTAMP/*'"
echo ""
echo "Stop fleet (pause billing):"
for i in $(seq 0 $((NUM_VMS - 1))); do
    echo "  az vm deallocate -g $RG -n qaoa-vm-${i} --no-wait"
done
echo ""
echo "Restart fleet (after eviction/stop):"
for i in $(seq 0 $((NUM_VMS - 1))); do
    echo "  az vm start -g $RG -n qaoa-vm-${i} --no-wait"
done
echo ""
echo "Destroy everything:"
echo "  az group delete -n $RG --yes"
