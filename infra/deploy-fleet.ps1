<#
.SYNOPSIS
    Deploy a fleet of Azure VMs to run the QAOA sweep in parallel.
    Each VM runs a subset of (k,D) pairs independently.
    Results sync to Azure Blob every 10 minutes.

.PARAMETER Subscription
    Azure subscription ID (required)

.PARAMETER ResourceGroup
    Resource group name (default: qaoa-fleet)

.PARAMETER Location
    Azure region (default: eastus2)

.PARAMETER PMax
    Maximum QAOA depth (default: 12)

.PARAMETER NumVMs
    Number of VMs (default: 5)

.PARAMETER VMSize
    VM size (default: auto from PMax)

.PARAMETER DryRun
    Print without executing

.EXAMPLE
    .\infra\deploy-fleet.ps1 -Subscription "12345-abcde" -PMax 12
    .\infra\deploy-fleet.ps1 -Subscription "12345-abcde" -PMax 13 -DryRun
#>
param(
    [Parameter(Mandatory)][string]$Subscription,
    [string]$ResourceGroup = "qaoa-fleet",
    [string]$Location = "eastus2",
    [int]$PMax = 12,
    [int]$NumVMs = 5,
    [string]$VMSize = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# Auto-size VM
if (-not $VMSize) {
    $VMSize = switch ($PMax) {
        { $_ -le 11 } { "Standard_E4as_v5"  } # 32GB
        12             { "Standard_E8as_v5"  } # 64GB
        13             { "Standard_E32as_v5" } # 256GB
        14             { "Standard_E64as_v5" } # 512GB
        default        { throw "p=$PMax — use deploy-vm for single big VM" }
    }
}

# The 15 (k,D) pairs
$AllPairs = @(
    "3,4","3,5","3,6","3,7","3,8",
    "4,5","4,6","4,7","4,8",
    "5,6","5,7","5,8",
    "6,7","6,8",
    "7,8"
)

# Round-robin distribution
$vmPairs = @{}
for ($i = 0; $i -lt $NumVMs; $i++) { $vmPairs[$i] = @() }
for ($i = 0; $i -lt $AllPairs.Count; $i++) { $vmPairs[$i % $NumVMs] += $AllPairs[$i] }

# Storage account name (must be lowercase alphanumeric, ≤24 chars)
$storageSuffix = ($ResourceGroup -replace '[^a-z0-9]','').Substring(0, [Math]::Min(10, ($ResourceGroup -replace '[^a-z0-9]','').Length))
$StorageName = "qaoa$storageSuffix$(Get-Date -Format 'dd')"
$ContainerName = "results"
$Timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmss")

Write-Host "=== QAOA Fleet Deployment ===" -ForegroundColor Cyan
Write-Host "Subscription:  $Subscription"
Write-Host "Resource Group: $ResourceGroup"
Write-Host "Location:       $Location"
Write-Host "VM Size:        $VMSize"
Write-Host "Num VMs:        $NumVMs"
Write-Host "P_MAX:          $PMax"
Write-Host "Storage:        $StorageName"
Write-Host ""
Write-Host "Pair distribution:" -ForegroundColor Yellow
for ($i = 0; $i -lt $NumVMs; $i++) {
    Write-Host "  VM-$i`: $($vmPairs[$i] -join ', ')"
}
Write-Host ""

function Invoke-AzCmd {
    param([string]$Description, [string[]]$Arguments)
    Write-Host "$Description..." -ForegroundColor DarkCyan
    if ($DryRun) {
        Write-Host "  [dry-run] az $($Arguments -join ' ')" -ForegroundColor DarkGray
    } else {
        $result = & az @Arguments 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "az command returned exit code $LASTEXITCODE"
            Write-Warning ($result | Out-String)
        }
        return $result
    }
}

# ── Create infrastructure ─────────────────────────────────────────────────────
Invoke-AzCmd "Setting subscription" @("account", "set", "-s", $Subscription)
Invoke-AzCmd "Creating resource group" @("group", "create", "-n", $ResourceGroup, "-l", $Location, "-o", "none")
Invoke-AzCmd "Creating storage account" @("storage", "account", "create", "-g", $ResourceGroup, "-n", $StorageName, "-l", $Location, "--sku", "Standard_LRS", "--kind", "StorageV2", "-o", "none")

$StorageKey = ""
if (-not $DryRun) {
    $StorageKey = (az storage account keys list -g $ResourceGroup -n $StorageName --query '[0].value' -o tsv)
    az storage container create -n $ContainerName --account-name $StorageName --account-key $StorageKey -o none 2>$null
}

# ── Deploy VMs ────────────────────────────────────────────────────────────────
for ($i = 0; $i -lt $NumVMs; $i++) {
    $vmName = "qaoa-vm-$i"
    $pairs = $vmPairs[$i] -join " "

    # Build cloud-init script
    $initScript = @"
#!/usr/bin/env bash
set -eo pipefail
export HOME=`${HOME:-/root}
exec > /var/log/qaoa-setup.log 2>&1
echo "=== VM-$i setup at `$(date -u) ==="
echo "Pairs: $pairs"

# Install Julia
curl -fsSL https://install.julialang.org | sh -s -- --yes
export PATH="`$HOME/.juliaup/bin:`$PATH"

# Install az CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | bash 2>/dev/null

# Clone and setup
cd /home/azureuser
git clone https://github.com/johnazariah/qaoa-xorsat.git
cd qaoa-xorsat
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
mkdir -p results/logs

# ── run-pairs.sh: resume-aware pair runner ────────────────────────────────────
cat > /home/azureuser/qaoa-xorsat/run-pairs.sh << 'RUNSCRIPT'
#!/usr/bin/env bash
set -eo pipefail
export HOME=`${HOME:-/root}
export PATH="`$HOME/.juliaup/bin:`$PATH"
cd /home/azureuser/qaoa-xorsat
P_MAX=$PMax
THREADS=`$(nproc)

# Find max completed p for a given (k,D) across all run CSVs
max_completed_p() {
    local k=`$1 d=`$2
    local max_p=0
    for csv in .project/results/optimization/runs/*/results.csv; do
        [ -f "`$csv" ] || continue
        # CSV col 8=k, 9=D, 10=p (1-indexed); filter matching k,D rows, get max p
        local p=`$(awk -F, -v k="`$k" -v d="`$d" 'NR>1 && `$8==k && `$9==d {if(`$10+0 > max) max=`$10+0} END {print max+0}' "`$csv")
        [ "`$p" -gt "`$max_p" ] && max_p=`$p
    done
    echo "`$max_p"
}

for PAIR in PAIRS_PLACEHOLDER; do
    K=`$(echo `$PAIR | cut -d, -f1)
    D=`$(echo `$PAIR | cut -d, -f2)
    LOG="results/logs/pair-k`${K}-d`${D}.log"

    # Resume: find where we left off
    DONE_P=`$(max_completed_p `$K `$D)
    P_MIN=`$((DONE_P + 1))

    if [ "`$P_MIN" -gt "`$P_MAX" ]; then
        echo "`$(date -u): (k=`$K, D=`$D) already complete through p=`$DONE_P, skipping"
        continue
    fi

    if [ "`$DONE_P" -gt 0 ]; then
        echo "`$(date -u): (k=`$K, D=`$D) resuming from p=`$P_MIN (completed through p=`$DONE_P)"
    else
        echo "`$(date -u): (k=`$K, D=`$D) starting fresh p=1..`$P_MAX"
        P_MIN=1
    fi

    julia --project=. -t `$THREADS scripts/optimize_qaoa.jl `$K `$D `$P_MIN `$P_MAX 2 320 1234 true adjoint >> "`$LOG" 2>&1
    echo "`$(date -u): Completed (k=`$K, D=`$D)"
done

echo "=== All pairs complete at `$(date -u) ==="
RUNSCRIPT
sed -i "s/PAIRS_PLACEHOLDER/$pairs/" /home/azureuser/qaoa-xorsat/run-pairs.sh
chmod +x /home/azureuser/qaoa-xorsat/run-pairs.sh

# ── blob-sync.sh: periodic result upload ──────────────────────────────────────
cat > /home/azureuser/qaoa-xorsat/blob-sync.sh << 'BLOBSCRIPT'
#!/usr/bin/env bash
while true; do
    sleep 600
    cd /home/azureuser/qaoa-xorsat
    for f in .project/results/optimization/runs/*/results.csv; do
        [ -f "`$f" ] || continue
        bn="`$(basename `$(dirname `$f))/results.csv"
        az storage blob upload -f "`$f" -c "$ContainerName" \
            -n "$Timestamp/vm-$i/`$bn" \
            --account-name "$StorageName" --account-key "$StorageKey" \
            --overwrite 2>/dev/null || true
    done
    tail -100 results/logs/pair-*.log > /tmp/vm-progress.txt 2>/dev/null || true
    az storage blob upload -f /tmp/vm-progress.txt -c "$ContainerName" \
        -n "$Timestamp/vm-$i/progress.txt" \
        --account-name "$StorageName" --account-key "$StorageKey" \
        --overwrite 2>/dev/null || true
done
BLOBSCRIPT
chmod +x /home/azureuser/qaoa-xorsat/blob-sync.sh

# ── systemd services ─────────────────────────────────────────────────────────
cat > /etc/systemd/system/qaoa.service << 'SVCEOF'
[Unit]
Description=QAOA-XORSAT Optimization
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/home/azureuser/qaoa-xorsat/run-pairs.sh
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
SVCEOF

cat > /etc/systemd/system/qaoa-sync.service << 'SYNCEOF'
[Unit]
Description=QAOA-XORSAT Blob Sync
After=network-online.target

[Service]
Type=simple
ExecStart=/home/azureuser/qaoa-xorsat/blob-sync.sh
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
SYNCEOF

systemctl daemon-reload
systemctl enable qaoa.service qaoa-sync.service
systemctl start qaoa.service qaoa-sync.service

echo "=== VM-$i setup complete at `$(date -u) ==="
"@

    # Write cloud-init to temp file
    $initFile = Join-Path $env:TEMP "qaoa-init-$i.sh"
    [System.IO.File]::WriteAllBytes($initFile, [System.Text.Encoding]::UTF8.GetBytes($initScript))

    Invoke-AzCmd "Creating $vmName (pairs: $pairs)" @(
        "vm", "create",
        "-g", $ResourceGroup, "-n", $vmName, "-l", $Location,
        "--size", $VMSize,
        "--image", "Canonical:ubuntu-24_04-lts:server:latest",
        "--admin-username", "azureuser",
        "--generate-ssh-keys",
        "--custom-data", $initFile,
        "--os-disk-size-gb", "64",
        "--priority", "Spot",
        "--eviction-policy", "Deallocate",
        "--max-price", "-1",
        "-o", "none"
    )

    Remove-Item $initFile -ErrorAction SilentlyContinue
}

# ── Print monitoring commands ─────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Fleet Deployed ===" -ForegroundColor Green
Write-Host ""

if (-not $DryRun) {
    Write-Host "VM IPs:" -ForegroundColor Yellow
    for ($i = 0; $i -lt $NumVMs; $i++) {
        $ip = az vm show -g $ResourceGroup -n "qaoa-vm-$i" -d --query publicIps -o tsv 2>$null
        if (-not $ip) { $ip = "pending" }
        Write-Host "  qaoa-vm-$i`: $ip (pairs: $($vmPairs[$i] -join ', '))"
    }
}

Write-Host ""
Write-Host "Monitor setup (~5 min after deploy):" -ForegroundColor Yellow
for ($i = 0; $i -lt $NumVMs; $i++) {
    Write-Host "  ssh azureuser@<vm-$i-ip> 'tail -5 /var/log/qaoa-setup.log'"
}
Write-Host ""
Write-Host "Monitor progress:" -ForegroundColor Yellow
Write-Host "  az storage blob list -c $ContainerName --account-name $StorageName --prefix '$Timestamp/' -o table"
Write-Host ""
Write-Host "Download results:" -ForegroundColor Yellow
Write-Host "  az storage blob download-batch -d ./results-fleet/ -s $ContainerName --account-name $StorageName --pattern '$Timestamp/*'"
Write-Host ""
Write-Host "Stop fleet (pause billing):" -ForegroundColor Yellow
for ($i = 0; $i -lt $NumVMs; $i++) {
    Write-Host "  az vm deallocate -g $ResourceGroup -n qaoa-vm-$i --no-wait"
}
Write-Host ""
Write-Host "Restart fleet:" -ForegroundColor Yellow
for ($i = 0; $i -lt $NumVMs; $i++) {
    Write-Host "  az vm start -g $ResourceGroup -n qaoa-vm-$i --no-wait"
}
Write-Host ""
Write-Host "Destroy everything:" -ForegroundColor Red
Write-Host "  az group delete -n $ResourceGroup --yes"

# ── Launch monitor with auto-restart ──────────────────────────────────────────
if (-not $DryRun) {
    Write-Host ""
    Write-Host "Launching fleet monitor (auto-restarts evicted VMs)..." -ForegroundColor Green
    $monitorScript = Join-Path $PSScriptRoot "monitor-fleet.ps1"
    Start-Process pwsh -ArgumentList "-NoExit", "-File", $monitorScript, "-ResourceGroup", $ResourceGroup, "-NumVMs", $NumVMs -WindowStyle Normal
    Write-Host "  Monitor running in new window. Close it to stop monitoring."
}
