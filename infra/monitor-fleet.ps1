<#
.SYNOPSIS
    Monitor and auto-heal the QAOA VM fleet.
    Checks VM power state every 60s, auto-restarts evicted Spot VMs,
    and prints full progress every 5 minutes.

.DESCRIPTION
    Designed to be launched by deploy-fleet.ps1 after deployment,
    or run standalone. Also deployed as an Azure Automation runbook
    for resilience when the local machine is off.

.PARAMETER ResourceGroup
    Resource group containing the fleet (default: qaoa-fleet-2)

.PARAMETER NumVMs
    Number of VMs in the fleet (default: 5)

.PARAMETER IntervalSeconds
    Seconds between eviction checks (default: 60)

.PARAMETER ProgressEveryN
    Show full SSH progress every N ticks (default: 5)

.EXAMPLE
    .\infra\monitor-fleet.ps1
    .\infra\monitor-fleet.ps1 -ResourceGroup qaoa-fleet-3 -NumVMs 5
#>
param(
    [string]$ResourceGroup = "qaoa-fleet-2",
    [int]$NumVMs = 5,
    [int]$IntervalSeconds = 60,
    [int]$ProgressEveryN = 5
)

$ErrorActionPreference = "Continue"

# Build VM list from Azure
Write-Host "Discovering fleet in $ResourceGroup..." -ForegroundColor Cyan
$vms = @()
for ($i = 0; $i -lt $NumVMs; $i++) {
    $azName = "qaoa-vm-$i"
    $ip = az vm show -g $ResourceGroup -n $azName -d --query publicIps -o tsv 2>$null
    if (-not $ip) { $ip = "pending" }
    $vms += @{ Name = "VM-$i"; AzName = $azName; IP = $ip }
}

$tick = 0
Write-Host "Monitoring $NumVMs VMs. Eviction check every ${IntervalSeconds}s, progress every $($IntervalSeconds * $ProgressEveryN)s." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop.`n"

while ($true) {
    $tick++
    $now = Get-Date -Format 'HH:mm:ss'

    # Every tick: check power state and auto-restart evicted VMs
    $restarted = @()
    foreach ($vm in $vms) {
        $state = az vm get-instance-view -g $ResourceGroup -n $vm.AzName --query "instanceView.statuses[1].displayStatus" -o tsv 2>$null
        if ($state -eq "VM deallocated") {
            az vm start -g $ResourceGroup -n $vm.AzName --no-wait 2>$null
            $restarted += $vm.Name
            # Refresh IP after restart
            $newIp = az vm show -g $ResourceGroup -n $vm.AzName -d --query publicIps -o tsv 2>$null
            if ($newIp) { $vm.IP = $newIp }
        }
    }
    if ($restarted.Count -gt 0) {
        Write-Host "[$now] EVICTED - restarting: $($restarted -join ', ')" -ForegroundColor Red
    }

    # Every Nth tick: full progress report via SSH
    if ($tick % $ProgressEveryN -eq 1 -or $tick -eq 1) {
        Write-Host "`n=== Fleet Status @ $now ===" -ForegroundColor Cyan
        foreach ($vm in $vms) {
            # Refresh IP
            $newIp = az vm show -g $ResourceGroup -n $vm.AzName -d --query publicIps -o tsv 2>$null
            if ($newIp) { $vm.IP = $newIp }
            $line = ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 -o LogLevel=ERROR "azureuser@$($vm.IP)" "bash -c 'cat /home/azureuser/qaoa-xorsat/results/logs/pair-k[0-9]*-d[0-9]*.log 2>/dev/null | tail -n 1 || echo no_output'" 2>$null
            if (-not $line) { $line = "booting/unreachable" }
            Write-Host "  $($vm.Name): $line"
        }
    } else {
        Write-Host "[$now] tick $tick - fleet OK" -ForegroundColor DarkGray -NoNewline
        if ($restarted.Count -eq 0) { Write-Host "" }
    }

    Start-Sleep -Seconds $IntervalSeconds
}
