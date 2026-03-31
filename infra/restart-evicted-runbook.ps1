<#
.SYNOPSIS
    Azure Automation runbook: auto-restart evicted QAOA Spot VMs.
    Runs on a schedule (every 5 min) in Azure — no local machine needed.

.DESCRIPTION
    Deploy via:
      .\infra\setup-automation.ps1 -Subscription <id> -ResourceGroup qaoa-fleet-2

    Or manually:
      1. Create Automation Account in Azure Portal
      2. Import this as a PowerShell runbook
      3. Create a schedule (every 5 min)
      4. Link schedule to runbook with parameters:
         FleetResourceGroup = "qaoa-fleet-2"
         NumVMs = 5
#>
param(
    [string]$FleetResourceGroup = "qaoa-fleet-2",
    [int]$NumVMs = 5
)

# Authenticate using Automation Account's managed identity
try {
    Connect-AzAccount -Identity -ErrorAction Stop | Out-Null
    Write-Output "Authenticated via managed identity"
} catch {
    Write-Error "Failed to authenticate: $_"
    throw
}

$restarted = @()
for ($i = 0; $i -lt $NumVMs; $i++) {
    $vmName = "qaoa-vm-$i"
    $vm = Get-AzVM -ResourceGroupName $FleetResourceGroup -Name $vmName -Status -ErrorAction SilentlyContinue
    if (-not $vm) {
        Write-Output "  $vmName`: not found"
        continue
    }

    $powerState = ($vm.Statuses | Where-Object Code -like "PowerState/*").DisplayStatus
    Write-Output "  $vmName`: $powerState"

    if ($powerState -eq "VM deallocated") {
        Write-Output "  -> Restarting $vmName..."
        Start-AzVM -ResourceGroupName $FleetResourceGroup -Name $vmName -NoWait -ErrorAction SilentlyContinue
        $restarted += $vmName
    }
}

if ($restarted.Count -gt 0) {
    Write-Output "Restarted $($restarted.Count) VMs: $($restarted -join ', ')"
} else {
    Write-Output "All VMs running — no action needed"
}
