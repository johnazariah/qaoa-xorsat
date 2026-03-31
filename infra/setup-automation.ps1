<#
.SYNOPSIS
    Set up Azure Automation to auto-restart evicted QAOA Spot VMs.
    Creates an Automation Account, imports the runbook, and schedules it.

.PARAMETER Subscription
    Azure subscription ID

.PARAMETER ResourceGroup
    Resource group for the Automation Account (same as fleet)

.PARAMETER FleetResourceGroup
    Resource group containing the VM fleet (default: same as ResourceGroup)

.PARAMETER Location
    Azure region (default: eastus2)

.EXAMPLE
    .\infra\setup-automation.ps1 -Subscription "9c75eb67-..." -ResourceGroup qaoa-fleet-2
#>
param(
    [Parameter(Mandatory)][string]$Subscription,
    [string]$ResourceGroup = "qaoa-fleet-2",
    [string]$FleetResourceGroup = "",
    [string]$Location = "eastus2"
)

if (-not $FleetResourceGroup) { $FleetResourceGroup = $ResourceGroup }

$AutomationAccount = "qaoa-automation"
$RunbookName = "RestartEvictedVMs"
$ScheduleName = "EveryFiveMinutes"

Write-Host "=== QAOA Automation Setup ===" -ForegroundColor Cyan
Write-Host "Subscription:    $Subscription"
Write-Host "Resource Group:  $ResourceGroup"
Write-Host "Fleet RG:        $FleetResourceGroup"
Write-Host "Automation Acct: $AutomationAccount"
Write-Host ""

# Set subscription
az account set -s $Subscription

# Create Automation Account with system-assigned managed identity
Write-Host "Creating Automation Account..." -ForegroundColor DarkCyan
az automation account create `
    -g $ResourceGroup -n $AutomationAccount -l $Location `
    --assign-identity `
    -o none 2>$null

# Get the managed identity's principal ID
$principalId = az automation account show -g $ResourceGroup -n $AutomationAccount `
    --query "identity.principalId" -o tsv

Write-Host "Managed identity principal: $principalId"

# Grant the managed identity VM Contributor on the fleet resource group
Write-Host "Assigning VM Contributor role..." -ForegroundColor DarkCyan
$fleetRgId = az group show -n $FleetResourceGroup --query id -o tsv
az role assignment create `
    --assignee-object-id $principalId `
    --assignee-principal-type ServicePrincipal `
    --role "Virtual Machine Contributor" `
    --scope $fleetRgId `
    -o none 2>$null

# Import the runbook
Write-Host "Importing runbook..." -ForegroundColor DarkCyan
$runbookPath = Join-Path $PSScriptRoot "restart-evicted-runbook.ps1"
az automation runbook create `
    -g $ResourceGroup --automation-account-name $AutomationAccount `
    -n $RunbookName --type PowerShell `
    -o none 2>$null

az automation runbook replace-content `
    -g $ResourceGroup --automation-account-name $AutomationAccount `
    -n $RunbookName --content @$runbookPath `
    -o none 2>$null

az automation runbook publish `
    -g $ResourceGroup --automation-account-name $AutomationAccount `
    -n $RunbookName `
    -o none 2>$null

# Create a schedule (every 5 minutes)
Write-Host "Creating schedule..." -ForegroundColor DarkCyan
$startTime = (Get-Date).AddMinutes(10).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
az automation schedule create `
    -g $ResourceGroup --automation-account-name $AutomationAccount `
    -n $ScheduleName `
    --frequency Minute --interval 5 `
    --start-time $startTime `
    -o none 2>$null

# Link schedule to runbook
Write-Host "Linking schedule to runbook..." -ForegroundColor DarkCyan
az automation job-schedule create `
    -g $ResourceGroup --automation-account-name $AutomationAccount `
    --runbook-name $RunbookName --schedule-name $ScheduleName `
    --parameters "FleetResourceGroup=$FleetResourceGroup" "NumVMs=5" `
    -o none 2>$null

Write-Host ""
Write-Host "=== Automation Setup Complete ===" -ForegroundColor Green
Write-Host "Runbook '$RunbookName' will check VM state every 5 minutes."
Write-Host "This runs in Azure — works even when your laptop is off."
Write-Host ""
Write-Host "To disable: az automation schedule update -g $ResourceGroup --automation-account-name $AutomationAccount -n $ScheduleName --is-enabled false"
Write-Host "To delete:  az automation account delete -g $ResourceGroup -n $AutomationAccount --yes"
