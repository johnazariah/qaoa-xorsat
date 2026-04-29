# Canonical local startup script for MaxCut sweeps on Windows.
#
# Usage:
#   .\scripts\start-maxcut-local.ps1 [[-D] <D|all>] [[-PMax] <Int|auto>] [[-Seed] <Int>]
#
# Examples:
#   .\scripts\start-maxcut-local.ps1 all auto 42
#   .\scripts\start-maxcut-local.ps1 8 12 42

param(
    [string]$D = "all",
    [string]$PMax = "auto",
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoDir

$LogDir = Join-Path $RepoDir "logs\maxcut-local"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogFile = Join-Path $LogDir "maxcut-local-$Timestamp.log"
Start-Transcript -Path $LogFile -Append | Out-Null

try {
    $Threads = if ($env:QAOA_THREADS) { [int]$env:QAOA_THREADS } else { [Environment]::ProcessorCount }
    $MemoryGb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)

    Write-Host "=== Local MaxCut startup ==="
    Write-Host "Repo:    $RepoDir"
    Write-Host "Host:    $env:COMPUTERNAME"
    Write-Host "Threads: $Threads"
    Write-Host "Seed:    $Seed"
    Write-Host "Log:     $LogFile"
    Write-Host "Date:    $(Get-Date -Format u)"
    Write-Host ""

    if (-not (Get-Command julia -ErrorAction SilentlyContinue)) {
        Write-Host "Julia not found; installing with winget..."
        winget install julia -e --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) { throw "winget install julia failed with exit code $LASTEXITCODE" }
        $env:PATH = "$env:LOCALAPPDATA\Programs\Julia\bin;$env:PATH"
    }

    Write-Host "Julia: $(julia --version)"
    Write-Host "Instantiating project..."
    julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
    if ($LASTEXITCODE -ne 0) { throw "Julia package setup failed with exit code $LASTEXITCODE" }
    Write-Host ""

    if ($PMax -eq "auto") {
        if ($MemoryGb -ge 140) { $ResolvedPMax = 14 }
        elseif ($MemoryGb -ge 40) { $ResolvedPMax = 13 }
        else { $ResolvedPMax = 12 }
        Write-Host "Memory ${MemoryGb}GB: targeting p=$ResolvedPMax"
    } else {
        $ResolvedPMax = [int]$PMax
    }

    $DValues = if ($D -eq "all") { @(3, 4, 5, 6, 7, 8) } else { @([int]$D) }

    foreach ($Degree in $DValues) {
        Write-Host ""
        Write-Host "============================================"
        Write-Host "=== MaxCut D=$Degree through p=$ResolvedPMax ==="
        Write-Host "============================================"
        julia --project=. -t $Threads scripts/maxcut_sweep.jl $Degree $ResolvedPMax $Seed
        if ($LASTEXITCODE -ne 0) { throw "MaxCut D=$Degree failed with exit code $LASTEXITCODE" }
    }

    Write-Host ""
    Write-Host "=== Local MaxCut complete ==="
    Write-Host "Results: results\maxcut-k2-d*-sweep.csv"
    Write-Host "Log:     $LogFile"
} finally {
    Stop-Transcript | Out-Null
}
