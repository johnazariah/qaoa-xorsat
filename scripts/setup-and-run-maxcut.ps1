# Setup and run MaxCut sweep on Windows.
#
# Usage:
#   1. Clone or copy this repo
#   2. Open PowerShell in the repo root
#   3. Run: .\scripts\setup-and-run-maxcut.ps1
#
# Prerequisites: none (downloads Julia if needed)

$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoDir

Write-Host "=== MaxCut Sweep Setup ==="
Write-Host "Host: $env:COMPUTERNAME"
Write-Host "CPUs: $([Environment]::ProcessorCount)"
$mem = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
Write-Host "Memory: ${mem}GB"
Write-Host "Date: $(Get-Date -Format u)"
Write-Host ""

# ── Install Julia if not present ──────────────────────────────────
$juliaCmd = Get-Command julia -ErrorAction SilentlyContinue
if ($juliaCmd) {
    Write-Host "Julia found: $(julia --version)"
} else {
    Write-Host "Installing Julia 1.12..."
    $juliaUrl = "https://julialang-s3.julialang.org/bin/winnt/x64/1.12/julia-1.12.5-win64.zip"
    $zipPath = "$env:TEMP\julia.zip"
    $installDir = "$env:LOCALAPPDATA\Julia"

    Write-Host "Downloading from $juliaUrl ..."
    Invoke-WebRequest -Uri $juliaUrl -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $installDir -Force
    Remove-Item $zipPath

    $juliaBin = Get-ChildItem -Path $installDir -Recurse -Filter "julia.exe" | Select-Object -First 1
    $juliaDir = $juliaBin.DirectoryName
    $env:PATH = "$juliaDir;$env:PATH"

    Write-Host "Installed: $(julia --version)"
    Write-Host ""
    Write-Host "Add to PATH permanently via System Properties > Environment Variables"
    Write-Host "  $juliaDir"
    Write-Host ""
}

# ── Install Julia packages ────────────────────────────────────────
Write-Host "Installing Julia packages..."
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
Write-Host "Packages ready."
Write-Host ""

# ── Determine settings ────────────────────────────────────────────
$threads = [Environment]::ProcessorCount
Write-Host "Running with $threads threads per sweep"

if ($mem -ge 140) { $pmax = 14 }
elseif ($mem -ge 40) { $pmax = 13 }
else { $pmax = 12 }
Write-Host "Memory ${mem}GB: targeting p=$pmax"
Write-Host ""

# ── Run MaxCut sweeps ─────────────────────────────────────────────
foreach ($D in 3, 4, 5, 6, 7, 8) {
    Write-Host "============================================"
    Write-Host "=== MaxCut D=$D, p=1..$pmax ==="
    Write-Host "============================================"
    julia --project=. -t $threads scripts/maxcut_sweep.jl $D $pmax 42
    Write-Host ""
}

Write-Host "=== All sweeps complete ==="
Write-Host "Results in results\maxcut-k2-d*-sweep.csv"
