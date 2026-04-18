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
$totalCPUs = [Environment]::ProcessorCount
Write-Host "CPUs: $totalCPUs, Memory: ${mem}GB"
Write-Host ""

$logDir = Join-Path $RepoDir "results"

function Start-Sweep-And-Wait($D, $threads, $pmax) {
    $logFile = Join-Path $logDir "maxcut-k2-d${D}-sweep.log"
    $errFile = Join-Path $logDir "maxcut-k2-d${D}-sweep.err"
    Write-Host "  D=$D  threads=$threads  p_max=$pmax  (log: $logFile)"
    $proc = Start-Process -FilePath julia `
        -ArgumentList "--project=.", "-t", $threads, "scripts/maxcut_sweep.jl", $D, $pmax, 42 `
        -WorkingDirectory $RepoDir `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError $errFile `
        -PassThru -NoNewWindow
    Write-Host "  PID: $($proc.Id)"
    $proc.WaitForExit()
    $exit = $proc.ExitCode
    $status = if ($exit -eq 0) { "OK" } else { "STOPPED (exit $exit)" }
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] D=$D finished: $status"
    Write-Host ""
}

# ── Sequential sweep: all D values solo with full resources ───────
# Order: most behind first, so the table fills evenly.
# p_max=12 for D=3..5 (memory allows solo); p_max=11 for D=6..8.
$sweeps = @(
    @{ D = 6; pmax = 11 },
    @{ D = 7; pmax = 11 },
    @{ D = 8; pmax = 11 },
    @{ D = 3; pmax = 12 },
    @{ D = 5; pmax = 12 },
    @{ D = 4; pmax = 12 }
)

foreach ($s in $sweeps) {
    Write-Host "============================================"
    Write-Host "=== MaxCut D=$($s.D), p_max=$($s.pmax), solo ==="
    Write-Host "============================================"
    Start-Sweep-And-Wait $s.D $totalCPUs $s.pmax
}

# ── Final report ──────────────────────────────────────────────────
Write-Host ""
Write-Host "=== All sweeps complete ==="
foreach ($D in 3,4,5,6,7,8) {
    $csv = Join-Path $logDir "maxcut-k2-d${D}-sweep.csv"
    $rows = if (Test-Path $csv) { (Get-Content $csv | Where-Object { $_ -match '^\d' }).Count } else { 0 }
    Write-Host "  D=${D}: $rows depths computed"
}
Write-Host "Results in results\maxcut-k2-d*-sweep.csv"
Write-Host "Logs in results\maxcut-k2-d*-sweep.log"
