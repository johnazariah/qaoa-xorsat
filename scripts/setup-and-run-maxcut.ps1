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

if ($mem -ge 140) { $pmax = 14 }
elseif ($mem -ge 40) { $pmax = 13 }
else { $pmax = 12 }
Write-Host "CPUs: $totalCPUs, Memory: ${mem}GB, targeting p=$pmax"
Write-Host ""

$logDir = Join-Path $RepoDir "results"

function Wait-Sweeps($jobs) {
    while ($jobs | Where-Object { -not $_.Process.HasExited }) {
        Start-Sleep -Seconds 60
        foreach ($j in $jobs) {
            if ($j.Process.HasExited -and -not $j.Reported) {
                $exit = $j.Process.ExitCode
                $status = if ($exit -eq 0) { "OK" } else { "FAILED (exit $exit)" }
                Write-Host "[$(Get-Date -Format 'HH:mm:ss')] D=$($j.D) finished: $status"
                $j.Reported = $true
            }
        }
    }
}

function Start-Sweep($D, $threads) {
    $logFile = Join-Path $logDir "maxcut-k2-d${D}-sweep.log"
    Write-Host "  D=$D  threads=$threads  (log: $logFile)"
    $proc = Start-Process -FilePath julia `
        -ArgumentList "--project=.", "-t", $threads, "scripts/maxcut_sweep.jl", $D, $pmax, 42 `
        -WorkingDirectory $RepoDir `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError (Join-Path $logDir "maxcut-k2-d${D}-sweep.err") `
        -PassThru -NoNewWindow
    return @{ D = $D; Process = $proc; Log = $logFile }
}

# ── Run sweeps in pairs (2 at a time, half CPUs each) ─────────────
$allD = @(3, 4, 5, 6, 7, 8)
$threads2 = [math]::Floor($totalCPUs / 2)

for ($i = 0; $i -lt $allD.Count; $i += 2) {
    $pair = @($allD[$i])
    if ($i + 1 -lt $allD.Count) { $pair += $allD[$i + 1] }
    $nPair = $pair.Count
    $thr = [math]::Floor($totalCPUs / $nPair)

    Write-Host ""
    Write-Host "=== Batch: D=$($pair -join ',') parallel, $thr threads each ==="
    $jobs = @()
    foreach ($D in $pair) {
        $jobs += Start-Sweep $D $thr
    }
    Write-Host "PIDs: $($jobs | ForEach-Object { $_.Process.Id })"
    Wait-Sweeps $jobs
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
