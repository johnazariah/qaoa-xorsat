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

function Start-Sweep($D, $threads, $pmax) {
    $logFile = Join-Path $logDir "maxcut-k2-d${D}-sweep.log"
    Write-Host "  D=$D  threads=$threads  p_max=$pmax  (log: $logFile)"
    $proc = Start-Process -FilePath julia `
        -ArgumentList "--project=.", "-t", $threads, "scripts/maxcut_sweep.jl", $D, $pmax, 42 `
        -WorkingDirectory $RepoDir `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError (Join-Path $logDir "maxcut-k2-d${D}-sweep.err") `
        -PassThru -NoNewWindow
    return @{ D = $D; Process = $proc; Log = $logFile }
}

# ── Memory budget per depth ───────────────────────────────────────
# p=10: ~2 GB/eval  → 6 in parallel fine
# p=11: ~8 GB/eval  → 3 in parallel (with semaphore + headroom)
# p=12: ~19 GB/eval → 1 at a time (solo, needs ~32 GB)
# p=13: ~84 GB/eval → 1 at a time (solo, needs ~128 GB)

# ── Phase 1: Catch everyone up to p=10 (3 at a time) ─────────────
Write-Host ""
Write-Host "=== Phase 1: Catch up D=6,7,8 to p=10 ==="
$catchUp = @(6, 7, 8)
$thr = [math]::Floor($totalCPUs / $catchUp.Count)
$jobs = @()
foreach ($D in $catchUp) {
    $jobs += Start-Sweep $D $thr 10
}
Write-Host "PIDs: $($jobs | ForEach-Object { $_.Process.Id })"
Wait-Sweeps $jobs

# ── Phase 2: All 6 to p=11 (3 at a time) ─────────────────────────
Write-Host ""
Write-Host "=== Phase 2: All D to p=11 (3 at a time) ==="
$allD = @(3, 4, 5, 6, 7, 8)
for ($i = 0; $i -lt $allD.Count; $i += 3) {
    $batch = @()
    for ($j = $i; $j -lt [math]::Min($i + 3, $allD.Count); $j++) { $batch += $allD[$j] }
    $thr = [math]::Floor($totalCPUs / $batch.Count)

    Write-Host ""
    Write-Host "--- Batch: D=$($batch -join ','), $thr threads each, p_max=11 ---"
    $jobs = @()
    foreach ($D in $batch) {
        $jobs += Start-Sweep $D $thr 11
    }
    Write-Host "PIDs: $($jobs | ForEach-Object { $_.Process.Id })"
    Wait-Sweeps $jobs
}

# ── Phase 3: D=3,4,5 to p=12 (one at a time) ────────────────────
Write-Host ""
Write-Host "=== Phase 3: D=3,4,5 to p=12 (sequential, all CPUs) ==="
foreach ($D in 3, 4, 5) {
    Write-Host ""
    Write-Host "--- D=$D solo, $totalCPUs threads, p_max=12 ---"
    $jobs = @(Start-Sweep $D $totalCPUs 12)
    Write-Host "PID: $($jobs[0].Process.Id)"
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
