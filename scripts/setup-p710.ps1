# Setup and run QAOA swarm on Windows (P710)
# Run this in PowerShell as Administrator

Write-Host "=== QAOA Swarm Setup for P710 ===" -ForegroundColor Green

# 1. Install Julia via juliaup
Write-Host "Installing Julia..." -ForegroundColor Yellow
if (-not (Get-Command julia -ErrorAction SilentlyContinue)) {
    winget install julia -e --accept-package-agreements --accept-source-agreements
    $env:PATH = "$env:LOCALAPPDATA\Programs\Julia\bin;$env:PATH"
}
julia --version

# 2. Clone repo
Write-Host "Cloning repo..." -ForegroundColor Yellow
cd C:\Users\johnaz
if (-not (Test-Path "qaoa-xorsat")) {
    git clone https://github.com/johnazariah/qaoa-xorsat.git
}
cd qaoa-xorsat
git pull origin main

# 3. Install Julia deps
Write-Host "Installing Julia dependencies..." -ForegroundColor Yellow
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"

# 4. Smoke test
Write-Host "Smoke test..." -ForegroundColor Yellow
julia --project=. -e "using QaoaXorsat; println(`"OK`")"

Write-Host ""
Write-Host "=== Setup complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Now run these in separate PowerShell windows:" -ForegroundColor Cyan
Write-Host "  cd C:\Users\johnaz\qaoa-xorsat"
Write-Host "  julia --project=. -t 16 scripts\swarm_chain.jl 5 7 12 100 10 20 42"
Write-Host ""
Write-Host "  cd C:\Users\johnaz\qaoa-xorsat"
Write-Host "  julia --project=. -t 16 scripts\swarm_chain.jl 5 8 12 100 10 20 42"
Write-Host ""
Write-Host "Results appear in results\swarm-k5d7.csv and results\swarm-k5d8.csv"
