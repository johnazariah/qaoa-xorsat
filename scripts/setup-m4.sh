#!/usr/bin/env bash
# Setup script for running QAOA-XORSAT on a bare macOS M4 host.
# Usage: bash scripts/setup-m4.sh
set -euo pipefail

echo "=== QAOA-XORSAT M4 Host Setup ==="

# 1. Check macOS and Apple Silicon
echo ""
echo "--- System ---"
uname -a
sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "(CPU brand unavailable)"
echo "Physical cores: $(sysctl -n hw.physicalcpu)"
echo "Logical cores:  $(sysctl -n hw.logicalcpu)"
echo "Memory:         $(sysctl -n hw.memsize | awk '{printf "%.0f GB\n", $1/1073741824}')"
echo "GPU cores:      $(system_profiler SPDisplaysDataType 2>/dev/null | grep 'Total Number of Cores' | awk -F: '{print $2}' | tr -d ' ' || echo 'unknown')"

# 2. Install juliaup if not present
echo ""
echo "--- Julia ---"
if command -v juliaup &>/dev/null; then
    echo "juliaup already installed: $(juliaup --version)"
    juliaup update
else
    echo "Installing juliaup..."
    curl -fsSL https://install.julialang.org | sh -s -- --yes
    export PATH="$HOME/.juliaup/bin:$PATH"
fi

julia --version
echo "Julia threads available: $(julia -e 'println(Sys.CPU_THREADS)')"

# 3. Clone or update the repo
echo ""
echo "--- Repository ---"
REPO_DIR="${QAOA_REPO_DIR:-$HOME/work/qaoa-xorsat}"

if [ -d "$REPO_DIR/.git" ]; then
    echo "Repo exists at $REPO_DIR — pulling latest"
    cd "$REPO_DIR"
    git pull --ff-only origin main
else
    echo "Cloning to $REPO_DIR"
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone git@github.com:johnazariah/qaoa-xorsat.git "$REPO_DIR"
    cd "$REPO_DIR"
fi

echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"

# 4. Install Julia dependencies
echo ""
echo "--- Dependencies ---"
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
echo "Dependencies installed and precompiled."

# 5. Run tests to verify everything works
echo ""
echo "--- Test Suite ---"
julia --project=. -t auto -e 'using Pkg; Pkg.test()'

# 6. Benchmark single evaluation at key depths
echo ""
echo "--- Benchmarks ---"
julia --project=. -t auto -e '
using QaoaXorsat

println("Single-evaluation benchmarks (k=3, D=4):")
println("p | N=2^(2p+1) | time")
println("--|-----------|-----")

for p in [5, 8, 10, 12]
    params = TreeParams(3, 4, p)
    angles = QAOAAngles(rand(p), rand(p))

    # Warm up
    basso_expectation(params, angles)

    # Time
    t0 = time_ns()
    for _ in 1:3
        basso_expectation(params, angles)
    end
    t = (time_ns() - t0) / 3.0e9

    N = 2^(2*p+1)
    println("$p | $N | $(round(t, sigdigits=3))s")
end
'

# 7. Report ready
echo ""
echo "=== Setup Complete ==="
echo "Repo:    $REPO_DIR"
echo "Julia:   $(julia --version)"
echo "Threads: $(julia -e 'println(Sys.CPU_THREADS)')"
echo ""
echo "To run optimization:"
echo "  cd $REPO_DIR"
echo "  julia --project=. -t 10 scripts/optimize_qaoa.jl 3 4 1 12 4 200 1234 true"
echo ""
echo "Key flags:"
echo "  -t 10    : use 10 threads (all performance cores)"
echo "  args:    : k D p_min p_max restarts maxiters seed preserve"
