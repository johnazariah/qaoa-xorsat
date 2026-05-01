# Reproducing Paper Results

Step-by-step guide to reproducing the results presented in Shutty et al. (arXiv:2604.24633).

## Prerequisites

- Julia 1.11+ (1.12 recommended)
- 32 GB RAM minimum (for p≤12), 64 GB recommended
- Multi-core CPU recommended (use `-t auto` for all cores)

## Installation

```bash
git clone https://github.com/johnazariah/qaoa-xorsat.git
cd qaoa-xorsat
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Running the Test Suite

Verify the installation:

```bash
julia --project=. -t auto -e 'using Pkg; Pkg.test()'
```

## Table 1: Max-3-XORSAT (k=3, D=4) through p=12

This is the primary result — QAOA satisfaction fractions for random 4-regular 3-uniform XORSAT.

```bash
julia --project=. -t auto scripts/run_xorsat.jl 3 4 12
```

Output: `results/xorsat-k3-d4-sweep.csv`

Expected values at key depths:
- p=1: c̃ ≈ 0.8040
- p=8: c̃ ≈ 0.8610
- p=11: c̃ ≈ 0.8725 (exceeds DQI+BP = 0.871)
- p=12: c̃ ≈ 0.8769 (exceeds Prange = 0.875)

**Time**: ~40 minutes on a 10-core Mac Studio.

## Table 2: All 15 (k,D) Pairs

Run all 15 combinations from the D-regular k-uniform table:

```bash
# Run each pair individually
for k_d in "3 4" "3 5" "3 6" "3 7" "3 8" \
           "4 5" "4 6" "4 7" "4 8" \
           "5 6" "5 7" "5 8" \
           "6 7" "6 8" \
           "7 8"; do
    julia --project=. -t auto scripts/run_xorsat.jl $k_d 12
done
```

**Time**: ~7 hours total on a Mac Studio (dominated by large branching-factor pairs).

## Table 3: MaxCut Validation

MaxCut (k=2) is used to validate the implementation against known results:

```bash
julia --project=. -t auto scripts/run_maxcut.jl 3 12
julia --project=. -t auto scripts/run_maxcut.jl 4 12
julia --project=. -t auto scripts/run_maxcut.jl 5 12
```

Output: `results/maxcut-k2-d{3,4,5}-sweep.csv`

Expected validation: D=3, p=1 gives c̃ = 0.6925 (matches Farhi et al. 2014 exactly).

## Using Docker

```bash
docker build -t qaoa-xorsat .
docker run --rm qaoa-xorsat julia --project=. -t auto scripts/run_maxcut.jl 3 8
```

## Notes

- All scripts support **resume**: if interrupted, re-running reads the existing CSV and continues from the last completed depth.
- Results are appended to CSV files, so you can run incrementally.
- Use `Double64` precision (automatic for k≥6) for large branching factors.
- The optimizer uses warm-start angle seeding: each depth p initialises from the p-1 optimum.
