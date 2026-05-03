# QAOA-XORSAT

**Exact QAOA performance on D-regular Max-k-XORSAT via tensor network contraction on the light-cone tree**

*John S Azariah — Centre for Quantum Software and Information, UTS*\
*ORCID: [0009-0007-9870-1970](https://orcid.org/0009-0007-9870-1970)*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19211958.svg)](https://doi.org/10.5281/zenodo.19211958)
[![CI](https://github.com/johnazariah/qaoa-xorsat/actions/workflows/ci.yml/badge.svg)](https://github.com/johnazariah/qaoa-xorsat/actions/workflows/ci.yml)
![Tests](https://img.shields.io/badge/tests-1741%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Julia](https://img.shields.io/badge/Julia-1.11+-purple)
![License](https://img.shields.io/badge/license-MIT-blue)

Reference implementation accompanying Shutty, Azariah & Jordan, *"Quantum Approximate Optimization Algorithm for D-Regular Max-k-XORSAT"* ([arXiv:2604.24633](https://arxiv.org/abs/2604.24633)).

This code computes **exact** QAOA satisfaction fractions $\tilde{c}(p)$ for depth-$p$ QAOA on $D$-regular Max-$k$-XORSAT, using tensor network contraction on the light-cone tree. The key algorithmic innovation — a Walsh-Hadamard factorisation of the $k$-body constraint fold — reduces the cost from $O(4^{kp})$ to $O(p^2 \cdot 4^p)$ for any $k$, making depths $p \leq 12$ tractable on a single workstation.

---

## Results

Best-found QAOA satisfaction fractions at finite D, computed on Apple M4 Mac Studio (64 GB), Azure fleet (5× E8as_v5, 256 GB), and Windows Server (40-core, 181 GB). Full data in [`results/`](results/).

### Primary target: (k=3, D=4) through p=12

| p | c̃(p) | Δc̃ | Wall time |
|---|-------|------|-----------|
| 1 | 0.6761 | — | 72 ms |
| 2 | 0.7391 | +0.0631 | 3 ms |
| 3 | 0.7771 | +0.0380 | 12 ms |
| 4 | 0.8022 | +0.0251 | 24 ms |
| 5 | 0.8205 | +0.0183 | 120 ms |
| 6 | 0.8344 | +0.0139 | 0.8 s |
| 7 | 0.8453 | +0.0109 | 5.2 s |
| 8 | 0.8541 | +0.0088 | 41 s |
| 9 | 0.8613 | +0.0072 | 3.6 min |
| 10 | 0.8674 | +0.0060 | 11 min |
| 11 | 0.8725 | +0.0051 | 10 min |
| **12** | **0.8769** | **+0.0044** | **~40 min** |
| **13** | **0.8807** | **+0.0038** | **~84 hr** |

### All 15 (k,D) pairs — Max-k-XORSAT

| (k,D) | p | c̃ | (k,D) | p | c̃ | (k,D) | p | c̃ |
|-------|---|------|-------|---|------|-------|---|------|
| (3,4) | 13 | 0.881 | (4,5) | 11 | 0.861 | (5,6) | 9 | 0.838 |
| (3,5) | 13 | 0.843 | (4,6) | 9 | 0.821 | (5,7) | 9 | 0.808 |
| (3,6) | 12 | 0.809 | (4,7) | 9 | 0.798 | (5,8) | 9 | 0.805 |
| (3,7) | 11 | 0.779 | (4,8) | 9 | 0.780 | (6,7) | 9 | 0.855 |
| (3,8) | 11 | 0.768 | | | | (6,8) | 8 | 0.802 |
| | | | | | | (7,8) | 8 | 0.819 |

QAOA surpasses DQI+BP for 13 of 15 pairs. To our knowledge, no prior exact finite-D QAOA evaluation has been performed for k ≥ 3.

### MaxCut (k=2) Validation — D=3 through D=8

First exact finite-D QAOA MaxCut satisfaction fractions at these depths.
Previous work (Basso et al.) computed only the large-D asymptotic coefficient ν_p^[k];
our values are exact at each finite D with no O(1/D) approximation.

| p | D=3 | D=4 | D=5 | D=6 | D=7 | D=8 |
|---|------|------|------|------|------|------|
| 1 | 0.6925 | 0.6624 | 0.6431 | 0.6294 | 0.6190 | 0.6108 |
| 2 | 0.7559 | 0.7161 | 0.6907 | 0.6726 | 0.6589 | 0.6480 |
| 3 | 0.7924 | 0.7486 | 0.7199 | 0.6993 | 0.6836 | 0.6711 |
| 4 | 0.8169 | 0.7690 | 0.7386 | 0.7165 | 0.6996 | 0.6861 |
| 5 | 0.8364 | 0.7841 | 0.7523 | 0.7292 | 0.7114 | 0.6972 |
| 6 | 0.8499 | 0.7949 | 0.7624 | 0.7386 | 0.7202 | 0.7055 |
| 7 | 0.8598 | 0.8034 | 0.7705 | 0.7460 | 0.7272 | 0.7121 |
| 8 | 0.8674 | 0.8099 | 0.7771 | 0.7519 | 0.7328 | 0.7174 |
| 9 | 0.8735 | 0.8152 | 0.7829 | 0.7568 | 0.7374 | 0.7217 |
| 10 | 0.8784 | 0.8196 | 0.7879 | 0.7608 | 0.7412 | 0.7253 |
| 11 | 0.8825 | 0.8233 | 0.7921 | 0.7641 | 0.7446 | 0.7284 |
| 12 | **0.8859** | 0.8264 | 0.7957 | 0.7670 | 0.7475 | 0.7311 |

## Technical Contributions

1. **Walsh-Hadamard factorisation**: The k-body constraint fold is a convolution on Z₂^{2p+1}. The WHT diagonalises it, reducing cost from O(4^{kp}) to O(p²·4^p) for any k. For k=3, p=8 this is 65,000× faster.

2. **Manual adjoint differentiation**: Reverse-mode gradients through the full evaluation pipeline at ~1.6× a single forward evaluation, independent of p. The WHT is self-adjoint; β gradients use a log-derivative trick. 12× faster than forward-mode AD at p=8.

3. **Normalized branch tensor recurrence**: Threshold-based normalization prevents Float64 overflow at high (k,D,p) by tracking scale in log space. For k≥6 where Float64 precision is insufficient, Double64 arithmetic (~31 digits) is used via DoubleFloats.jl with ~3-5× overhead.

4. **Generic fold engine**: The Basso-Farhi branch-tensor contraction is a catamorphism over the light-cone tree, parametrised by a cost algebra. MaxCut and Max-k-XORSAT are different instantiations of the same interface — validated by reproducing Farhi et al. (2025) MaxCut results with no code changes.

5. **Plateau detection**: Per-iteration Optim.jl callback with a 30-value circular buffer. Stops the optimizer when the objective range plateaus below g_abstol, reducing p=12 wall time from 2+ hours to ~40 minutes.

6. **Swarm/memetic optimizer**: Population-based basin discovery for rugged landscapes at high (k,D). 100 random candidates, short L-BFGS bursts, cull/crossover, early exit when stagnant, full L-BFGS polish on winner. Finds basins that standard multi-start L-BFGS misses — (7,8) went from failing at p=3 to valid results at p=8+.

## Quick Start

```bash
# Install Julia 1.11+ via juliaup
curl -fsSL https://install.julialang.org | sh

# Clone and set up
git clone https://github.com/johnazariah/qaoa-xorsat.git
cd qaoa-xorsat
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run tests (1741 tests, 100% coverage)
julia --project=. -t auto -e 'using Pkg; Pkg.test()'

# Quick evaluation
julia --project=. -e '
using QaoaXorsat
result = basso_expectation(TreeParams(3, 4, 1), QAOAAngles([0.3], [0.5]))
println("k=3, D=4, p=1: $result")
'
```

## Usage

Two clean entry-point scripts are provided:

### MaxCut (k=2)

```bash
# D=3, sweep p=1..12 (validates against Farhi et al. 2025)
julia --project=. -t auto scripts/run_maxcut.jl 3 12

# D=4, sweep p=1..10
julia --project=. -t auto scripts/run_maxcut.jl 4 10
```

Output: `results/maxcut-k2-d<D>-sweep.csv`

### Max-k-XORSAT

```bash
# k=3, D=4, sweep p=1..12 (primary paper target)
julia --project=. -t auto scripts/run_xorsat.jl 3 4 12

# k=4, D=5, sweep p=1..10
julia --project=. -t auto scripts/run_xorsat.jl 4 5 10
```

Output: `results/xorsat-k<K>-d<D>-sweep.csv`

### Advanced: TOML Config + General Purpose Script

```bash
# Single (k,D) pair with CLI args
julia --project=. -t 12 scripts/optimize_qaoa.jl 3 4 1 12 2 320 1234 true adjoint

# TOML config with resume from previous run
julia --project=. -t 12 scripts/optimize_qaoa.jl experiments/single-pair.toml

# Full 15-pair table
julia --project=. -t 12 scripts/run_full_table.jl 12
```

All scripts support **resume**: if interrupted, re-running reads the existing CSV and continues from the last completed depth.

### Docker

```bash
docker build -t qaoa-xorsat .
docker run --rm qaoa-xorsat julia --project=. -t auto scripts/run_maxcut.jl 3 8
```

### Memory and Hardware Requirements

| p | Vector size | Adjoint cache | Minimum RAM | Suitable hardware |
|---|-----------|--------------|-------------|-------------------|
| ≤10 | ≤2M | ≤2 GB | 8 GB | Any laptop |
| 11 | 8M | 8 GB | 16 GB | Any desktop |
| 12 | 33M | 19 GB | 32 GB | Mac Studio, gaming PC |
| 13 | 134M | 84 GB | 128 GB | Cloud VM |
| 14 | 537M | 394 GB | 512 GB | Large cloud VM |

All results through p=12 across all 15 (k,D) pairs were computed on an Apple M4 Mac Studio (64 GB). See [docs/computational-limits.md](docs/computational-limits.md) for a detailed scaling analysis.

## Reproducing Paper Results

See [docs/reproducing-results.md](docs/reproducing-results.md) for step-by-step instructions.

**Table 1** — (k=3, D=4) through p=12:
```bash
julia --project=. -t auto scripts/run_xorsat.jl 3 4 12    # ~40 min
```

**Table 2** — All 15 (k,D) pairs:
```bash
julia --project=. -t auto scripts/run_full_table.jl 12     # ~7 hr
```

**Table 3** — MaxCut validation (D=3..8):
```bash
for D in 3 4 5 6 7 8; do
    julia --project=. -t auto scripts/run_maxcut.jl $D 12
done
```

## Documentation

- **[Ten innovations](docs/innovations.md)** — the full technical story: WHT factorisation, manual adjoint, normalisation, cost algebra, swarm optimizer, gradient checkpointing, and the correctness framework that makes the results trustworthy
- [Computational limits](docs/computational-limits.md) — what drives the cost, what the WHT fixed, what still blows up
- [Reproducing results](docs/reproducing-results.md) — step-by-step guide for all paper tables

## Architecture

```
src/
  QaoaXorsat.jl          # Module entry point
  tree.jl                # Tree structure (k, D, p parameters)
  tensors.jl             # QAOAAngles{T}, hyperindex operations
  basso_finite_d.jl      # Exact finite-D evaluator + WHT
  wht.jl                 # Walsh-Hadamard transform (cache-oblivious, in-place)
  adjoint.jl             # Manual reverse-mode differentiation
  checkpointed_adjoint.jl # √p-memory checkpointed adjoint for high p
  cost_algebra.jl        # MaxCut / XORSAT algebra dispatch
  optimization.jl        # L-BFGS + swarm optimizer, warm-start, plateau detection
  qaoa.jl                # Public API: evaluate_qaoa, optimize_angles
  reduced_basis.jl       # Spectral truncation for large (k,D)
  spectral_analysis.jl   # Effective rank analysis
  transfer_oracles.jl    # Raw transfer matrix oracles
  maxcut_transfer.jl     # MaxCut-specific transfer recursion
  gpu_*.jl               # GPU acceleration (Metal/CUDA, auto-detected)

test/                    # 21 test files, 1741 tests, 100% coverage

scripts/
  run_maxcut.jl          # Clean MaxCut entry point
  run_xorsat.jl          # Clean Max-k-XORSAT entry point
  optimize_qaoa.jl       # General-purpose optimizer (CLI args or TOML)
  run_full_table.jl      # All 15 (k,D) pairs

experiments/             # TOML configs for reproducible runs
results/                 # Curated CSV data (canonical results)
docs/                    # Public documentation
```

The evaluator is generic over the angle element type `T <: Real` via `QAOAAngles{T}`, enabling ForwardDiff dual numbers to propagate through the full pipeline.

## Testing

```bash
julia --project=. -t auto -e 'using Pkg; Pkg.test()'
```

21 test files covering:
- Tree structure and node counting
- Tensor primitives and hyperindex operations
- Brute-force oracle validation against known results
- WHT correctness (round-trip, convolution theorem)
- Manual adjoint vs finite-difference gradients (< 1e-8)
- Checkpointed adjoint matching full adjoint (< 1e-12)
- Optimizer convergence to known optima
- Cost algebra dispatch (MaxCut and XORSAT)
- Overflow protection and normalization
- GPU kernels (forward, backward, WHT, checkpointed)

## Key Design Decisions

- **Parametric types** (`QAOAAngles{T}`): one-line change enabled automatic differentiation through 500+ lines of evaluation code
- **Multiple dispatch**: the adjoint was added as a new method without modifying the existing evaluator
- **Threaded comprehensions**: 9.5× speedup on 10-core M4 at p=8 for table precomputation
- **Precomputed tables**: f_table and constraint kernel computed once, reused across p iteration steps

## References

1. Shutty, Azariah, Jordan (2026) — [arXiv:2604.24633](https://arxiv.org/abs/2604.24633) — **This paper**: QAOA for D-regular Max-k-XORSAT
2. Farhi, Goldstone, Gutmann (2014) — [arXiv:1411.4028](https://arxiv.org/abs/1411.4028) — Original QAOA
3. Basso, Farhi, Marwaha, Villalonga, Zhou (2021) — [arXiv:2110.14206](https://arxiv.org/abs/2110.14206) — Branch-tensor recurrence
4. Farhi, Gutmann, Ranard, Villalonga (2025) — [arXiv:2503.12789](https://arxiv.org/abs/2503.12789) — Exact MaxCut evaluator
5. Jordan et al. (2025) — [Nature 646:831-836](https://doi.org/10.1038/s41586-024-08033-4) — DQI comparison target

## Citation

If you use this software, please cite:

```bibtex
@article{shutty2026qaoa,
  author = {Shutty, Noah and Azariah, John S and Jordan, Stephen P},
  title = {Quantum Approximate Optimization Algorithm for D-Regular Max-k-XORSAT},
  year = {2026},
  eprint = {2604.24633},
  archivePrefix = {arXiv},
  primaryClass = {quant-ph}
}

@software{azariah2026qaoa_code,
  author = {Azariah, John S},
  title = {QAOA-XORSAT: Exact QAOA Performance on D-Regular Max-k-XORSAT},
  year = {2026},
  url = {https://github.com/johnazariah/qaoa-xorsat},
  doi = {10.5281/zenodo.19211958}
}
```

## License

MIT
