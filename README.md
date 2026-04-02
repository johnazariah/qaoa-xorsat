# QAOA-XORSAT

**Exact QAOA performance on D-regular Max-k-XORSAT via generic tree folding**

*John S Azariah — Centre for Quantum Software and Information, UTS*\
*ORCID: [0009-0007-9870-1970](https://orcid.org/0009-0007-9870-1970)*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19211958.svg)](https://doi.org/10.5281/zenodo.19211958)
![Tests](https://img.shields.io/badge/tests-1741%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Julia](https://img.shields.io/badge/Julia-1.12+-purple)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## Results

Best-found QAOA satisfaction fractions at finite D, computed on Apple M4 Mac Studio (64 GB) and Azure fleet (5× E8as_v5, 256 GB). Full data in [`results/qaoa-best-values.csv`](results/qaoa-best-values.csv).

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

### All 15 (k,D) pairs

| (k,D) | p | c̃ | (k,D) | p | c̃ | (k,D) | p | c̃ |
|-------|---|------|-------|---|------|-------|---|------|
| (3,4) | 12 | 0.877 | (4,5) | 11 | 0.861 | (5,6) | 10 | 0.849 |
| (3,5) | 11 | 0.835 | (4,6) | 10 | 0.827 | (5,7) | 9 | 0.813 |
| (3,6) | 11 | 0.807 | (4,7) | 10 | 0.806 | (5,8) | 9 | 0.801 |
| (3,7) | 11 | 0.779 | (4,8) | 10 | 0.800 | (6,7) | 8 | 0.819 |
| (3,8) | 11 | 0.768 | | | | (6,8) | 8 | 0.799 |
| | | | | | | (7,8) | 8 | 0.823 |

QAOA surpasses DQI+BP for 11 of 15 pairs. At four pairs — (3,6), (3,7), (3,8), (4,8) — QAOA also exceeds Regev+FGUM. To our knowledge, no prior exact finite-D QAOA evaluation has been performed for k ≥ 3.

## Technical Contributions

1. **Walsh-Hadamard factorisation**: The k-body constraint fold is a convolution on Z₂^{2p+1}. The WHT diagonalises it, reducing cost from O(4^{kp}) to O(p²·4^p) for any k. For k=3, p=8 this is 65,000× faster.

2. **Manual adjoint differentiation**: Reverse-mode gradients through the full evaluation pipeline at ~1.6× a single forward evaluation, independent of p. The WHT is self-adjoint; β gradients use a log-derivative trick. 12× faster than forward-mode AD at p=8.

3. **Normalized branch tensor recurrence**: Per-step normalization prevents Float64 overflow at high (k,D,p) by tracking scale in log space. Enables exact evaluation at all 15 pairs through p=15 on high-memory nodes.

3. **Generic fold engine**: The Basso-Farhi branch-tensor contraction is a catamorphism over the light-cone tree, parametrised by a cost algebra. MaxCut and Max-k-XORSAT are different instantiations of the same interface — validated by reproducing Farhi et al. (2025) MaxCut results with no code changes.

## Quick Start

```bash
# Install Julia 1.12+ via juliaup
curl -fsSL https://install.julialang.org | sh

# Clone and set up
git clone https://github.com/johnazariah/qaoa-xorsat.git
cd qaoa-xorsat
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run tests (1741 tests, 100% coverage)
julia --project=. -t auto -e 'using Pkg; Pkg.test()'

# Smoke test
julia --project=. -e '
using QaoaXorsat
result = basso_expectation(TreeParams(3, 4, 1), QAOAAngles([0.3], [0.5]))
println("k=3, D=4, p=1: $result")
'
```

## Running Experiments

```bash
# Single (k,D) pair with CLI args
julia --project=. -t 12 scripts/optimize_qaoa.jl 3 4 1 12 2 320 1234 true adjoint

# TOML config with resume from previous run
julia --project=. -t 12 scripts/optimize_qaoa.jl experiments/single-pair.toml

# Full 15-pair table at p=11
julia --project=. -t 12 scripts/run_full_table.jl 11

# Deploy script (handles logging, caffeinate, thread detection)
./scripts/deploy.sh experiments/full-table.toml 16
```

### Docker

```bash
docker build -t qaoa-xorsat .
docker run --rm -v $(pwd)/results:/workspace/results qaoa-xorsat \
  scripts/optimize_qaoa.jl experiments/full-table.toml
```

### Memory Requirements

| p | Vector size | Adjoint cache | Minimum RAM |
|---|-----------|--------------|-------------|
| ≤10 | ≤2M | ≤2 GB | 8 GB |
| 11 | 8M | 8 GB | 16 GB |
| 12 | 33M | 19 GB | 32 GB |
| 13 | 134M | 84 GB | 128 GB |
| 14 | 537M | 394 GB | 512 GB |

## Documentation

See [`docs/learning/`](docs/learning/) for background material and design notes:
- [Problem statement](docs/learning/problem-statement.md) — what we're solving and why
- [WHT factorisation](docs/learning/wht-factorisation.md) — the core algorithmic insight
- [Performance optimization](docs/learning/performance-optimization.md) — 100× in 11 stages
- [Differentiation strategies](docs/learning/differentiation-strategies.md) — why manual adjoint

## Architecture

```
src/
  QaoaXorsat.jl          # Module entry point
  tree.jl                # Tree structure (k, D, p parameters)
  tensors.jl             # QAOAAngles{T}, hyperindex operations
  basso_finite_d.jl      # Tier 2: exact finite-D evaluator + WHT
  wht.jl                 # Walsh-Hadamard transform (generic, in-place)
  adjoint.jl             # Manual reverse-mode differentiation
  optimization.jl        # L-BFGS optimizer with thread-parallel restarts
  qaoa.jl                # Public API: parity_expectation, qaoa_expectation
  transfer_oracles.jl    # Raw transfer matrix oracles
  maxcut_transfer.jl     # MaxCut-specific transfer recursion
```

The evaluator is generic over the angle element type `T <: Real` via `QAOAAngles{T}`, enabling ForwardDiff dual numbers to propagate through the full pipeline.

## Key Design Decisions

- **Parametric types** (`QAOAAngles{T}`): one-line change enabled automatic differentiation through 500+ lines of evaluation code
- **Multiple dispatch**: the adjoint was added as a new method without modifying the existing evaluator
- **Threaded comprehensions**: 9.5× speedup on 10-core M4 at p=8 for table precomputation
- **Precomputed tables**: f_table and constraint kernel computed once, reused across p iteration steps

## References

1. Farhi, Goldstone, Gutmann (2014) — [arXiv:1411.4028](https://arxiv.org/abs/1411.4028) — Original QAOA
2. Basso, Farhi, Marwaha, Villalonga, Zhou (2021) — [arXiv:2110.14206](https://arxiv.org/abs/2110.14206) — Branch-tensor recurrence
3. Farhi, Gutmann, Ranard, Villalonga (2025) — [arXiv:2503.12789](https://arxiv.org/abs/2503.12789) — Exact MaxCut evaluator
4. Jordan et al. (2025) — [Nature 646:831-836](https://doi.org/10.1038/s41586-024-08033-4) — DQI comparison target

## Citation

```bibtex
@software{azariah2026qaoa,
  author = {Azariah, John S},
  title = {QAOA-XORSAT: Exact QAOA Performance on D-Regular Max-k-XORSAT},
  year = {2026},
  url = {https://github.com/johnazariah/qaoa-xorsat},
  doi = {10.5281/zenodo.19211958}
}
```

## License

MIT
