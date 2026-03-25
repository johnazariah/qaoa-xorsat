# QAOA-XORSAT

**Exact QAOA performance on D-regular Max-k-XORSAT via generic tree folding**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19211958.svg)](https://doi.org/10.5281/zenodo.19211958)
![Tests](https://img.shields.io/badge/tests-714%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Julia](https://img.shields.io/badge/Julia-1.12+-purple)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## Results

Exact QAOA satisfaction fractions for Max-3-XORSAT on 4-regular hypergraphs, computed on a single Apple M4 Max workstation (64 GB RAM, 10 threads):

| p | c̃(p) | Δc̃ | Wall time | Gap to DQI+BP |
|---|-------|------|-----------|---------------|
| 1 | 0.6761 | — | 1.7 s | 0.195 |
| 2 | 0.7391 | +0.0630 | 5 ms | 0.132 |
| 3 | 0.7771 | +0.0380 | 25 ms | 0.094 |
| 4 | 0.8022 | +0.0251 | 100 ms | 0.069 |
| 5 | 0.8205 | +0.0183 | 260 ms | 0.050 |
| 6 | 0.8344 | +0.0139 | 1.3 s | 0.037 |
| 7 | 0.8453 | +0.0109 | 10 s | 0.026 |
| 8 | 0.8541 | +0.0088 | 88 s | 0.017 |
| 9 | 0.8613 | +0.0072 | 6.5 min | 0.010 |
| 10 | 0.8674 | +0.0060 | 87 min | **0.003** |

Previous published state of the art for exact finite-D evaluation at k ≥ 3 was p ≤ 5.

## Technical Contributions

1. **Walsh-Hadamard factorisation**: The k-body constraint fold is a convolution on Z₂^{2p+1}. The WHT diagonalises it, reducing cost from O(4^{kp}) to O(p²·4^p) for any k. For k=3, p=8 this is 65,000× faster.

2. **Manual adjoint differentiation**: Reverse-mode gradients through the full evaluation pipeline at ~1.6× a single forward evaluation, independent of p. The WHT is self-adjoint; β gradients use a log-derivative trick. 12× faster than forward-mode AD at p=8.

3. **Generic fold engine**: The Basso-Farhi branch-tensor contraction is a catamorphism over the light-cone tree, parametrised by a cost algebra. MaxCut and Max-k-XORSAT are different instantiations of the same interface — validated by reproducing Farhi et al. (2025) MaxCut results with no code changes.

## Quick Start

```bash
# Install Julia 1.12+ via juliaup
curl -fsSL https://install.julialang.org | sh

# Clone and set up
git clone https://github.com/johnazariah/qaoa-xorsat.git
cd qaoa-xorsat
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run tests (714 tests, 100% coverage)
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
# XORSAT (k=3, D=4) p=1-10 with adjoint gradients, 10 threads
julia --project=. -t 10 scripts/optimize_qaoa.jl 3 4 1 10 2 320 1234 true adjoint

# MaxCut validation (k=2, D=3)
julia --project=. -t 10 scripts/optimize_qaoa.jl 2 3 1 5 8 200 1234 true adjoint

# Gradient method toggle: adjoint (default), forward (ForwardDiff), finite (FD)
julia --project=. -t 10 scripts/optimize_qaoa.jl 3 4 1 8 2 320 1234 true forward
```

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
