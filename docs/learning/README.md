# Learning Documents

Background material and design notes for the QAOA-XORSAT computation engine.

## Background

| Document | Description |
|----------|-------------|
| [foundations.md](foundations.md) | Prerequisites: qubits, QAOA, graphs, approximation ratios |
| [problem-statement.md](problem-statement.md) | What we're solving, why, and the DQI comparison landscape |
| [basso-iteration.md](basso-iteration.md) | The Basso et al. branch-tensor recurrence (what we build on) |
| [farhi-tensor-method.md](farhi-tensor-method.md) | The Farhi et al. exact tensor contraction (what we generalise) |

## Key Insights

| Document | Description |
|----------|-------------|
| [tensor-derivation.md](tensor-derivation.md) | Hyperindex convention and contraction ordering |
| [wht-factorisation.md](wht-factorisation.md) | The WHT diagonalisation of the k-body constraint fold — the core algorithmic contribution |
| [differentiation-strategies.md](differentiation-strategies.md) | Finite differences vs ForwardDiff vs manual adjoint |
| [performance-optimization.md](performance-optimization.md) | The 100× optimization journey: 11 stages from baseline to production |
