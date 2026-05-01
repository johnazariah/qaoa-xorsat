# Computational Limits

An honest analysis of what drives the cost, what the WHT breakthrough fixed, and what still blows up.

## What Drives the Cost

The fundamental cost of evaluating QAOA at depth `p` on `(k, D)` is:

$$\text{vector size} = 4^p$$

This is the hyperindex dimension — the branch tensor at each step lives in $\mathbb{R}^{4^p}$. Nothing can change this; it's the number of distinct variable configurations visible in the QAOA light cone.

| p | Vector elements | Memory (Float64) | Memory (Double64) |
|---|----------------|-------------------|---------------------|
| 8 | 65,536 | 512 KB | 1 MB |
| 10 | 1,048,576 | 8 MB | 16 MB |
| 12 | 16,777,216 | 128 MB | 256 MB |
| 13 | 67,108,864 | 512 MB | 1 GB |
| 14 | 268,435,456 | 2 GB | 4 GB |
| 15 | 1,073,741,824 | 8 GB | 16 GB |
| 16 | 4,294,967,296 | 32 GB | 64 GB |

## What the WHT Fixed

Before WHT: each constraint fold at depth `j` cost $O(4^{kp})$ — a sum over $2^{k-1}$ signed parity contributions. For k=3, p=8 this was $4^{24} = 2.8 \times 10^{14}$ operations.

After WHT: the constraint fold is a pointwise operation in the WHT-transformed domain. Cost per fold: $O(4^p)$. The WHT itself costs $O(p \cdot 4^p)$. Net: $O(p^2 \cdot 4^p)$ for the full tree, independent of k.

**The WHT doesn't reduce the $4^p$ vector size. It reduces the cost *per step* from $O(4^{kp})$ to $O(4^p)$.**

## What Still Blows Up

Three things scale exponentially and cannot be optimised away:

### 1. Forward Pass Memory

The branch tensor is a vector of $4^p$ elements. At p=16, this is 32 GB (Float64) or 64 GB (Double64). There is **one** of these per step, and we have $b^j$ branches at level $j$ — but we process them as element-wise powers, so only one vector is live at a time. Memory = $O(4^p)$.

### 2. Adjoint (Gradient) Cache

The standard reverse-mode adjoint caches all $p+1$ intermediate branch tensors for the backward pass. Memory = $O(p \cdot 4^p)$. At p=12 this is $(13)(128 \text{ MB}) \approx 1.7$ GB. At p=16 it would be $(17)(32 \text{ GB}) = 544$ GB.

**Mitigation**: Checkpointed adjoint with $\sqrt{p}$ checkpoints. Stores only $\sqrt{p}$ tensors, recomputes segments on the backward pass. Memory: $O(\sqrt{p} \cdot 4^p)$. At p=16: $4 \times 32\text{ GB} = 128$ GB instead of 544 GB. Cost: ~2× the forward pass instead of ~1.6×.

### 3. Optimisation Iterations

At p=12, the optimizer makes ~320 evaluations. Each evaluation includes a forward pass + gradient. At p=16, each forward+backward on Double64 takes ~hours. Total optimisation: days to weeks. **The swarm optimizer multiplies this by the population size** (typically 100 at low-p, but we can only afford single-start L-BFGS at high-p).

## The Tree Fan-Out Doesn't Help (But Doesn't Hurt Either)

The branching factor is $b = (D-1)(k-1)$:

| (k,D) | b | Branches at p=12 | Note |
|--------|---|-------------------|------|
| (2,3) | 2 | 4,096 | MaxCut — manageable |
| (3,4) | 6 | 2.2 billion | Primary target |
| (4,5) | 12 | 8.9 trillion | Large k |
| (7,8) | 42 | $4.2 \times 10^{19}$ | Massive |

But this doesn't affect memory! The key insight: all branches at the same depth are identical (by regularity), so we process them as element-wise exponentiation of a single vector. The cost is $O(p \cdot 4^p)$, not $O(b^p \cdot 4^p)$.

The tree fan-out *does* affect numerical precision — the exponentiation accumulates floating-point error, which is why Double64 is needed for (k,D) with large branching factors.

## What Hardware Can Do What

| Target | RAM needed | Time estimate | Suitable hardware |
|--------|-----------|---------------|-------------------|
| Any (k,D) at p≤10 | 8 GB | < 1 hr | Any laptop |
| Any (k,D) at p=12 | 32 GB | 40 min–7 hr | Mac Studio, gaming PC |
| (k=3, D=4) at p=13 | 128 GB | ~84 hr | Cloud VM (E8as_v5) |
| Any (k,D) at p=14 | 512 GB | days | Large cloud VM |
| Any (k,D) at p=15 | 16 GB (GPU) | hours | A100/H100 GPU |
| (k=3, D=4) at p=16 | 64 GB (GPU) | days | H100 GPU (80 GB) |

## The Mac Story

All results up to p=12 across all 15 (k,D) pairs were computed on an Apple M4 Mac Studio with 64 GB unified memory. This is notable because:

- The unified memory architecture means no CPU↔GPU copy overhead
- The 10 P-cores provide genuine parallel throughput for multi-start optimisation
- At p=12, the 19 GB adjoint cache fits comfortably in 64 GB
- Julia's GC + stack allocation means the actual memory footprint is well below the theoretical maximum

Beyond p=12, we moved to Google Cloud (1.4 TB RAM nodes) and are exploring H100 GPU acceleration. The Mac *could* do p=13 with the checkpointed adjoint (128 GB of cache reduced to ~40 GB with √13 ≈ 4 checkpoints), but optimisation convergence at 4-minute-per-evaluation pace makes it impractical.
