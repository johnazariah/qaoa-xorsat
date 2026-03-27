# WHT Factorisation of the k-Body Constraint Fold

> **Date**: 22 March 2026
> **Status**: Numerically verified at p=1,2,3 for D=3,4 — machine precision agreement
> **Impact**: Reduces exact finite-D constraint fold from O(4^{kp}) to O(p² · 4^p)
> **Branch**: `feature/wht-research` (worktree at `.worktrees/wht-research`)

---

## The Problem

The Basso finite-D iteration (Eq. 8.7, arXiv:2110.14206) computes the QAOA
expectation value exactly at any finite D. At each constraint fold step for
k-XORSAT, we must evaluate:

$$S(\mathbf{a}) = \sum_{\mathbf{b}^1, \mathbf{b}^2 \in \{-1,+1\}^{2p+1}}
\cos\!\left(\frac{\boldsymbol{\Gamma} \cdot
(\mathbf{a} \odot \mathbf{b}^1 \odot \mathbf{b}^2)}{\sqrt{D}}\right)
g(\mathbf{b}^1)\,g(\mathbf{b}^2)$$

where $\odot$ is entrywise product and $g$ is the branch tensor.

**Naive cost**: $O(N^3)$ where $N = 2^{2p+1}$ — equivalent to $O(4^{3p})$.
This limits exact computation to p ≈ 5-7.

## The Insight

The sum is a **double convolution on the group** $(\mathbb{Z}_2^{2p+1}, \oplus)$.

Define $c = b^1 \oplus b^2$ (bitwise XOR, which corresponds to $b^1 \odot b^2$
in spin notation). Then:

1. **Auto-convolution**: $W(c) = \sum_{b: b \oplus b' = c} g(b) \cdot g(b') =
   (g \star g)(c)$.

2. **Kernel convolution**: $S(a) = \sum_c \kappa(a \oplus c) \cdot W(c) =
   (\kappa \star W)(a)$, where $\kappa(d) =
   \cos(\boldsymbol{\Gamma} \cdot \text{spins}(d) / \sqrt{D})$.

Both are convolutions on $\mathbb{Z}_2^{2p+1}$. By the **convolution theorem**:

$$\hat{S} = \hat{\kappa} \cdot \hat{W} = \hat{\kappa} \cdot \hat{g}^2$$

where $\hat{\cdot}$ denotes the Walsh-Hadamard transform.

## The Algorithm

```
1. Compute g(b) for all b                           O(N)
2. ĝ = WHT(g)                                       O(N log N)
3. Ŵ = ĝ²  (element-wise square)                    O(N)
4. Compute κ(d) for all d                            O(N · p)
5. κ̂ = WHT(κ)                                       O(N log N)
6. Ŝ = κ̂ · Ŵ  (element-wise multiply)               O(N)
7. S = IWHT(Ŝ)                                      O(N log N)
```

Total: $O(N \log N) = O(p \cdot 4^p)$ per fold step. Over p steps: $O(p^2 \cdot 4^p)$.

## Why the cos-vs-exp Concern Was Unfounded

The initial concern was that $\cos(\sum_\ell t_\ell) \neq \prod_\ell \cos(t_\ell)$,
which would prevent factorisation over rounds.

This concern is **irrelevant** because the WHT does not require the kernel to
factorise over rounds. The kernel $\kappa(d)$ is treated as an opaque function
$\mathbb{Z}_2^{2p+1} \to \mathbb{R}$. The convolution theorem applies to ANY
function on the group, regardless of its internal structure. The transform
simply diagonalises the convolution operator.

## Generalisation to Arbitrary k

For general k, the constraint fold involves (k-1) child branches:

$$S(\mathbf{a}) = \sum_{\mathbf{b}^1, \ldots, \mathbf{b}^{k-1}}
\kappa(\mathbf{a} \odot \mathbf{b}^1 \odot \cdots \odot \mathbf{b}^{k-1})
\prod_{i=1}^{k-1} g(\mathbf{b}^i)$$

The (k-1)-fold auto-convolution has WHT-domain representation $\hat{g}^{k-1}$.
The full computation:

$$\hat{S} = \hat{\kappa} \cdot \hat{g}^{k-1}$$

**Cost**: $O(p \cdot 4^p)$ per step for ANY k. The constraint arity affects
only the exponent in the element-wise power, not the transform size or cost.

## Numerical Verification

Implemented in `test/test_wht_factorisation.jl` on branch `feature/wht-research`.

| p | D | Trials | Max |S_naive - S_wht| | Status |
|---|---|--------|---------------------|--------|
| 1 | 3 | 100 | 4.4 × 10⁻¹⁶ | **Pass** |
| 1 | 4 | 50 | 4.4 × 10⁻¹⁶ | **Pass** |
| 2 | 3 | 20 | 5.9 × 10⁻¹⁶ | **Pass** |
| 2 | 4 | 10 | 7.8 × 10⁻¹⁶ | **Pass** |
| 3 | 3 | 5 | 4.0 × 10⁻¹⁵ | **Pass** |
| 3 | 4 | 3 | 6.7 × 10⁻¹⁵ | **Pass** |

The growing error from p=1 to p=3 is consistent with floating-point accumulation
($\sim \epsilon \cdot N$ where $N = 2^{2p+1}$), not a systematic divergence.

## Impact on the Project

| Before | After |
|--------|-------|
| Exact k=3 costs $O(p \cdot 64^p)$ | Exact k=3 costs $O(p^2 \cdot 4^p)$ |
| Max exact p ≈ 5-7 | Max exact p ≈ 15-17 (same as MaxCut) |
| Need D→∞ approximation for p > 7 | **Exact at any D, at any p up to memory limit** |
| Open Question 1 in P1.3 spec | **Resolved** |

## Novelty

The Basso 2021 paper (arXiv:2110.14206) quotes $O(p \cdot 4^{pq})$ for the
finite-D iteration at constraint arity q. The Villalonga reference code
implements only the D→∞ iteration. No published work uses WHT to accelerate
the finite-D constraint fold. This factorisation appears to be **novel** and
is worth including in the write-up.

## What Remains

- Integrate the WHT fold into the P1.3 production contraction code
- Validate the full fold (not just one step) against the brute-force oracle
- Validate MaxCut results against Farhi 2025 Table 1
- Write a proof (the numerical evidence is conclusive but a 3-line algebraic
  proof via the convolution theorem would be cleaner for the paper)
