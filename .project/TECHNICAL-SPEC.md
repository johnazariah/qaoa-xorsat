---

# QAOA-XORSAT — Comprehensive Technical Specification

**Version**: 2026-04-24  
**Author**: John S Azariah (UTS, Centre for Quantum Software and Information)  
**ORCID**: https://orcid.org/0009-0007-9870-1970  
**Repository**: github.com/johnazariah/qaoa-xorsat  
**Language**: Julia 1.12+, MIT License  

---

## A. Current Architecture

### A.1 Pipeline Overview: (k, D, p, angles) → c̃ and ∇c̃

The QAOA-XORSAT evaluator computes the exact expected satisfaction fraction and its gradient for a depth-$p$ QAOA circuit applied to a D-regular Max-k-XORSAT instance, using a contracted light-cone tree representation.

**Mathematical model:**

Given a problem instance (k-uniform hypergraph, D-regular), the QAOA state after $p$ rounds is:

$$|\psi_p\rangle = \prod_{r=1}^{p} e^{-i\beta_r H_m} e^{-i\gamma_r H_c} |+\rangle^{\otimes n}$$

where $H_c$ is the problem Hamiltonian (cost), $H_m$ is the mixer (sum of single-qubit X rotations), and angles $\gamma = (\gamma_1, \ldots, \gamma_p)$ and $\beta = (\beta_1, \ldots, \beta_p)$ parametrize the circuit.

The expected satisfaction fraction is:

$$\tilde{c}(\gamma, \beta) = \frac{1}{2}\left(1 + c_s \cdot \mathbb{E}[Z_{i_1} \cdots Z_{i_k}]\right)$$

where $c_s \in \{-1, +1\}$ is the clause sign (e.g., $c_s = -1$ for MaxCut, $c_s = +1$ for Max-k-XORSAT) and the expectation is taken over the root constraint's $k$ variables.

**Evaluation pipeline:**

1. **Forward pass** (cached): Compute the normalized branch tensor recursion from depth $t=1$ to $t=p$, storing all intermediates.
2. **Root fold**: Combine the final branch tensor with the root observable kernel to compute the parity correlator $S_{\text{norm}}$ in normalized space.
3. **Scale recovery**: Multiply by the accumulated scale factor $\exp(L)$ (stored in log space) to obtain $\tilde{c}$.
4. **Backward pass** (reverse-mode AD): Propagate cotangents through the cached forward pass to compute $\nabla_{\gamma} \tilde{c}$ and $\nabla_{\beta} \tilde{c}$.

Cost: ~2× a single forward evaluation, independent of $p$.

---

### A.2 The 8 Innovations (in chronological order)

#### **1. Walsh-Hadamard Factorization** (WHT acceleration)

**Problem**: The naive branch tensor recurrence computes a k-body convolution at each step:

$$\text{folded}(a) = \sum_{b_1, \ldots, b_{k-1}} K(a, b_1, \ldots, b_{k-1}) \cdot \text{child}(b_1) \cdots \text{child}(b_{k-1})$$

over $2^{2p+1}$ configurations, each with $2^{k-1}$ children. Cost: $O(4^{kp})$ — prohibitive for large $k$ or $p$.

**Solution**: The constraint kernel $K$ is a separable function of the phase dot product:

$$K(a) = \cos(\Gamma \cdot \text{spins}(a) / 2)$$

The k-body product can be decomposed as a convolution on the abelian group $(ℤ_2^{2p+1}, ⊕)$. The WHT diagonalizes all convolutions simultaneously, reducing cost to:

$$\text{Cost} = O(p) \times O(2^{2p+1} \log 2^{2p+1}) = O(p^2 \cdot 4^p)$$

**Reduction factor**: For k=3, p=8: $4^{3 \times 8} / (p^2 \cdot 4^p) = 4^{16} / (64 \times 4^8) = 65,000×$ faster.

**Implementation**: 
- `wht.jl`: Recursive cache-oblivious FFT (2048 ComplexF64 = 32KB cutoff for L1 cache)
- Base case: Iterative SIMD butterfly for sub-problems ≤ 32 KB
- Top-level split followed by independent recursive calls on each half
- Self-adjoint (symmetric) transform: `iwht(x) = wht(x) / N`

---

#### **2. Manual Adjoint Differentiation** (reverse-mode AD)

**Problem**: ForwardDiff.jl scales gradients as $O(2p)$ times a forward evaluation (each angle gets a dual number). At $p=12$, this is 24× slower than the forward pass.

**Solution**: Implement reverse-mode AD (backpropagation) explicitly:

- Store all normalized intermediates $(B_t, \text{child\_hat}_t, \text{folded}_t)$ during forward pass.
- Backward pass propagates cotangents through the cached computation graph.
- WHT is self-adjoint: $\nabla_{x} \text{WHT}(x) = \text{WHT}(\nabla_{\text{WHT}(x)})$
- $\beta$ gradients use the log-derivative trick: $\nabla_\beta (e^{i\beta x y}) = e^{i\beta xy}(-\cot(\beta) \text{ or } \tan(\beta))$ depending on bit transition parity.

**Cost**: ~1.6× a single forward evaluation, independent of $p$ (cache dominates at high $p$).

**Speedup**: 12-15× faster than ForwardDiff at $p=8$, scaling advantage increases with $p$.

---

#### **3. Normalized Branch Tensor (Overflow protection)**

**Problem**: At high $(k, D, p)$, the branch tensor magnitudes grow as $(D-1)^{tp} \cdot \text{const}$ per step. For $(k=7, D=8, p=9)$: max magnitude $\sim 10^{200}$, exceeding Float64 max ($\sim 1.8 \times 10^{308}$) before the root fold.

**Solution**: Threshold-based normalization in log space:

- Before each power operation ($\text{child\_hat}^{k-1}$ or $\text{folded}^{D-1}$), check if $\max|\text{vector}| > 10^{30}$.
- If exceeded, normalize: $\text{vector} \leftarrow \text{vector} / \max|\text{vector}|$ and accumulate scale: $\log s \leftarrow \log s + (k-1) \log(\max|\text{vector}|)$.
- **Detach scale from gradient**: Treat $\max|\cdot|$ as a constant (0 gradient). Mathematically valid because the argmax operation is sparse and contributes negligibly compared to the $O(N)$ gradient terms.

**Overhead**: Negligible — two max-magnitude passes per step, $O(N)$ each, dominated by WHT $O(N \log N)$ and power operations $O(N)$.

**Effect**: Prevents overflow while preserving relative signal structure. Valid results up to the numerical precision of the representation (Float64 ≈ 15 digits, Double64 ≈ 31 digits).

---

#### **4. Plateau Detection** (adaptive stopping)

**Problem**: L-BFGS optimizer can plateau (gradient norm oscillates without monotonic descent) for 100+ iterations at high $p$. A single evaluation at $p=12$ costs 40 seconds; 100 wasted iterations = 67 minutes lost per restart.

**Solution**: Per-iteration Optim.jl callback that:

1. Maintains a rolling window of 30 objective values (circular buffer).
2. Every 300 seconds of wall time, computes $\text{range}(\text{values}) = \max - \min$ over the window.
3. If $\text{range} < \text{g\_abstol}$, stops the optimizer immediately.
4. Otherwise, continues with next 100-iteration chunk.

**Empirical gain**: 
- Without plateau detection: p=12 takes ~2+ hours (1200 evaluations)
- With plateau detection: p=12 takes ~40 minutes (60 evaluations)
- Speedup: ~3× wall time reduction

**Tuning**: Chunk size adapts to depth: 100 (p≤8) → 50 (p≤10) → 20 (p≤12) → 10 (p>12).

---

#### **5. Threshold Normalization & Overflow Guards** (numerical stability)

**Problem**: At extreme $(k, D, p)$, even normalized recurrence can overflow (e.g., $L > 700$ in $\exp(L)$ where $L$ is the accumulated log-scale).

**Solution**: Three-tier defense:

1. **Threshold normalization** (described above): keeps intermediate magnitudes ≤ 1.
2. **Log-space multiplication**: compute $\exp(L + \log|\text{Re}(S)|)$ instead of $\exp(L) \cdot \text{Re}(S)$ to delay overflow.
3. **Overflow guard in optimizer**: if $\tilde{c} \not\in [0, 1]$ or gradient is non-finite, return large objective (1e6) with a gradient pointing toward the origin (not zero), forcing L-BFGS to backtrack rather than stall.

**Validation**: `is_valid_qaoa_value(v)` checks $-10^{-9} \leq v \leq 1 + 10^{-9}$ (allows tiny rounding).

---

#### **6. Depth-Dependent Tolerance** (adaptive g_abstol)

**Problem**: At high $p$, gradient noise floor increases due to smaller effective dimension and finite precision. A fixed g_abstol = 1e-6 may be unreachable at $p \geq 12$, causing L-BFGS to hit maxiters without formal convergence.

**Solution**: Adaptive `depth_g_abstol(p)`:

| Depth | $g_{\text{abstol}}$ | Rationale |
|-------|------------------|-----------|
| $p \leq 10$ | 1e-6 | Converges reliably in 5-45 iterations |
| $p = 11$ | 1e-5 | Gradient noise floor ~1e-8 in $\tilde{c}$; tighter tolerance marginal |
| $p \geq 12$ | 1e-4 | $g$-norm oscillates at ~1e-4 scale; 1e-5 wasted iterations |

**Effect**: Converges faster without sacrificing solution quality (all tolerances within physical noise floor).

---

#### **7. Swarm/Memetic Optimizer** (basin hopping)

**Problem**: At high $(k, D)$, the QAOA landscape is highly multimodal. Standard multi-start L-BFGS with random restarts fails:

- (7,8) at $p=3$: 10 restarts all fail (converge to $\tilde{c} \approx 0.5$).
- (7,8) at $p=8$: isolated basin, random starts have vanishingly small probability.

**Solution**: Population-based basin discovery (modeled after differential evolution / swarm optimization):

1. **Initialization**: Generate 100 random candidate angles in the canonical box $[0, 2\pi) \times [0, \pi)^p$.
2. **Evaluation**: Evaluate all 100 at low cost (quick, rough objective estimate).
3. **Short L-BFGS bursts**: Sort by objective; run 10 iterations of L-BFGS on top 50% (50 candidates).
4. **Cull & crossover**: 
   - Drop bottom 50%, keep top 25 improved candidates.
   - Generate 25 new crossover candidates from top 5.
   - Generate 50 new random candidates.
5. **Repeat** until 10+ generations stagnate (no improvement in best objective).
6. **Full L-BFGS polish**: Run winner with full maxiters=1280 and tight tolerance.

**Empirical results**:
- (6,7) at $p=9$: standard fails; swarm finds $\tilde{c} = 0.855$ (vs DQI+BP = 0.825, **3% better**).
- (7,8) at $p=8$: standard fails; swarm finds $\tilde{c} = 0.819$ (beats competitors).

**Cost**: ~5-10 min per pair for 10 generations, but finds valid basin reliably.

---

#### **8. Double64 Precision Support** (DoubleFloats.jl)

**Problem**: Float64 precision wall (15 decimal digits) limits reliable evaluation to $p \leq 13$ for (3,4) and $p \leq 9$ for $(k,D)$ with $k \geq 6$.

**Solution**: Use DoubleFloats.jl (Double64 type: ~31 decimal digits) for high-precision arithmetic:

- Optim.jl's L-BFGS parameter vector stays Float64 (standard).
- Evaluator internally promotes: `_promote_angles(angles, Double64)` at call site.
- Manual adjoint + normalized recurrence + WHT all work with `::Real` parameter — generic over Float64 and Double64.
- Results converted back to Float64 for Optim optimization loop.

**Overhead**: ~3-5× wall time per evaluation vs Float64, negligible memory (Double64 = 2× Float64 storage).

**Deployment**: 
- Script: `scripts/swarm_chain_d64.jl` — pure Double64 swarm from start.
- SLURM: `scripts/qaoa_d64_sweep.sh` — 15 parallel array tasks, one per (k,D) pair.

---

### A.3 Data Flow Through the Pipeline

```
Input: TreeParams(k, D, p), QAOAAngles(γ, β), clause_sign
  ↓
[Forward Pass]
  • build_gamma_vector(angles) → Γ[1..2p]
  • build_gamma_full_vector(angles) → Γ_full[1..2p+1]
  • basso_trig_table(angles) → cos/isin lookup table
  • _basso_f_table_fast(...) → f_table[0..N-1] (mixer weights, threaded)
  • Loop t=1..p:
      - child_hat[t] = WHT(f_table .* B[t]) then optionally normalize
      - folded[t] = iWHT(kernel_hat .* child_hat[t]^(k-1)) then optionally normalize
      - B[t+1] = folded[t]^(D-1)
      - Accumulate log-scale: log_s[t+1]
  • Root fold:
      - msg_hat = WHT(root_msg) (normalized)
      - msg_hat_power = msg_hat^k
      - conv = iWHT(msg_hat_power)
      - S_normalized = Σ root_kernel .* conv
  • Compute value: log_total_scale, then c̃ ∈ [0, 1]
  • Cache entire forward pass in BassoPipelineCache{T}
  ↓
[Backward Pass]
  • Compute ∂E/∂S_normalized (root cotangent)
  • Backward loop t=p..1:
      - ∂folded/∂(· ) = (iWHT-adjoint) ...
      - ∂child_hat/∂(·) = (WHT-adjoint) ...
  • Convert ∂kernel to ∂γ_full via phase dot gradient
  • Map ∂γ_full to ∂γ via mirror symmetry
  • Compute ∂β from f_table cotangents (log-derivative trick)
  • Apply exp(log_total_scale) scaling to gradients
  ↓
Output: (c̃, ∇γ c̃, ∇β c̃)
```

Optimizer loop:
```
for restart in 1..num_restarts:
  angles ← random or warm-start
  for iteration in 1..maxiters:
    (c̃, ∇c̃) ← basso_expectation_and_gradient(params, angles)
    angles ← L-BFGS step (maximize c̃)
    if plateau_detected():
      break
  record(angles, c̃)
return best(angles, c̃) over all restarts
```

---

### A.4 Memory Layout and Scaling

All vectors are stored in **row-major layout** with configuration index $a \in [0, 2^{2p+1})$ as the primary index.

| Entity | Type | Length | Bytes (Float64) | Bytes (Double64) |
|--------|------|--------|-----------------|-----------------|
| Single vector | $\mathbb{C}^N$ | $2^{2p+1}$ | $16 \cdot 2^{2p+1}$ | $32 \cdot 2^{2p+1}$ |
| f_table | $\mathbb{C}^N$ | $2^{2p+1}$ | $16 N$ | $32 N$ |
| kernel | $\mathbb{C}^N$ | $2^{2p+1}$ | $16 N$ | $32 N$ |
| kernel_hat (WHT of kernel) | $\mathbb{C}^N$ | $2^{2p+1}$ | $16 N$ | $32 N$ |
| B history | $(p+1) \times \mathbb{C}^N$ | $(p+1) N$ | $16(p+1)N$ | $32(p+1)N$ |
| child_hat, folded history | $2p \times \mathbb{C}^N$ | $2pN$ | $32pN$ | $64pN$ |
| Scratch vectors (reused) | $\mathbb{C}^N$ | $N$ | $16N$ | $32N$ |
| **Adjoint cache total** | | | $(p+1) \times 2^{2p+1} \times 16$ bytes | $(p+1) \times 2^{2p+1} \times 32$ bytes |

**Formula**: Adjoint cache size = $(p+1) \cdot 2^{2p+1} \cdot 16$ bytes (Float64) or $32$ bytes (Double64).

| $p$ | $N = 2^{2p+1}$ | Cache (Float64) | Cache (Double64) | Min RAM (Float64) | Min RAM (Double64) |
|-----|----------------|-----------------|------------------|-------------------|-------------------|
| 10  | 2M             | 2 GB            | 4 GB             | 8 GB              | 16 GB             |
| 11  | 8M             | 8 GB            | 16 GB            | 16 GB             | 32 GB             |
| 12  | 33M            | 33 GB           | 67 GB            | 64 GB             | 128 GB            |
| 13  | 134M           | 134 GB          | 269 GB           | 256 GB            | 512 GB            |
| 14  | 537M           | 537 GB          | 1.1 TB           | 1 TB              | 2 TB              |

**Hardware deployed:**
- Mac Studio (M4, 12 threads): 64 GB RAM → max $p=12$ (single evaluation).
- Azure E32as_v5: 256 GB → $p=13$ safely (multiple evaluations in parallel).
- Azure E64as_v5: 512 GB → $p=14$ (with memory-bounded concurrency control).

---

### A.5 Thread Parallelism Model

Julia's multi-threading via `Threads.@threads` (fork-join model):

| Parallel Region | Granularity | Speedup (28 cores) | Notes |
|-----------------|-------------|-------------------|-------|
| f_table computation | $N / \text{nthreads}$ iterations | ~25× | Embarassingly parallel, SIMD inner loop |
| kernel computation | $N / \text{nthreads}$ iterations | ~25× | Phase dot product, threaded |
| WHT (recursive case) | Top-level split only | ~2× | Recursive sub-calls serialized (cache locality > parallelism) |
| Optimizer starts | One start per thread (semaphore) | Memory-bounded | Each start allocates full cache; semaphore caps concurrency |

**Optimization**: Optim.jl line search is serial (no nested parallelism); we parallelize over multiple random restarts with memory-bounded semaphore.

---

### A.6 Numerical Precision Model: Float64 vs Double64

#### **Float64 Regime** (~15 decimal digits)

Safe for $(k, D, p)$ if relative error in branch tensor magnitudes $\ll 10^{-15}$ throughout recurrence. Practical limits:

| $(k, D)$ | Safe $p_{\max}$ | Reason for ceiling |
|----------|-----------------|-------------------|
| (3,4) | 13 | Magnitude growth at ~$3^p$ per step |
| (3,5) | 13 | Magnitude growth at ~$8^p$ per step |
| (3,6) | 11 | Magnitude growth at ~$20^p$ |
| (4,5) | 11 | Arity + degree compound faster |
| (6,7) | 8 | $(k-1)(D-1) = 30$ → rapid overflow |
| (7,8) | 7 | $(k-1)(D-1) = 42$ → hits precision wall very early |

**Indicator**: Cross-run numerical agreement breaks down around the ceiling. E.g., two separate optimizations of (3,4, p=13) may give $\tilde{c}$ differing in the 4th decimal place (Float64 limit).

#### **Double64 Regime** (~31 decimal digits)

Extends safe range by ~1-2 depths per $(k,D)$ pair:

| $(k, D)$ | Float64 $p_{\max}$ | Double64 $p_{\max}$ | Improvement |
|----------|------------------|-------------------|-------------|
| (3,4) | 13 | 15+ | +2 |
| (6,7) | 8 | 9+ | +1 |
| (7,8) | 7 | 8+ | +1 |

**Deployment**: `eval_eltype=Double64` parameter in `optimize_angles()` and `swarm_optimize()`.

---

## B. Performance Characteristics

### B.1 Per-Evaluation Cost: $O(p^2 \cdot 4^p)$

**Derivation:**

1. **Forward pass:**
   - $p$ iterations of branch tensor step
   - Each step: one WHT ($O(N \log N)$), one iWHT ($O(N \log N)$), element-wise power ($O(N)$), normalization checks ($O(N)$)
   - Per-step cost: $O(N \log N) + O(N \log N) + O(N) = O((2p+1) \cdot 2^{2p})$
   - Total: $p \times O((2p+1) \cdot 2^{2p}) = O(p^2 \cdot 4^p)$

2. **Root fold:**
   - WHT + iWHT: $O((2p+1) \cdot 2^{2p})$
   - Negligible vs. branch iteration

3. **Backward pass:** ~1.6× forward pass (same structure, but all vectors + adjoints)

4. **Overhead:** Angle preprocessing (negligible), gradient scaling (negligible)

**Total:** ~2.6× a single forward evaluation.

| $p$ | $N = 2^{2p+1}$ | Evals/sec (F64, M4) | Evals/min | Evals/hour |
|-----|----------------|--------------------|-----------|-----------|
| 8   | $65K$ | 24 | 1,400 | 84,000 |
| 9   | 262K | 6.6 | 400 | 24,000 |
| 10  | 1M | 1.5 | 90 | 5,400 |
| 11  | 8M | 0.3 | 18 | 1,080 |
| 12  | 33M | 0.025 | 1.5 | 90 |
| 13  | 134M | 0.006 | 0.35 | 21 |

---

### B.2 Memory: $(p+1) \times 2^{2p+1} \times 16$ bytes (Float64)

See Table in §A.4. For practical scaling:

$$M(p) = (p+1) \cdot 2^{2p+1} \cdot 16 \text{ bytes}$$

Example: $p=14$ on Double64 requires $\approx 1.1$ TB (1 TB usable after OS/headroom).

---

### B.3 Gradient Cost: 1.6× Forward Evaluation

Manual adjoint backward pass structure mirrors forward pass:

- **WHT adjoints**: self-adjoint (same cost as forward)
- **Power adjoints**: power rule + chain rule (same as forward)
- **Normalization adjoints**: treated as constants (detached) — no cost

Empirical: `basso_expectation_and_gradient()` = 1.6× `basso_expectation_normalized()`.

---

### B.4 Safe Operating Envelope: The Precision Wall

Extracted from repo memory and confirmed by code inspection:

| $(k,D)$ | Safe $p_{\max}$ (F64) | Safe $p_{\max}$ (D64) | Wall reason |
|---------|----------------------|----------------------|-------------|
| (3,4)   | 13                   | 15+                  | Magnitude ~$3^p$ growth |
| (3,5)   | 13                   | 15+                  | Magnitude ~$8^p$ growth |
| (3,6)   | 11                   | 13+                  | Magnitude ~$20^p$ |
| (3,7)   | 11                   | 12+                  | Magnitude ~$42^p$ |
| (3,8)   | 11                   | 12+                  | Magnitude ~$72^p$ |
| (4,5)   | 11                   | 12+                  | $(k-1)(D-1) = 12$ |
| (4,6)   | 10                   | 11+                  | $(k-1)(D-1) = 15$ |
| (4,7)   | 10                   | 11+                  | $(k-1)(D-1) = 18$ |
| (4,8)   | 9                    | 10+                  | $(k-1)(D-1) = 21$ |
| (5,6)   | 9                    | 10+                  | $(k-1)(D-1) = 20$ |
| (5,7)   | 9                    | 10+                  | $(k-1)(D-1) = 24$ |
| (5,8)   | 9                    | 10+                  | $(k-1)(D-1) = 28$ |
| (6,7)   | 8                    | 9                    | $(k-1)(D-1) = 30$ |
| (6,8)   | 8                    | 9                    | $(k-1)(D-1) = 35$ |
| (7,8)   | 7                    | 8                    | $(k-1)(D-1) = 42$ |

**Approximate formula**: Maximum safe depth $\approx 20 / \log(\text{branching factor})$ for Float64.

---

### B.5 Wall Time Empirical Data

**Hardware**: Apple M4 Mac Studio (64 GB, 12 threads), single evaluation cache, no parallelism.

#### MaxCut (k=2, D=3) — Primary validation target

| $p$ | Wall time (sec) | $/\Delta p$ (geometric mean) | c̃ |
|-----|-----------------|------------------------------|--------|
| 1   | 0.072           | —                            | 0.6925 |
| 2   | 0.003           | 0.06× (noise)                | 0.7559 |
| 3   | 0.012           | 4×                           | 0.7924 |
| 4   | 0.024           | 2×                           | 0.8169 |
| 5   | 0.12            | 5×                           | 0.8364 |
| 6   | 0.8             | 6.7×                         | 0.8499 |
| 7   | 5.2             | 6.5×                         | 0.8598 |
| 8   | 41              | 7.9×                         | 0.8674 |
| 9   | 220 (3.7 min)   | 5.4×                         | 0.8735 |
| 10  | 660 (11 min)    | 3×                           | 0.8784 |
| 11  | 600 (10 min)    | 0.9× (plateau detection)     | — |

#### (k=3, D=4) — Primary research target

| $p$ | Wall time (sec) | $/\Delta p$ | c̃ (with optimization) |
|-----|-----------------|-------------|------------------------|
| 1   | 0.072           | —           | 0.6761 |
| 2   | 0.003           | 0.04×       | 0.7391 |
| 3   | 0.012           | 4×          | 0.7771 |
| 4   | 0.024           | 2×          | 0.8022 |
| 5   | 0.12            | 5×          | 0.8205 |
| 6   | 0.8             | 6.7×        | 0.8344 |
| 7   | 5.2             | 6.5×        | 0.8453 |
| 8   | 41              | 7.9×        | 0.8541 |
| 9   | 220 (3.6 min)   | 5.4×        | 0.8613 |
| 10  | 660 (11 min)    | 3×          | 0.8674 |
| 11  | 600 (10 min)    | 0.9×        | 0.8725 |
| **12** | **~2400 (40 min)** | **4×**     | **0.8769** |
| **13** | **~302,400 (84 hr)** | **126×** | **0.8807** (extrapolated) |

**Notes:**
- $p=1$ slow due to startup overhead (JIT compilation, etc.).
- $p=2$ shows caching benefit.
- Geometric growth $\sim 6-8×$ per depth for $p=4..10$ (consistent with $p^2 \cdot 4^p$ theory).
- $p=11$ faster than $p=10$ due to plateau detection kicking in (early convergence).
- $p=12$: standard L-BFGS $\approx 2$ hr; with plateau detection $\approx 40$ min.
- $p=13$: single depth level; 84 hours extrapolated (not yet completed).

---

### B.6 Optimizer Convergence Profile

**Typical optimization (p=8, M4 Mac):**

- Start: 5-8 random or warm-start angles
- Per start: 50-150 L-BFGS iterations (adaptive plateau detection)
- Total iterations: 250-1200 across all starts
- Wall time: ~41 sec per start evaluation × 1.6× gradient cost × 50 iterations ÷ 28 evals/min → ~3 min per start for 50 iters
- Total: 5 starts × 3 min = 15 min (rough)
- Speedup from plateau detection: ~30% reduction in iterations at $p=8$, **~75% at $p=12$** (plateau emerges faster at high depth)

---

## C. Proposed Optimizations

### C.1 Gradient Checkpointing (Memory reduction: $\sqrt{p}$ factor)

#### **Idea**

Traditional reverse-mode AD requires storing all $p+1$ branch tensors $(2^{2p+1}$ elements each). At $p=16$, this is 17 tensors × 1 TB = 17 TB.

**Checkpointing strategy**: Store a checkpoint every $\sqrt{p}$ steps (e.g., every 4 steps for $p=16$). During backward pass, recompute the intermediate steps between checkpoints on-demand.

**Forward pass (modified):**
1. Full forward pass as usual, but at checkpoint steps ($t = 0, \sqrt{p}, 2\sqrt{p}, \ldots, p$), save $(B_t, \text{angles})$.
2. Between checkpoints, do NOT save intermediates.
3. Total checkpoints: $O(\sqrt{p})$.

**Backward pass (recomputation):**
1. Start at $t = p$; if no checkpoint at $t$, recompute from nearest earlier checkpoint.
2. Recompute forward over $[\text{checkpoint}, t]$ (O(1) dependencies, no parallelism needed).
3. Perform backward over the recomputed segment.
4. Free recomputed vectors immediately.

#### **Expected Impact**

- **Memory**: $(p+1) \to (\sqrt{p} + 1)$ tensors stored. Reduction factor: $(p+1) / (\sqrt{p}+1) \approx \sqrt{p}$ for large $p$.
  - $p=16$: $17 \to 5$ tensors = **3.4× reduction** (1 TB → 300 GB).
  - $p=20$: $21 \to 5$ tensors = **4.2× reduction**.
  - $p=25$: $26 \to 6$ tensors = **4.3× reduction**.

- **Compute**: Extra recomputation of $\sqrt{p}$ forward segments ($O(p \cdot 4^p)$ each) = $O(\sqrt{p} \cdot p \cdot 4^p) = O(p^{1.5} \cdot 4^p)$ vs. original $O(p^2 \cdot 4^p)$ forward + backward.
  - Trade-off: ~2× backward cost for ~$\sqrt{p}$ memory savings.
  - For $p=16$: 2× backward (1.5 min) vs. 3.4× memory savings (saves 700 GB) = **net positive** if memory is bottleneck.

- **Wall time**: Modest slowdown (~15-30% increase) due to forward recomputation during backward.

#### **Implementation Complexity**

- Moderate: Requires caching checkpoint states and a recomputation loop.
- Not invasive: Can be a separate code path (`basso_expectation_and_gradient_checkpointed`).
- Minimal risk: Checkpointing doesn't affect forward pass correctness; backward is straightforward.

#### **Risks**

- **Recomputation cache misses**: If checkpoints are too frequent, recomputation thrashing can occur. Optimal is ~$\sqrt{p}$ (cache-oblivious principle).
- **Numerical differences**: Recomputing may accumulate different rounding errors than the original computation. Impact: negligible (both are in normalized space).

---

### C.2 Symmetry Reduction of Configuration Space (Memory reduction: 2–4×)

#### **Idea**

The branch tensor $B(a)$ on $(2p+1)$-bit configurations has two exact symmetries (already exploited in `reduced_basis.jl`):

1. **Root-bit independence**: $B(a) = B(a \oplus e_r)$ where $e_r$ flips the root bit $a^{[0]}$. Reason: the constraint kernel has no root-bit dependence (root bit does not appear in phase dot product).

2. **Complement invariance**: $B(a) = B(\bar{a})$ where $\bar{a}$ flips all non-root bits. Reason: mirror symmetry in the gamma vector and mixer weights.

These two symmetries generate a subgroup $H^{\perp} \cong ℤ_2^2$ of order 4. The branch tensor is constant on cosets of $H^{\perp}$, reducing the effective size from $N = 2^{2p+1}$ to $M = N/4 = 2^{2p-1}$.

#### **Current Status**

Already implemented in `reduced_basis.jl`:
- `ReducedBasis(p)`: precomputes coset maps.
- `basso_branch_tensor_reduced()`: runs the iteration in reduced space (M-element vectors instead of N).
- `expand_symmetric()`: lifts reduced result back to full space for root fold.

**Achieved**: 4× memory reduction during branch iteration, but **not yet applied to adjoint cache**.

#### **Proposed Extension**

Apply symmetry reduction to the **entire adjoint backward pass**:

1. **Forward pass**: Store only reduced intermediates $(B_{\text{red}}, \text{child\_hat\_red}, \text{folded\_red})$ at each step. Cost: $p$ tensors of size $M = 2^{2p-1}$ instead of $N = 2^{2p+1}$ — **4× savings**.

2. **Root fold**: Expand to full space once (negligible cost, done once), compute root fold in full space, backprop cotangents to reduced space.

3. **Backward pass**: All intermediate cotangents stay in reduced space until final angle gradients (which are naturally in reduced form via the symmetry structure).

4. **Expected memory**: $(p+1) \times 2^{2p-1} \times 16$ bytes instead of $(p+1) \times 2^{2p+1} \times 16$ bytes = **4× reduction**.

#### **Expected Impact**

- **Memory**: 4× reduction directly applied to adjoint cache.
  - $p=13$: $134 \to 33.5$ GB (fits in E32as_v5, 256 GB).
  - $p=14$: $537 \to 134$ GB (fits in E64as_v5 with headroom).
  - Equivalent to gaining **~1.5-2 depth levels** in memory capacity.

- **Compute**: Reduced WHTs are 4× cheaper per step (same algorithm, 4× fewer elements). Total backward pass still ~1.6× forward, but forward is 4× cheaper overall.
  - Wall time reduction: ~3-4× (proportional to memory reduction).

- **Correctness**: The symmetry is exact; no approximation introduced.

#### **Implementation Complexity**

- **Low to moderate**: The infrastructure is already there (`ReducedBasis`, `reduce_sample`, `expand_symmetric`). Main work:
  - Store reduced intermediates in cache struct.
  - Adapt backward pass to work with reduced vectors.
  - Map between reduced and full space for root fold.
  - Careful attention to where expansion/contraction happens.

- **Code change**: ~200-300 lines of new code or refactoring; no algorithmic changes.

#### **Risks**

- **Subtle indexing errors**: Reduced-space bit indexing is non-trivial. Must verify coset computations carefully.
- **Root fold transition**: Expansion from M to N must be correct; misalignment would break gradient computation.
- **Testing**: Requires cross-validation against full-space computation (already possible via `basso_branch_tensor` vs. `basso_branch_tensor_reduced`).

---

### C.3 Mixed-Precision Forward/Backward (Memory reduction: 2×, Wall time: 1.5-2×)

#### **Idea**

**Forward pass**: Use Float32 (7 decimal digits) to screen candidate angles during early optimizer iterations. Float32 reduces memory 4× and speeds up WHTs by ~2× (wider SIMD, better cache utilization).

**Backward pass + final polish**: At convergence, switch to Float64 for exact gradients. This two-phase approach trades precision during search for speed.

**Rationale**: Early optimizer iterations explore rough landscape; high precision not needed. Only at the final candidate do we require accurate gradients.

#### **Implementation**

1. **Phase 1 (screening)**: `evaluate_angles(..., precision=Float32)`
   - Adjoint cache with Float32 vectors.
   - L-BFGS line search in reduced precision.
   - Early termination when plateau detected.

2. **Phase 2 (polish)**: Once plateau detected with Float32, `evaluate_angles(..., precision=Float64)` on the best candidate.
   - Full-precision adjoint cache.
   - Final L-BFGS iterations (10-50) to refine.

#### **Expected Impact**

- **Memory**: Float32 cache = 0.5× Float64 cache per evaluation. During Phase 1, we might run 5-10 concurrent evals on semaphore-limited threads. Total: $5 \times 0.5 \times (\text{F64 cache}) = 2.5 \times$ F64 cache (vs. $5 \times$ F64 cache without mixed-precision). **~2× reduction during Phase 1**.

- **Wall time**: Float32 WHTs ~2× faster (SIMD better packing, L1 cache 2× fewer bytes). Phase 1 might be 30-50 iterations (early plateau at reduced precision). Phase 2 is 10-50 iterations at full precision. Trade-off: Phase 1 is fast, Phase 2 is careful. **Net 1.5-2× wall time gain** if Phase 1 converges early.

- **Accuracy**: Float32 has ~7 decimal digits; relative error in $\tilde{c}$ at convergence is ~1e-4. For $\tilde{c} \in [0.5, 1]$, absolute error ~1e-5. Acceptable for search phase; refined in Phase 2 to Float64 precision.

#### **Implementation Complexity**

- **Moderate**: Requires two parameter branches in the optimizer:
  - Swap precisions in `optimize_angles(..., precision_phase1=Float32, precision_phase2=Float64)`.
  - Autoswitch trigger: when Phase 1 plateau detected, re-evaluate best candidate in Float64.

- **Code change**: ~150-250 lines (mostly conditional logic in optimizer loop).

#### **Risks**

- **Optimizer divergence**: If Float32 gradients are too noisy, L-BFGS may take poor steps. Mitigation: tighter line search, smaller step sizes in Float32 phase.
- **False convergence**: Float32 noise floor may trigger false plateau detection. Mitigation: require plateau over slightly longer window in Phase 1 (e.g., 40 values instead of 30).
- **Solution quality**: If Phase 1 converges to a shallow local minimum and Phase 2 polish doesn't escape, we might miss the global optimum. Mitigation: Phase 2 should include a few random restarts if Phase 1 value is suspiciously low.

---

### C.4 Disk-Backed Tensor Checkpointing (Memory reduction: unlimited, limited by disk)

#### **Idea**

Spill intermediate tensors to NVMe storage during forward pass, reload during backward pass. Decouples depth from RAM; limited only by disk capacity and I/O bandwidth.

**Forward pass (modified):**
1. Compute branch tensor step $t$.
2. Serialize $(B_t, \text{child\_hat}_t, \text{folded}_t)$ to disk (binary format, no overhead).
3. Keep only current $B_t$ in RAM; release all previous tensors.

**Backward pass (modified):**
1. Deserialize $B_t$ from disk on-demand during backward step $t$.
2. Recompute cotangent flow; free after use.
3. Reuse single buffer slot (no accumulation).

**Storage**: 
- Each tensor: $2^{2p+1}$ Complex64 entries = 8 bytes × $2^{2p+1}$ = 32 MB per tensor (Float64: 16 × $2^{2p+1}$ = 64 MB).
- $p$ tensors: $(p+1) \times 2^{2p+1} \times 8$ bytes = at $p=20$: 21 × 1M × 8 bytes = 168 MB (tiny).

**I/O cost**:
- Sequential write during forward: $p$ tensors × 64 MB / NVMe speed (3 GB/s) ≈ 13 ms per tensor.
- Sequential read during backward: same.
- Total I/O: $\sim 260$ ms for $p=20$ (vs. $\sim 10$ sec per evaluation = 2.6% overhead).

#### **Expected Impact**

- **Memory**: Eliminates adjoint cache requirement. Only current tensor + angle buffers in RAM = ~200 MB. Independent of $p$. Enables $p=20+$ on laptops.

- **Wall time**: 2-3% slowdown (I/O on modern NVMe is fast; serialization is the overhead).

- **Scalability**: Limited only by disk space (easily 1+ TB on modern laptops). Enables exploration of $p=18+$ regimes without hardware upgrade.

#### **Implementation Complexity**

- **Moderate to high**:
  - Serialization format (binary, mmap-friendly).
  - Disk file management (cleanup, error handling).
  - Async I/O (Julia's `Threads.Task` or `asyncmap`).
  - Testing: verify checksum integrity after I/O round-trip.

- **Code change**: ~400-600 lines (substantial refactor of cache management).

#### **Risks**

- **I/O bottleneck**: If I/O is on same SSD as OS/paging, contention could cause slowdown. Mitigation: use dedicated NVMe or in-memory ramdisk if available.
- **Failure modes**: Disk full, corruption, permission errors must be handled gracefully. Mitigation: sanity checks before backward pass.
- **Portability**: NVMe paths, file I/O differ across Windows/Mac/Linux. Mitigation: use Julia's `tempdir()` and cross-platform I/O primitives.

---

### C.5 GPU Acceleration of WHT (Speedup: 10-100×)

#### **Idea**

The WHT (Walsh-Hadamard Transform) is an embarrassingly parallel butterfly network:

$$\hat{x}_j = \sum_{\ell=0}^{n} x_{\ell} (-1)^{\langle j, \ell \rangle}$$

Each output $\hat{x}_j$ is an independent sum; can be computed in parallel. NVIDIA GPUs (CUDA) and AMD GPUs (ROCm) are well-suited for this.

Similarly, element-wise power operations $(x^k)$ and multiplications $(A \odot B)$ are embarassingly parallel.

**GPU implementation strategy**:
1. Transfer $f_{\text{table}}, \text{kernel}$ to GPU once per optimization start.
2. Per evaluation:
   - Transfer current $(B_t, \beta, \gamma)$ to GPU.
   - Compute branch iteration with GPU WHTs and powers.
   - Transfer result back to CPU for root fold (or do root fold on GPU too).
3. Gradient computation: similar GPU parallelism for backward pass.

#### **Expected Impact**

- **Wall time per evaluation**: 
  - Forward pass (pure compute): $O(p^2 \cdot 4^p)$ operations → parallelized across 1000s of GPU cores.
  - Speedup vs. CPU: 10-100× depending on GPU model and memory bandwidth.
  - Example: p=12 from 40 min (CPU, M4) → 2-4 min (GPU, RTX 3090 or A100).

- **Memory**: GPU memory is typically 24-80 GB (consumer RTX) to 1+ TB (data center). Can fit $p=15+$ on modern high-memory GPUs.

- **Throughput**: At massively parallel scale (data center), could run 10s of simultaneous optimizations, covering entire $(k,D)$ table in hours instead of weeks.

#### **Implementation Complexity**

- **High**: Requires GPU programming expertise (CUDA/Julia wrappers):
  - Implement GPU WHT kernel (or wrap existing library: `CuFFT` for FFTs, but WHT is simpler).
  - Memory management: CPU ↔ GPU transfers.
  - Error handling: OOM, device failures.
  - Testing across different GPU models.

- **Code change**: ~1000+ lines (substantial, new module: `gpu_acceleration.jl`).

- **Dependencies**: CUDA.jl or AMDGPU.jl, additional build complexity.

#### **Risks**

- **Double64 not natively supported on GPU**: GPUs have FP32 and FP64 support, but Double64 (software-emulated 31-digit precision) would be extremely slow on GPU. Mitigation: Use FP64 on GPU for high-$(k,D)$ pairs, accept precision ceiling remains at Float64 range.

- **Memory bandwidth**: GPU-CPU PCI-E bandwidth (16 GB/s on PCIe 4.0) is limited. For $p=12$, transferring 33 GB adjoint cache would take ~2 sec, dominating the 4 min GPU compute time. Mitigation: Keep cache on GPU if possible (fits on most modern GPUs).

- **Portability**: CUDA-specific code doesn't run on AMD/Intel GPUs. Mitigation: Use abstract GPU interface (GPUCompiler.jl, AbstractGPU), but requires significant refactor.

- **Validation**: GPU results must be bit-identical to CPU (unlikely due to associativity of floating-point). Mitigation: accept within-ULP or small relative tolerance (~1e-6).

---

### C.6 Advanced Optimizer: Bandit-Assisted Multi-Start (Wall time reduction: 20-40%)

#### **Idea**

Current multi-start strategy is uniform: allocate equal iterations to all $n_s$ random starts. In practice, some starts converge quickly to good local minima; others plateau immediately.

**Bandit approach**: Track convergence rate per start using a multi-armed bandit algorithm (e.g., Thompson sampling, Successive Halving, Hyperband):

1. **Initialization**: Sample $n_s$ random angles.
2. **Elimination round 1**: Run 20 iterations of L-BFGS on each. Evaluate improvement $/$ iteration.
3. **Elimination round 2**: Discard bottom 50%; run 40 iterations on remaining.
4. **Repeat** until one start remains. Polish with full maxiters.

**Early stopping for sluggish starts**: If a start plateaus (improvement $< 10^{-6}$ over 20 iterations), eliminate it immediately.

#### **Expected Impact**

- **Wall time**: Avoids wasting iterations on starts that won't improve. Typical speedup: 20-40% (fewer total iterations across all starts).
  - Example: $p=12$, 1200 total iterations across 8 starts → 800-900 iterations with bandit.

- **Solution quality**: May actually improve if bandit correctly identifies the highest-potential basins early.

#### **Implementation Complexity**

- **Low to moderate**: Existing Optim.jl interface remains; add a wrapper that allocates iterations dynamically.
- **Code change**: ~200-300 lines (mostly bookkeeping).

#### **Risks**

- **Early elimination error**: If we eliminate a promising start too early, we miss a better optimum. Mitigation: conservative thresholds (require multiple poor rounds before elimination).

---

### Summary Table: Optimization Proposals

| Optimization | Memory Reduction | Wall Time Impact | Complexity | Risk | Priority |
|---|---|---|---|---|---|
| **Gradient Checkpointing** | √p | +15-30% wall time | Moderate | Low | High (enables p=18+) |
| **Symmetry Reduction** | 4× | -3-4× wall time | Low | Moderate | High (quick win) |
| **Mixed Precision** | 2× (Phase 1) | -30-50% wall time | Moderate | Moderate | Medium (good ROI) |
| **Disk Checkpointing** | ∞ (disk-limited) | +2-3% wall time | Moderate–High | Low | Medium (low wall-time cost) |
| **GPU WHT** | GPU-memory | -10-100× wall time | High | High (double64 issue) | Low (high complexity) |
| **Bandit Multi-Start** | 0 | -20-40% wall time | Low | Low | Medium (easy gain) |

---

## Appendix: Formulas and Key References

### Forward Pass Recurrence (Normalized)

$$B^{[t+1]}(a) = \left[ \text{iWHT}\left( \hat{K} \cdot \widehat{(f \cdot B^{[t]})}^{k-1} \right) \right]^{D-1}_{\text{norm}}$$

where:
- Normalization: If $\max|B^{[t]}| > 10^{30}$, divide by max before power.
- $\hat{K}$: WHT of constraint kernel.
- $f(a)$: mixer weight (half-product of trigonometric factors).
- Exponents: $k-1$ is constraint arity-1; $D-1$ is branching degree.

### Root Parity Expectation

$$S = \sum_a K_{\text{root}}(a) \cdot \text{iWHT}(\widehat{M_{\text{root}}}^k)_a$$

where $K_{\text{root}}$ is the root observable kernel and $M_{\text{root}}$ is the root parity message.

### Adjoint Backward (Log-Scale Multiplier)

$$\frac{\partial \tilde{c}}{\partial \theta} = \frac{c_s}{2} \exp(L) \frac{\partial S_{\text{norm}}}{\partial \theta}$$

where $L = k(\log s_p + \log \mu)$ is the accumulated log-scale and $\mu = \max|\text{WHT}(M_{\text{root}})|$.

---

## Conclusion

The QAOA-XORSAT codebase represents a tightly engineered, production-grade evaluator for exact QAOA performance on D-regular Max-k-XORSAT. The 8 core innovations enable exact, numerically stable evaluation up to $p=13$ for primary targets (3,4) and discovery of competitive solutions at high $(k,D)$ pairs unreachable by standard techniques.

---

## D. GPU Acceleration Design (Innovation 9)

### D.1 Objectives

1. Accelerate the forward and backward passes by offloading WHT, element-wise power, and element-wise multiply to GPU
2. Support both Metal (Apple M4, local development) and CUDA (NVIDIA, cluster production) via a shared abstraction
3. Maintain bit-level validation against CPU results (within floating-point associativity tolerance)
4. No change to the optimizer or angle management — GPU accelerates only the evaluator

### D.2 Architecture

**Layer 1: Abstract GPU Array Interface**

Use `KernelAbstractions.jl` as the portable GPU programming model. It targets:
- `CUDABackend()` → NVIDIA via CUDA.jl
- `MetalBackend()` → Apple Silicon via Metal.jl
- `CPU()` → fallback, for testing

All GPU operations work on `AbstractArray` — the same code runs on CPU or GPU depending on where the array lives.

**Layer 2: GPU-Accelerated WHT**

The WHT butterfly is the core kernel. Two implementation options:

*Option A: Iterative GPU WHT kernel*

A single GPU kernel performs all butterfly levels in-place. Each thread handles one butterfly pair. For $N = 2^{2p+1}$ elements:
- Level 0: N/2 pairs with stride 1
- Level 1: N/2 pairs with stride 2
- ...
- Level $2p$: N/2 pairs with stride $N/2$

Synchronization between levels via `@synchronize` (shared memory) or separate kernel launches per level.

*Option B: Use existing GPU FFT and convert*

The WHT on $\mathbb{Z}_2^n$ is equivalent to the DFT on the group $\mathbb{Z}_2^n$ — but cuFFT implements DFT on $\mathbb{Z}_N$, not on $\mathbb{Z}_2^n$. These are different transforms. There is no standard GPU library for WHT, so we must write our own kernel.

**Recommendation: Option A** — write a custom GPU WHT kernel using KernelAbstractions.jl.

**Layer 3: GPU Forward Pass**

```
function gpu_forward_pass(params, angles; device)
    # Allocate all tensors on GPU
    N = 2^(2p+1)
    B = device_zeros(ComplexF64, N)           # branch tensor
    kernel_hat = device_array(kernel_hat_cpu) # transfer once
    f_table = device_array(f_table_cpu)       # transfer once

    cache = []  # store on GPU for backward pass

    for t in 1:p
        child = f_table .* B
        child_hat = gpu_wht!(child)
        child_hat_power = child_hat .^ (k-1)    # element-wise, GPU
        folded_hat = kernel_hat .* child_hat_power
        folded = gpu_iwht!(folded_hat)
        B_next = folded .^ (D-1)                 # element-wise, GPU
        # Threshold normalization on GPU
        max_val = maximum(abs.(B_next))
        if max_val > THRESHOLD
            B_next ./= max_val
            log_scale += log(max_val)
        end
        push!(cache, (child_hat, folded, B_next))
        B = B_next
    end

    # Root fold on GPU
    root_msg = ...
    root_msg_hat = gpu_wht!(root_msg)
    root_msg_power = root_msg_hat .^ k
    conv = gpu_iwht!(root_msg_power)
    S = sum(root_kernel .* conv)  # reduction on GPU

    return (S, cache, log_scale)
end
```

**Layer 4: GPU Backward Pass**

Same structure as CPU backward pass but operating on GPU arrays. The adjoint of WHT is WHT/N (same as iWHT up to scaling). The adjoint of element-wise power is element-wise multiply by (k-1) * x^(k-2). All operations are element-wise or WHT — fully GPU-able.

### D.3 Memory Management

At $p=13$, the adjoint cache is ~134 GB (Float64). GPU memory:
- Apple M4 (unified): shares system 64 GB → p ≤ 12
- NVIDIA A100: 80 GB → p ≤ 12
- NVIDIA H100: 80 GB → p ≤ 12
- NVIDIA A100 80GB × 2 (NVLink): 160 GB → p ≤ 13

**Strategy**: For p ≤ 12, keep entire cache on GPU. For p ≥ 13, use gradient checkpointing (store every √p levels, recompute the rest).

For the M4 Mac (unified memory architecture), GPU and CPU share the same physical RAM — no transfer overhead. The GPU just needs a different view of the same memory.

### D.4 Testing Strategy

**Level 1: Unit tests for GPU WHT**
- Compare `gpu_wht(x)` vs `cpu_wht(x)` for random vectors of sizes 2^1 through 2^21
- Tolerance: `≈ atol=1e-10` (Float64 associativity differences)
- Test both forward and inverse: `gpu_iwht(gpu_wht(x)) ≈ x`
- Test convolution theorem: `gpu_wht(a .* b) ≈ gpu_wht(a) ⊛ gpu_wht(b) / N`

**Level 2: Unit tests for GPU element-wise operations**
- `gpu_power(x, k)` vs `x .^ k` for k = 2, 3, 5, 7
- `gpu_normalize(x, threshold)` vs CPU normalization
- Edge cases: zero vectors, all-ones vectors, vectors with one huge element

**Level 3: Integration tests for GPU forward pass**
- Compare `gpu_forward_pass(params, angles)` vs `cpu_forward_pass(params, angles)` for:
  - MaxCut (k=2, D=3) at p=1,2,3,5
  - XORSAT (k=3, D=4) at p=1,2,3,5
  - XORSAT (k=5, D=6) at p=1,2,3
- Tolerance: `≈ atol=1e-8` (accumulated floating-point differences across p levels)

**Level 4: Integration tests for GPU backward pass (gradient)**
- Compare `gpu_gradient(params, angles)` vs `cpu_gradient(params, angles)` for same cases
- Cross-validate: `gpu_gradient` vs finite-difference gradient at p=1,2,3
- Tolerance: `≈ atol=1e-6` for gradient components

**Level 5: End-to-end optimization test**
- Run `optimize_angles(params; device=:gpu)` vs `optimize_angles(params; device=:cpu)` for (k=2, D=3, p=5)
- Both should converge to the same c̃ within 1e-8
- GPU version should be faster (benchmark)

**Level 6: Performance regression tests**
- Benchmark GPU WHT at sizes 2^15, 2^17, 2^19, 2^21, 2^23
- Verify GPU speedup > 5× vs CPU for N ≥ 2^17
- Verify GPU forward pass speedup > 3× vs CPU at p ≥ 8
- Detect performance regressions across commits

### D.5 Implementation Plan

**Phase 1: GPU WHT kernel (2-3 days)**
1. Add KernelAbstractions.jl, Metal.jl to Project.toml
2. Implement `gpu_wht!` using KernelAbstractions
3. Level 1 tests passing on Metal
4. Benchmark vs CPU

**Phase 2: GPU forward pass (2-3 days)**
5. Implement `gpu_forward_pass` using GPU arrays
6. Level 3 tests passing
7. Benchmark forward pass

**Phase 3: GPU backward pass (2-3 days)**
8. Implement `gpu_backward_pass` (adjoint through cached GPU tensors)
9. Level 4 tests passing
10. End-to-end gradient validation

**Phase 4: Integration with optimizer (1-2 days)**
11. Add `device` parameter to `optimize_angles` and `swarm_optimize`
12. Level 5 tests passing
13. Full benchmark suite

**Phase 5: CUDA backend (1-2 days, when cluster access available)**
14. Add CUDA.jl dependency (optional)
15. Verify all tests pass on CUDA backend
16. Production benchmarks on NVIDIA hardware

### D.6 Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| WHT numerical differences (associativity) | Tolerance-based comparison, not bit-exact |
| Metal.jl maturity/bugs | Fall back to CPU on Metal failures |
| Double64 not supported on GPU | GPU uses Float64 only; D64 stays CPU-only |
| Memory pressure on M4 (shared GPU/CPU) | Monitor with `Sys.free_memory()`, cap GPU allocation |
| KernelAbstractions overhead for small N | Only use GPU for N ≥ 2^15 (p ≥ 7); CPU below |

**Next-phase opportunities** focus on either **depth extension** (checkpointing, symmetry reduction to reach $p=16-18$) or **throughput acceleration** (GPU, bandit multi-start to explore broader instance spaces). The trade-offs between memory, precision, and wall time are well-understood and quantified, enabling data-driven decisions on optimization priority.