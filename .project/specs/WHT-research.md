# Research Spec: Walsh-Hadamard Factorisation of the Constraint Fold

**Date**: 22 March 2026
**Status**: Hypothesis — awaiting verification
**Blocks**: Tier 2 implementation cost reduction (P1.3)
**Sources**: Basso 2021 §8.2 Eq. (8.7), Boolean Fourier analysis

---

## 1. Problem Statement

### The sum

At each iteration step $m$ of the Basso finite-$D$ iteration for Max-$k$-XORSAT
(Eq. 8.7, arXiv:2110.14206), the inner sum at a constraint node for $k = 3$ is:

$$S(\mathbf{a}) = \sum_{\mathbf{b}^1, \mathbf{b}^2 \in \{-1,+1\}^{2p+1}}
\cos\!\left(\frac{\boldsymbol{\Gamma} \cdot
(\mathbf{a} \odot \mathbf{b}^1 \odot \mathbf{b}^2)}{\sqrt{D}}\right)
\, g(\mathbf{b}^1)\, g(\mathbf{b}^2)$$

where:
- $\mathbf{a} \in \{-1,+1\}^{2p+1}$ is the parent variable's hyperindex
- $\mathbf{b}^1, \mathbf{b}^2 \in \{-1,+1\}^{2p+1}$ are the two child variables' hyperindices
- $\odot$ is entry-wise (Hadamard) product
- $\boldsymbol{\Gamma} = (\gamma_1, \ldots, \gamma_p, 0, -\gamma_p, \ldots, -\gamma_1)$
  is the $(2p+1)$-component angle vector
- $g(\mathbf{b}) = f(\mathbf{b}) \cdot H_D^{(m-1)}(\mathbf{b})$ is the **dressed
  branch tensor** (mixer sandwich times previous-iteration branch tensor)
- $D$ is Basso's branching factor (our degree minus 1)

The branch tensor update is then $H_D^{(m)}(\mathbf{a}) = S(\mathbf{a})^D$.

### The cost

$S(\mathbf{a})$ sums over $2^{2p+1} \times 2^{2p+1} = 4^{2p+1}$ child pairs
$(\mathbf{b}^1, \mathbf{b}^2)$. Evaluating $S$ for all $2^{2p+1}$ parent values
$\mathbf{a}$ costs $O(4^{3p})$ per iteration step.

At $k=3$, $p$ iteration steps plus the final evaluation (Eq. 8.8, also $O(4^{3p})$)
give total cost $O(p \cdot 4^{3p})$.

| $p$ | $4^{3p}$ | Wall time (est.) |
|-----|----------|------------------|
| 3   | $2.6 \times 10^5$ | milliseconds |
| 5   | $1.1 \times 10^9$ | seconds–minutes |
| 7   | $4.4 \times 10^{12}$ | hours (cluster) |
| 10  | $1.2 \times 10^{18}$ | infeasible |

### Why it matters

If the sum can be factorised to $O(p \cdot 4^p)$ or even $O(4^{2p})$, we push
the feasible $p$ frontier from $\sim 5$ to $\sim 10$ or $\sim 15$, matching the
reach of DQI results from Stephen Jordan and enabling meaningful comparison.

---

## 2. The WHT Factorisation Hypothesis

### Step 1: Change of variables

Define the **parity vector** $\mathbf{c} = \mathbf{b}^1 \odot \mathbf{b}^2$.
Since each $b^i_\ell \in \{-1,+1\}$, we have $c_\ell = b^1_\ell b^2_\ell \in \{-1,+1\}$.

Rewrite:

$$S(\mathbf{a}) = \sum_{\mathbf{c} \in \{-1,+1\}^{2p+1}}
\cos\!\left(\frac{\boldsymbol{\Gamma} \cdot (\mathbf{a} \odot \mathbf{c})}{\sqrt{D}}\right)
W(\mathbf{c})$$

where:

$$W(\mathbf{c}) = \sum_{\substack{\mathbf{b}^1, \mathbf{b}^2 \\ \mathbf{b}^1 \odot \mathbf{b}^2 = \mathbf{c}}} g(\mathbf{b}^1)\, g(\mathbf{b}^2)$$

### Step 2: Computing $W(\mathbf{c})$ via WHT — cost $O(p \cdot 4^p)$

The map $(\mathbf{b}^1, \mathbf{b}^2) \mapsto \mathbf{b}^1 \odot \mathbf{b}^2$
is point-wise multiplication in $\{-1,+1\}^n$, which is equivalent to addition in
$\mathbb{Z}_2^n$ under the isomorphism $x \mapsto (1 - x)/2$.

Therefore $W = g \ast g$ is the **convolution** on the group $\mathbb{Z}_2^{2p+1}$.
By the convolution theorem for the Walsh-Hadamard transform (WHT):

$$\hat{W} = \hat{g}^2$$

where $\hat{g}$ denotes the WHT of $g$. Computing $\hat{g}$ costs
$O(n \cdot 2^n) = O(p \cdot 4^p)$, squaring is $O(4^p)$, and inverse WHT is
$O(p \cdot 4^p)$. Total: $O(p \cdot 4^p)$.

**This step is clean and rigorous.** No approximation involved.

### Step 3: Computing $S(\mathbf{a})$ from $W(\mathbf{c})$ — THE CRITICAL STEP

We now need:

$$S(\mathbf{a}) = \sum_{\mathbf{c}} \cos\!\left(\frac{\boldsymbol{\Gamma} \cdot (\mathbf{a} \odot \mathbf{c})}{\sqrt{D}}\right) W(\mathbf{c})$$

Write $\boldsymbol{\Gamma} \cdot (\mathbf{a} \odot \mathbf{c}) = \sum_\ell \Gamma_\ell a_\ell c_\ell$.

**The hypothesis**: this is also a convolution on $\mathbb{Z}_2^{2p+1}$, and can
be computed via WHT in $O(p \cdot 4^p)$.

**The obstacle**: define $K(\mathbf{a}, \mathbf{c}) = \cos\!\left(\frac{1}{\sqrt{D}} \sum_\ell \Gamma_\ell a_\ell c_\ell\right)$. For this to be a convolution kernel, we would need:

$$K(\mathbf{a}, \mathbf{c}) = \kappa(\mathbf{a} \odot \mathbf{c})$$

for some function $\kappa$. Indeed, let $\kappa(\mathbf{d}) = \cos\!\left(\frac{1}{\sqrt{D}} \boldsymbol{\Gamma} \cdot \mathbf{d}\right)$. Then:

$$K(\mathbf{a}, \mathbf{c}) = \kappa(\mathbf{a} \odot \mathbf{c})$$

and

$$S(\mathbf{a}) = \sum_{\mathbf{c}} \kappa(\mathbf{a} \odot \mathbf{c})\, W(\mathbf{c}) = (\kappa \ast W)(\mathbf{a})$$

This IS a convolution on $\mathbb{Z}_2^{2p+1}$! By the convolution theorem:

$$\hat{S} = \hat{\kappa} \cdot \hat{W}$$

So $S$ can be computed as:
1. Compute $\hat{\kappa}$ (WHT of the cosine kernel): $O(p \cdot 4^p)$
2. Multiply $\hat{\kappa} \cdot \hat{W}$: $O(4^p)$
3. Inverse WHT: $O(p \cdot 4^p)$

**Total cost of the full constraint fold: $O(p \cdot 4^p)$.**

### Wait — is this really correct?

The key question is whether Step 3 is valid. Let me spell out the algebra explicitly.

We have:

$$S(\mathbf{a}) = \sum_{\mathbf{c}} \kappa(\mathbf{a} \odot \mathbf{c})\, W(\mathbf{c})$$

In the $\mathbb{Z}_2$ isomorphism ($x \leftrightarrow (1-x)/2$), entry-wise
multiplication in $\{-1,+1\}$ corresponds to addition in $\mathbb{Z}_2$. So
$\mathbf{a} \odot \mathbf{c}$ corresponds to $\tilde{\mathbf{a}} + \tilde{\mathbf{c}}$ (mod 2).

Therefore:

$$S(\tilde{\mathbf{a}}) = \sum_{\tilde{\mathbf{c}}} \kappa(\tilde{\mathbf{a}} + \tilde{\mathbf{c}})\, W(\tilde{\mathbf{c}})$$

This is the definition of convolution on $\mathbb{Z}_2^{2p+1}$:
$(κ \ast W)(\tilde{\mathbf{a}}) = \sum_{\tilde{\mathbf{c}}} \kappa(\tilde{\mathbf{a}} + \tilde{\mathbf{c}}) W(\tilde{\mathbf{c}})$, which equals
$\sum_{\tilde{\mathbf{c}}} \kappa(\tilde{\mathbf{a}} - \tilde{\mathbf{c}}) W(\tilde{\mathbf{c}})$ since in $\mathbb{Z}_2$, addition and subtraction coincide.

Alternatively: the group $\mathbb{Z}_2^n$ is self-dual, so the convolution theorem
applies and $\hat{S} = \hat{\kappa} \cdot \hat{W}$.

**There is no cos-vs-exp issue here.** The cosine kernel $\kappa$ does NOT need
to factorise as a product over rounds. It is treated as an opaque function
$\{-1,+1\}^{2p+1} \to \mathbb{R}$, and the WHT handles the convolution structure.

---

## 3. Where the Argument Might Break

### 3.1. Does the cosine need to factorise? NO.

The original concern was that $\cos(\sum_\ell t_\ell) \neq \prod_\ell \cos(t_\ell)$,
which would block a round-by-round factorisation. But the WHT approach does NOT
require the kernel to factorise over rounds. The WHT operates on the full
$(2p+1)$-bit string simultaneously. The non-separability of cosine is irrelevant.

**The cos vs exp concern is a red herring for this approach.** It would matter if
we tried to factor the sum round-by-round (as in the Farhi 2025 MaxCut trick).
The WHT approach instead factors over the two child branches, not over rounds.

### 3.2. Does $\kappa(\mathbf{a} \odot \mathbf{c})$ give a valid convolution? YES.

The function $(\mathbf{a}, \mathbf{c}) \mapsto \kappa(\mathbf{a} \odot \mathbf{c})$
is a convolution kernel on $\mathbb{Z}_2^{2p+1}$ because $\odot$ is the group
operation. This is exactly the structure the WHT convolution theorem handles.

### 3.3. Is $W(\mathbf{c})$ really a convolution of $g$ with itself? YES.

$W(\mathbf{c}) = \sum_{\mathbf{b}^1 \odot \mathbf{b}^2 = \mathbf{c}} g(\mathbf{b}^1) g(\mathbf{b}^2)$.
Substituting $\mathbf{b}^2 = \mathbf{b}^1 \odot \mathbf{c}$ (valid since
$\odot$ is invertible — each element is its own inverse in $\mathbb{Z}_2$):

$$W(\mathbf{c}) = \sum_{\mathbf{b}^1} g(\mathbf{b}^1) g(\mathbf{b}^1 \odot \mathbf{c})
= (g \star g)(\mathbf{c})$$

where $\star$ denotes correlation/convolution on $\mathbb{Z}_2^{2p+1}$ (they
coincide because every element is its own inverse). In the WHT domain:
$\hat{W} = \hat{g}^2$ (or $\hat{g} \cdot \overline{\hat{g}}$ if $g$ is complex,
but since convolution not correlation, it's $\hat{g}^2$).

**Subtlety**: if $g$ is complex-valued (which it may be since $f(\mathbf{b})$
involves mixer matrix elements with factors of $-i \sin\beta$), then
$\hat{W} = \hat{g}^2$, NOT $|\hat{g}|^2$. The autoconvolution uses $\hat{g}^2$;
autocorrelation would use $|\hat{g}|^2$. Since we want the sum
$\sum_{\mathbf{b}^1} g(\mathbf{b}^1) g(\mathbf{b}^1 \odot \mathbf{c})$
(not $g(\mathbf{b}^1) \overline{g(\mathbf{b}^1 \odot \mathbf{c})}$), this is
convolution and $\hat{W} = \hat{g}^2$ is correct.

### 3.4. Potential numerical issues

- **Cancellation**: the WHT of $\kappa$ involves $\cos$ of sums of $\Gamma_\ell / \sqrt{D}$
  values. At large $p$, these may have small magnitudes with large relative error.
  Needs empirical check.
- **Complex arithmetic**: $g$ is complex, so WHT must use complex arithmetic.
  Standard WHT implementations handle this.
- **Normalisation convention**: WHT conventions vary (factor of $1/2^n$ on forward
  vs inverse). Must be consistent. The un-normalised convention
  $\hat{g}(s) = \sum_x g(x) (-1)^{\langle s, x \rangle}$ with inverse
  $g(x) = \frac{1}{2^n} \sum_s \hat{g}(s) (-1)^{\langle s, x \rangle}$ gives
  $\widehat{f \ast g} = \hat{f} \cdot \hat{g}$ when convolution is defined as
  $(f \ast g)(x) = \sum_y f(y) g(x + y)$.  **But**: with normalised WHT
  ($\hat{g} = \frac{1}{\sqrt{2^n}} \sum_x g(x) (-1)^{\langle s,x\rangle}$)
  the convolution theorem picks up a factor $\sqrt{2^n}$. Must track carefully.

### 3.5. Does this generalise to $k > 3$?

For general $k$, the constraint fold sums over $k-1$ child branches. The parity
change-of-variables gives $\mathbf{c} = \mathbf{b}^1 \odot \cdots \odot \mathbf{b}^{k-1}$
and the marginal $W(\mathbf{c})$ is the $(k-1)$-fold convolution of $g$.

In the WHT domain: $\hat{W} = \hat{g}^{k-1}$. Cost: still $O(p \cdot 4^p)$ for
the WHT plus $O(4^p)$ for the power.

The $S(\mathbf{a})$ convolution with $\kappa$ is identical. So **the full
factorisation to $O(p \cdot 4^p)$ works for all $k$**, not just $k=3$.

---

## 4. Complete Algorithm (if the factorisation holds)

### Iteration step: compute $H_D^{(m)}$ from $H_D^{(m-1)}$

**Input**: branch tensor $H^{(m-1)} \in \mathbb{C}^{2^{2p+1}}$, angles $(\gamma, \beta)$, parameters $(k, D)$.

1. **Build dressed branch tensor**: $g(\mathbf{b}) = f(\mathbf{b}) \cdot H^{(m-1)}(\mathbf{b})$ — $O(4^p)$
2. **WHT of $g$**: $\hat{g} = \mathrm{WHT}(g)$ — $O(p \cdot 4^p)$
3. **$(k-1)$-fold autoconvolution**: $\hat{W} = \hat{g}^{k-1}$ (entry-wise power) — $O(4^p)$
4. **Inverse WHT**: $W = \mathrm{IWHT}(\hat{W})$ — $O(p \cdot 4^p)$
5. **Build cosine kernel**: $\kappa(\mathbf{d}) = \cos\!\left(\frac{\boldsymbol{\Gamma} \cdot \mathbf{d}}{\sqrt{D}}\right)$ for all $\mathbf{d}$ — $O(p \cdot 4^p)$
6. **WHT of $\kappa$**: $\hat{\kappa} = \mathrm{WHT}(\kappa)$ — $O(p \cdot 4^p)$ (can be precomputed once per angle set)
7. **Pointwise multiply**: $\hat{S} = \hat{\kappa} \cdot \hat{W}$ — $O(4^p)$
8. **Inverse WHT**: $S = \mathrm{IWHT}(\hat{S})$ — $O(p \cdot 4^p)$
9. **Element-wise power**: $H^{(m)}(\mathbf{a}) = S(\mathbf{a})^D$ — $O(4^p)$

**Per-step cost**: $O(p \cdot 4^p)$.
**Total for $p$ steps**: $O(p^2 \cdot 4^p)$.
**Memory**: $O(4^p)$.

### Final evaluation (Eq. 8.8)

The final sum (Eq. 8.8) has the same structure — sum over $k$ branch copies with
a $\sin$ kernel instead of $\cos$. The same WHT trick applies with
$\kappa_{\sin}(\mathbf{d}) = \sin\!\left(\frac{\boldsymbol{\Gamma} \cdot \mathbf{d}}{\sqrt{D}}\right)$.

Cost: $O(p \cdot 4^p)$ (one convolution).

### Overall complexity

| Component | Cost |
|-----------|------|
| Per iteration step | $O(p \cdot 4^p)$ |
| $p$ steps | $O(p^2 \cdot 4^p)$ |
| Final evaluation | $O(p \cdot 4^p)$ |
| **Total** | **$O(p^2 \cdot 4^p)$** |

This matches the cost of the $D \to \infty$ iteration — but is **exact** at finite $D$.

| $p$ | $p^2 \cdot 4^p$ | Feasibility |
|-----|-----------------|-------------|
| 5   | $2.6 \times 10^4$ | instant |
| 10  | $1.0 \times 10^8$ | seconds |
| 15  | $2.4 \times 10^{11}$ | minutes–hours |
| 20  | $6.4 \times 10^{14}$ | cluster |

If correct, p=15 becomes feasible — matching the Farhi et al. (2025) MaxCut reach.

---

## 5. Verification Protocol

### 5.1. Numerical verification at $p = 1$

At $p=1$: hyperindex has $2p+1 = 3$ bits, so $2^3 = 8$ values. The naive sum has
$8 \times 8 \times 8 = 512$ terms ($8$ parent values, $8 \times 8$ child pairs
each). The WHT operates on 8-element vectors.

**Procedure**:
1. Choose random $\gamma_1 \in [0, \pi]$, $\beta_1 \in [0, \pi/2]$.
2. Compute $H^{(0)} = \mathbf{1}$, so $g(\mathbf{b}) = f(\mathbf{b})$.
3. Compute $S_{\text{naive}}(\mathbf{a})$ by brute-force triple loop.
4. Compute $S_{\text{WHT}}(\mathbf{a})$ via the algorithm in §4.
5. Check: $\|S_{\text{naive}} - S_{\text{WHT}}\|_\infty < 10^{-12}$.

Repeat for 100 random angle sets.

**Also verify the full iteration**: compute $H^{(1)}$ both ways and verify the
final expectation value matches the Tier 1 brute-force simulator.

### 5.2. Numerical verification at $p = 2$

At $p=2$: hyperindex has 5 bits, $2^5 = 32$ values. Naive sum: $32^3 = 32768$
terms per parent value, $32 \times 32768 \approx 10^6$ total. WHT on 32-element
vectors.

**Procedure**: same as §5.1, but also:
- Verify across both iteration steps ($m=1$ and $m=2$).
- Compare final expectation value against Tier 1 at $p=2$ (if Tier 1 reaches —
  for k=3, D=4, p=2 requires 129 qubits in brute-force, so use smaller $(k,D)$
  or verify only the inner sum $S$).
- For k=2, D=3 at p=2: Tier 1 is feasible (14 qubits). Cross-validate.

### 5.3. Verification at $p = 3$

At $p=3$: hyperindex has 7 bits, $2^7 = 128$ values. Naive sum: $128^3 \approx 2 \times 10^6$
terms per parent value, total $\sim 2.7 \times 10^8$. Still feasible (seconds).

Cross-validate naive vs WHT for 10 random angle sets.

### 5.4. Decision tree

```
                    p=1 naive == WHT?
                   /               \
                 YES                NO
                 /                   \
          p=2 naive == WHT?      Identify failing step:
         /               \        - Is W(c) correct?
       YES                NO       - Is S(a) correct?
       /                   \       → debug & report
  p=3 naive == WHT?      Same diagnosis
  /               \
YES                NO
 |                  \
 v                   v
PROVEN              Partial break (identify p-dependent issue)
(write rigorous proof)
```

### 5.5. What to check if they disagree

If the WHT result disagrees with brute force:

1. **Isolate Step 2 vs Step 3**: Compute $W(\mathbf{c})$ by brute force and by
   WHT. If these disagree, the convolution theorem application is wrong
   (normalisation issue or misidentified group operation).

2. **Isolate the $\kappa$ convolution**: With correct $W$, compute $S$ by brute
   force summation and by WHT. If these disagree, the kernel is not a valid
   convolution (which would mean the algebra in §2 Step 3 is wrong).

3. **Check normalisation**: the most likely failure mode. Verify:
   - $\mathrm{IWHT}(\mathrm{WHT}(g)) = g$ (round-trip)
   - $\mathrm{IWHT}(\hat{f} \cdot \hat{g})$ equals the convolution $\sum_y f(y) g(x \oplus y)$
     for known test functions

---

## 6. Partial Factorisation Alternatives

If the full factorisation fails for some reason, consider these partial approaches:

### 6.1. WHT for $W$ only, brute-force for $S$

Use WHT to compute $W(\mathbf{c})$ in $O(p \cdot 4^p)$, then compute $S(\mathbf{a})$
by brute force sum over $\mathbf{c}$: $O(4^{2p})$ total. This is already a
square-root speedup from $O(4^{3p})$.

### 6.2. Complex exponential approach

Write $\cos(\theta) = \frac{1}{2}(e^{i\theta} + e^{-i\theta})$ and split:

$$S(\mathbf{a}) = \frac{1}{2} \sum_{\mathbf{c}} \left[e^{i \Gamma \cdot (a \odot c) / \sqrt{D}} + e^{-i \Gamma \cdot (a \odot c) / \sqrt{D}}\right] W(\mathbf{c})$$

$$= \frac{1}{2}\left[(\kappa_+ \ast W)(\mathbf{a}) + (\kappa_- \ast W)(\mathbf{a})\right]$$

where $\kappa_\pm(\mathbf{d}) = e^{\pm i \boldsymbol{\Gamma} \cdot \mathbf{d} / \sqrt{D}}$.
These are complex exponentials, and:

$$e^{i \boldsymbol{\Gamma} \cdot \mathbf{d} / \sqrt{D}} = \prod_\ell e^{i \Gamma_\ell d_\ell / \sqrt{D}}$$

This DOES factorise over rounds! So $\kappa_+$ is a character of $\mathbb{Z}_2^{2p+1}$
(well, not exactly — $\kappa_+(\mathbf{d})$ depends on the actual values
$d_\ell \in \{-1,+1\}$, not just the $\mathbb{Z}_2$ class). But it is a
multiplicative function, which means its WHT has a known closed form. This could
simplify computation but doesn't change the asymptotic cost.

### 6.3. Truncated WHT

If only low-order Walsh coefficients of $g$ are significant (spectral concentration),
truncate the WHT to the largest coefficients. This gives an approximation with
tunable accuracy. Relevant if exact results are not needed.

### 6.4. Round-by-round decomposition

The cosine can be expanded as:

$$\cos\!\left(\sum_\ell t_\ell\right) = \operatorname{Re}\!\left(\prod_\ell e^{it_\ell}\right) = \sum_{S \subseteq [2p+1]} (-1)^{|S|/2} \prod_{\ell \in S} \sin(t_\ell) \prod_{\ell \notin S} \cos(t_\ell)$$

(via the real part of product expansion). This has $2^{2p+1}$ terms, each of which
factorises over rounds. For each such term the sum over $(\mathbf{b}^1, \mathbf{b}^2)$
factors. Cost: $O(4^p \cdot 2^{2p+1}) = O(4^p \cdot 4^p) = O(16^p)$ — worse than
the WHT approach but better than $O(64^p)$.

Actually this expansion is not quite right. The correct identity is:

$$\cos\!\left(\sum_\ell t_\ell\right) = \operatorname{Re}\prod_\ell (\cos t_\ell + i \sin t_\ell)$$

Expanding the product gives $2^{2p+1}$ terms, each a product of cosines and sines
with a phase $i^{(\text{number of sines})}$. The real part selects terms where the
number of sines is even. This gives $2^{2p}$ terms, each factoring over rounds.
Total: $O(4^p \cdot 4^p) = O(16^p)$.

---

## 7. Literature and Relevant Techniques

### 7.1. Walsh-Hadamard Transform and convolution theorem

The WHT on $\mathbb{Z}_2^n$ is:

$$\hat{f}(s) = \sum_{x \in \mathbb{Z}_2^n} f(x)\, (-1)^{\langle s, x \rangle}$$

with inverse $f(x) = \frac{1}{2^n} \sum_s \hat{f}(s)\, (-1)^{\langle s, x \rangle}$.

Convolution: $(f \ast g)(x) = \sum_y f(y)\, g(x \oplus y)$.

Convolution theorem: $\widehat{f \ast g} = \hat{f} \cdot \hat{g}$.

This is standard; see e.g. O'Donnell, *Analysis of Boolean Functions* (2014), Ch. 1.

### 7.2. Boolean Fourier analysis

The functions $g, \kappa, W$ are all real- or complex-valued functions on
$\{-1,+1\}^{2p+1} \cong \mathbb{Z}_2^{2p+1}$. The Walsh-Fourier expansion is:

$$g(x) = \sum_{S \subseteq [2p+1]} \hat{g}(S)\, \chi_S(x)$$

where $\chi_S(x) = \prod_{i \in S} x_i$ are the characters. Convolution in this
basis is pointwise multiplication of Fourier coefficients.

Reference: O'Donnell, *Analysis of Boolean Functions*, Cambridge University Press (2014).

### 7.3. Tensor network contractions via FFT-type transforms

Using spectral transforms to speed up tensor contractions is a well-known technique:
- **MERA** (Multiscale Entanglement Renormalisation Ansatz): uses disentanglers
  that can be viewed as FFT-like operations.
- **Belief propagation** on factor graphs: the sum-product algorithm uses a similar
  parity decomposition for XOR constraints (see: Richardon-Urbanke,
  *Modern Coding Theory*, Ch. 2).

### 7.4. Fast subset convolution

Björklund et al. (2007) give $O(2^n n^2)$ algorithms for subset convolution over
the OR semiring. Our problem uses XOR (group convolution), which is simpler — the
standard WHT suffices.

### 7.5. QAOA-specific references

- Basso et al. 2021 (arXiv:2110.14206), §8: states the $O(p \cdot 4^{pq})$ cost
  but does not mention WHT factorisation.
- Farhi et al. 2025 (arXiv:2503.12789): achieves $O(p \cdot 4^p)$ for $k=2$ via
  a different trick (the 2-body gate factorises directly). Does not address $k \geq 3$.
- No published work (as of March 2026) appears to use WHT to reduce the constraint
  fold cost for $k \geq 3$ QAOA on hypergraphs.

---

## 8. Summary and Recommendation

### The argument appears to be correct.

The factorisation rests on two applications of the convolution theorem on
$\mathbb{Z}_2^{2p+1}$:

1. $W = g \ast g$ → $\hat{W} = \hat{g}^2$ — **uncontroversial**.
2. $S = \kappa \ast W$ → $\hat{S} = \hat{\kappa} \cdot \hat{W}$ — **also
   uncontroversial**, since $\kappa(\mathbf{a} \odot \mathbf{c})$ is manifestly
   a convolution kernel.

The "cos doesn't factorise" concern was about round-by-round factorisation, which
is NOT what the WHT approach does. The WHT treats the cosine kernel as a black-box
function on $\{-1,+1\}^{2p+1}$ and convolves it with $W$. No factorisation of the
cosine is needed.

### Next steps

1. **Numerical verification at $p=1,2,3$** (§5) to confirm the algebra is correct
   and there are no normalisation or sign errors.
2. **Write a short proof** (1 page) formalising the argument for the paper.
3. **Implement in Julia** using FFTW or a custom in-place WHT.
4. **Benchmark**: verify $O(p^2 \cdot 4^p)$ scaling empirically.
5. **Push to p=10–15** for (k=3, D=4) and compare against DQI.

### Risk assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Normalisation error in WHT | Medium | Low (fixable) | Step 5.5 diagnosis |
| Numerical precision at large p | Low | Medium | Use Float128 or BigFloat |
| Fundamental algebra error | Very low | High | p=1 verification will catch it |
| Inapplicable to final sum (Eq 8.8) | Very low | Medium | Same structure as iteration |

---

## Appendix A: Notation Reference

| Symbol | Meaning |
|--------|---------|
| $k$ | Constraint arity (= Basso's $q$) |
| $D$ | Basso's branching factor (= our degree $-$ 1) |
| $p$ | QAOA depth |
| $\mathbf{a}, \mathbf{b}$ | Hyperindex vectors in $\{-1,+1\}^{2p+1}$ |
| $\odot$ | Entry-wise (Hadamard) product |
| $\boldsymbol{\Gamma}$ | Angle vector $(\gamma_1,\ldots,\gamma_p, 0, -\gamma_p,\ldots,-\gamma_1)$ |
| $f(\mathbf{b})$ | Product of mixer matrix elements |
| $g(\mathbf{b})$ | Dressed branch tensor: $f(\mathbf{b}) \cdot H^{(m-1)}(\mathbf{b})$ |
| $W(\mathbf{c})$ | Autoconvolution of $g$: $\hat{W} = \hat{g}^2$ |
| $\kappa(\mathbf{d})$ | Cosine kernel: $\cos(\boldsymbol{\Gamma} \cdot \mathbf{d} / \sqrt{D})$ |
| $S(\mathbf{a})$ | Constraint fold result: $\kappa \ast W$ |
| $\hat{\cdot}$ | Walsh-Hadamard transform |

## Appendix B: WHT Implementation Notes

For $n = 2p+1$ bits, the in-place WHT is:

```
for i in 0:(n-1)
    for j in 0:(2^n - 1)
        if j & (1 << i) == 0
            x = f[j]
            y = f[j | (1 << i)]
            f[j]            = x + y
            f[j | (1 << i)] = x - y
        end
    end
end
```

Inverse: same butterfly, then divide by $2^n$.

In Julia, use `using FFTW` — but FFTW doesn't have WHT. Options:
- **Hadamard.jl**: provides `hadamard(x)` for power-of-2 lengths
- **Custom implementation**: 10 lines, no dependencies
- **Via real FFT trick**: WHT on $\{-1,+1\}^n$ maps to DFT on $\mathbb{Z}_2^n$

For our sizes ($2^{2p+1}$ up to $2^{31}$ at $p=15$), a custom implementation is
fine. At $p=15$ the vector has $2^{31} \approx 2 \times 10^9$ entries — fits in
~16 GB of RAM (complex Float64).
