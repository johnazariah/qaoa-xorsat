# Root Interface Derivation for the Basso Finite-D Evaluator

> **Purpose**: Derive the exact root contraction formula connecting Basso branch tensors to the QAOA expectation value, in the repo's conventions.
> **Audience**: Developer agent implementing the root fold.
> **Date**: 22 March 2026

---

## 1. What the Paper Says

From Basso 2021, Eq. (8.16), the root evaluation after all branch sums have been absorbed into $H_D^{(p)}$ is:

$$\langle Z_1 Z_2 \cdots Z_q \rangle = \sum_{\mathbf{z}^1, \ldots, \mathbf{z}^q} z_1^{[0]} z_2^{[0]} \cdots z_q^{[0]} \exp\!\left(-\frac{i}{\sqrt{D}} \boldsymbol{\Gamma} \cdot (\mathbf{z}^1 \mathbf{z}^2 \cdots \mathbf{z}^q)\right) \prod_{j=1}^{q} f(\mathbf{z}^j) H_D^{(p)}(\mathbf{z}^j)$$

where:
- Each $\mathbf{z}^j$ is a $(2p+1)$-component vector with entries in $\{-1, +1\}$
- $z_j^{[0]}$ is the 0-th component (the "physical" / observable bit) of the $j$-th root variable
- $\boldsymbol{\Gamma} \cdot (\mathbf{z}^1 \mathbf{z}^2 \cdots \mathbf{z}^q) = \sum_{\ell} \Gamma_\ell \prod_{j=1}^{q} z_j^{[\ell]}$ is the entry-wise product dot
- $D$ here is Basso's $D$ = our `params.D - 1` (branching degree)

The second form uses $\sin$ instead of $\exp$ (due to the $z^{[0]}$ parity extraction):

$$= -i \sum_{\mathbf{z}^1, \ldots, \mathbf{z}^q} \sin\!\left(\frac{1}{\sqrt{D}} \boldsymbol{\Gamma} \cdot (\mathbf{z}^1 \cdots \mathbf{z}^q)\right) \prod_{j=1}^{q} z_j^{[0]} f(\mathbf{z}^j) H_D^{(p)}(\mathbf{z}^j)$$

## 2. Structure of the Root Sum

Define the **root message** for leg $j$:

$$m(\mathbf{z}^j) = z_j^{[0]} \cdot f(\mathbf{z}^j) \cdot H_D^{(p)}(\mathbf{z}^j)$$

And the **root kernel**:

$$\kappa(\mathbf{d}) = -i \sin\!\left(\frac{1}{\sqrt{D}} \boldsymbol{\Gamma} \cdot \text{spins}(\mathbf{d})\right)$$

where $\mathbf{d} = \mathbf{z}^1 \odot \mathbf{z}^2 \odot \cdots \odot \mathbf{z}^q$ (entry-wise product = XOR in binary).

Then the parity expectation is:

$$\langle Z_1 \cdots Z_q \rangle = \sum_{\mathbf{z}^1, \ldots, \mathbf{z}^q} \kappa(\mathbf{z}^1 \odot \cdots \odot \mathbf{z}^q) \prod_{j=1}^{q} m(\mathbf{z}^j)$$

This is a **q-fold XOR convolution** — exactly the same structure as the branch fold!

$$\langle Z_1 \cdots Z_q \rangle = \sum_{\mathbf{d}} \kappa(\mathbf{d}) \cdot (m \star m \star \cdots \star m)(\mathbf{d}) = \text{IWHT}\!\left(\hat{\kappa} \cdot \hat{m}^q\right)[0]$$

Wait — not quite index [0]. The sum is over all $\mathbf{d}$, weighted by $\kappa$. Let me be precise:

$$\langle Z_1 \cdots Z_q \rangle = \sum_{\mathbf{d}} \kappa(\mathbf{d}) \cdot W_q(\mathbf{d})$$

where $W_q = m^{\star q}$ is the q-fold auto-convolution. By Parseval/convolution theorem:

$$= \frac{1}{N} \sum_{\mathbf{s}} \hat{\kappa}(\mathbf{s}) \cdot \hat{m}(\mathbf{s})^q$$

where $N = 2^{2p+1}$ and $\hat{\cdot}$ is the (unnormalized) WHT.

Or equivalently:

$$= \text{sum}\!\left(\text{IWHT}\!\left(\hat{\kappa} \cdot \hat{m}^q\right)\right) \cdot \frac{1}{N}$$

Hmm, let me be more careful with the normalization. The XOR convolution is:

$$(f \star g)(d) = \sum_{a \oplus b = d} f(a) g(b) = \sum_a f(a) g(a \oplus d)$$

And the convolution theorem with the *unscaled* WHT ($\hat{f}(s) = \sum_a (-1)^{s \cdot a} f(a)$, $f(a) = \frac{1}{N} \sum_s (-1)^{s \cdot a} \hat{f}(s)$) says:

$$\widehat{f \star g} = \hat{f} \cdot \hat{g}$$

So:

$$\sum_d \kappa(d) \cdot W_q(d) = \frac{1}{N} \sum_s \hat{\kappa}(s) \cdot \hat{W}_q(s) = \frac{1}{N} \sum_s \hat{\kappa}(s) \cdot \hat{m}(s)^q$$

## 3. Does It Factorize Over Root Legs?

**No.** The root sum is a q-fold convolution, not a product. The $\sin$ kernel couples all q legs through the joint parity $\mathbf{z}^1 \odot \cdots \odot \mathbf{z}^q$. However, the q-fold convolution can be computed efficiently via WHT:

$$\langle Z_1 \cdots Z_q \rangle = \frac{1}{N} \sum_s \hat{\kappa}(s) \cdot \hat{m}(s)^q$$

Cost: one WHT of $\kappa$, one WHT of $m$, element-wise power, element-wise multiply, sum. Total: $O(N \log N) = O(p \cdot 4^p)$.

## 4. The Root Message

The message per root leg is:

$$m(a) = \text{basso\_root\_parity}(a) \cdot f(a) \cdot H_D^{(p)}(a)$$

where `basso_root_parity(a)` extracts $z^{[0]}$ (the physical bit, which is element $p+1$ in Basso's $(2p+1)$-vector using the ordering $\{p, p-1, \ldots, 1, 0, -1, \ldots, -p\}$).

This is already implemented as `basso_root_message` in the repo — it computes $f(a) \cdot H_D^{(p)}(a)$ but **does not include the $z^{[0]}$ parity factor**.

**Fix**: the root message must be $z^{[0]} \cdot f(a) \cdot H^{(p)}_D(a)$, i.e., multiply by `basso_root_parity(a)`.

## 5. The Root Kernel

$$\kappa(d) = -i \cdot \text{clause\_sign} \cdot \sin\!\left(\frac{\text{clause\_sign}}{\sqrt{D_\text{basso}}} \sum_\ell \Gamma_\ell \cdot \text{spin}(d_\ell)\right)$$

Wait — let me be careful about the sign. From Eq. (8.16):

$$\langle Z_1 \cdots Z_q \rangle = -i \sum \sin\!\left(\frac{1}{\sqrt{D}} \Gamma \cdot (\mathbf{z}^1 \cdots \mathbf{z}^q)\right) \prod_j z_j^{[0]} f(\mathbf{z}^j) H^{(p)}(\mathbf{z}^j)$$

The argument of $\sin$ is $\frac{1}{\sqrt{D}} \Gamma \cdot (\mathbf{z}^1 \cdots \mathbf{z}^q)$. Since $\mathbf{z}^1 \cdots \mathbf{z}^q$ means entry-wise product, and $d = z^1 \oplus z^2 \oplus \cdots \oplus z^q$ in binary, the spin of $d$ at position $\ell$ is $\prod_j z_j^{[\ell]}$.

So $\Gamma \cdot (\mathbf{z}^1 \cdots \mathbf{z}^q) = \Gamma \cdot \text{spins}(d)$.

Therefore:

$$\kappa(d) = -i \sin\!\left(\frac{1}{\sqrt{D_\text{basso}}} \sum_\ell \Gamma_\ell \cdot \text{spin}_\ell(d)\right)$$

For odd-clause conventions (MaxCut, clause_sign = -1), the cost function has an extra sign that modifies the $\Gamma$ or the final result. In the Basso setup, after gauge-fixing all $J=-1$, the expression (8.5) gives:

$$\frac{1}{|E|} \langle C^{XOR}_J \rangle = \frac{1}{2} - \frac{1}{2} \langle Z_1 \cdots Z_q \rangle$$

So the satisfaction fraction is $(1 - \langle Z_1 \cdots Z_q \rangle)/2$ regardless of clause_sign, because Basso has already absorbed the sign into the gauge fixing.

**But** the repo's Tier 1 evaluator uses `clause_sign` to handle this. We need to match conventions. For the Basso evaluator:

- All couplings are $J = -1$ (gauge-fixed)
- $\langle Z_1 \cdots Z_q \rangle$ is computed directly
- Satisfaction = $(1 + \text{clause\_sign} \cdot \langle Z_1 \cdots Z_q \rangle) / 2$

For MaxCut (clause_sign = -1): satisfaction = $(1 - \langle Z_1 Z_2 \rangle)/2$ ✓
For XORSAT (clause_sign = +1): satisfaction = $(1 + \langle Z_1 Z_2 Z_3 \rangle)/2$ ✓

## 6. The Complete Root Contraction (WHT form)

```
Input: branch tensor H_D^(p), angles, params
Output: ⟨Z₁···Zₖ⟩

1. Build root message:
   m(a) = basso_root_parity(a, p) * f(a) * H_D^(p)(a)
   [N = 2^{2p+1} entries]

2. Build root kernel:
   κ(d) = -i * sin( (1/√D_basso) * Σ_ℓ Γ_ℓ * spin_ℓ(d) )
   [N entries]

3. WHT both:
   m̂ = WHT(m)
   κ̂ = WHT(κ)

4. Combine:
   result = (1/N) * Σ_s κ̂(s) * m̂(s)^k

5. The parity expectation:
   ⟨Z₁···Zₖ⟩ = real(result)
   (imaginary part should be zero to machine precision)

6. Satisfaction fraction:
   c = (1 + clause_sign * ⟨Z₁···Zₖ⟩) / 2
```

## 7. Worked Example: k=3, D=2, p=1

At p=1:
- $2p+1 = 3$ bits per configuration → $N = 8$ configurations
- Basso ordering: $(z^{[1]}, z^{[0]}, z^{[-1]})$ with entries in $\{-1, +1\}$
- $D_\text{basso} = D - 1 = 1$
- Root bit: $z^{[0]}$ at position index 2 (1-indexed middle of 3)

$\Gamma = (\gamma_1, 0, -\gamma_1)$ for the (ket₁, physical, bra₁) slots.

The f function for p=1:

$$f(z^{[1]}, z^{[0]}, z^{[-1]}) = \frac{1}{2} \langle z^{[1]} | e^{i\beta_1 X} | z^{[0]} \rangle \langle z^{[0]} | e^{-i\beta_1 X} | z^{[-1]} \rangle$$

At specific angles γ=0.31, β=0.17, the brute-force evaluator gives:
- `parity_expectation(TreeParams(3, 2, 1), QAOAAngles([0.31], [0.17]); clause_sign=1)` =  some reference value

The Basso evaluator should match this exactly.

**Key implementation note**: For $D_\text{basso} = 1$ (i.e., $D = 2$), each variable has only 1 branching hyperedge, so the branch tensor after p=1 steps with the D-th power exponent being 1 is just the inner sum without exponentiation. This is the simplest non-trivial case.

## 8. What the Developer Needs to Change

Based on this analysis, the current `basso_root_parity_sum` needs:

1. **Include the parity factor in the root message**: multiply each $m(a)$ by `basso_root_parity(a, p)` (the $z^{[0]}$ spin value). Currently `basso_root_message` computes $f(a) \cdot H^{(p)}(a)$ but omits the $z^{[0]}$ factor.

2. **Use the correct root kernel**: $\kappa(d) = -i \sin(\frac{1}{\sqrt{D_\text{basso}}} \Gamma \cdot \text{spins}(d))$. Check whether `basso_root_problem_kernel` computes exactly this.

3. **Use WHT-based k-fold convolution at the root**: The sum is $\frac{1}{N} \sum_s \hat{\kappa}(s) \hat{m}(s)^k$. This is the same `basso_root_fold` pattern but verify the normalization.

4. **Verify against brute-force** at (k=3, D=2, p=1) and (k=2, D=3, p=1,2).

## 9. Summary Answers

**Does the root factorize?** No — it's a k-fold convolution, genuinely coupling all k legs through the sin kernel. But the WHT makes it O(p·4^p) regardless.

**Role of mixer at root**: The mixer is encoded entirely in the f(a) function. The root observable $Z_1 \cdots Z_k$ is diagonal — it contributes only the $z^{[0]}$ parity factor in the message. There is no separate "mixer dressing" of the observable; instead, the f(a) factors in the message already contain all mixer matrix elements from the complete-set insertion.

**Why local ansätze failed**: The root contraction is NOT a local operation on each leg independently. The sin kernel couples all k legs. Any factorized ansatz (per-leg observable × per-leg mixer) necessarily misses the cross-leg coupling. The correct object is the k-fold convolution $\sum_d \kappa(d) (m^{\star k})(d)$.

---

## 10. CRITICAL UPDATE: Convention Mismatch (Numerically Verified)

### The symptom

The formula from Section 6/7, applied literally to
`(k=3, D=2, p=1, γ=0.31, β=0.17, clause_sign=1)`, gives **-0.4089** instead
of the brute-force reference **0.2486**. Sweeping overall sign, clause_sign
insertion, and phase scale does not fix it.

### The diagnosis

The mismatch is **not at the root** — it's a **convention conflict between the
Basso branch iteration and the physical QAOA circuit**.

Basso 2021 defines a **rescaled** cost function (Eq. 8.3):

$$C_\text{Basso} = \frac{1}{\sqrt{D}} \sum_{(i_1,\ldots,i_q)} J_{i_1\ldots i_q} Z_{i_1} \cdots Z_{i_q}$$

The QAOA state $|\gamma, \beta\rangle$ in the paper is prepared using this $C_\text{Basso}$.
The inserted-complete-set derivation (Eq. 8.11) then gives the phase per clause as:

$$\exp\!\left(-\frac{i}{\sqrt{D}} \boldsymbol{\Gamma} \cdot (\mathbf{z}^{i_1} \cdots \mathbf{z}^{i_q})\right)$$

Meanwhile, the **physical** QAOA circuit (as implemented in `src/qaoa.jl`) uses:

$$\exp\!\left(-\frac{i \gamma}{2} \cdot \text{clause\_sign} \cdot Z_{i_1} \cdots Z_{i_q}\right)$$

per clause per round. In the complete-set picture this gives a phase scale of
$\gamma/2$ per Γ component, NOT $\gamma/\sqrt{D}$.

### The code situation

The repo's `basso_constraint_kernel` uses `phase_scale = inv(sqrt(float(branch_degree)))`,
i.e., Basso's $1/\sqrt{D}$ convention. But `QAOAAngles` contains the **physical**
$\gamma$ values. The branch tensor is therefore computed in a hybrid: Basso's
structure with physical angles, which is neither the Basso convention nor the
physical convention.

### Numerical evidence

At `(k=2, D=3, p=1, γ=0.31, β=0.17)` ($D_\text{basso}=2$):
- Root with $1/\sqrt{D}$ root phase: parity = -0.2188 (reference: -0.1740, error 0.045)
- Root with physical $-\gamma/2$ root phase: parity = 0.1573 (error 0.331)
- Neither matches.

At `(k=3, D=2, p=1)` ($D_\text{basso}=1$, so $1/\sqrt{D}=1$):
- All scale variants give -0.409 vs reference 0.249.
- The $-0.5\gamma$ variant gives 0.215 (closest, error 0.034, but still wrong).

### The fix (two options)

**Option A — Rescale angles at the interface:**

Map physical $\gamma_\text{phys}$ to Basso $\gamma_\text{B}$ before the iteration:

$$\gamma_\text{B} = \gamma_\text{phys} \cdot \frac{\sqrt{D}}{2}$$

so that $\gamma_\text{B} / \sqrt{D} = \gamma_\text{phys} / 2$ at every constraint.
Then use Basso's $1/\sqrt{D}$ convention throughout (branch and root).
The $\sqrt{D}$ factors cancel and the physical circuit is recovered.

**Option B — Rewrite iteration in physical convention (recommended):**

Replace `phase_scale = inv(sqrt(float(branch_degree)))` with `phase_scale = 0.5`
everywhere — in the branch kernel AND in the root kernel. The iteration algebra
is identical; only the phase scale changes. This avoids the Basso normalization
entirely and matches the oracle directly.

The branch tensor power `.^ D_basso` is unaffected (it accounts for identical
branches, not the phase convention). The f(a) function is also unaffected (it
encodes mixer matrix elements, which use the physical β directly).

### What the developer should do

1. Create a parallel `physical_constraint_kernel` with `phase_scale = 0.5`
2. Create `physical_branch_tensor_step` using that kernel
3. Create `physical_root_kernel` also using `phase_scale = 0.5`
4. Test: `physical_root_fold` at (k=3, D=2, p=1) must give 0.2486
5. Test: at (k=2, D=3, p=1) must give -0.1740 (MaxCut parity)
6. If both match, replace the Basso-scaled iteration with the physical one
7. The WHT acceleration is unaffected — it operates on the kernel/message
   structure, not on the phase scale value
