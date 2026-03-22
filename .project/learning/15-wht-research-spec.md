# WHT Factorisation of the k-Body Constraint Sum: Research Specification

> **Date**: 22 March 2026
> **Status**: Research — rigorous analysis, no code
> **References**:
> - Basso et al. 2021, §8.2, Eq. (8.7): `papers/text/basso2021-qaoa-high-depth.txt`
> - Learning file 14: `learning/14-deep-dive-basso2021.md`
> - Learning file 12: `learning/12-transfer-contraction-k-body.md`
> - P1.3 spec: `specs/P1.3-contraction.md`

---

## 1. Problem Statement

We compute exact QAOA expectation values on D-regular Max-k-XORSAT at depth p.
The core bottleneck is the Basso finite-D iteration (Eq. 8.7, arXiv:2110.14206).
For k=3, one step of the iteration is:

$$H_D^{(m)}(\mathbf{a}) = \left[\sum_{\mathbf{b}^1, \mathbf{b}^2} \cos\!\left(\frac{\boldsymbol{\Gamma} \cdot (\mathbf{a} \odot \mathbf{b}^1 \odot \mathbf{b}^2)}{\sqrt{D}}\right) g(\mathbf{b}^1)\, g(\mathbf{b}^2)\right]^D$$

where:
- $\mathbf{a}, \mathbf{b}^1, \mathbf{b}^2 \in \{-1,+1\}^{2p+1}$, so each takes $2^{2p+1}$ values
- $g(\mathbf{b}) = f(\mathbf{b})\, H_D^{(m-1)}(\mathbf{b})$, a known function tabulated as a vector of $2^{2p+1}$ values
- $\boldsymbol{\Gamma} = (\gamma_1, \ldots, \gamma_p, 0, -\gamma_p, \ldots, -\gamma_1) \in \mathbb{R}^{2p+1}$
- $\odot$ denotes entry-wise (Hadamard) product
- The exponent $D$ is (our degree $-$ 1), i.e., Basso's $D$ convention

**Naive cost**: For each of the $2^{2p+1}$ values of $\mathbf{a}$, sum over $2^{2p+1} \times 2^{2p+1}$ pairs $(\mathbf{b}^1, \mathbf{b}^2)$. Total: $O(2^{3(2p+1)}) = O(8^{2p+1}) = O(64^p)$ per iteration step.

**Target**: Reduce to $O(p \cdot 4^p)$, matching the k=2 (MaxCut) cost.

### The Mathematical Question

> **Can the inner sum $S(\mathbf{a}) = \sum_{\mathbf{b}^1, \mathbf{b}^2} \cos\!\left(\frac{\boldsymbol{\Gamma} \cdot (\mathbf{a} \odot \mathbf{b}^1 \odot \mathbf{b}^2)}{\sqrt{D}}\right) g(\mathbf{b}^1)\, g(\mathbf{b}^2)$ be evaluated for all $\mathbf{a}$ in $O(p \cdot 4^p)$ time?**

---

## 2. Algebraic Setup

### 2.1. The Group

The domain $\{-1,+1\}^{2p+1}$ with entry-wise multiplication $\odot$ forms an abelian group isomorphic to $(\mathbb{Z}/2\mathbb{Z})^{2p+1}$. We denote this group $G$, with $|G| = 2^{2p+1}$.

**Convolution** on $G$: For functions $\phi, \psi : G \to \mathbb{C}$,

$$(\phi * \psi)(\mathbf{c}) = \sum_{\mathbf{x} \in G} \phi(\mathbf{x})\, \psi(\mathbf{x}^{-1} \odot \mathbf{c})$$

Since every element is its own inverse in this group ($\mathbf{x}^{-1} = \mathbf{x}$), this simplifies to:

$$(\phi * \psi)(\mathbf{c}) = \sum_{\mathbf{x} \in G} \phi(\mathbf{x})\, \psi(\mathbf{x} \odot \mathbf{c})$$

This is also called **correlation**, and on $\mathbb{Z}_2^n$ convolution and correlation coincide.

**Walsh-Hadamard Transform (WHT)**: The characters of $G$ are $\chi_\mathbf{s}(\mathbf{x}) = \prod_\ell x_\ell^{s_\ell}$ for $\mathbf{s} \in \{0,1\}^{2p+1}$, or equivalently $\chi_\mathbf{s}(\mathbf{x}) = (-1)^{\mathbf{s} \cdot \mathbf{t}}$ where $x_\ell = (-1)^{t_\ell}$. The Fourier transform is:

$$\hat{\phi}(\mathbf{s}) = \sum_{\mathbf{x} \in G} \phi(\mathbf{x})\, \chi_\mathbf{s}(\mathbf{x})$$

The **convolution theorem** states: $\widehat{\phi * \psi} = \hat{\phi} \cdot \hat{\psi}$ (pointwise product).

The WHT and its inverse can be computed in $O(n \cdot 2^n)$ time via the butterfly algorithm, where $n = 2p+1$.

### 2.2. Step 1 — Factor by $\mathbf{c} = \mathbf{b}^1 \odot \mathbf{b}^2$

**Claim**: Define $W(\mathbf{c}) = \sum_{\mathbf{b}^1 \odot \mathbf{b}^2 = \mathbf{c}} g(\mathbf{b}^1)\, g(\mathbf{b}^2)$.

Then:

$$S(\mathbf{a}) = \sum_{\mathbf{c} \in G} \cos\!\left(\frac{\boldsymbol{\Gamma} \cdot (\mathbf{a} \odot \mathbf{c})}{\sqrt{D}}\right) W(\mathbf{c})$$

**Proof**: Substitute $\mathbf{c} = \mathbf{b}^1 \odot \mathbf{b}^2$ and rewrite the double sum by grouping all pairs $(\mathbf{b}^1, \mathbf{b}^2)$ that yield the same $\mathbf{c}$:

$$S(\mathbf{a}) = \sum_{\mathbf{b}^1, \mathbf{b}^2} \cos\!\left(\frac{\boldsymbol{\Gamma} \cdot (\mathbf{a} \odot \mathbf{b}^1 \odot \mathbf{b}^2)}{\sqrt{D}}\right) g(\mathbf{b}^1)\, g(\mathbf{b}^2) = \sum_\mathbf{c} \cos\!\left(\frac{\boldsymbol{\Gamma} \cdot (\mathbf{a} \odot \mathbf{c})}{\sqrt{D}}\right) \sum_{\mathbf{b}^1 \odot \mathbf{b}^2 = \mathbf{c}} g(\mathbf{b}^1)\, g(\mathbf{b}^2)$$

This is valid because the cosine depends on $(\mathbf{b}^1, \mathbf{b}^2)$ only through $\mathbf{c} = \mathbf{b}^1 \odot \mathbf{b}^2$. $\square$

### 2.3. Step 2 — $W$ is a Convolution

**Claim**: $W = g * g$ (convolution on $G$).

**Proof**: By definition of convolution on our group (where $\mathbf{x}^{-1} = \mathbf{x}$):

$$(g * g)(\mathbf{c}) = \sum_{\mathbf{b}^1 \in G} g(\mathbf{b}^1)\, g(\mathbf{b}^1 \odot \mathbf{c})$$

We need this to equal $\sum_{\mathbf{b}^1 \odot \mathbf{b}^2 = \mathbf{c}} g(\mathbf{b}^1)\, g(\mathbf{b}^2)$.

The constraint $\mathbf{b}^1 \odot \mathbf{b}^2 = \mathbf{c}$ is equivalent to $\mathbf{b}^2 = \mathbf{b}^1 \odot \mathbf{c}$ (multiply both sides by $\mathbf{b}^1$, using $\mathbf{b}^1 \odot \mathbf{b}^1 = \mathbf{1}$). So:

$$W(\mathbf{c}) = \sum_{\mathbf{b}^1} g(\mathbf{b}^1)\, g(\mathbf{b}^1 \odot \mathbf{c}) = (g * g)(\mathbf{c}) \quad \square$$

**Consequence**: By the convolution theorem, $\hat{W} = \hat{g}^2$ (pointwise square). So $W$ can be computed by:

1. Compute $\hat{g}$ via WHT: $O((2p+1) \cdot 2^{2p+1})$ time
2. Square pointwise: $\hat{W}(\mathbf{s}) = [\hat{g}(\mathbf{s})]^2$, $O(2^{2p+1})$ time
3. Inverse WHT to get $W$: $O((2p+1) \cdot 2^{2p+1})$ time

**Total for $W$**: $O(p \cdot 4^p)$. $\checkmark$

### 2.4. Step 3 — Is the Remaining Sum a Convolution?

We now need to evaluate, for all $\mathbf{a} \in G$:

$$S(\mathbf{a}) = \sum_{\mathbf{c} \in G} h(\mathbf{a} \odot \mathbf{c})\, W(\mathbf{c})$$

where $h(\mathbf{x}) = \cos\!\left(\frac{\boldsymbol{\Gamma} \cdot \mathbf{x}}{\sqrt{D}}\right)$.

**This is precisely the convolution** $(h * W)(\mathbf{a})$:

$$(h * W)(\mathbf{a}) = \sum_{\mathbf{c}} h(\mathbf{a} \odot \mathbf{c})\, W(\mathbf{c}) = S(\mathbf{a})$$

So if we can compute $\hat{h}$, then $\hat{S} = \hat{h} \cdot \hat{W}$ and $S$ follows from an inverse WHT.

The question reduces to: **Can $\hat{h}$ be computed efficiently?**

### 2.5. Step 4 — Computing $\hat{h}$ (The Hard Part)

We need:

$$\hat{h}(\mathbf{s}) = \sum_{\mathbf{x} \in G} \cos\!\left(\frac{\boldsymbol{\Gamma} \cdot \mathbf{x}}{\sqrt{D}}\right) \chi_\mathbf{s}(\mathbf{x})$$

where $\boldsymbol{\Gamma} \cdot \mathbf{x} = \sum_{\ell=0}^{2p} \Gamma_\ell x_\ell$ and $\chi_\mathbf{s}(\mathbf{x}) = \prod_\ell x_\ell^{s_\ell}$.

This is just the WHT of the function $h$, which takes $O(p \cdot 4^p)$ time by the butterfly algorithm. Since $h$ has $2^{2p+1}$ entries and the WHT operates in $O(n \cdot 2^n)$ with $n = 2p+1$, this is $O(p \cdot 4^p)$.

However, before reaching for the generic butterfly, let us check whether $\hat{h}$ has a **closed form**, since $h$ has special structure. This is instructive for understanding why the factorisation works.

---

## 3. Detailed Analysis of $\hat{h}$

### 3.1. The Exponential Route

Write $\cos(\theta) = \frac{1}{2}(e^{i\theta} + e^{-i\theta})$ with $\theta = \boldsymbol{\Gamma} \cdot \mathbf{x} / \sqrt{D}$. Define:

$$h_+(\mathbf{x}) = e^{i \boldsymbol{\Gamma} \cdot \mathbf{x} / \sqrt{D}}, \qquad h_-(\mathbf{x}) = e^{-i \boldsymbol{\Gamma} \cdot \mathbf{x} / \sqrt{D}}$$

So $h = \frac{1}{2}(h_+ + h_-)$ and $\hat{h} = \frac{1}{2}(\hat{h}_+ + \hat{h}_-)$.

Now, $h_+$ factors over positions:

$$h_+(\mathbf{x}) = e^{i \sum_\ell \Gamma_\ell x_\ell / \sqrt{D}} = \prod_\ell e^{i \Gamma_\ell x_\ell / \sqrt{D}}$$

**This is the critical factorisation.** Since $x_\ell \in \{-1, +1\}$, each factor takes only two values:

$$e^{i \Gamma_\ell x_\ell / \sqrt{D}} = \begin{cases} e^{i \Gamma_\ell / \sqrt{D}} & \text{if } x_\ell = +1 \\ e^{-i \Gamma_\ell / \sqrt{D}} & \text{if } x_\ell = -1 \end{cases}$$

### 3.2. WHT of a Factored Function

When $\phi(\mathbf{x}) = \prod_\ell \phi_\ell(x_\ell)$ factors over positions, its WHT also factors:

$$\hat{\phi}(\mathbf{s}) = \prod_\ell \hat{\phi}_\ell(s_\ell)$$

where $\hat{\phi}_\ell(s) = \sum_{x \in \{-1,+1\}} \phi_\ell(x) \cdot x^s$ is a 1D WHT (2-point).

**Proof**:

$$\hat{\phi}(\mathbf{s}) = \sum_{\mathbf{x}} \prod_\ell \phi_\ell(x_\ell) \cdot \prod_\ell x_\ell^{s_\ell} = \sum_{\mathbf{x}} \prod_\ell \left[\phi_\ell(x_\ell) \cdot x_\ell^{s_\ell}\right] = \prod_\ell \left[\sum_{x_\ell \in \{-1,+1\}} \phi_\ell(x_\ell) x_\ell^{s_\ell}\right]$$

The sum over $\mathbf{x}$ splits into independent sums because the summand factors. $\square$

### 3.3. WHT of $h_+$

Apply §3.2 with $\phi_\ell(x) = e^{i \Gamma_\ell x / \sqrt{D}}$:

$$\hat{h}_{+,\ell}(0) = e^{i\Gamma_\ell/\sqrt{D}} + e^{-i\Gamma_\ell/\sqrt{D}} = 2\cos(\Gamma_\ell / \sqrt{D})$$

$$\hat{h}_{+,\ell}(1) = e^{i\Gamma_\ell/\sqrt{D}} \cdot (+1) + e^{-i\Gamma_\ell/\sqrt{D}} \cdot (-1) = 2i\sin(\Gamma_\ell / \sqrt{D})$$

So:

$$\hat{h}_+(\mathbf{s}) = \prod_\ell \hat{h}_{+,\ell}(s_\ell) = \prod_\ell \begin{cases} 2\cos(\Gamma_\ell/\sqrt{D}) & \text{if } s_\ell = 0 \\ 2i\sin(\Gamma_\ell/\sqrt{D}) & \text{if } s_\ell = 1 \end{cases}$$

$$= 2^{2p+1} \prod_{\ell: s_\ell = 0} \cos(\Gamma_\ell/\sqrt{D}) \prod_{\ell: s_\ell = 1} i\sin(\Gamma_\ell/\sqrt{D})$$

Similarly:

$$\hat{h}_-(\mathbf{s}) = 2^{2p+1} \prod_{\ell: s_\ell = 0} \cos(\Gamma_\ell/\sqrt{D}) \prod_{\ell: s_\ell = 1} (-i)\sin(\Gamma_\ell/\sqrt{D})$$

### 3.4. WHT of $h = \frac{1}{2}(h_+ + h_-)$

$$\hat{h}(\mathbf{s}) = \frac{1}{2}\left(\hat{h}_+(\mathbf{s}) + \hat{h}_-(\mathbf{s})\right)$$

Let $|\mathbf{s}| = \sum_\ell s_\ell$ (Hamming weight). The $i^{|\mathbf{s}|}$ and $(-i)^{|\mathbf{s}|}$ terms combine:

$$\hat{h}(\mathbf{s}) = 2^{2p} \prod_{\ell: s_\ell = 0} \cos(\Gamma_\ell/\sqrt{D}) \prod_{\ell: s_\ell = 1} \sin(\Gamma_\ell/\sqrt{D}) \cdot \left(i^{|\mathbf{s}|} + (-i)^{|\mathbf{s}|}\right)$$

Now:

$$i^n + (-i)^n = i^n(1 + (-1)^n) = \begin{cases} 2i^n & \text{if } n \text{ even} \\ 0 & \text{if } n \text{ odd} \end{cases}$$

And for even $n$: $i^n = (-1)^{n/2}$.

**Result**:

$$\hat{h}(\mathbf{s}) = \begin{cases} 2^{2p+1} (-1)^{|\mathbf{s}|/2} \prod_{\ell: s_\ell = 0} \cos(\Gamma_\ell/\sqrt{D}) \prod_{\ell: s_\ell = 1} \sin(\Gamma_\ell/\sqrt{D}) & \text{if } |\mathbf{s}| \text{ even} \\ 0 & \text{if } |\mathbf{s}| \text{ odd} \end{cases}$$

**This is a closed-form expression.** Each $\hat{h}(\mathbf{s})$ can be computed in $O(p)$ time, and there are $2^{2p+1}$ values of $\mathbf{s}$, so computing all $\hat{h}$ values takes $O(p \cdot 4^p)$.

However, we don't even need the closed form — a standard WHT butterfly on the $2^{2p+1}$ entries of $h$ costs $O(p \cdot 4^p)$ regardless. The closed form is useful for verification and for understanding the sparsity (half the spectrum is zero).

---

## 4. The Complete Algorithm

Combining §2.3, §2.4, and §3.4, the algorithm for computing $S(\mathbf{a})$ for all $\mathbf{a}$ is:

### Algorithm: WHT-accelerated constraint fold

**Input**: $g : G \to \mathbb{C}$ (tabulated, $2^{2p+1}$ values), angles $\boldsymbol{\Gamma}$, degree $D$

**Output**: $S(\mathbf{a})$ for all $\mathbf{a} \in G$

1. **Compute $\hat{g}$** via forward WHT of $g$. Cost: $O(p \cdot 4^p)$.

2. **Compute $\hat{W}$** by pointwise squaring: $\hat{W}(\mathbf{s}) = [\hat{g}(\mathbf{s})]^2$. Cost: $O(4^p)$.

3. **Compute $\hat{h}$** either by:
   - (a) Tabulating $h(\mathbf{x}) = \cos(\boldsymbol{\Gamma} \cdot \mathbf{x} / \sqrt{D})$ for all $\mathbf{x}$, then forward WHT. Cost: $O(p \cdot 4^p)$.
   - (b) Using the closed form from §3.4 directly. Cost: $O(p \cdot 4^p)$.

4. **Compute $\hat{S}$** by pointwise product: $\hat{S}(\mathbf{s}) = \hat{h}(\mathbf{s}) \cdot \hat{W}(\mathbf{s})$. Cost: $O(4^p)$.

5. **Recover $S$** via inverse WHT. Cost: $O(p \cdot 4^p)$.

6. **Raise to power $D$**: $H_D^{(m)}(\mathbf{a}) = [S(\mathbf{a})]^D$ pointwise. Cost: $O(4^p)$.

**Total cost per iteration step**: $O(p \cdot 4^p)$.

Over $p$ iteration steps: $O(p^2 \cdot 4^p)$.

**Memory**: $O(4^p)$ (several vectors of size $2^{2p+1}$).

---

## 5. Rigorous Verification: Does $\text{Re}(\cdot)$ Commute With the Convolution?

This is the key concern raised in the problem statement. Let us be precise about what happens.

### 5.1. The Concern

The concern was: "$\cos(\sum_\ell t_\ell) \neq \prod_\ell \cos(t_\ell)$, so the cosine doesn't factor over positions. Does taking Re() break the convolution structure?"

### 5.2. Why It Does NOT Break

The convolution $S(\mathbf{a}) = \sum_\mathbf{c} h(\mathbf{a} \odot \mathbf{c})\, W(\mathbf{c})$ is a **linear operation** on $h$. Since $h = \frac{1}{2}(h_+ + h_-)$ and convolution distributes over addition:

$$S(\mathbf{a}) = \frac{1}{2} \sum_\mathbf{c} h_+(\mathbf{a} \odot \mathbf{c})\, W(\mathbf{c}) + \frac{1}{2} \sum_\mathbf{c} h_-(\mathbf{a} \odot \mathbf{c})\, W(\mathbf{c})$$

$$= \frac{1}{2}(h_+ * W)(\mathbf{a}) + \frac{1}{2}(h_- * W)(\mathbf{a})$$

Each of $h_+ * W$ and $h_- * W$ can be computed via WHT. The $\text{Re}(\cdot)$ operation was applied **before** the convolution (in the definition of $h$), not after. So there is no commutativity issue.

More precisely: we never need $h_+$ or $h_-$ to individually have real WHTs. The sum $h = \frac{1}{2}(h_+ + h_-)$ is a real-valued function, and the WHT of a real function is well-defined. The convolution theorem applies to $h$ directly — no step requires that the exponential and the convolution exchange order with a real-part operation.

### 5.3. What Would Break

The concern would be valid if we were trying to do something like:

$$\text{Re}\left(\prod_\ell (\text{something}_\ell)\right) = \prod_\ell \text{Re}(\text{something}_\ell)$$

This is indeed false. But our algorithm never requires this. We compute $h(\mathbf{x}) = \cos(\boldsymbol{\Gamma} \cdot \mathbf{x} / \sqrt{D})$ as a **black-box tabulated function** and take its WHT. The internal structure of $h$ (that it's a cosine of a sum) is only exploited for the optional closed-form computation of $\hat{h}$ in §3.4, which is independently verified.

### 5.4. Summary

**The factorisation works.** There is no obstruction. The cosine-of-a-sum structure is handled cleanly by:
1. Recognising that $S(\mathbf{a})$ is a convolution of $h$ and $W$ over $G$.
2. Using the convolution theorem: $\hat{S} = \hat{h} \cdot \hat{W}$.
3. Computing $\hat{h}$ and $\hat{W}$ each in $O(p \cdot 4^p)$ time.

No step requires the cosine to factor over positions. The WHT handles the cosine as a general function.

---

## 6. Formal Proof of Correctness

**Theorem.** For all $\mathbf{a} \in \{-1,+1\}^{2p+1}$:

$$S(\mathbf{a}) = \sum_{\mathbf{b}^1, \mathbf{b}^2 \in G} \cos\!\left(\frac{\boldsymbol{\Gamma} \cdot (\mathbf{a} \odot \mathbf{b}^1 \odot \mathbf{b}^2)}{\sqrt{D}}\right) g(\mathbf{b}^1)\, g(\mathbf{b}^2)$$

can be computed for all $\mathbf{a}$ simultaneously in $O(p \cdot 4^p)$ time and $O(4^p)$ space.

**Proof.**

*Step 1.* Group the double sum by $\mathbf{c} = \mathbf{b}^1 \odot \mathbf{b}^2$:

$$S(\mathbf{a}) = \sum_{\mathbf{c} \in G} h(\mathbf{a} \odot \mathbf{c})\, W(\mathbf{c})$$

where $h(\mathbf{x}) = \cos(\boldsymbol{\Gamma} \cdot \mathbf{x} / \sqrt{D})$ and $W(\mathbf{c}) = \sum_{\mathbf{b}^1 \odot \mathbf{b}^2 = \mathbf{c}} g(\mathbf{b}^1)\, g(\mathbf{b}^2)$.

This rearrangement is valid because the cosine depends only on $\mathbf{a} \odot \mathbf{c}$, not on $\mathbf{b}^1, \mathbf{b}^2$ individually (proven in §2.2).

*Step 2.* $W = g * g$ is a convolution on $G = (\{-1,+1\}^{2p+1}, \odot)$:

$$W(\mathbf{c}) = \sum_{\mathbf{b}^1} g(\mathbf{b}^1)\, g(\mathbf{b}^1 \odot \mathbf{c}) = (g * g)(\mathbf{c})$$

proven in §2.3. By the convolution theorem, $\hat{W} = \hat{g}^2$.

Cost to compute $W$: one forward WHT of $g$, one pointwise square, one inverse WHT. Total: $O(p \cdot 4^p)$.

*Step 3.* $S = h * W$ is a convolution on $G$:

$$S(\mathbf{a}) = \sum_{\mathbf{c}} h(\mathbf{a} \odot \mathbf{c})\, W(\mathbf{c}) = (h * W)(\mathbf{a})$$

By the convolution theorem, $\hat{S} = \hat{h} \cdot \hat{W}$.

Cost: forward WHT of $h$ (or closed form from §3.4): $O(p \cdot 4^p)$. Pointwise multiply: $O(4^p)$. Inverse WHT: $O(p \cdot 4^p)$.

*Total*: $O(p \cdot 4^p)$. Memory: $O(4^p)$ (constant number of arrays of size $|G| = 2^{2p+1}$). $\square$

---

## 7. Generalisation to Arbitrary $k$

For general $k$, the inner sum involves $(k-1)$ children $\mathbf{b}^1, \ldots, \mathbf{b}^{k-1}$:

$$S(\mathbf{a}) = \sum_{\mathbf{b}^1, \ldots, \mathbf{b}^{k-1}} \cos\!\left(\frac{\boldsymbol{\Gamma} \cdot (\mathbf{a} \odot \mathbf{b}^1 \odot \cdots \odot \mathbf{b}^{k-1})}{\sqrt{D}}\right) \prod_{i=1}^{k-1} g(\mathbf{b}^i)$$

Define $\mathbf{c} = \mathbf{b}^1 \odot \cdots \odot \mathbf{b}^{k-1}$. The generalised weight function is:

$$W_{k-1}(\mathbf{c}) = \sum_{\mathbf{b}^1 \odot \cdots \odot \mathbf{b}^{k-1} = \mathbf{c}} \prod_{i=1}^{k-1} g(\mathbf{b}^i)$$

This is the $(k-1)$-fold autoconvolution of $g$. By the convolution theorem:

$$\hat{W}_{k-1} = \hat{g}^{k-1}$$

(pointwise $(k-1)$-th power). The rest of the argument is identical: $S = h * W_{k-1}$, so $\hat{S} = \hat{h} \cdot \hat{g}^{k-1}$.

**Cost for any $k$**: $O(p \cdot 4^p)$, independent of $k$. The constraint arity $k$ enters only through the exponent in $\hat{g}^{k-1}$, which is a pointwise $O(4^p)$ operation.

---

## 8. The Full Iteration and Final Expectation

### 8.1. One Iteration Step

Given $H_D^{(m-1)}$, compute $H_D^{(m)}$:

1. Form $g(\mathbf{b}) = f(\mathbf{b}) \cdot H_D^{(m-1)}(\mathbf{b})$ — pointwise multiply, $O(4^p)$
2. WHT: $\hat{g} = \text{WHT}(g)$ — $O(p \cdot 4^p)$
3. $\hat{W} = \hat{g}^{k-1}$ — pointwise, $O(4^p)$
4. Tabulate $h(\mathbf{x}) = \cos(\boldsymbol{\Gamma}_m \cdot \mathbf{x} / \sqrt{D})$ for all $\mathbf{x}$ — $O(p \cdot 4^p)$

   (Note: the angle vector $\boldsymbol{\Gamma}$ might depend on the step $m$; in Basso's formulation it does not — it is the same $\boldsymbol{\Gamma}$ at every step. But we compute $h$ once and reuse.)
5. $\hat{h} = \text{WHT}(h)$ — $O(p \cdot 4^p)$
6. $\hat{S} = \hat{h} \cdot \hat{W}$ — pointwise, $O(4^p)$
7. $S = \text{IWHT}(\hat{S})$ — $O(p \cdot 4^p)$
8. $H_D^{(m)}(\mathbf{a}) = [S(\mathbf{a})]^D$ — pointwise, $O(4^p)$

Cost per step: $O(p \cdot 4^p)$. Over $p$ steps: $O(p^2 \cdot 4^p)$.

**Optimisation**: If $\boldsymbol{\Gamma}$ is the same for all $m$ (which it is — Basso's iteration uses the same angle vector throughout), then $h$ and $\hat{h}$ can be precomputed once. This saves constant factors but does not change the asymptotic complexity.

### 8.2. Final Expectation (Eq. 8.8)

The final expectation involves a sum over $k$ roots:

$$\nu_p^{[k]}(D, \gamma, \beta) = i\sqrt{\frac{D}{2k}} \sum_{\mathbf{a}^1, \ldots, \mathbf{a}^k} \sin\!\left(\frac{\boldsymbol{\Gamma} \cdot (\mathbf{a}^1 \odot \cdots \odot \mathbf{a}^k)}{\sqrt{D}}\right) \prod_{j=1}^k a_0^j\, f(\mathbf{a}^j)\, H_D^{(p)}(\mathbf{a}^j)$$

Define $\tilde{g}(\mathbf{a}) = a_0 \cdot f(\mathbf{a}) \cdot H_D^{(p)}(\mathbf{a})$ and $h_{\sin}(\mathbf{x}) = \sin(\boldsymbol{\Gamma} \cdot \mathbf{x} / \sqrt{D})$.

Then:

$$\nu = i\sqrt{D/(2k)} \sum_\mathbf{c} h_{\sin}(\mathbf{c})\, W_k(\mathbf{c})$$

where $W_k = \tilde{g}^{*k}$ ($k$-fold autoconvolution). By the convolution theorem, $\hat{W}_k = \hat{\tilde{g}}^k$.

But this sum is a **scalar** (sum over all $\mathbf{c}$), which equals $|G|^{-1} \hat{h}_{\sin}(\mathbf{0}) \cdot \hat{W}_k(\mathbf{0})$... no, let's be more careful.

Actually, $\nu = i\sqrt{D/(2k)} \sum_\mathbf{c} h_{\sin}(\mathbf{c}) W_k(\mathbf{c})$, which is just the inner product $\langle h_{\sin}, W_k \rangle$. By Parseval's theorem on $G$:

$$\sum_\mathbf{c} h_{\sin}(\mathbf{c}) W_k(\mathbf{c}) = \frac{1}{|G|} \sum_\mathbf{s} \hat{h}_{\sin}(\mathbf{s})\, \overline{\hat{W}_k(\mathbf{s})}$$

But since $W_k$ is real-valued (it's a $k$-fold convolution of a real function with itself, as $g$ involves $f$ and $H$ which are generally complex... actually, let's check: $g$ may be complex).

Regardless of whether $g$ is real or complex, we can compute $\hat{\tilde{g}}$ in $O(p \cdot 4^p)$, raise to the $k$-th power pointwise, and compute $\hat{h}_{\sin}$ in $O(p \cdot 4^p)$. The inner product in Fourier space is $O(4^p)$.

**Cost for final expectation**: $O(p \cdot 4^p)$.

---

## 9. Complexity Summary

| Component | Naive cost | WHT cost |
|-----------|-----------|----------|
| One iteration step ($W$ computation) | $O(4^{kp})$ | $O(p \cdot 4^p)$ |
| One iteration step ($S$ from $W$) | $O(4^{2p})$ | $O(p \cdot 4^p)$ |
| One iteration step (total) | $O(4^{kp})$ | $O(p \cdot 4^p)$ |
| Full iteration ($p$ steps) | $O(p \cdot 4^{kp})$ | $O(p^2 \cdot 4^p)$ |
| Final expectation | $O(4^{kp})$ | $O(p \cdot 4^p)$ |
| **Total** | $O(p \cdot 4^{kp})$ | $O(p^2 \cdot 4^p)$ |

For k=3:
| $p$ | Naive $4^{3p}$ | WHT $p^2 \cdot 4^p$ | Speedup |
|-----|----------------|----------------------|---------|
| 1 | 64 | 4 | 16× |
| 3 | 262,144 | 576 | 455× |
| 5 | $10^9$ | 25,600 | 40,000× |
| 7 | $4 \times 10^{12}$ | $6.4 \times 10^5$ | $6 \times 10^6$× |
| 10 | $10^{18}$ | $10^8$ | $10^{10}$× |
| 15 | $10^{27}$ | $2.4 \times 10^{11}$ | $4 \times 10^{15}$× |

With WHT factorisation, p=15 becomes feasible (minutes on laptop). p=17 is reachable on a cluster.

---

## 10. Where Could This Go Wrong?

Despite the clean algebra, several potential failure modes should be checked:

### 10.1. Numerical Stability

The WHT involves sums and differences of large numbers. At high $p$, the dynamic range of $g$ values could span many orders of magnitude, leading to catastrophic cancellation. This is an implementation concern, not a mathematical one.

**Mitigation**: Compare WHT results against naive summation at small $p$ using high-precision arithmetic. Monitor condition number of the transform.

### 10.2. Is $f(\mathbf{b})$ Correctly Handled?

The function $f(\mathbf{b})$ is a product of mixer matrix elements. It is complex-valued in general (the matrix elements of $e^{-i\beta X}$ are $\cos\beta$ and $-i\sin\beta$). The WHT works over $\mathbb{C}$, so this is fine algebraically. But the WHT of a complex function requires tracking both real and imaginary parts.

### 10.3. Convention: WHT Normalisation

There are multiple WHT conventions (with or without $1/|G|$ prefactor). The convolution theorem statement depends on the chosen convention. If we use the convention $\hat{\phi}(\mathbf{s}) = \sum_\mathbf{x} \phi(\mathbf{x}) \chi_\mathbf{s}(\mathbf{x})$ (no normalisation), then:

$$\widehat{\phi * \psi}(\mathbf{s}) = \hat{\phi}(\mathbf{s}) \cdot \hat{\psi}(\mathbf{s})$$

and the inverse is:

$$\phi(\mathbf{x}) = \frac{1}{|G|} \sum_\mathbf{s} \hat{\phi}(\mathbf{s}) \chi_\mathbf{s}(\mathbf{x})$$

The convolution as we defined it ($(\phi * \psi)(\mathbf{c}) = \sum_\mathbf{x} \phi(\mathbf{x}) \psi(\mathbf{x} \odot \mathbf{c})$) matches this convention. **Verify the normalisation in the implementation by checking $W(\mathbf{1}) = \left(\sum_\mathbf{b} g(\mathbf{b})\right)^2$**, which is the total weight when all parities are $+1$.

### 10.4. The $\Gamma_0 = 0$ Slot

In Basso's convention, $\boldsymbol{\Gamma}$ has a zero in the middle slot (position 0):
$\boldsymbol{\Gamma} = (\gamma_1, \ldots, \gamma_p, 0, -\gamma_p, \ldots, -\gamma_1)$.

This does not affect the algebra. The middle position contributes $\cos(0) = 1$ and $\sin(0) = 0$ to the closed-form $\hat{h}$, effectively making that position inert.

---

## 11. Numerical Test Protocol

To validate the WHT approach before building the full iteration, implement both methods at small $p$ and compare:

### Test 1: $W(\mathbf{c})$ computation (p=1)

At $p = 1$: $\mathbf{x} \in \{-1,+1\}^3$, so $|G| = 8$.

1. Choose random $g$ values: draw 8 complex numbers $g(\mathbf{b})$.
2. **Naive**: For each of the 8 values of $\mathbf{c}$, compute $W(\mathbf{c}) = \sum_{\mathbf{b}^1 \odot \mathbf{b}^2 = \mathbf{c}} g(\mathbf{b}^1) g(\mathbf{b}^2)$ by iterating over all 64 pairs.
3. **WHT**: Compute $\hat{g}$, square pointwise, inverse WHT.
4. Assert: $|W_{\text{naive}}(\mathbf{c}) - W_{\text{WHT}}(\mathbf{c})| < \epsilon$ for all $\mathbf{c}$.

### Test 2: $S(\mathbf{a})$ computation (p=1)

1. Choose random $g$ (8 values), random angles $\gamma_1, \beta_1$.
2. **Naive**: For each of 8 values of $\mathbf{a}$, sum over all 64 pairs $(\mathbf{b}^1, \mathbf{b}^2)$.
3. **WHT**: Full algorithm from §4.
4. Assert agreement.

### Test 3: Full iteration step (p=1, p=2)

1. Set $H_D^{(0)} = 1$, compute $H_D^{(1)}$ both ways.
2. For p=2 ($|G| = 32$): compute $H_D^{(1)}$ then $H_D^{(2)}$ both ways.
3. Assert agreement at each step.

### Test 4: Final expectation (p=1)

1. Compare $\nu_p^{[3]}$ computed by naive triple-sum vs. WHT-accelerated.
2. Cross-validate against existing brute-force simulator at $p=1$, $k=3$, $D=4$.

### Test 5: Closed-form $\hat{h}$ vs. butterfly WHT

1. Tabulate $h(\mathbf{x})$ for all $\mathbf{x}$.
2. Compute $\hat{h}$ by butterfly WHT.
3. Compute $\hat{h}$ by the closed form in §3.4.
4. Assert agreement.

### Test configuration

- Use Float64 arithmetic; tolerance $\epsilon = 10^{-12}$.
- Use random angles drawn uniformly from $[0, 2\pi)$.
- Repeat with 50+ random angle configurations to avoid accidental coincidences.
- Test with $D \in \{2, 3, 4\}$ to check $D$-dependence.

---

## 12. Conclusions

### 12.1. The Answer

**Yes, the WHT factorisation works.** The cost of the Basso finite-D iteration (Eq. 8.7) for arbitrary $k$ reduces from $O(p \cdot 4^{kp})$ to $O(p^2 \cdot 4^p)$.

The key insight is that the inner sum factors into two convolutions over the group $(\{-1,+1\}^{2p+1}, \odot) \cong \mathbb{Z}_2^{2p+1}$:
1. $W = g^{*(k-1)}$ — the $(k-1)$-fold autoconvolution of $g$
2. $S = h * W$ — convolution of the cosine kernel with $W$

Both are computable via WHT in $O(p \cdot 4^p)$ time.

The "known obstacle" ($\cos(\text{sum}) \neq \prod \cos(\text{terms})$) is a **non-issue**. We never need the cosine to factor. The cosine function $h$ enters the convolution as a black-box function; its WHT is computed either by the generic butterfly algorithm or by the closed form in §3.4. The $\text{Re}(\cdot)$ operator does not need to commute with anything — it was already applied at the level of defining $h$.

### 12.2. Impact on the Project

If validated numerically:
- **k=3, D=4 becomes feasible at p=15–17**, up from p=5–7.
- This suffices to observe QAOA performance approaching or exceeding the asymptotic regime.
- The comparison with DQI can be made at much higher circuit depth.
- The complexity is **independent of k**, so k=4, k=5 are also in reach.

### 12.3. What Is NOT Claimed

- We do not claim this is a new mathematical result. The convolution theorem on $\mathbb{Z}_2^n$ is classical. The observation that the Basso sum factors this way may or may not be in the literature — we have not checked. The Basso paper itself does not mention this optimisation for the finite-D iteration (they focus on the $D \to \infty$ limit for $k \geq 3$).
- We have not verified numerical stability at large $p$. This requires empirical testing.
- The angle optimisation cost (many evaluations of $\nu_p$) is a separate concern. Each evaluation now costs $O(p^2 \cdot 4^p)$ instead of $O(p \cdot 4^{kp})$, but with $O(10^3)$ L-BFGS iterations × $O(10)$ restarts, total work at p=15 is $\sim 10^4 \times 15^2 \times 4^{15} \approx 10^{4} \times 225 \times 10^{9} \approx 2 \times 10^{15}$ ops — still a significant computation.

---

## Appendix A: WHT Convention Reference

For $G = \{-1,+1\}^n$ with entry-wise multiplication:

**Forward WHT**:
$$\hat{\phi}(\mathbf{s}) = \sum_{\mathbf{x} \in G} \phi(\mathbf{x}) \prod_{\ell=1}^n x_\ell^{s_\ell}, \qquad \mathbf{s} \in \{0,1\}^n$$

**Inverse WHT**:
$$\phi(\mathbf{x}) = \frac{1}{2^n} \sum_{\mathbf{s} \in \{0,1\}^n} \hat{\phi}(\mathbf{s}) \prod_{\ell=1}^n x_\ell^{s_\ell}$$

**Convolution** (on $G$):
$$(\phi * \psi)(\mathbf{c}) = \sum_{\mathbf{x} \in G} \phi(\mathbf{x}) \psi(\mathbf{x} \odot \mathbf{c})$$

**Convolution theorem**: $\widehat{\phi * \psi}(\mathbf{s}) = \hat{\phi}(\mathbf{s}) \cdot \hat{\psi}(\mathbf{s})$

**Parseval**: $\sum_\mathbf{x} \phi(\mathbf{x}) \overline{\psi(\mathbf{x})} = \frac{1}{2^n} \sum_\mathbf{s} \hat{\phi}(\mathbf{s}) \overline{\hat{\psi}(\mathbf{s})}$

**Proof of convolution theorem**:

$$\widehat{\phi * \psi}(\mathbf{s}) = \sum_\mathbf{c} \left[\sum_\mathbf{x} \phi(\mathbf{x}) \psi(\mathbf{x} \odot \mathbf{c})\right] \chi_\mathbf{s}(\mathbf{c})$$

Substitute $\mathbf{y} = \mathbf{x} \odot \mathbf{c}$ (so $\mathbf{c} = \mathbf{x} \odot \mathbf{y}$, $\chi_\mathbf{s}(\mathbf{c}) = \chi_\mathbf{s}(\mathbf{x}) \chi_\mathbf{s}(\mathbf{y})$):

$$= \sum_\mathbf{x} \phi(\mathbf{x}) \chi_\mathbf{s}(\mathbf{x}) \sum_\mathbf{y} \psi(\mathbf{y}) \chi_\mathbf{s}(\mathbf{y}) = \hat{\phi}(\mathbf{s}) \cdot \hat{\psi}(\mathbf{s}) \quad \square$$

## Appendix B: Worked Example at $p=1$, $k=3$

At $p=1$: $n = 2p+1 = 3$, $|G| = 8$, $\boldsymbol{\Gamma} = (\gamma_1, 0, -\gamma_1)$.

Label the 8 elements of $G$ as $(x_1, x_0, x_{-1}) \in \{-1,+1\}^3$.

**$h(\mathbf{x}) = \cos\left(\frac{\gamma_1 x_1 + 0 \cdot x_0 - \gamma_1 x_{-1}}{\sqrt{D}}\right) = \cos\left(\frac{\gamma_1(x_1 - x_{-1})}{\sqrt{D}}\right)$**

Since $x_1, x_{-1} \in \{-1,+1\}$:
- $x_1 = x_{-1}$: $h = \cos(0) = 1$
- $x_1 = +1, x_{-1} = -1$: $h = \cos(2\gamma_1/\sqrt{D})$
- $x_1 = -1, x_{-1} = +1$: $h = \cos(-2\gamma_1/\sqrt{D}) = \cos(2\gamma_1/\sqrt{D})$

So $h$ takes only two distinct values: $1$ (when $x_1 = x_{-1}$, 4 configurations) and $\cos(2\gamma_1/\sqrt{D})$ (when $x_1 \neq x_{-1}$, 4 configurations).

**Naive $S(\mathbf{a})$**: For each of 8 values of $\mathbf{a}$, sum over 64 pairs: 512 evaluations total.

**WHT $S(\mathbf{a})$**: Three WHTs of length 8 (each: $3 \times 8 = 24$ ops) + two pointwise operations ($2 \times 8 = 16$ ops) = ~88 ops. Speedup: ~6×.

At $p=2$ ($|G| = 32$): naive = $32^3 = 32768$. WHT = $3 \times 5 \times 32 + 2 \times 32 = 544$. Speedup: ~60×.

---

*End of research specification.*
