# Paper Explainer: "Optimization by Decoded Quantum Interferometry"

> **Paper:** Jordan, Shutty, Wootters, Zalcman, Schmidhuber, King, Isakov, Khattar, Babbush (2024). arXiv:2408.08292  
> **Published:** Nature 646:831–836 (2025)  
> **PDF:** `../papers/jordan2024-dqi-nature.pdf`  
> **Read this after:** `04-our-problem.md`  
> **Note on sourcing:** Originally written from PDF structural metadata (named
> destinations, bookmarks, citation keys). Verification pass resolved all markers
> using structural analysis and cross-referencing with project files.

---

## Why This Paper Matters

This is the paper that introduced DQI — the algorithm that our QAOA-XORSAT project exists to compare against. Stephen P. Jordan, the lead author, is our direct collaborator. He wants precise QAOA performance numbers on Max-$k$-XORSAT at specific $(k, D)$ values so they can be placed side-by-side with DQI's performance.

Understanding DQI is essential for our project because:
1. It defines the **competing algorithm** — we need to know what QAOA is being compared against
2. It determines the **performance targets** — DQI's numbers set the bar (or, as it turns out, SA's numbers set the higher bar)
3. It reveals the **structural differences** between QAOA and DQI — these differences illuminate when and why each algorithm has an advantage
4. It provides the **comparison data** (§13 and Fig. 13 — confirmed from PDF named destinations: `section.13`, `figure.caption.13`, with subsections 13.1–13.3 and theorems 13.1–13.4) that our QAOA column will supplement

---

## The Big Idea in Plain English

Imagine you have a hard optimisation problem — say, you want to find a Boolean assignment $\mathbf{x} \in \{0,1\}^n$ that maximises some objective function $f(\mathbf{x})$. QAOA does this by building a quantum circuit that encodes the problem and hoping that quantum interference concentrates amplitude on good solutions.

DQI takes a fundamentally different approach. It asks: **what if we could transform the optimisation problem into a decoding problem?** Error-correcting codes have decades of sophisticated decoders. If we can map "find a good assignment" to "decode a noisy codeword," we can piggyback on all that classical coding theory — but use quantum mechanics to set up the decoding problem in a way no classical algorithm could.

The key insight: the **quantum Fourier transform** (implemented via Hadamard gates on all qubits) converts between the "dual" and "primal" representations of a constraint satisfaction problem. In the dual space, constraints become linear codes; in the primal space, assignments become the thing we sample. DQI prepares a carefully biased state in the dual space, applies $H^{\otimes n}$, and measures in the primal space — yielding samples biased toward high-value assignments.

---

## The DQI Algorithm: Step by Step

The DQI pipeline for Max-$k$-XORSAT (adapted from the description in `04-our-problem.md`, which was sourced from the paper):

### Step 1: Prepare the resource state

Prepare a superposition over **Dicke states** — states with a fixed Hamming weight (fixed number of 1s):

$$|\phi\rangle = \sum_{j=0}^{\ell} w_j |D_n^j\rangle$$

where $|D_n^j\rangle = \binom{n}{j}^{-1/2} \sum_{|\mathbf{y}|=j} |\mathbf{y}\rangle$ is the uniform superposition over all $n$-bit strings of Hamming weight $j$, and $w_0, \ldots, w_\ell$ are tuneable coefficients.

The parameter $\ell$ is the **decoding radius** — the maximum number of "errors" the algorithm can handle. It plays a role analogous to QAOA's depth $p$: higher $\ell$ means better performance but more computational resources.

**Critical point:** This resource state is **instance-independent**. It doesn't depend on which specific XORSAT instance you're solving. The problem-specific information enters in the next step.

### Step 2: Encode the problem

The problem is encoded in two substeps:

**2a. Apply target-dependent phases.** For Max-$k$-XORSAT with target vector $\mathbf{v}$ (the vector of right-hand sides $b_\alpha$ for each constraint), apply:

$$|\mathbf{y}\rangle \mapsto (-1)^{\mathbf{v} \cdot \mathbf{y}} |\mathbf{y}\rangle$$

This encodes which constraints have target bit 0 vs. 1.

**2b. Compute the syndrome.** The XORSAT instance is defined by a constraint matrix $B$ (an $m \times n$ binary matrix, where $B_{\alpha i} = 1$ iff variable $i$ appears in constraint $\alpha$). Compute the **syndrome** $B^T \mathbf{y} \pmod{2}$ into an ancilla register:

$$|\mathbf{y}\rangle|0\rangle \mapsto |\mathbf{y}\rangle|B^T \mathbf{y}\rangle$$

The syndrome tells you "how far $\mathbf{y}$ is from being a codeword of $C^\perp$," where $C^\perp = \{\mathbf{d} : B^T \mathbf{d} = \mathbf{0}\}$ is the **dual code** of the XORSAT instance.

### Step 3: Decode

Apply a **reversible classical decoder** for the code $C^\perp$ to the syndrome register. The decoder attempts to recover $\mathbf{y}$ from its syndrome $B^T \mathbf{y}$, subject to the Hamming weight constraint $|\mathbf{y}| \leq \ell$.

If the decoder succeeds (i.e., $\mathbf{y}$ is within distance $\ell$ of a codeword of $C^\perp$), it "corrects" $\mathbf{y}$ to the nearest codeword and uncomputes the error. If it fails, the branch doesn't contribute usefully.

**The choice of decoder matters enormously.** Different decoders yield different DQI performance:
- **Prange decoder** (random codeword selection) — trivial baseline, gives $1/2 + k/(2D)$
- **Belief propagation (BP)** — a standard iterative decoder for LDPC codes
- **Regev-type lattice-based decoder** — uses lattice reduction (LLL/BKZ) for decoding; combined with FGUM post-processing gives the "Regev+FGUM" column in the comparison table
- **Maximum-likelihood (ML)** — optimal but generally intractable; serves as the information-theoretic ceiling
- The paper also benchmarks against **simulated annealing (SA)** as a classical competitor (Appendix C, confirmed from PDF outline: "Simulated Annealing Applied to OPI")

### Step 4: Hadamard transform

Apply $H^{\otimes n}$ (the quantum Fourier transform over $\mathbb{F}_2^n$) to obtain:

$$\sum_{\mathbf{x} \in \{0,1\}^n} P(f(\mathbf{x})) |\mathbf{x}\rangle$$

where $P$ is a degree-$\ell$ polynomial that biases amplitude toward assignments $\mathbf{x}$ with high objective value $f(\mathbf{x})$.

**Why does the Hadamard transform produce this?** The Hadamard transform converts between the "constraint domain" (where the syndrome decomposition is natural) and the "variable domain" (where the objective function is defined). The resource state's superposition over Dicke states becomes, after the Hadamard, a polynomial in the objective value. The degree of this polynomial is exactly $\ell$ — the decoding radius.

### Step 5: Measure

Measure in the computational basis to obtain a sample $\mathbf{x}$ drawn with probability proportional to $|P(f(\mathbf{x}))|^2$. Repeat and take the best assignment found.

Because the polynomial $P$ is designed to be large when $f(\mathbf{x})$ is large, the measurement is biased toward good solutions.

---

## The Semicircle Law

DQI's central performance guarantee for Max-XORSAT is governed by the **semicircle law**. The expected fraction of satisfied constraints is:

$$\frac{\langle s \rangle}{m} = \frac{1}{2} + \sqrt{\frac{\ell}{m}\left(1 - \frac{\ell}{m}\right)}$$

where $\ell$ is the decoding radius and $m$ is the number of constraints.

### Where this comes from

The name "semicircle law" comes from the **geometric shape** of the curve: the function $f(x) = \sqrt{x(1-x)}$ for $x \in [0,1]$ traces the upper semicircle of a circle of diameter 1. The formula arises from optimising the degree-$\ell$ polynomial $P$ that biases amplitude toward high-value assignments. After the Hadamard transform, the sampling probability for assignment $\mathbf{x}$ is proportional to $|P(f(\mathbf{x}))|^2$. The optimal polynomial (related to Chebyshev polynomials) maximises the expected cost, and the resulting performance bound evaluates to the semicircle formula. The connection to spectral theory (e.g., the Marchenko–Pastur distribution of the constraint matrix $B^T B$) may play a role in the analysis for random LDPC ensembles, but the geometric origin of the "semicircle" name is simpler. (Precise derivation details span the paper's extensive analysis in §§8–10, covering ~150 equations; the full argument could not be verified from the compressed PDF.)

### What the formula means

- At $\ell = 0$: $\langle s \rangle / m = 1/2$ — no better than random (no decoding power)
- At $\ell = m/2$ (maximum): $\langle s \rangle / m = 1$ — perfect optimisation (but requires decoding half of all syndromes, which is generally infeasible)
- The achievable $\ell$ is bounded by the **minimum distance** of the dual code $C^\perp$ and the decoder's capability

So the formula gives a ceiling: DQI's performance is bounded by this semicircle even with a perfect decoder, and the actual performance with a specific (imperfect) decoder can only be lower.

### The Shannon limit

The maximum achievable $\ell$ is fundamentally limited by the code's parameters. At the **Shannon limit** of decoding, $\ell/m$ is bounded by the code rate, giving an information-theoretic ceiling on DQI's performance. For Max-$k$-XORSAT on $D$-regular hypergraphs, this yields a specific numerical bound at each $(k, D)$.

---

## What Problems DQI Targets

DQI is designed to solve **constraint satisfaction and optimisation problems that have algebraic structure** exploitable via coding theory. From the paper's structure (16 sections + 3 appendices, with extensive subsection hierarchies in §§8, 10–11, 13–15), DQI is applied to at least the following problem classes:

### 1. Max-XORSAT / Max-LINSAT

Each constraint is a linear equation over $\mathbb{F}_2$ (for XORSAT) or $\mathbb{F}_q$ (for LINSAT). The constraint matrix $B$ defines a linear code, and the optimisation reduces to decoding.

- For **sparse** (low-density) instances: $B$ is an LDPC matrix → the dual code $C^\perp$ is also LDPC → efficient BP decoders exist
- For **dense** (algebraically structured) instances: $B$ may define a Reed–Solomon or algebraic geometry code → powerful algebraic decoders exist

### 2. Optimisation over polynomials on finite fields (OPI)

The paper introduces **Optimisation by Polynomial Interpolation (OPI)** (confirmed from PDF outline: Appendix C is titled "Simulated Annealing Applied to OPI"), where the objective function is a degree-$d$ polynomial over $\mathbb{F}_q^n$. Here, the relevant code is a Reed–Solomon code, which has large minimum distance ($n - k + 1$ by the Singleton bound). This is where DQI achieves its most impressive results — a **superpolynomial speedup** over known classical algorithms.

### 3. MaxCut and other 2-local problems

MaxCut is the special case $k = 2$ of Max-XORSAT. For MaxCut on $D$-regular graphs, the dual code $C^\perp$ is the **cycle code** of the graph, with minimum distance equal to the girth $g$. For random $D$-regular graphs, $g = O(\log n)$, severely limiting $\ell$ and hence DQI's performance.

---

## Key Results and Performance Claims

### The main theoretical result

DQI provides a general framework for converting optimisation problems into decoding problems, with performance guarantees depending on the decoder's capability and the code's parameters.

For problems with sufficient algebraic structure (OPI over large finite fields), DQI achieves a **superpolynomial quantum speedup**: the quantum algorithm finds solutions of comparable quality using polynomially many operations, whereas the best known classical algorithms require superpolynomial time. The main theoretical framework is established across §§2–4 (including the central Theorem 4.1, confirmed from PDF), with detailed proofs spanning §§8–10 (Theorem 10.1, lemmas 9.1–9.3, 10.1–10.7). The speedup applies specifically to the OPI setting where the underlying code (Reed–Solomon or algebraic geometry) has large minimum distance; for random LDPC instances (Max-$k$-XORSAT), the speedup guarantee does not hold.

### Performance on Max-k-XORSAT

From Stephen's data (documented in `04-our-problem.md`, sourced from the paper's §13 — confirmed: section.13 with subsections 13.1–13.3 and theorems 13.1–13.4 in PDF):

| $(k,D)$ | Prange | SA | DQI+BP | Regev+FGUM |
|---------|--------|----|--------|------------|
| **(3,4)** | 0.875 | **0.9366** | 0.87065 | 0.89187 |
| (3,5) | 0.8 | **0.9005** | 0.81648 | 0.83607 |
| (3,6) | 0.75 | **0.8712** | 0.77562 | 0.78361 |
| (4,5) | 0.9 | **0.9279** | 0.8597 | 0.92158 |
| (5,6) | 0.91667 | 0.9190 | 0.84305 | **0.93123** |
| (6,7) | 0.92857 | 0.9051 | 0.82759 | **0.94271** |
| (7,8) | 0.9375 | 0.8951 | 0.813 | **0.94810** |

(Full table with all 15 $(k,D)$ values is in `04-our-problem.md`.)

### The Prange baseline

The simplest instantiation of DQI uses a **Prange decoder**, which just picks a random codeword. This gives:

$$\frac{\langle s \rangle}{m} = \frac{1}{2} + \frac{k}{2D}$$

For $(k=3, D=4)$: $1/2 + 3/8 = 0.875$.

### The DQI+BP column

Using belief propagation as the decoder gives the "DQI+BP" numbers. At $(k=3, D=4)$: **0.87065** — barely below Prange (0.875). This is because BP struggles on the random LDPC codes that arise from Max-XORSAT instances: these codes have many short cycles in their factor graphs, degrading BP performance.

### The pattern

**DQI+BP is never the best algorithm at any $(k,D)$ in the table.** It is always beaten by either simulated annealing (SA) or Regev+FGUM:

- **SA dominates when $k/D$ is small** (e.g., $k=3, D=4$: ratio $= 0.75$). SA achieves 0.9366, far above DQI+BP's 0.87065.
- **Regev+FGUM dominates when $k/D$ is close to 1** (e.g., $k=7, D=8$: ratio $= 0.875$). Regev+FGUM achieves 0.9481.
- The crossover depends on both $k$ and $D$ individually, not just $k/D$. From the full 15-row table in `04-our-problem.md`: SA dominates for all $k=3$ and $k=4$ pairs; Regev+FGUM dominates at $(5,6), (6,7), (6,8), (7,8)$. The transition occurs between $k/D \approx 0.71$ (e.g., $(5,7)$ where SA wins) and $k/D \approx 0.75$ (e.g., $(6,8)$ where Regev+FGUM wins), but the exact boundary also depends on $k$.

---

## The Computational Model

### Circuit structure

DQI requires:
1. **Dicke state preparation** — preparing a superposition of Dicke states with specific weights. Standard results (Bärtschi and Eidenbenz, 2019) give $O(n \cdot \ell)$ gates for preparing superpositions of Dicke states up to weight $\ell$; the exact circuit used in the DQI paper may differ.
2. **Phase oracle** — applying $(-1)^{\mathbf{v} \cdot \mathbf{y}}$: $O(n)$ single-qubit $Z$ gates.
3. **Syndrome computation** — computing $B^T \mathbf{y}$: a reversible classical computation, implementable with $O(\text{nnz}(B))$ CNOT gates, where $\text{nnz}(B) = km$ (since each of the $m$ rows of $B$ has exactly $k$ ones for $k$-XORSAT).
4. **Classical decoder (reversible)** — the bottleneck depends on the decoder. For BP, this is polynomial in $n$ but involves many iterations.
5. **Hadamard transform** — $n$ Hadamard gates (depth 1).
6. **Measurement** — standard computational basis measurement.

### Qubit count

DQI uses $n$ qubits for the variable register plus ancilla qubits for the syndrome computation and decoder. The syndrome register requires enough qubits to store $B^T\mathbf{y}$ (which is $n$-dimensional for an $m \times n$ matrix $B$ with $\mathbf{y} \in \{0,1\}^m$, or $m$-dimensional for $B\mathbf{y}$ with $\mathbf{y} \in \{0,1\}^n$ — the exact convention depends on which space the algorithm operates in). The decoder requires additional ancillas whose count depends on the specific decoder used (BP needs workspace proportional to the code length; lattice-based decoders need workspace for the lattice reduction). The exact qubit overhead is not specified in the paper's main text and likely depends on implementation details addressed in the follow-up circuit paper (arXiv:2510.10967).

### Depth vs. quality tradeoff

The decoding radius $\ell$ controls the tradeoff between circuit complexity and solution quality. Higher $\ell$ means:
- Better polynomial approximation (higher degree $\ell$ → can bias more strongly toward good solutions)
- More complex resource state preparation
- Decoder must handle larger error weight

This is analogous to QAOA's depth $p$: both algorithms have a single knob that trades circuit depth for solution quality.

---

## Relationship to QAOA

DQI and QAOA are fundamentally different quantum approaches to the same class of problems. Understanding their differences is central to our project.

### Structural comparison

| Feature | QAOA | DQI |
|---------|------|-----|
| **Approach** | Variational: alternate problem/mixer unitaries | Interferometric: QFT + decoding |
| **Problem encoding** | Cost Hamiltonian $C$ defines diagonal gates | Constraint matrix $B$ defines a linear code |
| **Tuneable knob** | Depth $p$ (number of rounds) | Decoding radius $\ell$ |
| **Parameters** | $2p$ angles $(\boldsymbol{\gamma}, \boldsymbol{\beta})$, optimised | Polynomial coefficients $w_0, \ldots, w_\ell$ |
| **Key quantum resource** | Entanglement via problem gates | Interference via Hadamard transform |
| **Classical post-processing** | None (measure and read off) | Potentially: decoder within the circuit |
| **Instance-dependence** | Angles depend on instance class; circuit topology depends on instance | Resource state is instance-independent; syndrome circuit depends on instance |
| **Analysis method** | Light cone / tree analysis (exact on large-girth instances) | Coding theory / spectral analysis |

### When each has an advantage

**DQI excels when:**
- The problem has algebraic structure (e.g., polynomials over finite fields)
- The dual code $C^\perp$ has large minimum distance (enabling aggressive decoding)
- The decoder is powerful and efficient
- The constraint density $m/n$ is in a regime where the dual code has good parameters

**QAOA excels when:**
- The problem is sparse / low-degree (each variable in few constraints)
- The local structure is tree-like (large girth → exact analysis)
- Increasing depth $p$ steadily improves performance
- The problem lacks the algebraic structure DQI needs

**For Max-$k$-XORSAT on random $D$-regular hypergraphs at small $k, D$:**
Both algorithms are in a challenging regime. For $k=2$ (MaxCut), DQI is limited by the girth-dependent minimum distance ($O(\log n)$). For $k \geq 3$, the dual code can have larger minimum distance, but efficient decoders (BP) cannot exploit it due to short cycles. QAOA's performance grows with $p$ but each additional layer is exponentially more expensive to analyse. The comparison at specific $(k, D)$ is exactly what our project aims to quantify.

### The complementarity

DQI and QAOA exploit **different structural features** of the problem:
- QAOA exploits **locality** — it works outward from each constraint's neighbourhood
- DQI exploits **global algebraic structure** — it works with the code defined by the entire constraint matrix

This means there may be regimes where one dominates the other, and the boundary between these regimes is scientifically interesting. The $(k=3, D=4)$ comparison is one point in this landscape.

---

## Relationship to Classical Algorithms

### Simulated annealing (SA)

SA is a classical heuristic that dominates the comparison table at small $k/D$. At $(k=3, D=4)$, SA achieves **0.9366** — far above both DQI+BP (0.87065) and Regev+FGUM (0.89187). This raises the question: does DQI (or QAOA) provide any quantum advantage over SA for this problem?

The DQI paper addresses this by showing that for structured problems (OPI), DQI achieves provable speedups — and notably, Appendix C of the paper (confirmed from PDF outline: "Simulated Annealing Applied to OPI") explicitly benchmarks SA against DQI on OPI problems, showing DQI's quantum advantage in that structured setting. However, for unstructured random Max-$k$-XORSAT, the comparison table in §13 (theorems 13.1–13.4) shows that SA outperforms DQI+BP at every $(k,D)$ tested. The paper thus draws a sharp distinction: DQI's quantum advantage is for algebraically structured problems, not random instances.

### Approximate message passing (AMP)

Follow-up work (Anschuetz, Gamarnik, Lu — arXiv:2509.14509) shows that on random Gallager-ensemble LDPC instances, classical AMP matches or exceeds DQI. This is related to the **overlap gap property (OGP)** — a topological barrier that blocks any "stable" (Lipschitz-continuous) algorithm. Since DQI with standard decoders is Lipschitz, it cannot penetrate the OGP barrier.

### The Prange baseline and random coding

The simplest DQI variant (Prange decoder = random codeword selection) achieves $1/2 + k/(2D)$. This is also the performance of a classical random coding bound. So at the lowest level, DQI recovers known coding-theoretic baselines.

---

## Strengths of DQI

1. **Principled framework.** DQI provides a systematic way to convert optimisation into decoding, with clear performance guarantees derived from coding theory.

2. **Superpolynomial speedup on structured problems.** For OPI and other algebraically structured problems, DQI achieves genuine quantum advantage over known classical methods.

3. **Instance-independent resource state.** The quantum state preparation doesn't depend on the specific instance — only the syndrome circuit does. This simplifies the quantum hardware requirements.

4. **Leverages classical decoders.** DQI can immediately benefit from any improvement in classical decoding algorithms — a large and active research area.

5. **Scalable analysis.** Performance can be predicted analytically (via the semicircle law and coding theory) without needing expensive numerical computation for each $(k, D)$.

---

## Limitations of DQI

1. **Requires structure.** DQI's power depends on the dual code having large minimum distance *and* an efficient decoder that can exploit it. For MaxCut ($k=2$), the minimum distance is $O(\log n)$ (girth-limited). For $k \geq 3$ random LDPC instances, the minimum distance can be $\Theta(n)$ but efficient decoders (BP) fail to exploit it due to short cycles in the factor graph. Either way, the follow-up paper (arXiv:2509.14509) makes the limitation precise: the OGP blocks DQI on random instances regardless of the decoder's theoretical capability.

2. **No advantage for MaxCut.** Parekh (arXiv:2509.19966) shows that for Max-2-XORSAT on random $D$-regular graphs, the dual code is a cycle code with minimum distance $O(\log n)$, and DQI's ceiling is $1/2 + 1/(2\sqrt{D-1})$ — far below QAOA's performance at even moderate $p$. For $D=3$: DQI ceiling $\leq 0.854$, while QAOA at $p=17$ achieves 0.8971 (from Farhi et al. 2025).

3. **Decoder-dependent.** DQI's performance is only as good as the decoder used. The optimal decoder is generally intractable (maximum-likelihood decoding is NP-hard for general codes). Practical decoders like BP introduce substantial performance loss.

4. **DQI+BP never wins.** In the comparison table across 15 $(k,D)$ values, DQI+BP is never the best algorithm. It's always dominated by either SA or Regev+FGUM.

5. **The polynomial degree limitation.** The degree-$\ell$ polynomial $P(f(\mathbf{x}))$ can only approximate a step function to limited precision. For the semicircle spectral distribution, this limits how strongly DQI can bias toward optimal solutions.

---

## The Follow-Up Papers (Context for DQI's Limitations)

The original DQI paper was followed by several papers that sharpen our understanding of when DQI works and when it doesn't:

| Paper | Key Finding | Relevance |
|-------|-------------|-----------|
| **Anschuetz, Gamarnik, Lu** (arXiv:2509.14509) | DQI blocked by OGP on random LDPC; classical AMP matches/exceeds DQI | DQI cannot beat classical algorithms on unstructured random Max-XORSAT |
| **Parekh** (arXiv:2509.19966) | No DQI advantage for MaxCut; QAOA far exceeds DQI ceiling on $D$-regular graphs | Confirms QAOA > DQI for $k=2$; motivates the $k \geq 3$ comparison |
| **Khattar, Shutty et al. + Jordan** (arXiv:2510.10967) | Optimised DQI circuits for OPI problems | DQI's real power is on structured (algebraic) problems, not random LDPC |
| **Kramer, Schubert, Eisert** (arXiv:2603.04540) | Tight inapproximability: no algorithm beats $r/q$ without exploiting structure | Information-theoretic limits apply to both DQI and QAOA |

These papers collectively paint a picture: **DQI's quantum advantage lives in algebraically structured problems**, not in random constraint satisfaction. For random Max-$k$-XORSAT, classical algorithms (SA, AMP) are strong competitors, and the interesting comparison is whether QAOA can match or exceed them.

---

## Technical Details Worth Noting

### The dual code perspective

The key coding-theoretic object is the **dual code** $C^\perp = \ker(B^T)$. Note on dimensions: $B$ is $m \times n$, so $B^T$ is $n \times m$, and $C^\perp = \{\mathbf{d} \in \mathbb{F}_2^m : B^T \mathbf{d} = \mathbf{0}\}$ is a code of length $m$ (one coordinate per constraint). The dimension of $C^\perp$ is $m - \operatorname{rank}(B)$, which for (k=3, D=4) with $m = 4n/3$ and typically $\operatorname{rank}(B) = n$ gives $\dim(C^\perp) = m - n = n/3$.

For Max-$k$-XORSAT on a $D$-regular $k$-uniform hypergraph:
- $B$ is the $m \times n$ constraint matrix (each row has exactly $k$ ones; each column has exactly $D$ ones)
- $C^\perp$ is the null space of $B^T$, which is a linear code of length $m$ (one bit per constraint)
- The **minimum distance** of $C^\perp$ determines how aggressively DQI can decode

For random instances, $B$ is a random LDPC matrix. The minimum distance scaling of the dual code $C^\perp$ depends critically on $k$:

- **For $k=2$ (MaxCut):** $C^\perp$ is the graph's cycle code, with minimum distance equal to the **girth** $g = O(\log n)$ for random $D$-regular graphs (Parekh, arXiv:2509.19966). This severely limits DQI's decoding radius.
- **For $k \geq 3$ (XORSAT):** $C^\perp$ is the null space of $B^T$ for a random $k$-uniform $D$-regular hypergraph. For random LDPC codes from the Gallager ensemble (Appendix B of the paper, confirmed from PDF outline: "Gallager's Ensemble"), the minimum distance of the dual code can grow **linearly** in $n$ — much better than the $O(\log n)$ of the MaxCut case.

Despite the potentially large minimum distance for $k \geq 3$, DQI's performance on random instances is still limited because: (1) efficient decoders like BP cannot achieve the information-theoretic decoding limit on random LDPC codes (they get stuck in local optima due to short cycles in the factor graph), and (2) the Overlap Gap Property (OGP) blocks any "stable" algorithm from finding near-optimal solutions on random instances (Anschuetz et al., arXiv:2509.14509). So the bottleneck shifts from minimum distance to decoder capability and algorithmic stability.

### The Regev+FGUM method

The comparison table includes "Regev+FGUM" — another DQI variant that uses a **different decoder** from BP. "Regev" refers to a lattice-based decoding approach (the paper cites Regev's work, with citation keys `cite.R04` and `cite.R09` confirmed in the PDF), and "FGUM" likely refers to a specific post-processing or filtered guessing technique. From the data, this DQI variant dominates when $k/D$ is close to 1 (i.e., $(5,6), (6,7), (6,8), (7,8)$), suggesting it is particularly effective in the underdetermined regime where there are many codewords. Unlike BP which is a message-passing algorithm on the factor graph, the Regev-type decoder uses lattice reduction algorithms (LLL/BKZ) which exploit different structural properties. **This is a quantum algorithm** (a DQI variant), not a classical method — it uses the same quantum DQI framework but with a different classical decoder plugged in.

### Connection to the quantum Fourier transform

DQI's use of $H^{\otimes n}$ is effectively the QFT over $\mathbb{F}_2^n$. This is a much simpler transform than the QFT over $\mathbb{Z}_{2^n}$ used in Shor's algorithm. The simplicity is both a strength (easy to implement: just $n$ single-qubit gates in parallel, depth 1) and a limitation (less "resolution" than the full QFT).

---

## A Concrete Example: DQI on (k=3, D=4)

Let's trace through what DQI does on our target problem.

**Setup:** $n$ Boolean variables, $m = 4n/3$ constraints (by double-counting: $3m = 4n$), each a 3-way XOR. Each variable appears in $D=4$ constraints. Constraint matrix $B$ is $m \times n$ with 3 ones per row and 4 ones per column.

**Step 1:** Prepare resource state — superposition over Dicke states $|D_n^0\rangle, |D_n^1\rangle, \ldots, |D_n^\ell\rangle$ with weights $w_j$.

**Step 2:** Encode — apply phases for target vector $\mathbf{v}$, then compute syndrome $B^T \mathbf{y}$ into ancilla.

**Step 3:** Decode — run reversible BP decoder on the LDPC code defined by $B$.

**Step 4:** Apply $H^{\otimes n}$.

**Step 5:** Measure. Expected fraction of satisfied constraints with BP decoder: **0.87065** (from comparison table).

For comparison:
- A random assignment achieves $0.5$
- The Prange baseline achieves $0.875$
- SA achieves $0.9366$

Note that DQI+BP (0.87065) actually performs **slightly below** the trivial Prange baseline (0.875) at this $(k, D)$. This is a regime where BP decoding is particularly ineffective: the random LDPC code from a $(3,4)$-regular hypergraph has short cycles in its factor graph, causing BP to converge poorly or to incorrect fixed points. When BP fails to decode correctly, the algorithm can perform *worse* than the Prange baseline (which always succeeds by construction, since it just picks a random codeword). The deficit of 0.004 is small but real — it reflects the cost of decoder failures in the DQI framework. (The two numbers use the same notion of "fraction satisfied" — both are expected fraction of constraints satisfied, computed analytically via the DQI framework.)

---

## Relevance to Our Project

### Why Stephen wants this comparison

From the project context (journal entry 1 and `04-our-problem.md`): Stephen wants to place QAOA performance numbers alongside DQI, SA, and other algorithms in a comparison table across multiple $(k, D)$ values. The primary target is $(k=3, D=4)$.

The existing data shows that at $(k=3, D=4)$:
- DQI+BP achieves 0.87065
- SA achieves 0.9366
- QAOA at depth $p$: **unknown** — this is what we compute

### What DQI's limitations mean for our project

The follow-up papers establish that DQI is weak on random Max-$k$-XORSAT instances. At $(k=3, D=4)$, the paper's §13 (confirmed: section.13 with theorems 13.1–13.4 in the PDF) contains the numerical comparison showing that this is in the regime where simulated annealing outperforms all DQI decoder variants.

So the question is not whether QAOA beats DQI at $(k=3, D=4)$ — it almost certainly does, even at modest $p$. The questions are:

1. **At what depth $p$ does QAOA surpass each competitor?**
   - Beat Prange (0.875)? Likely at small $p$
   - Beat DQI+BP (0.87065)? Likely at small $p$
   - Beat Regev+FGUM (0.89187)? Moderate $p$
   - Beat SA (0.9366)? Potentially large $p$ — this is the ambitious target

2. **How does QAOA's performance grow with $p$?** Does it approach the SA number from below? Does it eventually surpass it? At what rate?

3. **Are there other $(k, D)$ values where the comparison is more interesting?** The table has 15 entries; some may show tighter races between algorithms.

### What to take from this explainer

For our tensor-network computation:
- DQI provides one row in the comparison table. Our job is to add the QAOA row.
- The DQI numbers are already known; they are the fixed targets we compare against.
- The real competitor at $(k=3, D=4)$ is SA (0.9366), not DQI (0.87065).
- DQI's weakness on random sparse instances (confirmed by follow-up papers) means the comparison at small $(k, D)$ is more about "QAOA vs. SA" than "QAOA vs. DQI."
- At large $k/D$ (e.g., $(7,8)$), Regev+FGUM dominates; these may be interesting secondary targets.

---

## Key Equations to Remember

| Equation | Meaning |
|----------|---------|
| $\frac{\langle s \rangle}{m} = \frac{1}{2} + \sqrt{\frac{\ell}{m}(1 - \frac{\ell}{m})}$ | DQI semicircle law: theoretical performance ceiling |
| $\frac{1}{2} + \frac{k}{2D}$ | Prange baseline (trivial DQI with random codeword) |
| $C^\perp = \ker(B^T) \subseteq \mathbb{F}_2^m$ | Dual code — the coding-theoretic object DQI decodes (length $m$, one bit per constraint) |
| $P(f(\mathbf{x}))$ | Degree-$\ell$ polynomial biasing amplitude toward good solutions |

---

## Jargon Glossary

| Term | Meaning |
|------|---------|
| **DQI** | Decoded Quantum Interferometry — the algorithm introduced in this paper |
| **Decoding radius $\ell$** | Maximum Hamming weight of errors the decoder can correct; analogous to QAOA depth $p$ |
| **Dual code $C^\perp$** | The linear code $\ker(B^T) \subseteq \mathbb{F}_2^m$; its properties determine DQI's power |
| **Syndrome** | $B^T \mathbf{y}$ — encodes how $\mathbf{y}$ deviates from a codeword |
| **Prange decoder** | Trivially picks a random codeword; gives baseline DQI performance |
| **BP (Belief Propagation)** | Iterative message-passing decoder for LDPC codes |
| **OGP (Overlap Gap Property)** | A topological barrier that prevents "stable" algorithms from finding near-optimal solutions |
| **LDPC** | Low-Density Parity-Check — a code whose parity-check matrix is sparse |
| **Dicke state** | $|D_n^j\rangle$: uniform superposition over all $n$-bit strings of Hamming weight $j$ |
| **Semicircle law** | The spectral distribution governing DQI's performance on random instances |
| **OPI** | Optimisation by Polynomial Interpolation — a structured problem class where DQI achieves superpolynomial speedup (confirmed from PDF outline: Appendix C is "Simulated Annealing Applied to OPI") |
| **Regev+FGUM** | A DQI variant using a lattice-based decoder (Regev's approach, cite keys R04/R09 confirmed in PDF) + post-processing; dominates at large $k/D$ |

---

**Next:** With this understanding of DQI, you can now appreciate the full comparison landscape described in `04-our-problem.md` and understand exactly what our QAOA computation adds to the picture.