# Our Problem: Connecting Everything Together

> **Read this after:** All three paper explainers  
> **Purpose:** Synthesise what you've learned and understand exactly what we're doing and why

---

## The One-Paragraph Summary

Stephen Jordan — lead author of the DQI (Decoded Quantum Interferometry) paper published in Nature — wants to compare QAOA against DQI on Max-3-XORSAT on 4-regular hypergraphs at specific small $(k,D)$, particularly $(k{=}3, D{=}4)$. The existing QAOA analysis (Basso et al.) is only accurate for large $D$ and gives $O(1/D)$ errors — too imprecise at $D=4$. The Farhi et al. 2025 paper demonstrates an exact tensor network method for MaxCut ($k=2$) that works at any $D$. Our job is to **adapt that exact method to k=3 XORSAT on D=4 regular hypergraphs** and compute QAOA performance at each depth $p$, pushing $p$ as high as computationally feasible.

---

## What is Max-k-XORSAT, Precisely?

### The problem instance

An instance of Max-$k$-XORSAT on a $D$-regular $k$-uniform hypergraph consists of:

- **$n$ Boolean variables** $x_1, x_2, \ldots, x_n \in \{0,1\}$
- **$m$ constraints** (hyperedges), each involving exactly $k$ variables
- Each constraint $\alpha$ has a **target bit** $b_\alpha \in \{0,1\}$
- Constraint $\alpha$ on variables $(x_{i_1}, x_{i_2}, \ldots, x_{i_k})$ is **satisfied** when $x_{i_1} \oplus x_{i_2} \oplus \cdots \oplus x_{i_k} = b_\alpha$
- Each variable appears in exactly $D$ constraints ($D$-regularity)

The goal: find an assignment of $(x_1, \ldots, x_n)$ that **maximises the number of satisfied constraints**.

### Our specific case: k=3, D=4

- Each constraint is a 3-way XOR: e.g., "$x_2 \oplus x_5 \oplus x_9 = 1$"
- Each variable appears in exactly 4 constraints
- By double-counting: $3m = 4n$, so $m = 4n/3$ (there are more constraints than variables)
- The **fraction of satisfied constraints** is $c/m$ where $c$ is the number satisfied

### Connection to MaxCut

MaxCut is the special case $k=2$:
- Each "constraint" is an edge $(i,j)$ asserting "$x_i \oplus x_j = 1$" (i.e., the endpoints differ)
- The fraction of satisfied constraints = the cut fraction

So everything the papers do for MaxCut carries over to k-XORSAT with appropriate modifications to handle the $k$-body interaction.

---

## What is DQI?

DQI stands for **Decoded Quantum Interferometry** — a quantum algorithm introduced by Stephen P. Jordan, Noah Shutty, Mary Wootters, Adam Zalcman, Alexander Schmidhuber, Robbie King, Sergei V. Isakov, Tanuj Khattar, and Ryan Babbush. Published in **Nature 646:831-836 (2025)** ([arXiv:2408.08292](https://arxiv.org/abs/2408.08292)).

DQI uses the **quantum Fourier transform** to reduce optimisation problems to **decoding problems**. The key idea:
1. Encode the optimisation problem so that good solutions correspond to codewords of a classical error-correcting code
2. Use quantum interference (via QFT) to amplify near-codeword states
3. Apply a classical decoder to extract good solutions upon measurement

For problems with algebraic structure (e.g., polynomial optimisation over finite fields), DQI achieves a **superpolynomial speedup** over known classical algorithms. For sparse-clause problems like Max-XORSAT, the optimisation reduces to decoding LDPC codes, for which powerful classical decoders exist.

Stephen has DQI performance numbers for Max-k-XORSAT at specific $(k, D)$ values. The comparison question is: **how does QAOA at depth $p$ compare against DQI?** Does QAOA match or surpass DQI performance at some feasible $p$?

This comparison is significant because QAOA and DQI are fundamentally different quantum approaches to the same problem — comparing them illuminates which structural features each exploits.

### How DQI works in more detail

The DQI pipeline for Max-$k$-XORSAT:

1. **Prepare a resource state** — a superposition over Dicke states (states with a fixed number of 1s), weighted by coefficients $w_0, \ldots, w_\ell$. This is instance-independent.
2. **Encode the problem** — apply phases $(-1)^{\mathbf{v} \cdot \mathbf{y}}$ encoding the target vector $\mathbf{v}$, then compute the syndrome $B^T \mathbf{y}$ into an ancilla register (where $B$ is the constraint matrix of the XORSAT instance).
3. **Decode** — use a reversible classical decoder for the code $C^\perp = \{\mathbf{d} : B^T \mathbf{d} = \mathbf{0}\}$ to recover $\mathbf{y}$ from $B^T\mathbf{y}$ and uncompute the error. This is syndrome decoding with a Hamming weight constraint $|\mathbf{y}| \le \ell$.
4. **Hadamard transform** — apply $H^{\otimes n}$ to obtain a state $\sum_{\mathbf{x}} P(f(\mathbf{x})) |\mathbf{x}\rangle$, where $P$ is a degree-$\ell$ polynomial biasing toward high objective values.
5. **Measure** — sample $\mathbf{x}$ with probability $\propto P(f(\mathbf{x}))^2$.

The key parameter is $\ell$ — the decoding radius. Higher $\ell$ means stronger bias toward optimal solutions, but the decoder must be able to correct $\ell$ errors.

### The semicircle law

DQI's central performance guarantee: for Max-XORSAT, the expected satisfaction fraction is governed by the **semicircle law**:

$$\frac{\langle s \rangle}{m} = \frac{1}{2} + \sqrt{\frac{\ell}{m}\left(1 - \frac{\ell}{m}\right)}$$

where $\ell$ is the number of decodable errors and $m$ is the number of constraints. At the Shannon limit of decoding, this gives an information-theoretic ceiling on DQI's performance.

### DQI's limitations on unstructured Max-k-XORSAT

A series of follow-up papers (see below) have shown:

- **DQI requires structure** (Anschuetz, Gamarnik, Lu — arXiv:2509.14509): On random Gallager-ensemble LDPC instances, DQI is blocked by the **overlap gap property (OGP)** — a topological barrier that prevents any "stable" (Lipschitz) algorithm from finding near-optimal solutions. Classical AMP (approximate message passing) matches or exceeds DQI.
- **No quantum advantage for MaxCut** (Parekh — arXiv:2509.19966): For Max-2-XORSAT, the dual code $C^\perp$ is a cycle code with minimum distance $O(\log n)$, severely limiting DQI's decoding radius. QAOA at $p=17$ achieves 0.8971 on 3-regular graphs; DQI's upper bound is $1/2 + 1/(2\sqrt{D-1})$.
- **DQI's power lies in algebraic structure**: Reed-Solomon/OPI problems, where the dual code has large minimum distance, are where DQI achieves superpolynomial speedup.

### What this means for $(k=3, D=4)$

From the original DQI paper (Fig. 13 and §13): $(k=3, D=4)$ is in the regime (the "orange region") where **even the information-theoretic upper bound on DQI with classical decoders is exceeded by simulated annealing and QAOA**. The paper explicitly states that QAOA exceeds DQI's upper bound at $k=2$ and $k=3$.

So Stephen likely wants precise QAOA numbers to **quantify by how much** QAOA exceeds DQI at this specific point — not to discover whether it does, but to measure the gap precisely.

---

## The Known Landscape: Stephen's Current Data

Stephen has shared performance data across multiple $(k,D)$ values. Here are the actual numbers (bold = best for each row):

| $(k,D)$ | Prange | Simulated Annealing | DQI+BP | Regev+FGUM |
|---------|--------|-------------------|--------|------------|
| **(3,4)** | 0.875 | **0.9366** | 0.87065 | 0.89187 |
| (3,5) | 0.8 | **0.9005** | 0.81648 | 0.83607 |
| (3,6) | 0.75 | **0.8712** | 0.77562 | 0.78361 |
| (3,7) | 0.71428 | **0.8492** | 0.74727 | 0.76024 |
| (3,8) | 0.6875 | **0.8287** | 0.72351 | 0.72943 |
| (4,5) | 0.9 | **0.9279** | 0.8597 | 0.92158 |
| (4,6) | 0.83333 | **0.9024** | 0.82062 | 0.86144 |
| (4,7) | 0.78571 | **0.8771** | 0.78862 | 0.82645 |
| (4,8) | 0.75 | **0.8587** | 0.76539 | 0.79021 |
| (5,6) | 0.91667 | 0.9190 | 0.84305 | **0.93123** |
| (5,7) | 0.85714 | **0.8965** | 0.81422 | 0.88529 |
| (5,8) | 0.8125 | **0.8740** | 0.78752 | 0.84403 |
| (6,7) | 0.92857 | 0.9051 | 0.82759 | **0.94271** |
| (6,8) | 0.875 | 0.8875 | 0.80327 | **0.89619** |
| (7,8) | 0.9375 | 0.8951 | 0.813 | **0.94810** |

### Algorithms explained

- **Prange** = $1/2 + k/(2D)$ — trivial baseline from random coding theory (DQI with Prange decoder, i.e., random codeword selection)
- **Simulated Annealing** — classical heuristic; dominates at small $k$ relative to $D$
- **DQI+BP** — DQI with belief propagation decoder; never the best at any $(k,D)$ in this table
- **Regev+FGUM** — another quantum/quantum-inspired method; dominates when $k/D$ is large (close to 1)

### Key observations

1. **DQI+BP is never the winner** — always beaten by SA or Regev+FGUM
2. **SA dominates at small $k$** ($k=3,4$): the gap is substantial (e.g., 0.9366 vs 0.89187 at $(3,4)$)
3. **Regev+FGUM dominates at large $k/D$** (when $k \approx D$): $(5,6), (6,7), (6,8), (7,8)$
4. **The crossover** is around $k/D \approx 0.8$

### What QAOA needs to achieve

For our primary target $(k=3, D=4)$:
- Beat Prange (0.875) — should be easy at modest $p$
- Beat DQI+BP (0.87065) — should also be easy
- Beat Regev+FGUM (0.89187) — likely achievable at moderate $p$
- **Beat SA (0.9366)** — this is the real target; requires significant $p$

Our computation adds a **QAOA** column to this table, with precise values at each depth $p$.

---

## The Factor Graph Tree for (k=3, D=4)

### Structure

At depth $p$, the QAOA light cone for a single constraint is a factor graph tree:

```
Level 0 (root):     [α₀]                    ← 1 root constraint (hyperedge)
                   / | \
Level 1:        (x₁)(x₂)(x₃)               ← k=3 variable nodes
               /|\  /|\  /|\
Level 2:    [·][·][·] [·][·][·] [·][·][·]   ← (D-1)=3 constraints per variable
           /|\ ...  /|\ ...                       = 9 constraints total
Level 3:  (·)(·)(·)(·)...                    ← (k-1)=2 variables per constraint
                                                  = 18 variables total
... continuing to depth 2p (p constraint layers, p variable layers)
```

**Node types alternate:**
- Even levels: constraint nodes (hyperedge factors), each connecting to $k$ variable nodes below
- Odd levels: variable nodes, each connecting to $(D-1)$ constraint nodes below

### Size growth

The branching factor per two-level step is $(D-1)(k-1) = 3 \times 2 = 6$ for our case.

| Level | Type | Count |
|-------|------|-------|
| 0 | Constraint | 1 |
| 1 | Variable | 3 |
| 2 | Constraint | 9 |
| 3 | Variable | 18 |
| 4 | Constraint | 54 |
| 5 | Variable | 108 |
| ... | ... | ... |

At level $2\ell$: $3 \times 6^{\ell-1}$ constraints (approximately). At level $2\ell+1$: $6 \times 6^{\ell-1} = 6^\ell$ variables.

At depth $p$ (meaning $p$ rounds of the QAOA, which involves $2p$ levels of the factor graph), the total number of **variable nodes** (qubits) is approximately $\sum_{\ell=0}^{p} 6^\ell \approx 6^p$.

But remember: **we don't need to store the full state.** The tensor contraction trick handles this.

### The contraction cost for k=3

For MaxCut ($k=2$), the Farhi 2025 paper achieves $O(4^p)$ cost, independent of $D$. For $k=3$, the tensor structure changes:

- Each constraint node involves a $k$-body interaction → the problem gate tensor has $k=3$ indices instead of 2
- The tensor "sandwich" at each layer of the contraction involves more indices

The exact cost depends on how the contraction is structured. A careful analysis is needed (this is Phase 1 of our work plan), but the rough expectation is:
- Time: $O(4^p)$ to $O(8^p)$ depending on the contraction strategy
- Space: similar

Even at $O(8^p)$:

| $p$ | $8^p$ | Feasibility |
|-----|-------|-------------|
| 5   | 32,768 | Trivial |
| 8   | ~$1.7 \times 10^7$ | Easy |
| 10  | ~$10^9$ | Feasible |
| 12  | ~$7 \times 10^{10}$ | Feasible (hours) |
| 15  | ~$3.5 \times 10^{13}$ | Hard (cluster, days) |
| 17  | ~$2.3 \times 10^{15}$ | Very hard |

So we're likely looking at $p_{\max}$ somewhere between 10 and 15 for $(k=3, D=4)$, depending on optimisation and available compute. This is a rough estimate — the actual cost needs to be determined by working through the tensor contraction in detail.

---

## What Exactly We Need to Compute

For each depth $p = 1, 2, 3, \ldots, p_{\max}$:

1. **Build the factor graph tree** for $(k=3, D=4)$ at depth $p$
2. **Set up the tensor network** for computing $\langle \boldsymbol{\gamma}, \boldsymbol{\beta} | C_\alpha | \boldsymbol{\gamma}, \boldsymbol{\beta} \rangle$ where $C_\alpha$ is the cost operator for the root constraint
3. **Contract the tensor network** using the tree contraction with element-wise exponentiation
4. **Optimise** $(\boldsymbol{\gamma}, \boldsymbol{\beta})$ over the $2p$ angles to maximise the expected fraction of satisfied constraints
5. **Report** the optimal fraction and the optimal angles

The result is a table:

| $p$ | Optimal fraction satisfied | Optimal $\boldsymbol{\gamma}$ | Optimal $\boldsymbol{\beta}$ |
|-----|---------------------------|-------------------------------|------------------------------|
| 1   | ???                       | ...                           | ...                          |
| 2   | ???                       | ...                           | ...                          |
| 3   | ???                       | ...                           | ...                          |
| ... | ...                       | ...                           | ...                          |

Stephen will then overlay this against his DQI numbers.

---

## The Cost Operator for k-XORSAT in Quantum Form

For a constraint $\alpha$ on variables $(i_1, i_2, i_3)$ with target bit $b_\alpha$:

$$C_\alpha = \frac{1 - (-1)^{b_\alpha} Z_{i_1} Z_{i_2} Z_{i_3}}{2}$$

**Why?** The operator $Z_{i_1} Z_{i_2} Z_{i_3}$ has eigenvalue $(-1)^{x_{i_1} \oplus x_{i_2} \oplus x_{i_3}}$ on computational basis state $|x\rangle$. So:

- If the XOR equals $b_\alpha$ (constraint satisfied): $(-1)^{b_\alpha} \cdot (-1)^{b_\alpha} = 1$, so $C_\alpha = 0$... 

Wait, that gives 0 for satisfied. Let me redo this. We want $C_\alpha = 1$ when satisfied and $C_\alpha = 0$ when not.

The constraint is satisfied when $x_{i_1} \oplus x_{i_2} \oplus x_{i_3} = b_\alpha$.

$Z_{i_1}Z_{i_2}Z_{i_3}$ on $|x_{i_1} x_{i_2} x_{i_3}\rangle$ gives $(-1)^{x_{i_1}+x_{i_2}+x_{i_3}}$. Note that $(-1)^{x_{i_1}+x_{i_2}+x_{i_3}} = (-1)^{x_{i_1} \oplus x_{i_2} \oplus x_{i_3}}$ when only looking at the parity (since $(-1)$ raised to an integer only depends on parity).

So $(-1)^{b_\alpha} Z_{i_1}Z_{i_2}Z_{i_3}$ gives $(-1)^{b_\alpha + x_{i_1} \oplus x_{i_2} \oplus x_{i_3}}$:
- When XOR = $b_\alpha$: this gives $(-1)^{2b_\alpha} = +1$
- When XOR $\neq$ $b_\alpha$: this gives $(-1)^{b_\alpha + 1 - b_\alpha} = (-1)^1 = -1$

Hence:

$$C_\alpha = \frac{1 + (-1)^{b_\alpha} Z_{i_1} Z_{i_2} Z_{i_3}}{2}$$

This gives $C_\alpha = 1$ when satisfied and $C_\alpha = 0$ when not. (Note the + sign, not the - sign I had before!)

**For random constraints** where $b_\alpha$ is equally likely 0 or 1, the average performance on a tree (with no information about $b_\alpha$ values of distant constraints) is the same regardless of the specific $b_\alpha$ choices. So in practice, we can set all $b_\alpha = 0$ or all $b_\alpha = 1$ without loss of generality (on a tree with no loops). This needs to be verified but is standard in the literature.

The full cost function is $C = \sum_\alpha C_\alpha$, and the QAOA problem unitary is:

$$U(C, \gamma) = e^{-i\gamma C} = \prod_\alpha e^{-i\gamma C_\alpha}$$

Each factor $e^{-i\gamma C_\alpha}$ is a diagonal $k$-body gate:
$$e^{-i\gamma(1 + Z_{i_1}Z_{i_2}Z_{i_3})/2} = e^{-i\gamma/2} \cdot e^{-i\gamma Z_{i_1}Z_{i_2}Z_{i_3}/2}$$

The global phase $e^{-i\gamma/2}$ is irrelevant. The essential gate is $e^{-i\gamma Z_{i_1}Z_{i_2}Z_{i_3}/2}$.

---

## Key Questions to Discuss with Stephen

1. **DQI data:** Can you share the specific DQI numbers for $(k=3, D=4)$? From your Nature paper Fig. 13, $(k=3, D=4)$ appears to be in the regime where QAOA exceeds DQI — are you looking to quantify the precise gap, or has the picture changed?

2. **DQI depth parameter:** DQI has a decoding radius $\ell$ analogous to QAOA's depth $p$. Are you comparing at a specific $\ell$, or at DQI's optimal $\ell$, or against the Shannon-limit ceiling?

3. **Which decoder?** The DQI numbers depend heavily on the decoder (BP, Prange, Shannon limit). Which comparison point matters most?

4. **Existing code:** Do you know of any public implementations of the exact tensor-network QAOA evaluation (from Farhi et al. 2025 or related work)? Their code is C++/OpenMP — is it available?

5. **Other $(k,D)$ values:** Beyond $(3,4)$, are there other $(k,D)$ pairs you want to compare? The same code should work for any $(k,D)$ — only the tree structure and problem gates change.

6. **XORSAT target bits:** Do the DQI bounds depend on the specific choice of $b_\alpha$ values, or are they averaged over random instances? (This determines whether we can simplify by fixing all $b_\alpha = 0$.)

7. **The "structure" question:** Given the results of Anschuetz et al. (DQI requires structure), is the comparison specifically about *unstructured* random instances, or instances where the constraint matrix has particular coding-theoretic properties?

---

## Summary: The Path from Papers to Code

```
┌──────────────────┐
│ Farhi 2014       │  QAOA framework: circuit structure, light cone idea
│ (arXiv:1411.4028)│  → defines the problem
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Basso et al. 2021│  Iterative formula for large D, generalises to k-XORSAT
│ (arXiv:2110.14206│  → proves feasibility, gives O(1/D) baseline
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐
│ Farhi et al. 2025    │  Exact tensor network method, O(4^p), independent of D
│ (arXiv:2503.12789)   │  → the method we adapt
└────────┬─────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│              OUR CONTRIBUTION                 │
│                                               │
│  1. Generalise tensor network from MaxCut     │
│     (k=2) to k-XORSAT (k=3)                  │
│  2. Implement in high-performance code        │
│  3. Run for (k=3, D=4) at p=1,2,...,p_max     │
│  4. Optimise angles at each p                 │
│  5. Compare against DQI bounds                │
└──────────────────────────────────────────────┘
```

---

## What Success Looks Like

A successful project delivers:

1. **A table of numbers**: QAOA optimal fraction of satisfied constraints for Max-3-XORSAT on 4-regular hypergraphs, at each $p$ up to some $p_{\max}$
2. **Validated code** that can also reproduce known MaxCut results as a sanity check
3. **A clear comparison** with Stephen's DQI numbers
4. **Understanding** of whether QAOA surpasses DQI at some finite $p$, and if so, where

Bonus outcomes:
- Results for other $(k,D)$ values
- Published paper or contribution to Stephen's paper
- Open-source code that others can use

---

## Further Reading: DQI Papers

In addition to the three QAOA papers and the original DQI paper, these follow-up papers provide crucial context:

| Paper | Key finding |
|-------|-------------|
| **Anschuetz, Gamarnik, Lu** (arXiv:2509.14509) | DQI is blocked by OGP on random LDPC instances; classical AMP matches/exceeds DQI |
| **Parekh** (arXiv:2509.19966) | No quantum advantage for DQI on MaxCut; QAOA far exceeds DQI's ceiling |
| **Khattar, Shutty, et al. + Jordan** (arXiv:2510.10967) | Optimised DQI circuits for OPI — where DQI's power actually lies (structured problems) |
| **Kramer, Schubert, Eisert** (arXiv:2603.04540) | Tight inapproximability: no algorithm beats $r/q$ without exploiting structure |

PDFs of all these are in `../papers/`.

---

**You now have enough context to read the actual papers and to hold a meaningful technical conversation with Stephen. Good luck!**
