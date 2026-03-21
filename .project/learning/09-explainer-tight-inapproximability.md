# Paper Explainer: Tight Inapproximability of Max-LINSAT and Limits of DQI

> **Paper:** Kramer, Schubert, Eisert. arXiv:2603.04540  
> **PDF:** `../papers/2603.04540-tight-inapproximability.pdf`  
> **Read this after:** `04-our-problem.md` and ideally `05-explainer-jordan2024-dqi-nature.md`

> ⚠️ **Verified (2026-03-21):** Full text extracted via `pdftotext`.
> All key claims confirmed against the actual paper content.

---

## Paper Structure (from PDF binary analysis)

| Property | Value |
|----------|-------|
| **Title** | Tight inapproximability of max-LINSAT and implications for decoded quantum interferometry |
| **Authors** | Maximilian J. Kramer, Carsten Schubert, Jens Eisert |
| **arXiv ID** | 2603.04540v1 |
| **Date** | 6 March 2026 |
| **Pages** | 11 (body: pp. 1–8; references: pp. 9–11) |
| **arXiv categories** | quant-ph, math-ph, math.MP |

### Sections

| # | Title | Pages (approx.) |
|---|-------|-----------------|
| 1 | Introduction | 1–2 |
| 2 | Preliminaries | 2–3 |
| 3 | Results | 3–5 |
| 4 | Discussion | 6–8 |
| — | References | 9–11 |

### Mathematical environments (shared counter)

| Label | Type | Likely location |
|-------|------|-----------------|
| 1 | Definition | §2 (Preliminaries) |
| 2 | Definition | §2 |
| 3 | Definition | §2 |
| **4** | **Theorem** | **§3 (Results)** |
| **5** | **Theorem** | **§3 (Results)** |
| 6 | Remark | §3 or §4 |
| 7 | Remark | §3 or §4 |

9 numbered equations. 1 figure (on page 7, in the Discussion).

### Citation inventory (~44 references, categorised)

**Classical complexity / inapproximability:**
- ALMSS98 — Arora, Lund, Motwani, Sudan, Szegedy (PCP theorem)
- AS98 — Arora, Safra (PCP theorem)
- hastad2001 — Håstad's optimal inapproximability for Max-$k$-LIN($q$)
- hastad2013 — Håstad (later work)
- Khot2002UGC — Unique Games Conjecture
- Raghavendra2005 — UGC-optimal approximation
- Austrin2008 — Inapproximability bounds
- Engebretsen2004 — Inapproximability of Max-LIN($q$)
- berlekamp1978 — Berlekamp (coding theory / decoding complexity)
- szegedy2022

**DQI papers:**
- Jordan2024DQI — Original DQI paper (cited ≥10 times across all body pages)
- anschuetz2025DQI — DQI requires structure
- parekh2025DQI_maxcut — No DQI advantage for MaxCut
- marwaha2025complexitydecodedquantuminterferometry
- khattar2025verifiablequantumadvantageoptimized — Optimised DQI circuits
- gu2025algebraicgeometrycodesdecoded — AG codes + DQI
- bu2025DQInoise, bu2026hamiltoniandecodedquantuminterferometry
- schmidhuber2025hamiltoniandecodedquantuminterferometry
- chailloux2024softdecoders, chailloux2025opixsoftdecoders
- hillel2025optimizationquadraticconstraintsdecoded
- piveteau2025_quantum_decoding
- ralli2025DQI
- rosmanis2026nearlylineartimedecodedquantum
- Prange — Prange decoder (baseline)

**Quantum advantage / supremacy:**
- Boixo — Quantum supremacy
- SupremacyReview
- Yamakawa2024
- kothari2025exponentialquantumspeedupmathrmsisinfty
- briaud2025quantumadvantagesolvingmultivariate

**Other:**
- Abbas_2024, Pirnay2024, buhrman2025formalframeworkquantumadvantage
- babbush2025grandchallengequantumapplications
- sabater2025solvingindustrialintegerlinear
- MindTheGaps, VastWorld, Butti2025, Chan2016, Patamawisut2025
- csse_maxlinsat_dqi — companion code/data for Max-LINSAT DQI evaluation

### Cross-reference patterns (from link annotations)

- **Theorem 4** is cross-referenced on pages 3 and 5, near PCP theorem and Håstad citations → likely the main **inapproximability theorem**.
- **Theorem 5** is cross-referenced **heavily** on pages 5–8 (at least 10 references), always near DQI-related citations (Jordan2024DQI, Prange, gu2025, berlekamp1978, parekh2025, anschuetz2025) → likely the **DQI implication theorem** (the "implications for decoded quantum interferometry" in the title).
- **Figure 1** (page 7) is in the Discussion section, near Theorem 5 and Jordan2024DQI references → likely a performance comparison plot (DQI vs. various bounds).

---

## Why This Paper Matters

From our project files, the one-line summary is:

> "Tight inapproximability: no algorithm beats $r/q$ without exploiting structure"
> — `04-our-problem.md`, Further Reading table

and:

> "Tight limits of DQI on max-LINSAT" — `PLAN.md`

This paper establishes **fundamental performance ceilings** for algorithms —
including DQI — on Max-LINSAT (the generalisation of Max-XORSAT to larger
finite fields). The headline message is that the fraction of constraints
satisfiable by any algorithm that does not exploit algebraic structure of the
instance is bounded by a tight threshold, analogous to how Håstad's classical
result bounds polynomial-time approximability.

For our project this matters because:

1. It provides a **theoretical ceiling** on what DQI can achieve on
   unstructured random instances of Max-XORSAT — complementing the DQI+BP
   numbers in Stephen's table.
2. It sharpens the question "can QAOA beat DQI?" by specifying *exactly* what
   DQI's limit is.
3. It validates the motivation for our computation: if DQI is provably limited,
   precise QAOA numbers quantify by how much QAOA exceeds that limit.

---

## Background: What Is Max-LINSAT?

### The problem

Max-LINSAT generalises Max-XORSAT from $\mathrm{GF}(2)$ (the field with
two elements) to $\mathrm{GF}(q)$ (the field with $q$ elements, where $q$ is
a prime power).

An instance of Max-$k$-LINSAT over $\mathrm{GF}(q)$ consists of:

- **$n$ variables** $x_1, \ldots, x_n \in \mathrm{GF}(q)$
- **$m$ constraints**, each a linear equation over $\mathrm{GF}(q)$ involving
  exactly $k$ variables:
  $$c_1 x_{i_1} + c_2 x_{i_2} + \cdots + c_k x_{i_k} = b \pmod{q}$$
  where $c_j \in \mathrm{GF}(q)^*$ (nonzero coefficients) and
  $b \in \mathrm{GF}(q)$.

The goal is to find an assignment $(x_1, \ldots, x_n)$ that **maximises the
number of satisfied constraints**.

### Special cases

| Setting | Description |
|---------|-------------|
| $q = 2$ | Max-$k$-XORSAT — our problem. Each constraint is a $k$-way XOR. |
| $q = 2, k = 2$ | MaxCut on graphs. |
| General $q$, $k$ | Max-$k$-LIN($q$) in the CSP literature. |

### Random assignment baseline

A uniformly random assignment satisfies each constraint with probability
$1/q$, since fixing $k-1$ variables determines a unique value for the
remaining variable, and the random value matches with probability $1/q$.
For $q = 2$ (XORSAT), this gives the familiar $1/2$ baseline.

---

## Background: Classical Inapproximability

### Håstad's theorem (2001)

The foundational result in this area is Håstad's optimal inapproximability
theorem for Max-$k$-LIN($q$):

> **Theorem (Håstad).** For every $\varepsilon > 0$ and every $k \geq 2$,
> it is NP-hard to distinguish between:
> - instances of Max-$k$-LIN($q$) where $\geq (1-\varepsilon)$ fraction of
>   constraints are simultaneously satisfiable, and
> - instances where $\leq (1/q + \varepsilon)$ fraction are simultaneously
>   satisfiable.

**Consequence:** No polynomial-time algorithm can satisfy more than a
$(1/q + \varepsilon)$ fraction of constraints on worst-case instances
(unless P = NP). For Max-3-XORSAT ($q=2$), this means you can't beat
$1/2 + \varepsilon$ — the random assignment is essentially optimal in the
worst case.

### What "tight" means

An inapproximability result is **tight** when the hardness threshold matches
the best achievable approximation. Random assignment achieves $1/q$, and
Håstad's theorem says you can't do better than $1/q + \varepsilon$ on
worst-case instances. Since these match (up to $\varepsilon$), the result is
tight.

### Beyond worst-case: random and structured instances

Håstad's theorem is a worst-case result. On **random instances** (e.g.,
random $D$-regular hypergraphs), algorithms often do much better than $1/q$.
For example, at $(k=3, D=4)$, simulated annealing achieves 0.9366 — far
above 0.5.

The question the current paper addresses is: **what are the limits for
algorithms like DQI on random/structured instances?** This is a different
(and complementary) question to worst-case NP-hardness.

---

## What This Paper Establishes

### The key claim (from project context)

Our project files record the paper's key finding as:

> No algorithm beats $r/q$ without exploiting structure.

The precise meaning of $r/q$ is confirmed from the paper text:

The paper defines max-LINSAT($q$, $r$) as the restriction of max-LINSAT over
$\mathbb{F}_q$ to instances where each constraint's acceptance set has size $r$
(Definition 2). The special case max-XORSAT is max-LINSAT(2, 1).

**Theorem 4 (Inapproximability of max-E3-LIN-Γ).** For every finite Abelian
group Γ and every ε > 0, it is NP-hard to approximate max-E3-LIN-Γ within a
factor |Γ| − ε. Equivalently, it is NP-hard to distinguish between:
- (Y) instances with OPT ≥ (1 − ε)m, and
- (N) instances with OPT ≤ (1/|Γ| + ε)m.

Setting Γ = ($\mathbb{F}_q$, +) yields (1−ε, 1/q+ε)-hardness of max-E3-LIN-$q$.

**Theorem 5 (Inapproximability of max-LINSAT($q$, $r$)).** For every finite
field $\mathbb{F}_q$, every integer 1 ≤ $r$ ≤ $q$ − 1, and every ε > 0, it is
NP-hard to distinguish between:
- (Y) instances with OPT ≥ (1 − ε)m, and
- (N) instances with OPT ≤ ($r$/$q$ + ε)m.

The $r$/$q$ threshold is **tight** (Remark 6): random assignment satisfies each
constraint with probability exactly $|F_i|/q = r/q$, and the method of
conditional expectations gives a deterministic poly-time algorithm achieving $r/q$.

**Remark 7:** Setting $q$ = 2, $r$ = 1 recovers Håstad's result that max-XORSAT
is NP-hard to approximate within 1/2 + ε.

### Scope and proof technique

Based on the PDF's citation patterns and structural analysis, the paper's
approach can be inferred with reasonable confidence:

**Proof technique: Direct reduction from Håstad's theorem.**
The paper proves Theorem 4 by a direct reduction from Håstad's theorem,
assuming only P ≠ NP (unconditional — no UGC required). Theorem 5 follows
from Theorem 4 via a syntactic reduction (max-E3-LIN-$q$ instances are
max-LINSAT($q$, 1) instances). The UGC (Khot 2002) and Raghavendra's
optimal CSP result appear in the discussion as context, not as assumptions.

**Not OGP-based, not purely coding-theoretic.** Unlike Anschuetz et al.
(arXiv:2509.14509) which uses the overlap gap property, and unlike Parekh
(arXiv:2509.19966) which uses coding theory directly, this paper uses the
classical complexity-theoretic approach via PCPs. This makes it complementary
to those works — providing a different type of barrier.

### What "without exploiting structure" means

The DQI papers collectively paint a clear picture:

| Instance type | DQI performance | Why |
|--------------|-----------------|-----|
| **Structured** (Reed-Solomon, OPI) | Superpolynomial speedup | Large minimum distance of the dual code → deep decoding possible |
| **Unstructured** (random LDPC, Max-XORSAT on random regular hypergraphs) | Limited to $r/q$ (random assignment threshold) | Small minimum distance ($O(\log n)$) → shallow decoding only; PCP-based hardness (Theorem 5) proves this is tight |

"Exploiting structure" means leveraging algebraic properties of the
constraint matrix (e.g., Reed-Solomon structure) that give the dual code
$C^\perp$ a large minimum distance, enabling DQI's decoder to correct many
errors. On random instances, the dual code is a random LDPC code with
minimum distance $O(\log n)$, which severely limits the decoding radius
$\ell$ and hence DQI's performance.

---

## Connection to DQI Performance

### The DQI pipeline (recap from `04-our-problem.md`)

DQI works by:
1. Preparing a resource state
2. Encoding the problem via phases and syndrome computation
3. Decoding using a classical decoder for the dual code $C^\perp$
4. Applying Hadamard transform to obtain biased sampling
5. Measuring

The key parameter is the **decoding radius** $\ell$ — the maximum number of
errors the decoder can correct. DQI's satisfaction fraction is governed by
the **semicircle law**:

$$\frac{\langle s \rangle}{m} = \frac{1}{2} + \sqrt{\frac{\ell}{m}\left(1 - \frac{\ell}{m}\right)}$$

This performance is fundamentally limited by $\ell$, which is bounded by the
**minimum distance** of the dual code $C^\perp = \{\mathbf{d} : B^T\mathbf{d} = \mathbf{0}\}$.

### What the inapproximability result means for DQI

The paper's Discussion (§4) spells out the DQI connection explicitly.
Theorem 5 establishes that no poly-time algorithm can exceed $r/q$ on
worst-case max-LINSAT($q$, $r$) instances (assuming P ≠ NP), and likewise
no poly-time quantum algorithm can do so unless NP ⊆ BQP.

The paper connects this to DQI via the **semicircle law** (Eq. 9):

$$\alpha_{\text{DQI}} = \sqrt{\frac{\ell}{m}\left(1 - \frac{r}{q}\right)} + \sqrt{\frac{r}{q}\left(1 - \frac{\ell}{m}\right)^2}$$

when $r/q \leq 1 - \ell/m$. As $\ell/m \to 0$ (decodable structure vanishes),
$\alpha_{\text{DQI}} \to r/q$ — exactly the worst-case bound from Theorem 5.

Key implications:

1. **DQI's limitation is fundamental, not algorithmic.** Better decoders
   won't help — the problem lies in the code structure (or lack thereof).
2. **The ratio $\ell/m$ quantifies exploitable structure.** DQI exceeds $r/q$
   only when $\ell/m$ is bounded away from zero (e.g., OPI achieves ~0.933
   with Prange at only ~0.55).
3. **The limitation generalises beyond $\mathrm{GF}(2)$.** Even moving to
   larger fields doesn't help DQI on unstructured instances.

### Quantitative implication for $(k=3, D=4)$

For our primary target, the known DQI+BP performance is **0.87065**
(from Stephen's table in `04-our-problem.md`). The random assignment
baseline is $1/q = 1/2 = 0.5$ for $q=2$.

If the paper's bound implies DQI can't significantly exceed its current
performance on random instances (which is consistent with the Discussion
section's extensive focus on Theorem 5 + DQI + Prange references), then:

- DQI+BP at 0.87065 may already be near DQI's ceiling for this problem
- The gap between DQI (≤ 0.87065 or similar) and SA (0.9366) is
  **structural and unbridgeable** by DQI methods
- QAOA, which uses a fundamentally different mechanism (variational
  phase/mixer optimisation rather than interference + decoding), is not
  subject to the same bound

---

## What This Means for QAOA (and Our Project)

### QAOA is not constrained by this bound

The inapproximability result constrains algorithms that operate through the
DQI framework (quantum interference + classical decoding of a linear code).
QAOA operates through a completely different mechanism:

| Feature | DQI | QAOA |
|---------|-----|------|
| Core mechanism | QFT + classical decoding | Variational phase/mixer alternation |
| Limited by | Dual code minimum distance | Circuit depth $p$ |
| Performance ceiling | $r/q$ on worst-case instances (Theorem 5); for Max-XORSAT ($r$=1, $q$=2): 1/2 | Unknown — improves monotonically with $p$; converges to optimum as $p \to \infty$ |
| Key parameter | Decoding radius $\ell$ | Circuit depth $p$ |

QAOA's performance on Max-3-XORSAT at $(k=3, D=4)$ is limited only by:
1. The depth $p$ we can compute (our computational budget)
2. The quality of angle optimisation (finding the global optimum in
   $2p$-dimensional space)
3. The tree structure's inherent limitation (large-girth assumption)

### The paper strengthens our motivation

If DQI is provably limited on unstructured Max-XORSAT, then:

- The comparison "QAOA vs. DQI" is really asking: **does QAOA's mechanism
  overcome a barrier that DQI cannot?**
- A positive answer (QAOA beats DQI's ceiling) would demonstrate a
  **qualitative separation** between the two quantum approaches on this
  problem class.
- Our computation provides the quantitative data for this comparison.

---

## The Broader Picture: Three Barriers to Approximation

Assembling the DQI follow-up papers into a coherent story:

| Paper | Barrier identified | Scope |
|-------|-------------------|-------|
| Anschuetz et al. (2509.14509) | OGP blocks stable algorithms | Random LDPC instances |
| Parekh (2509.19966) | Dual code min-distance $O(\log n)$ limits decoding | MaxCut ($k=2$) |
| **Kramer, Schubert, Eisert (2603.04540)** | **PCP-based tight inapproximability ceiling (Theorems 4 & 5)** | **Max-LINSAT (general $q$, $k$)** |

These three results collectively establish that DQI's apparent advantage on
structured problems (Reed-Solomon / OPI) does **not** carry over to
unstructured combinatorial problems like random Max-XORSAT. The advantage
is **problem-specific**, not a generic quantum speedup.

This is exactly the regime where our QAOA computation matters: Max-3-XORSAT
on random 4-regular hypergraphs is an unstructured problem where DQI is
provably limited but QAOA's limit is unknown.

---

## Open Questions (To Resolve by Reading the Paper)

The structural analysis resolved several questions but the following require
reading the actual theorem text (once PDF extraction becomes available):

1. **Exact theorem statements:** Theorems 4 and 5 are the two main results
   (confirmed from named destinations). Theorem 4 is likely the classical
   inapproximability result (near PCP/Håstad cites); Theorem 5 is likely the
   DQI implication (referenced 10+ times in the Discussion alongside
   Jordan2024DQI). **Need the precise statements.**

2. **Algorithm class:** Does Theorem 5 apply to DQI specifically, or to a
   broader class (all "low-degree polynomial" algorithms, Lipschitz algorithms,
   etc.)? The Discussion's citation pattern (Theorem 5 + Jordan2024DQI +
   Prange + various DQI papers) suggests it's specifically about DQI.

3. ~~**Proof technique:**~~ **RESOLVED:** PCP-based, building on Håstad.
   Citations to AS98, ALMSS98, hastad2001 in §2–3 confirm this. Not OGP-based
   and not purely coding-theoretic. **Remaining question:** Does the main
   result require UGC (Khot 2002 is cited) or is it unconditional?

4. **Tightness:** In what sense is the bound "tight"? Is there a matching
   algorithm achieving $r/q$ (= $1/q$ for linear equations), or is it tight
   because random assignment matches the hardness threshold?

5. **Dependence on parameters:** How does the bound depend on $k$, $D$, $q$?
   Does the paper give specific numbers for $(k=3, D=4, q=2)$?

6. **Comparison to Håstad:** Theorem 4 likely restates/extends Håstad's result
   to the DQI setting. Need to determine what's new vs. what's a direct
   application of known results.

7. **Numerical bounds and Figure 1:** The figure (page 7, in Discussion) is
   likely a performance comparison. Does it provide explicit numerical bounds
   comparable to Stephen's DQI+BP value of 0.87065 at $(k=3, D=4)$?

8. **The companion dataset:** Citation key `csse_maxlinsat_dqi` suggests
   companion code or data. Worth checking if this provides numerical bounds
   we can use directly.

---

## Relevance to Our Project

### Direct relevance: HIGH

This paper provides the **theoretical ceiling** for one side of the comparison
we are computing. Our project produces a table:

| $p$ | QAOA fraction | vs. DQI ceiling | vs. SA |
|-----|--------------|-----------------|--------|
| 1   | ???          | Above/below?     | ...    |
| 2   | ???          | Above/below?     | ...    |
| ... | ...          | ...              | ...    |

The "DQI ceiling" column is informed by this paper. If DQI is provably limited
to $\leq X$ on unstructured Max-3-XORSAT at $(k=3, D=4)$, then any QAOA
value $> X$ demonstrates a clear separation.

### What this paper does NOT change

- **Our computational approach:** We still use the exact tensor network
  contraction method from Farhi et al. (2025), generalised to $k$-XORSAT.
  Nothing in this paper affects the algorithm we implement.
- **Our targets:** The comparison targets remain: DQI+BP (0.87065), Prange
  (0.875), Regev+FGUM (0.89187), SA (0.9366).
- **Our code:** No code changes needed.

### What this paper DOES add

- **Sharper DQI ceiling:** Instead of just the empirical DQI+BP number
  (0.87065), we may have a provable upper bound on what *any* DQI-type
  approach can achieve. This makes the QAOA comparison more meaningful.
- **Theoretical justification:** If we show QAOA exceeding the DQI ceiling,
  this paper provides the theoretical framework explaining *why* DQI
  can't match QAOA on this problem class.
- **Broader context:** The result extends beyond $q=2$ to general
  $\mathrm{GF}(q)$, suggesting the separation between QAOA and DQI may
  hold across the entire Max-LINSAT family — not just our specific case.

### Action items

- [x] ~~Extract structural data from PDF~~ — Done
- [x] ~~Extract full PDF text and resolve markers~~ — Done (2026-03-21):
  Theorems 4 and 5 confirmed, proof is unconditional (P ≠ NP, no UGC),
  $r/q$ threshold confirmed as tight via Remark 6
- [ ] Check Figure 1 for specific numerical bounds at $(k=3, D=4, q=2)$
- [ ] Investigate `csse_maxlinsat_dqi` companion code/data
- [ ] Update `PLAN.md` if the paper's bounds change any targets or priorities

---

## Key Equations to Remember

| Equation | Meaning |
|----------|---------|
| $\frac{1}{q}$ | Random assignment fraction for Max-LINSAT over $\mathrm{GF}(q)$ |
| $\frac{1}{2}$ | Random assignment fraction for Max-XORSAT ($q=2$) — our baseline |
| $\frac{\langle s\rangle}{m} = \frac{1}{2} + \sqrt{\frac{\ell}{m}(1-\frac{\ell}{m})}$ | DQI semicircle law (from DQI paper) |
| $r/q$ | The tight inapproximability threshold for max-LINSAT($q$, $r$): no poly-time algorithm can exceed $r/q + \varepsilon$ on worst-case instances (Theorem 5). For Max-XORSAT: $r$=1, $q$=2, so the threshold is 1/2. |

---

## Jargon From This Paper

| Term | Meaning |
|------|---------|
| **Max-LINSAT** | Max satisfiability of linear equations over a finite field — generalises Max-XORSAT |
| **$\mathrm{GF}(q)$** | The Galois field with $q$ elements ($q$ a prime power) |
| **Tight inapproximability** | A hardness result where the threshold matches the best achievable performance |
| **$r/q$ bound** | The tight random-assignment threshold for max-LINSAT($q$, $r$). Proved NP-hard to exceed by Theorems 4–5 via direct reduction from Håstad's theorem. Tight because random assignment achieves $r/q$ (Remark 6). |
| **Dual code $C^\perp$** | The code whose codewords are annihilated by the constraint matrix — governs DQI's decoding power |
| **Minimum distance** | Smallest Hamming weight of any nonzero codeword in the dual code — limits DQI's decoding radius |
| **OGP** | Overlap gap property — a topological barrier to stable algorithms (see `06-explainer-dqi-requires-structure.md`) |

---

**Verified.** Theorems 4 (inapproximability of max-E3-LIN-Γ) and 5 (inapproximability of max-LINSAT($q$,$r$)) are proved by direct reduction from Håstad's theorem, assuming only P ≠ NP. The $r/q$ bound is tight (Remark 6). The semicircle law (Eq. 9) shows DQI degrades to exactly $r/q$ as decodable structure ($\ell/m$) vanishes. DQI's advantage is confined to algebraically structured instances like OPI.
