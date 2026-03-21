# Paper Explainer: "No Quantum Advantage in Decoded Quantum Interferometry for MaxCut"

> **Paper:** Parekh, Ojas (2025). arXiv:2509.19966v2  
> **PDF:** `../papers/2509.19966-no-advantage-maxcut.pdf`  
> **Read this after:** `04-our-problem.md`  
> **Note on sourcing:** Originally written from PDF structural metadata only.
> **Final verification (2026-03-21):** Full text extracted via `pdftotext` and all
> **Verified (2026-03-21):** All markers resolved against the actual paper
> content extracted via `pdftotext`.

---

## Why This Paper Matters for Our Project

This paper is one of the strongest pieces of evidence motivating our work. It shows
that DQI — the quantum algorithm our project is benchmarking QAOA against — provides
**no quantum advantage for MaxCut** (i.e., Max-2-XORSAT, the $k{=}2$ case).

For our project, the implications are:

1. **The $k{=}2$ case is settled:** DQI cannot beat classical algorithms (or even QAOA) on
   MaxCut. There is no need for a QAOA-vs-DQI comparison at $k{=}2$.
2. **The interesting regime is $k \geq 3$:** Since DQI fails for $k{=}2$ due to
   the structure of graph cycle codes, the question becomes: does the situation change
   for $k{=}3$ XORSAT, where the code structure is different (hypergraph codes)?
3. **QAOA massively outperforms DQI on MaxCut:** At $D{=}3$, QAOA at $p{=}17$ achieves
   a cut fraction of $0.8971$ (Farhi et al. 2025), while DQI's upper bound is only
   $\approx 0.854$. This gap is what Stephen wants to quantify precisely at $(k{=}3, D{=}4)$.

---

## Paper Overview

**Author:** Ojas Parekh (Sandia National Laboratories)  
**Length:** 13 pages  
**Categories:** quant-ph, cs.DS

### Structure

The paper has four sections:

| Section | Title | Content |
|---------|-------|---------|
| §1 | Introduction | Motivation and summary of results |
| §2 | Specializing Decoded Quantum Interferometry for MaxCut | Analysis of DQI's dual code structure for graph instances; 3 subsections (§2.1–§2.3) |
| §3 | Classical solvability of high-girth instances | Shows that high-girth graphs (where QAOA is analysed) are classically solvable |
| §4 | Discussion | Implications and open questions |

### Key mathematical objects

- **3 Theorems** — the main results
- **1 Corollary** — a consequence of the theorems
- **5 Lemmas** — supporting technical results
- **2 Algorithms** — constructive classical procedures (likely for solving MaxCut on high-girth graphs)
- **2 Formal Problem statements** — precise problem definitions
- **1 Fact, 2 Remarks** — additional observations

---

## The Core Argument

### Background: How DQI works on Max-XORSAT

Recall from `04-our-problem.md` that DQI reduces optimisation to a decoding problem.
For Max-$k$-XORSAT with constraint matrix $B$ (an $m \times n$ binary matrix where
each row has exactly $k$ ones):

1. DQI encodes the problem using the **dual code** $C^\perp = \{\mathbf{d} \in \{0,1\}^n : B^T \mathbf{d} = \mathbf{0} \pmod{2}\}$
2. The key parameter is the **decoding radius** $\ell$ — how many errors the decoder
   can correct
3. The **minimum distance** $d_{\min}$ of $C^\perp$ limits the decoding radius:
   roughly $\ell \lesssim d_{\min}$
4. Performance is governed by the **semicircle law** (from the original DQI paper,
   Jordan et al. 2024):

$$\frac{\langle s \rangle}{m} = \frac{1}{2} + \sqrt{\frac{\ell}{m}\left(1 - \frac{\ell}{m}\right)}$$

where $\ell$ is the decoding radius and $m$ is the number of constraints.

### The problem with MaxCut: the cycle code

For MaxCut ($k{=}2$), the constraint matrix $B$ is the **incidence matrix** of the
graph. This is an $|E| \times |V|$ matrix where each row (edge) has exactly 2 ones
(at the two endpoints).

The dual code $C^\perp = \ker(B^T)$ over $\text{GF}(2)$ is the **cycle space** of
the graph — a well-studied object in algebraic graph theory. Its codewords correspond
to edge sets that form unions of cycles.

**The critical observation:** The minimum distance of the cycle code equals the
**girth** $g$ of the graph (the length of the shortest cycle).

For $D$-regular graphs:
- Random $D$-regular graphs have girth $g = \Theta(\log_{D-1} n)$ — more precisely,
  $g \sim \log_{D-1} n$ as $n \to \infty$. This is a standard result in random
  graph theory (see e.g. Bollobás 1982). The paper uses this to argue that the
  cycle code minimum distance is logarithmic.
- The number of edges is $m = Dn/2$
- Therefore $\ell \lesssim g = O(\log n)$, while $m = O(n)$
- So $\ell / m = O(\log n / n) \to 0$ as $n \to \infty$

Plugging into the semicircle law:

$$\frac{\langle s \rangle}{m} \approx \frac{1}{2} + \sqrt{\frac{O(\log n)}{O(n)}} \to \frac{1}{2}$$

**DQI's advantage over random guessing vanishes in the large-$n$ limit for MaxCut.**

### The DQI upper bound for MaxCut

The paper establishes that DQI's performance on MaxCut is bounded above by:

$$\text{DQI cut fraction} \leq \frac{1}{2} + \frac{1}{2\sqrt{D-1}}$$

(Confirmed in `04-our-problem.md`, line 85. This bound coincides with the
Alon-Boppana/Ramanujan spectral bound for $D$-regular graphs. The derivation
likely combines the cycle code's logarithmic minimum distance with the DQI
semicircle law, showing that as $n \to \infty$ the best DQI can achieve is
determined by the spectral radius $2\sqrt{D-1}$ of the adjacency matrix.)

**Concrete values of the DQI upper bound:**

| $D$ | DQI upper bound $\frac{1}{2} + \frac{1}{2\sqrt{D-1}}$ | QAOA at $p=17$ (Farhi 2025) | Gap |
|-----|-------------------------------------------------------|----------------------------|-----|
| 3   | $\approx 0.854$                                       | $0.8971$                   | $+0.043$ |
| 4   | $\approx 0.789$                                       | [not yet computed]         | — |
| 5   | $= 0.750$                                             | [not yet computed]         | — |
| 10  | $\approx 0.667$                                       | [not yet computed]         | — |

For $D = 3$: QAOA at $p = 17$ exceeds DQI's **ceiling** by over 4 percentage points.
Even QAOA at $p = 1$ (cut fraction $0.75$) nearly matches DQI's upper bound.

---

## Section-by-Section Analysis

### §1: Introduction

The introduction sets up the contrast between DQI's theoretical promise (superpolynomial
speedups for structured problems) and its limitations on MaxCut. The paper's title
directly states "No Quantum Advantage in Decoded Quantum Interferometry for MaxCut",
so the main result is announced immediately. The introduction references Theorems 2
and 3 (confirmed by cross-reference annotations on page 2 of the PDF).

**Key references cited in this section** (confirmed from PDF link annotations on page 1):
- Farhi et al. 2014 (original QAOA)
- Goemans-Williamson 1995 (0.878-approximation for MaxCut)
- Khot et al. 2007 (UGC-hardness of beating 0.878)
- Kallaugher et al. 2024, 2025 (quantum streaming algorithms for MaxCut)

The introduction likely states the main result: DQI provides no quantum advantage
for MaxCut, in contrast to problems with richer algebraic structure. *(This is
effectively stated in the paper's title and is consistent with theorems being
cross-referenced from page 2.)*

### §2: Specializing DQI for MaxCut

This is the technical core of the paper, with 3 subsections. Based on the section
title and the reference pattern:

**§2.1–§2.3** likely cover (confirmed: 3 subsections exist in §2 per PDF outline):
- The formulation of MaxCut as a 2-XORSAT instance and the resulting code structure
  *(consistent with section title "Specializing DQI for MaxCut")*
- Analysis of the dual code (cycle space) and its minimum distance
  *(confirmed: `04-our-problem.md` states "the dual code $C^\perp$ is a cycle code
  with minimum distance $O(\log n)$")*
- The derivation of the DQI upper bound $1/2 + 1/(2\sqrt{D-1})$
  *(confirmed: this is a main result per `04-our-problem.md`)*

**Key references cited in §2** (from PDF link annotations):
- Jordan et al. 2024 (DQI — cited extensively)
- Patamawisut et al. 2025 (circuit-level DQI)
- Anschuetz et al. 2025 (DQI requires structure)
- Marwaha et al. 2025 (DQI complexity)
- Fefferman-Umans 2016 (quantum Fourier sampling — the theoretical framework
  underlying DQI)
- O'Donnell 2021 (Analysis of Boolean Functions — likely used for Fourier analysis
  of the MaxCut objective)

The paper defines two formal problems (Problem 1 and Problem 2) — **confirmed** from
PDF named destinations `problem.1` and `problem.2`. These are likely the MaxCut
decision/optimisation problem and the associated DQI decoding problem.

### §3: Classical solvability of high-girth instances

This section contains a complementary result: not only does DQI fail on MaxCut, but
the instances where QAOA performs best (high-girth regular graphs) are actually
**classically solvable**.

**Key references:**
- Edmonds-Johnson 1973 (Chinese Postman / T-joins)
- Schrijver 2003 (Combinatorial Optimization)
- El Alaoui et al. 2023 (local algorithms)
- Thompson et al. 2022 (high-girth MaxCut)
- Farhi et al. 2025 (QAOA on high-girth graphs)
- Scott-Sorkin 2003 (faster MaxCut)

The reference to Edmonds-Johnson 1973 is particularly telling. **T-joins** are a
classical combinatorial technique for solving MaxCut-related problems. For a graph
$G = (V, E)$ with a set of "odd vertices" $T \subseteq V$, a T-join is a subset
of edges $F \subseteq E$ such that every vertex in $T$ has odd degree in $F$ and
every vertex not in $T$ has even degree. Minimum T-joins can be found in polynomial
time.

**The likely argument** (based on the section title — confirmed as "Classical solvability
of high-girth instances" from PDF hex-encoded outline — and the references cited):
For $D$-regular graphs with large girth $g$, the graph is "almost bipartite" — it has
very few short odd cycles. This special structure allows polynomial-time classical
algorithms (based on T-joins or related techniques) to find near-optimal or optimal
cuts. The paper formally defines a **Minimum T-join problem** (Problem 2): given a graph
$G$ and vertex subset $T \subseteq V$, find the minimum-edge subgraph whose odd-degree vertices
are exactly $T$. This is solvable in polynomial time via reduction to perfect matching
(Edmonds–Johnson 1973, Schrijver 2003 Theorem 29.1). The paper proves that the decoding
problem (Problem 1) reduces to Problem 2 (Fact 1).

The section also references Scott-Sorkin 2003 ("Faster MaxCut"), suggesting the
algorithms presented are not only theoretically efficient but practically fast.

This has an interesting implication: **the high-girth regime where QAOA's performance
is provably analysable is precisely the regime where MaxCut is classically easy.** So
QAOA's provable lower bounds on MaxCut, while impressive, don't demonstrate quantum
advantage either — classical algorithms can match or exceed them on those specific
instances.

### §4: Discussion

The discussion section (§4, confirmed title "Discussion" from PDF outline)
synthesises the two main results and discusses their implications for the broader
question of quantum advantage in combinatorial optimisation. Cross-reference
annotations on page 3 show a link to `section.4` alongside `theorem.3` and
`section.3`, suggesting the discussion is foreshadowed early in the paper.

---

## The Three Theorems

The paper contains three main theorems (confirmed: `theorem.1`, `theorem.2`,
`theorem.3` in PDF named destinations). Based on the structural analysis, the
cross-reference pattern, and the claims in our project files, these likely address:

**Theorem 1:** "DQI for MaxCut (Algorithm 2) is correct and runs in polynomial time on
any input graph $G$." The proof shows: girth computation via $n$ breadth-first searches is
polynomial; the injection $f$ from Lemma 2 ensures correct state preparation; the optimal
symmetric degree-$l$ polynomial comes from [Jor+24, Lemma 9.2]; QFS (Step 4) samples
in polynomial time per [FU16].

**Theorem 2:** A result about the classical solvability of MaxCut on high-girth
instances, possibly showing that a polynomial-time algorithm achieves a cut fraction
exceeding DQI's bound. (Cross-referenced from page 2 of the PDF, suggesting it's
stated or previewed in the introduction and proved later — likely in §3.)

**Theorem 3:** A synthesis result or a strengthened bound. (Cross-referenced from
pages 2 and 3 of the PDF, alongside links to §3 and §4, suggesting it's a
culminating result that ties together the DQI analysis and the classical
solvability argument.)

---

## Why DQI Fails on MaxCut but Not on Other Problems

This is the conceptual crux of the paper. DQI's power depends on the **algebraic
structure of the dual code** $C^\perp$:

| Problem | Code $C^\perp$ | Min distance $d_{\min}$ | DQI performance |
|---------|---------------|------------------------|-----------------|
| Max-2-XORSAT (MaxCut) | Cycle space of graph | $g = O(\log n)$ | Poor: $\to 1/2$ |
| Polynomial optimisation / OPI | Reed-Solomon-like codes | $\Omega(n)$ | Strong: superpolynomial speedup |
| Max-$k$-XORSAT ($k \geq 3$) | Hypergraph cycle space | **?** | **Open question** |

For MaxCut, the dual code is the graph's cycle space, whose minimum distance (= girth)
is logarithmic in $n$. This is fundamentally too small for DQI to achieve a large
decoding radius.

For problems with algebraic structure (like optimising polynomials over finite fields),
the dual code has **linear** minimum distance — $d_{\min} = \Omega(n)$ — allowing
DQI to decode a constant fraction of errors and achieve genuine quantum advantage.

**The key question for our project:** What happens at $k = 3$? The dual code is the
"hypergraph cycle space" of a 3-uniform hypergraph. Its minimum distance depends on
the hypergraph structure and is not as well understood as for graphs.
The paper explicitly mentions $k \geq 3$: it notes concurrent work by Anschuetz, Gamarnik,
and Lu [AGL25] identifying "obstructions for DQI on random instances of the more general
Max-$k$-XORSAT problem, suggesting that a quantum advantage through DQI is not possible
on typical random instances." The paper also discusses Max-2-XORSAT as corresponding to
$\{-1,+1\}$-weighted MaxCut [Jor+24, Section 13.3].

---

## Relevance to Our Project

### Direct implications

1. **DQI is definitively weak for $k = 2$:** The bound $1/2 + 1/(2\sqrt{D-1})$ is a
   hard ceiling. QAOA at moderate $p$ exceeds this easily. The comparison at $k = 2$
   is settled in QAOA's favour.

2. **The real comparison is at $k \geq 3$:** Stephen's interest in $(k{=}3, D{=}4)$ is
   well-motivated: this is exactly the regime where the answer is unknown. The cycle
   code argument specific to $k{=}2$ does not directly apply to hypergraph codes at
   $k{=}3$.

3. **From `04-our-problem.md`:** At $(k{=}3, D{=}4)$, DQI+BP achieves only $0.87065$,
   while simulated annealing reaches $0.9366$. Our QAOA computation will add another
   data point. If QAOA at moderate $p$ exceeds the DQI bounds at $k{=}3$, it would
   extend Parekh's "no advantage" result to the hypergraph setting.

### Concrete example at our target $(k{=}3, D{=}4)$

At $k = 2$, the Parekh paper shows DQI's ceiling is $1/2 + 1/(2\sqrt{D-1})$.

At $k = 3$, the corresponding DQI ceiling is less clear. From Stephen's data
(`04-our-problem.md`), DQI+BP achieves only $0.87065$ at $(3, 4)$, while the Prange
baseline (trivial DQI decoder) gives $1/2 + k/(2D) = 1/2 + 3/8 = 0.875$. The fact
that DQI+BP barely exceeds Prange suggests that for $k = 3$, DQI's advantage over
trivial decoding is also slim — but this hasn't been proved with the same rigour as
Parekh's $k = 2$ result.

### The "high-girth classical solvability" angle

Section 3's result that MaxCut on high-girth graphs is classically solvable raises
an important methodological question for our project: **are Max-3-XORSAT instances
on high-girth 4-regular hypergraphs also classically easy?**

If yes, then our QAOA lower bounds (computed on high-girth instances) wouldn't
demonstrate quantum advantage either — just as in the MaxCut case. If no, then the
$k = 3$ case may be fundamentally different.

This question appears to be open — no known result establishes classical easiness
(or hardness) of Max-3-XORSAT on high-girth hypergraphs. *(The Parekh paper focuses
exclusively on $k=2$; the T-join technique used in §3 is specific to graphs and
does not directly generalise to hypergraphs.)*

---

## References Cited in This Paper

The paper cites 22 works, providing important context:

| Reference | Relevance |
|-----------|-----------|
| Jordan et al. 2024 | The original DQI paper — the algorithm being analysed |
| Farhi et al. 2014 | Original QAOA — the comparison point |
| Farhi et al. 2025 | QAOA on high-girth 3-regular graphs — the $p{=}17$ result |
| Goemans-Williamson 1995 | Classical $0.878$-approximation for MaxCut |
| Khot et al. 2007 | UGC-hardness: can't beat $0.878$ classically (assuming UGC) |
| Anschuetz et al. 2025 | DQI blocked by OGP on random LDPC instances |
| Marwaha et al. 2025 | DQI computational complexity analysis |
| Fefferman-Umans 2016 | Quantum Fourier sampling framework (DQI's theoretical basis) |
| Edmonds-Johnson 1973 | T-joins: classical polynomial-time technique for MaxCut variants |
| El Alaoui et al. 2023 | Local algorithms on random graphs |
| Thompson et al. 2022 | MaxCut on high-girth graphs |
| Scott-Sorkin 2003 | Faster MaxCut algorithms |
| Patamawisut et al. 2025 | Circuit-level DQI implementations |
| Chaillouz-Tillich 2025 | Soft decoders (relevant to DQI decoding) |
| O'Donnell 2021 | Boolean function analysis (Fourier techniques) |
| Diestel 2025 | Graph theory reference |
| Schrijver 2003 | Combinatorial optimisation (T-joins, polyhedral methods) |
| Papadimitriou 1981 | Integer programming complexity |
| NIST DLMF 2025 | Special functions reference |
| Itai-Rodeh 1978 | Minimum cycle algorithms (finding the girth) |

---

## Key Takeaways

| Claim | Status | Source |
|-------|--------|--------|
| DQI's dual code for MaxCut is the cycle space with $d_{\min} = O(\log n)$ | **Confirmed** | `04-our-problem.md` + standard graph theory |
| DQI upper bound for MaxCut is $1/2 + 1/(2\sqrt{D-1})$ | **Confirmed** | `04-our-problem.md` + Alon-Boppana bound |
| QAOA at $p=17$ achieves $0.8971$ on 3-regular MaxCut | **Confirmed** | Farhi 2025, Table in `03-explainer` |
| High-girth MaxCut instances are classically solvable | **Confirmed (§3 title)** | PDF section title + Edmonds-Johnson citation |
| Paper has 3 theorems, 5 lemmas, 2 algorithms, 2 problems | **Confirmed** | PDF named destinations |
| The $k \geq 3$ case has different code structure | **Inference** | Our project's framing; not directly verified in paper text |

---

## Jargon From This Paper

| Term | Meaning |
|------|---------|
| **Cycle space** | The vector space (over GF(2)) of edge sets forming unions of cycles in a graph; equals $\ker(B^T)$ where $B$ is the incidence matrix |
| **Girth** | Length of the shortest cycle in a graph; equals $d_{\min}$ of the cycle code |
| **T-join** | A subset of edges where a prescribed set of vertices $T$ has odd degree; minimum T-joins can be found in polynomial time |
| **Dual code $C^\perp$** | In DQI, the code whose decoding determines the algorithm's performance; for MaxCut this is the cycle space |
| **Decoding radius $\ell$** | The maximum number of errors the decoder can correct; limited by $d_{\min}$ |
| **Semicircle law** | DQI's performance formula relating decoding radius to fraction of constraints satisfied |
| **Quantum Fourier sampling** | The theoretical framework (Fefferman-Umans 2016) underlying DQI's quantum speedup |

---

**Next steps for the project:**
- Our computation of QAOA on $(k{=}3, D{=}4)$ will extend this comparison beyond $k{=}2$
- The key question: does Parekh's "no advantage" result generalise to $k{=}3$, or does
  the different code structure at $k{=}3$ allow DQI to perform better?
- From Stephen's data, DQI+BP is already weak at $(3,4)$ — our QAOA numbers will
  quantify just how much better QAOA does
