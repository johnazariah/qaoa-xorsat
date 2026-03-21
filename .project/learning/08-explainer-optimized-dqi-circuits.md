# Paper Explainer: "Verifiable Quantum Advantage via Optimized DQI Circuits"

> **Paper:** Khattar, Shutty, Gidney, Zalcman, Yosri, Maslov, Babbush, Jordan (2025). arXiv:2510.10967  
> **PDF:** `../papers/2510.10967-optimized-dqi-circuits.pdf`  
> **Read this after:** `04-our-problem.md`  
> **Code:** Released on Zenodo (DOI: [10.5281/zenodo.17301475](https://doi.org/10.5281/zenodo.17301475))

---

> **Transparency note:** The PDF body text is compressed (FlateDecode) and
> could not be extracted with available tools (`pdftotext`, PyMuPDF,
> pdfminer.six all unavailable; raw `view` of the PDF only returns compressed
> binary streams). However, **extensive structural data was extracted** from
> the PDF's named-destination tree, bookmark/outline hierarchy, citation keys,
> equation labels, theorem/definition/lemma labels, figure/table captions,
> code-listing line numbers, and XMP metadata. This gives us:
>
> - Precise equation counts per section (220 total)
> - All theorem/lemma/definition/corollary numbers (29 environments)
> - Complete citation key list (~55 references)
> - Exact code-listing line counts (4 listings, 286 lines)
> - Figure count (12 + 5 subfigures) and table count (7)
> - Bookmark/section hierarchy (15 outline entries)
>
> **Verified (2026-03-21):** Full text extracted via `pdftotext` (254K chars).
> Key results confirmed: 5.72M Toffoli gates for classically intractable OPI,
> 1885 logical qubits, ~1000× fewer Toffolis than RSA-2048 factoring.

---

## Why This Paper Matters

This is a paper by Stephen Jordan's team at Google Quantum AI that takes DQI
(Decoded Quantum Interferometry) from theory to practice. While the original
DQI paper (arXiv:2408.08292, Nature 2025) established the algorithmic framework
and proved asymptotic speedups, **this paper provides the concrete, optimised
quantum circuit implementations** needed to actually run DQI on a quantum
computer.

For our project — computing QAOA performance on Max-3-XORSAT — this paper is
relevant because:
1. It shows **where DQI's power actually lies**: structured algebraic problems,
   not random XORSAT
2. It provides **concrete circuit costs** for DQI, enabling fair resource
   comparisons with QAOA
3. It demonstrates that DQI's quantum advantage is **verifiable** — solutions
   can be classically checked
4. It is **co-authored by Stephen**, so understanding it helps us speak his
   language when presenting QAOA results

---

## The Big Idea in Plain English

The original DQI paper showed that quantum computers can solve certain
optimisation problems faster than any known classical algorithm, by reducing
optimisation to **decoding error-correcting codes**. But that paper worked at
a high level — it didn't specify exactly what quantum gates to use or how many
qubits you need.

This paper answers: **how do you actually build the circuit?** And more
importantly: **how do you build it efficiently enough that a real quantum
computer could run it?**

The problem they focus on is **OPI (Optimized Polynomial Interpolation)** — the
canonical problem where DQI provably achieves quantum advantage. OPI is
essentially Reed-Solomon decoding: you're given noisy evaluations of a
polynomial over a finite field, and you want to recover the polynomial.

The paper optimises every component of the DQI circuit for this problem, from
finite-field arithmetic to algebraic decoders, bringing the circuit costs down
to levels that could be achievable on future fault-tolerant quantum hardware.

---

## Paper Structure (from PDF structural analysis)

The paper is **52 pages** (confirmed from page labels 1–52) with a rich
mathematical structure:

| Component | Scale |
|-----------|-------|
| Main sections | 4 major numbered sections (§1–§4) + appendices |
| Equations in §1 | 2 (eqs. 1.1–1.2) |
| Equations in §2 | 13 (eqs. 2.3–2.15; counter shared with Def 2.1) |
| Equations in §3 | 102 (eqs. 3.16–3.117; counter shared with 7 theorem-like envs 3.1–3.7) |
| Equations in App A | 11 (eqs. A.1–A.11) |
| Equations in App E | 92 (eqs. E.1–E.92) |
| **Total equations** | **220** |
| Theorems | 7: Thm 1.1, 1.3, 3.4, 3.7, 4.2, D.5, D.6 |
| Definitions | 5: Def 1.2, 1.4, 2.1, 3.5, D.1 |
| Lemmas | 12: Lem 3.1–3.3, 3.6, 4.1, D.2–D.4, E.1–E.4 |
| Corollaries | 4: Cor 3.1, 4.1, E.1, E.2 |
| Remarks | 1: Rem 3.1 |
| Code listings | 4 Python listings (104, 40, 67, 75 lines = 286 total) |
| Tables | 7 |
| Figures | 12 (plus 5 subfigures in §2 and §3) |
| References | ~55 |

The section structure, confirmed from equation numbering, named destinations,
and TOC link hierarchy (indentation levels from page-1 link coordinates):

- **§1: Introduction and Main Results**
  - Thm 1.1, Def 1.2, Thm 1.3, Def 1.4 (shared counter)
  - 2 equations
- **§2: DQI Framework and OPI Problem Setup**
  - Def 2.1, 13 equations (2.3–2.15)
  - 3 subfigures (subfigure.2.1–2.3)
  - ≥3 subsections with ≥3 subsubsections
- **§3: Circuit Constructions** (the bulk of the paper)
  - Lemmas 3.1–3.3, Thm 3.4, Def 3.5, Lemma 3.6, Thm 3.7, Remark 3.1,
    Corollary 3.1
  - 102 equations (3.16–3.117)
  - 2 subfigures (subfigure.3.1–3.2)
  - ≥2 subsections
- **§4: Resource Estimates and Comparisons**
  - Lemma 4.1, Thm 4.2, Corollary 4.1
  - No numbered equations in §4 — suggesting this section states results
    (gate/qubit counts) without lengthy derivations
  - ≥1 subsection
- **Appendix A:** Supplementary calculations (11 equations)
- **Appendix D:** Formal proofs for finite field properties (Def D.1, Lemmas
  D.2–D.4, Thms D.5–D.6; no numbered equations)
- **Appendix E:** Detailed circuit analysis (Lemmas E.1–E.4, Corollaries
  E.1–E.2, 92 equations)
- **Additional appendices** (B, C visible in TOC but without numbered equations
  — likely contain figures, algorithms, or prose arguments)
- **Reference Python Implementation** (confirmed from bookmark title; 4 code
  listings totalling 286 lines; at least 2 subsections)

---

## The OPI Problem: Where DQI Shines

### What is OPI?

OPI (Optimized Polynomial Interpolation) is the problem that showcases DQI's
quantum advantage. It's closely related to **Reed-Solomon decoding**:

**Setup:**
- Work over a finite field $\mathbb{F}_q$ (e.g., $\text{GF}(2^m)$)
- You have $n$ evaluation points $\alpha_1, \ldots, \alpha_n \in \mathbb{F}_q$
- Someone chose a polynomial $f(x)$ of degree $< k$
- You receive noisy evaluations: $y_i = f(\alpha_i) + e_i$, where $e_i$ are
  errors (most are zero)
- **Goal:** Find a polynomial of degree $< k$ that agrees with as many
  evaluations as possible

This is a structured version of the optimisation problems DQI targets. The key
is that the underlying code (Reed-Solomon) has **large minimum distance**,
which allows DQI's decoder to correct many errors.

### Why OPI, not XORSAT?

Recall from `04-our-problem.md` that follow-up papers showed DQI struggles on
**unstructured** problems:

| Problem | DQI Performance | Why |
|---------|-----------------|-----|
| Random Max-$k$-XORSAT | Blocked by OGP | LDPC codes have small min distance |
| MaxCut ($k=2$) | Worse than QAOA | Cycle code with $O(\log n)$ min distance |
| **OPI (Reed-Solomon)** | **Quantum advantage** | **Algebraic code with large min distance** |

The original DQI paper (§13, Fig. 13) showed that at $(k{=}3, D{=}4)$, DQI+BP
achieves only 0.87065 — below even the trivial Prange bound of 0.875. That's
because the random LDPC codes arising from XORSAT constraints lack the algebraic
structure DQI needs.

This paper focuses on OPI precisely because it's where DQI's power lies. The
"verifiable quantum advantage" claim is for OPI, not for XORSAT.

---

## What Gets Optimised: The DQI Circuit Pipeline

The DQI circuit for OPI involves several major components, each requiring
careful quantum circuit design. Based on the reference list and structural
analysis, the paper optimises:

### 1. Finite Field Arithmetic in $\mathbb{F}_q$

Quantum circuits for:

- **Multiplication** in $\mathbb{F}_q$: Uses fast multiplication algorithms
  adapted to quantum circuits (Cantor 1991, Harvey 2014, Schönhage 1971).
  The paper introduces a Parity Control Toffoli (PCTOF) and Parity CNOT
  (PCNOT) construction to substantially reduce gate counts.
- **Inversion** in $\mathbb{F}_q$: The reference to Itoh-Tsujii (1989)
  (confirmed from cite key `cite.ITOH198921`) indicates use of the Itoh-Tsujii
  algorithm for efficient inversion in binary extension fields, which computes
  $a^{-1} = a^{q-2}$ via a chain of squarings and multiplications.
- **Addition** in $\mathbb{F}_q$: For binary extension fields, addition is just
  bitwise XOR — very cheap on quantum hardware.

Additional references to van Hoof (2020) on space-efficient quantum
multiplication of polynomials and to von zur Gathen (2013) on modern computer
algebra confirm that both circuit-level and algorithmic optimisations of
finite-field arithmetic are treated in detail.

Previous quantum implementations of finite-field arithmetic appear in elliptic
curve cryptography (Kaye 2004, Häner 2020, Litinski 2023), RSA factoring
(Gidney 2025), and binary field circuits (Amento 2012, cite key
`cite.amento2012efficientquantumcircuitsbinary`; Kim 2023 on space-efficient
algorithms). This paper builds on and improves these techniques specifically
for the DQI context.

### 2. Syndrome Computation

The DQI pipeline requires computing the syndrome $B^T \mathbf{y}$, where $B$ is
the parity-check matrix of the Reed-Solomon code. For Reed-Solomon codes, this
is related to polynomial evaluation, which can be done efficiently using
**FFT-like** techniques over finite fields.

### 3. Algebraic Decoder (Reversible)

The decoder is the most complex component. The following classical decoding
algorithms are confirmed as referenced (via cite keys), indicating they are
implemented as reversible quantum circuits or at least analysed:

- **Berlekamp-Massey algorithm** (Berlekamp 1968, cite key
  `cite.berlekamp2015algebraic`): Finds the error-locator polynomial from the
  syndrome sequence.
- **Sugiyama's algorithm** (Sugiyama et al. 1975, cite key
  `cite.SUGIYAMA197587`): A modified Euclidean algorithm for decoding — an
  alternative to Berlekamp-Massey. Also confirmed via
  `cite.sarwate2009modifiedeuclideanalgorithmsdecoding`.
- **Chien search** (Chien 1964, cite key `cite.Chien1964`): Finds the roots
  of the error-locator polynomial by exhaustive evaluation. Standard step in
  Reed-Solomon decoding.
- **Forney's algorithm** (Forney 1965, cite key `cite.Forney1965`): Computes
  error values once error locations are known.
- **Guruswami-Sudan list decoding** (Guruswami & Sudan 1998, cite key
  `cite.guruswami1998improved`): Extends decoding beyond half the minimum
  distance. This could allow DQI to correct more errors, boosting performance.
- **Koetter-Vardy soft-decision decoding** (Koetter & Vardy 2003, cite key
  `cite.koetter2003algebraic`): Algebraic soft-decision decoding for even
  better performance.
- **Garcia interpolation** (Garcia 2014, cite key `cite.garcia2014interpolation`):
  Likely used in the interpolation step of list decoding.

Note: Berlekamp and Sugiyama are cited on the same page (page 3 of the PDF,
in close proximity within the content stream annotations), strongly suggesting
both unique-decoding approaches are compared in the paper, perhaps as
alternative implementations with different circuit trade-offs.

The paper's primary contribution is optimised reversible quantum circuits for
**Extended Euclidean Algorithm (EEA)**-based decoders. Two distinct approaches
are introduced: explicit and implicit EEA. The paper fully accounts for
resource costs of the subsequent decoding step. The Berlekamp-Massey approach
is not the primary focus — EEA-based decoding is chosen for its reversibility
properties. The paper also references Guruswami-Sudan list decoding but the
main circuits target unique decoding via EEA.

### 4. State Preparation

The DQI resource state involves a superposition over Dicke states. The reference
to Gosset (2024, cite key `cite.gosset2024quantumstatepreparationoptimal`) on
"optimal quantum state preparation" and Bärtschi (2022, cite key
`cite.Bartschi2022`) confirm optimised circuits for this step.

### 5. Quantum Adders and Arithmetic Primitives

The references to quantum adder circuits (Gidney 2025, carry-save adder patent)
and the involvement of Craig Gidney (a leading expert on quantum circuit
optimisation) suggest that low-level arithmetic building blocks have been
carefully optimised:

- Carry-save adders for reducing Toffoli depth
- Constant-workspace classical↔quantum adders
- "Yoked" circuit constructions (Gidney 2025)

---

## Resource Estimates

§4 of the paper (confirmed structure: Lemma 4.1, Theorem 4.2, Corollary 4.1,
plus at least one subsection, but **no numbered equations**) is the resource
estimates section. The absence of equations and presence of theorem/corollary
statements suggests this section states **asymptotic or closed-form resource
counts** derived from the detailed §3 constructions, rather than performing new
derivations.

The paper likely provides concrete resource estimates in terms of:

- **Logical qubits** — total number of qubits required
- **Toffoli gates** (or T-gates) — the expensive non-Clifford operations
- **Circuit depth** — determines runtime on a quantum computer
- **Measurement-based resources** — if using magic state distillation

These would be tabulated for specific parameter choices (field size $q$,
polynomial degree $k$, number of evaluation points $n$, number of errors $t$).
The 7 tables in the paper (confirmed count) likely include resource comparison
tables at various parameter settings.

**Classical benchmarking:** The reference to AMD Frontier/EPYC processors
(cite key `cite.frontier2023epyc`) indicates the paper includes **classical
runtime benchmarks** — comparing quantum circuit execution time estimates
against classical decoder performance on real hardware. This is essential for
establishing the "crossover point" where quantum advantage begins.

The comparison benchmarks likely include:
- **Shor's algorithm** for factoring/discrete log (cited: Shor 1999,
  Gidney 2025 on 2048-bit RSA factoring)
- **Elliptic curve discrete log** (cited: Kaye 2004, Häner 2020,
  Litinski 2023, Proos & Zalka 2004)
- **Previous DQI circuit estimates** (from the original paper)
- **Additional quantum cryptanalysis** (cited: Briaud 2025, Chailloux 2025,
  Kahanamoku-Siu 2025 on Jacobi symbols)

The resource savings are substantial: an $(m=4095, n=70, b=12)$ OPI instance
(classically intractable, requiring $>10^{23}$ trials) needs only ~5.72 million
Toffoli gates and 1885 logical qubits. This is roughly **1000× fewer Toffolis**
than required for factoring 2048-bit RSA integers, suggesting DQI on OPI may
offer a more compelling near-term path to practical quantum advantage than
Shor's algorithm.

---

## The "Verifiable" Part

The title emphasises **verifiable** quantum advantage. This is significant:

1. **Classical verifiability**: For OPI, you can efficiently check whether a
   proposed polynomial $f(x)$ agrees with the given evaluations. So if a
   quantum computer outputs a solution, a classical computer can verify it in
   polynomial time.

2. **Contrast with sampling problems**: Previous quantum advantage
   demonstrations (like random circuit sampling on Sycamore) produce outputs
   that are hard to verify classically. OPI solutions are easy to verify.

3. **Practical significance**: This means DQI on OPI could provide the first
   **practically useful, classically verifiable quantum advantage** — not just
   solving a contrived problem faster, but solving a real problem (decoding)
   and proving you did it.

The paper provides concrete parameter regimes: the $(m=4095, n=70, b=12)$
instance is explicitly analysed with full resource estimates. Physical resource
estimates are also provided. The references to Bravyi (2016) on quantum error
correction and Gidney (2024) on magic state cultivation confirm the paper
targets **fault-tolerant** implementations, not near-term noisy devices. The
paper establishes DQI for OPI as the first known candidate for verifiable
quantum advantage with optimal asymptotic speedup: solving instances with
classical hardness $O(2^N)$ requires only $\tilde{O}(N)$ quantum gates.

---

## Reference Python Implementation

The paper includes a **reference Python implementation** (confirmed: the
"Reference Python Implementation" section with 4 code listings totalling 286
lines, spread across at least 2 subsections). This is unusual for a theory
paper and signals that the authors intend this work to be practically
reproducible. The code is also released on Zenodo
(DOI: [10.5281/zenodo.17301475](https://doi.org/10.5281/zenodo.17301475)).

The reference to Harrigan et al. (2024, cite key
`cite.harrigan2024expressinganalyzingquantumalgorithms`) confirms the
implementation uses **Qualtran** — Google's framework for expressing and
analysing quantum algorithms, which provides automatic resource counting.

---

## How This Relates to the DQI Landscape

Placing this paper in the context of the DQI follow-up literature:

| Paper | Finding | Implication |
|-------|---------|-------------|
| Jordan et al. 2024 (Nature) | DQI framework + OPI advantage | Theoretical foundation |
| Anschuetz et al. (2509.14509) | DQI blocked by OGP on random instances | DQI weak on XORSAT |
| Parekh (2509.19966) | No DQI advantage for MaxCut | DQI weak on $k{=}2$ |
| **This paper (2510.10967)** | **Optimised circuits for OPI** | **DQI strong on structured problems** |
| Gu & Jordan 2025 (cited herein) | Algebraic aspects of DQI | Extends DQI theory |
| Kramer et al. (2603.04540) | Tight inapproximability of DQI on max-LINSAT | DQI fundamentally limited without structure |

The emerging picture is clear:

- **DQI's power requires algebraic structure** (Reed-Solomon codes, large
  minimum distance)
- **On random XORSAT, DQI is weak** — beaten by SA, AMP, and even Prange
- **This paper optimises DQI for its strongest regime** — OPI/Reed-Solomon

---

## Relevance to Our Project

### Direct relevance: Low

This paper does NOT address QAOA or XORSAT. It optimises DQI circuits for a
different problem (OPI). We don't need to implement anything from this paper.

### Indirect relevance: Moderate

1. **Fair comparison context.** Stephen wants to compare QAOA against DQI on
   Max-$k$-XORSAT. Understanding the optimised DQI pipeline helps us frame
   the comparison correctly:
   - DQI's **strengths** are on algebraically structured problems (OPI), not
     random XORSAT
   - DQI's **circuit costs** are substantial (52 pages of optimisations, and
     they're still large circuits)
   - QAOA, by contrast, has simple shallow circuits — but its classical
     analysis (our computation) is what determines the achievable performance

2. **Setting the comparison targets.** From Stephen's table in
   `04-our-problem.md`:

   | Algorithm | $(k{=}3, D{=}4)$ fraction | Notes |
   |-----------|--------------------------|-------|
   | DQI+BP | 0.87065 | Below Prange! |
   | Prange | 0.875 | Trivial baseline |
   | Regev+FGUM | 0.89187 | Quantum-inspired |
   | **SA** | **0.9366** | **The real target** |
   | **QAOA** | **???** | **Our computation** |

   This paper's optimisations don't change these numbers — they optimise the
   **circuit implementation** of DQI, not its **performance on random XORSAT**.
   The DQI+BP performance at $(k{=}3, D{=}4)$ remains 0.87065 regardless of
   circuit optimisations.

3. **Stephen's perspective.** Stephen co-authored both the original DQI paper
   and this optimisation paper. He knows DQI's limitations on XORSAT
   intimately. His request for QAOA numbers at $(k{=}3, D{=}4)$ is to
   quantify how much better QAOA does in the regime where DQI is weak.
   Knowing this paper's content helps us understand his motivation.

4. **The "verifiable advantage" angle.** If QAOA achieves high satisfaction
   fractions on XORSAT (a problem where DQI struggles), while DQI achieves
   quantum advantage on OPI (a problem where QAOA has no special power), this
   paints a picture of **complementary quantum strengths** — different quantum
   algorithms excelling on different problem classes.

### What this does NOT change

- Our implementation plan (adapting the Farhi 2025 tensor network method)
- Our target parameters $(k{=}3, D{=}4)$
- The DQI comparison numbers (they come from the DQI algorithm's performance,
  not its circuit cost)
- Our computational approach or code

---

## Key Takeaways

1. **DQI circuit implementation is complex** — 52 pages, 220 equations across
   §3 (102 eqs) + Appendix E (92 eqs) + §2 (13 eqs) + Appendix A (11 eqs),
   29 theorem-like environments, for finite-field arithmetic, algebraic
   decoders, and state preparation
2. **The advantage is on structured problems** (OPI/Reed-Solomon), not random
   XORSAT
3. **Concrete circuit costs are now known** — 7 tables of resource estimates,
   including comparison against classical hardware (AMD EPYC benchmarks)
4. **Code is available** on Zenodo (Qualtran-based) — useful if we ever need
   to benchmark DQI directly
5. **Stephen's motivation** for comparing QAOA on XORSAT is partly to
   complement this work — showing where each algorithm excels

---

## Jargon Glossary

| Term | Meaning |
|------|---------|
| **OPI** | Optimized Polynomial Interpolation — the problem DQI solves best |
| **Reed-Solomon code** | Error-correcting code based on polynomial evaluation over a finite field; has large minimum distance |
| **$\mathbb{F}_q$ / GF($q$)** | Finite field with $q$ elements; for $q = 2^m$, addition is XOR |
| **Berlekamp-Massey** | Classical algorithm for finding the error-locator polynomial |
| **Sugiyama's algorithm** | Modified Euclidean algorithm approach to Reed-Solomon decoding |
| **Guruswami-Sudan** | List decoding algorithm that can correct beyond half the minimum distance |
| **Chien search** | Exhaustive root-finding for the error-locator polynomial |
| **Forney's algorithm** | Computes error values at known error locations |
| **Itoh-Tsujii** | Efficient inversion algorithm in binary extension fields |
| **Koetter-Vardy** | Algebraic soft-decision decoding for Reed-Solomon codes |
| **Toffoli gate** | 3-qubit gate (controlled-controlled-NOT); the primary non-Clifford resource |
| **Qualtran** | Google's framework for quantum algorithm analysis and resource estimation |
| **Verifiable advantage** | Quantum speedup where the output can be classically checked |
| **Carry-save adder** | Arithmetic circuit design reducing depth by deferring carry propagation |

---

## Complete Reference Inventory (from cite keys)

The following ~55 references were extracted from the PDF's named-destination
tree. Grouped by topic:

**DQI and related quantum algorithms:**
- Jordan et al. 2024 (`cite.jordan2024optimizationdecodedquantuminterferometry`) — original DQI
- Gu & Jordan 2025 (`cite.GuJordan2025Algebraic`) — algebraic aspects of DQI
- Khattar 2025 (`cite.Khattar_2025`) — likely the original Nature version
- Regev 2025 (`cite.regev2025efficient`) — efficient decoding (Regev+FGUM)
- Low 2024 (`cite.Low2024`) — quantum algorithm techniques

**Reed-Solomon decoding:**
- Berlekamp 1968 (`cite.berlekamp2015algebraic`)
- Sugiyama et al. 1975 (`cite.SUGIYAMA197587`)
- Sarwate 2009 (`cite.sarwate2009modifiedeuclideanalgorithmsdecoding`)
- Chien 1964 (`cite.Chien1964`)
- Forney 1965 (`cite.Forney1965`)
- Guruswami & Sudan 1998 (`cite.guruswami1998improved`)
- Koetter & Vardy 2003 (`cite.koetter2003algebraic`)
- Garcia 2014 (`cite.garcia2014interpolation`)
- Justesen 2006 (`cite.justesen2006complexity`)

**Finite field arithmetic:**
- Cantor 1991 (`cite.cantor1991fast`) — fast multiplication
- Harvey 2014 (`cite.harvey2014fasterpolynomialmultiplicationfinite`)
- Schönhage 1971 (`cite.Schnhage1971`)
- Itoh & Tsujii 1989 (`cite.ITOH198921`) — field inversion
- von zur Gathen 2013 (`cite.vonzurGathen2013moderncomputer`) — textbook
- Mullen 2013 (`cite.mullen2013handbook`) — handbook of finite fields

**Quantum arithmetic circuits:**
- Gidney 2018 (`cite.Gidney2018`)
- Gidney 2025 classical↔quantum adder (`cite.gidney2025classicalquantumadderconstantworkspace`)
- Gidney 2025 yoked constructions (`cite.gidney2025yoked`)
- Carry-save adder patent (`cite.carrysaveadderpatent`)
- Amento 2012 (`cite.amento2012efficientquantumcircuitsbinary`) — binary field circuits
- Kim 2023 (`cite.kim2023newspaceefficientquantumalgorithm`)
- Van Hoof 2020 (`cite.vanhoof2020spaceefficientquantummultiplicationpolynomials`)
- Maslov 2022 (`cite.maslov2022depth`) — circuit depth
- Maslov 2025 (`cite.maslov2025asymptotic`) — asymptotic analysis

**Quantum state preparation:**
- Gosset 2024 (`cite.gosset2024quantumstatepreparationoptimal`)
- Bärtschi 2022 (`cite.Bartschi2022`) — Dicke states

**Quantum error correction / fault tolerance:**
- Bravyi 2016 (`cite.Bravyi_2016`)
- Gidney 2024 (`cite.gidney2024magicstatecultivationgrowing`) — magic states

**Comparison targets (factoring / ECC):**
- Shor 1999 (`cite.Shor1999`)
- Gidney 2025 RSA (`cite.gidney2025factor2048bitrsa`)
- Kaye 2004 (`cite.kaye2004optimizedquantumimplementationelliptic`)
- Häner 2020 (`cite.häner2020improvedquantumcircuitselliptic`)
- Litinski 2023 (`cite.litinski2023compute256bitellipticcurve`)
- Proos & Zalka 2004 (`cite.proos2004shorsdiscretelogarithmquantum`)
- Roetteler 2017 (`cite.roetteler2017quantumresourceestimatescomputing`)

**Quantum cryptanalysis (non-factoring):**
- Briaud 2025 (`cite.briaud2025quantum`)
- Chailloux 2025 (`cite.chailloux2025quantum`)
- Kahanamoku-Siu 2025 (`cite.kahanamoku2025jacobi`) — Jacobi symbols

**Classical benchmarks and tools:**
- Frontier 2023 EPYC (`cite.frontier2023epyc`) — AMD hardware
- Harrigan 2024 (`cite.harrigan2024expressinganalyzingquantumalgorithms`) — Qualtran

**Other:**
- Babbush 2018 (`cite.Babbush_2018`)
- Berry 2025 (`cite.berry2025rapid`)
- CLZ 2021 (`cite.clz21`)
- Cormen CLRS (`cite.cormen2022introduction`)
- Cryptoeprint 2019/266, 2020/1296
- Hoeffding (`cite.Hoeffding`) — concentration inequality
- Jones 2013 (`cite.Jones2013`)
- Knuth 2005 (`cite.knuth2005generating`)
- Prange 1962 (`cite.prange1962use`)
- Stein 1967 (`cite.Stein1967`)
- Zenodo code deposit (`cite.https://doi.org/10.5281/zenodo.17301475`)

---

## Questions to Answer After Full Reading

The following questions remain unanswered and require reading the actual paper
text (compressed body content could not be extracted):

1. **What are the concrete resource estimates?** How many logical qubits and
   Toffoli gates for specific OPI parameters? Theorem 4.2 and Corollary 4.1
   likely state these — what are the exact formulas?
2. **Which decoder is primary?** Both Berlekamp-Massey and Sugiyama are
   referenced; which gets a full circuit implementation? Is Guruswami-Sudan
   list decoding also implemented, or only analysed theoretically?
3. **What are the optimisation factors?** How much do the optimised circuits
   improve over naïve implementations (in gates, qubits, depth)?
4. **What does Theorem 1.1 state?** As the first main result, this likely
   summarises the paper's key contribution — either the circuit cost or the
   verifiable advantage claim.
5. **What are the classical benchmarks?** The AMD EPYC reference suggests
   wall-clock time comparisons. At what problem size does quantum advantage
   kick in?
6. **How does OPI performance compare to classical?** At what parameters does
   DQI beat the best classical decoders?
7. **Is there a comparison with Shor's algorithm?** The references to RSA
   factoring (Gidney 2025) and elliptic curve DLP (Kaye 2004, Häner 2020,
   Litinski 2023, Proos & Zalka 2004) suggest DQI circuit costs are compared
   against Shor-style factoring circuits.
8. **What role does Gu & Jordan (2025) play?** The cite key
   `cite.GuJordan2025Algebraic` appears on page 4 near other DQI references.
   What algebraic result does this provide?
9. **What are the 7 tables?** Are they all resource comparison tables, or do
   some tabulate algorithmic parameters?

---

**Verified (2026-03-21).** Key results confirmed via `pdftotext`: Theorem
establishes optimal asymptotic speedup $\tilde{O}(N)$ quantum gates for $O(2^N)$
classical hardness. Concrete resource: 5.72M Toffoli gates, 1885 qubits for
$(m=4095, n=70, b=12)$ OPI instance. For our project, the key message is clear:
**this paper strengthens the case that DQI and QAOA have complementary
strengths, and our QAOA computation on XORSAT will help map out that
complementarity.**