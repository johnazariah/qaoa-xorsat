# Mathematics Learning Roadmap for John

> **Purpose**: Track the mathematical concepts needed to understand and explain
> the QAOA-XORSAT implementation. Ordered by priority and dependency.
> Updated as the project evolves.
>
> **Last updated**: 22 March 2026

---

## Tier 1: Concepts You Now Have (from today's coaching)

These you can explain in conversation. Review the learning docs to solidify.

- [x] **Boolean satisfiability / XOR constraints** — what Max-k-XORSAT is
- [x] **Superposition and measurement** — all $2^n$ assignments at once, probability = |amplitude|²
- [x] **QAOA circuit** — twist (problem phase) + mix (X rotation), p rounds
- [x] **Interference** — phase differences → amplitude differences via mixing
- [x] **Light cone / locality** — depth p sees p hops; on high-girth graph this is a tree
- [x] **Girth and cycles** — why cycles break the computation, why girth > 2p saves us
- [x] **The fold** — carry a branch tensor from leaves to root, one level at a time
- [x] **Branch tensor as summary** — $4^p$ entries, one per "path through p rounds"
- [x] **Element-wise exponentiation** — identical siblings combine by powering entries
- [x] **Gradient ascent for angle optimisation** — L-BFGS, smooth landscape, 2p parameters
- [x] **GF(2) arithmetic** — XOR is addition, XORSAT is linear algebra over {0,1}
- [x] **Hypergraphs** — edges connecting k>2 nodes, otherwise same as graphs

---

## Tier 2: Concepts You Understand Intuitively But Could Go Deeper

These you grasp at the "I know what it does" level. Worth sharpening for
Stephen conversations and paper writing.

- [ ] **Convolution** — weighted sum over shifted inputs; $O(n^2)$ naively
  - *You know*: the wine-stain analogy, polynomial multiplication
  - *Go deeper*: circular vs linear convolution, multi-dimensional convolution
  - *Reference*: any DSP textbook, ch. 1-2

- [ ] **DFT and FFT** — change of basis to frequency domain; convolution theorem
  - *You know*: the idea (evaluate, multiply, interpolate), $O(n \log n)$
  - *Go deeper*: roots of unity, butterfly diagram, Cooley-Tukey derivation
  - *Reference*: CLRS ch. 30, or 3Blue1Brown's FFT video

- [ ] **Walsh-Hadamard Transform** — DFT for XOR-land; $(-1)^{\langle a,b \rangle}$ basis
  - *You know*: it diagonalises XOR-convolution, same trick as FFT
  - *Go deeper*: Hadamard matrix, recursive structure, Boolean Fourier analysis
  - *Reference*: O'Donnell "Analysis of Boolean Functions" ch. 1

- [ ] **Tensor networks** — expressing quantum states as connected local tensors
  - *You know*: the fold IS a tensor contraction; branch tensor = contracted subtree
  - *Go deeper*: bond dimension, contraction ordering, MPS/MPO, tree tensor networks
  - *Reference*: Bridgeman & Chubb "Hand-waving and Interpretive Dance" (arXiv:1603.03039)

- [ ] **Ising spin glasses** — our problem IS one; spins, couplings, frustration
  - *You know*: the mapping from variables to spins
  - *Go deeper*: partition function, Boltzmann distribution, replica method, Parisi solution
  - *Reference*: Mézard & Montanari "Information, Physics, and Computation" ch. 1-3

---

## Tier 3: Concepts You'll Need for the Paper Write-Up

These are required to write equations and proofs, not just explain ideas.

- [ ] **Pauli algebra** — X, Y, Z matrices, commutation relations, tensor products
  - *Why*: the problem gate $e^{-i\gamma Z_1 Z_2 Z_3}$ and mixer $e^{-i\beta X}$
    are expressed in Pauli operators
  - *Quick version*: Z is diagonal (±1 eigenvalues), X flips bits, they anti-commute
  - *Reference*: Nielsen & Chuang §2.1, or any quantum info textbook

- [ ] **Unitary operators and matrix exponentials** — $e^{iHt}$ where $H$ is Hermitian
  - *Why*: every gate in the QAOA circuit is a matrix exponential
  - *Quick version*: $e^{i\theta Z} = \cos\theta \cdot I + i\sin\theta \cdot Z$
    (Euler formula for matrices)
  - *Reference*: N&C §2.2

- [ ] **Expectation values and the Born rule** — $\langle\psi|O|\psi\rangle$
  - *Why*: the satisfaction fraction IS an expectation value
  - *You know*: the word and the idea (average over measurements)
  - *Go deeper*: trace formulation, density matrices, POVM (probably overkill)
  - *Reference*: N&C §2.2.5

- [ ] **Group theory basics** — groups, abelian groups, characters, Pontryagin duality
  - *Why*: the WHT is the Fourier transform on the group $\mathbb{Z}_2^n$. The
    convolution theorem is a consequence of character orthogonality.
  - *Quick version*: a group is a set with an operation (for us: XOR). Characters
    are the "pure tones" of the group. WHT decomposes into characters.
  - *Reference*: Terras "Fourier Analysis on Finite Groups" ch. 1-2, or
    O'Donnell ch. 1

- [ ] **Approximation ratios vs. cut fractions** — subtly different measures
  - *You know*: cut fraction = absolute, approximation ratio = relative to optimum
  - *Go deeper*: Goemans-Williamson 0.878, unique games conjecture, inapproximability
  - *Reference*: Williamson & Shmoys "Design of Approximation Algorithms" ch. 5

- [ ] **LDPC codes and syndrome decoding** — for understanding DQI
  - *Why*: DQI reduces Max-XORSAT to decoding. The constraint matrix $B$ defines
    an LDPC code. Stephen's algorithm IS a decoder.
  - *Quick version*: $Bx = v$ over GF(2); the code is $\ker(B^T)$; decoding =
    finding the nearest codeword
  - *Reference*: Richardson & Urbanke "Modern Coding Theory" ch. 1, 3

---

## Tier 4: Nice-to-Have (Depth, Not Breadth)

These would strengthen your understanding but aren't blocking.

- [ ] **Belief propagation / message passing** — the fold IS belief propagation on a tree
  - Connection to statistical physics cavity method
  - Why it's exact on trees and approximate on graphs with cycles
  - Reference: Mézard & Montanari ch. 14

- [ ] **Adiabatic quantum computation** — QAOA is the digitised version
  - Quantum annealing, spectral gap, adiabatic theorem
  - Reference: Farhi et al. "Quantum Computation by Adiabatic Evolution" (2000)

- [ ] **Overlap Gap Property (OGP)** — why some problems are hard for local algorithms
  - QAOA at fixed p is a local/stable algorithm → subject to OGP
  - At (k=3, D=4) we're below the OGP barrier → QAOA has a chance
  - Reference: Gamarnik "The overlap gap property" survey (2021)

- [ ] **Spectral graph theory** — eigenvalues of adjacency/Laplacian matrices
  - Alon-Boppana bound, Ramanujan graphs, expansion
  - Connects to the DQI upper bound $1/2 + 1/(2\sqrt{D-1})$
  - Reference: Spielman's lecture notes (Yale)

- [ ] **Variational methods and barren plateaus** — challenges of variational quantum algorithms
  - Why QAOA works better than generic ansätze (problem structure prevents barren plateaus)
  - Reference: McClean et al. "Barren plateaus" (2018), Farhi & Harrow (2016)

---

## How to Use This

1. **For Stephen conversations**: Tiers 1-2 are sufficient. You can explain
   everything in the project with these.

2. **For writing the paper**: Tier 3 items need to be solid enough to write
   equations. Budget a few hours per item with a textbook.

3. **For your PhD more broadly**: Tier 4 connects this project to the wider
   landscape of quantum computing and optimisation.

4. **Suggestion**: pick ONE Tier 2 item per week and spend a couple of hours
   with the recommended reference. Start with the FFT (you almost have it)
   and the WHT (it's the key to our novel result).
