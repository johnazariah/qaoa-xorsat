# Foundational Concepts

> **Purpose:** This document covers the prerequisite concepts you need before reading the three key papers. Read this first. If you're already comfortable with quantum computing basics and graph theory, skim it and move on.

---

## 1. Qubits, States, and Measurement

### What is a qubit?

A classical bit is either 0 or 1. A **qubit** is a quantum system that can be in a **superposition** of 0 and 1:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$

where $\alpha, \beta$ are complex numbers with $|\alpha|^2 + |\beta|^2 = 1$.

When you **measure** a qubit, you get outcome 0 with probability $|\alpha|^2$ and outcome 1 with probability $|\beta|^2$. After measurement, the qubit collapses to the observed state.

### Multiple qubits

$n$ qubits live in a $2^n$-dimensional space. A general state is:

$$|\psi\rangle = \sum_{x \in \{0,1\}^n} \alpha_x |x\rangle$$

where $x$ is a bit string of length $n$, and $\sum_x |\alpha_x|^2 = 1$.

The **uniform superposition** is the state where all $2^n$ bit strings have equal amplitude:

$$|s\rangle = \frac{1}{\sqrt{2^n}} \sum_{x \in \{0,1\}^n} |x\rangle$$

This is the starting state for QAOA. You create it by applying a Hadamard gate to each qubit initialised in $|0\rangle$.

### Expectation values

Given an observable (a Hermitian operator) $C$ and a quantum state $|\psi\rangle$, the **expectation value** is:

$$\langle C \rangle = \langle\psi| C |\psi\rangle$$

This is the average value you'd get if you measured $C$ many times on copies of $|\psi\rangle$. For us, $C$ will be a cost function (like the number of cut edges), and maximising $\langle C \rangle$ is the goal.

---

## 2. Quantum Gates and Circuits

### Unitary operators

Quantum evolution is described by **unitary operators** $U$ (matrices with $U^\dagger U = I$). Applying $U$ to state $|\psi\rangle$ gives $U|\psi\rangle$.

### Key gates for QAOA

**Pauli operators** on a single qubit:

- $X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ — bit flip (swaps $|0\rangle$ and $|1\rangle$)
- $Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ — phase flip ($|0\rangle \to |0\rangle$, $|1\rangle \to -|1\rangle$)
- $Z_iZ_j$ — measures whether qubits $i$ and $j$ agree ($+1$) or disagree ($-1$)

**Rotation gates:**

- $e^{-i\gamma Z_i Z_j}$ — a diagonal gate that applies a phase depending on whether qubits $i,j$ agree or disagree. This is the building block of the "problem unitary" in QAOA for MaxCut.
- $e^{-i\beta X_j}$ — rotates qubit $j$ around the X-axis. This is the "mixer" gate.

### Quantum circuits

A quantum circuit is a sequence of gates applied to qubits. The depth of a circuit is the number of sequential layers (gates in the same layer can act in parallel on different qubits). QAOA circuits have depth proportional to $p$, the number of rounds.

---

## 3. Combinatorial Optimisation Problems

### MaxCut

**Input:** An undirected graph $G = (V, E)$ with vertices $V$ and edges $E$.

**Goal:** Partition the vertices into two sets $S$ and $\bar{S}$ to maximise the number of edges crossing between the two sets (the "cut").

**As a cost function:** Assign each vertex $i$ a bit $b_i \in \{0,1\}$. An edge $(i,j)$ is "cut" if $b_i \neq b_j$. The cost function is:

$$C(b) = \sum_{(i,j) \in E} (b_i \oplus b_j) = \sum_{(i,j) \in E} \frac{1 - Z_iZ_j}{2}$$

where the second form uses the quantum operator $Z_i$ (which has eigenvalue $+1$ on $|0\rangle$ and $-1$ on $|1\rangle$). When $b_i = b_j$, $Z_iZ_j = +1$ and the term contributes 0 (edge not cut). When $b_i \neq b_j$, $Z_iZ_j = -1$ and the term contributes 1 (edge cut).

### XOR Satisfiability (XORSAT)

**k-XORSAT:** You have $n$ Boolean variables and $m$ constraints. Each constraint involves exactly $k$ variables and asserts that their XOR (sum mod 2) equals some target bit. For example, with $k=3$: "$x_1 \oplus x_4 \oplus x_7 = 1$".

**Max-k-XORSAT:** Maximise the number of satisfied constraints (since not all may be simultaneously satisfiable).

**Connection to MaxCut:** MaxCut is the special case $k=2$! Each edge $(i,j)$ is a constraint "$b_i \oplus b_j = 1$" (asking for the endpoints to differ). So MaxCut = Max-2-XORSAT.

When $k=3$, each constraint involves 3 variables, and the underlying structure is a **hypergraph** rather than a graph.

### The "fraction of constraints satisfied" 

If there are $m$ constraints total and the algorithm satisfies $c$ of them in expectation, the **fraction of constraints satisfied** is $c/m$. This is the quantity Stephen wants to compute precisely for QAOA at each depth $p$.

A random assignment satisfies half the constraints on average (since each XOR is equally likely to be 0 or 1). So $1/2$ is the trivial baseline. Anything above $1/2$ is doing something useful.

---

## 4. Graphs, Hypergraphs, and Regularity

### Regular graphs

A graph is **$D$-regular** if every vertex has exactly $D$ neighbours. For example:
- A cycle is 2-regular
- The Petersen graph is 3-regular

### Hypergraphs

A **hypergraph** generalises a graph: edges can connect more than 2 vertices. A **hyperedge** of size $k$ connects $k$ vertices. A **$k$-uniform** hypergraph has all hyperedges of size $k$.

For Max-k-XORSAT on a $D$-regular $k$-uniform hypergraph:
- Each variable (vertex) participates in exactly $D$ constraints (hyperedges)
- Each constraint (hyperedge) involves exactly $k$ variables

**Our target:** $k=3, D=4$. Each variable appears in 4 constraints, each constraint involves 3 variables.

### Girth

The **girth** of a graph is the length of its shortest cycle. A graph with large girth looks locally like a tree — there are no short loops.

**Why girth matters for QAOA:** The QAOA at depth $p$ can only "see" the neighbourhood of radius $p$ around each constraint. If the girth is large enough ($g \geq 2p+2$), these neighbourhoods are **trees**. On trees, the computation simplifies dramatically because every branch is independent.

### Factor graphs

For k-XORSAT, the natural structure is a **factor graph** (also called a Tanner graph): a bipartite graph with two types of nodes:
- **Variable nodes** (circles): one per variable
- **Factor/constraint nodes** (squares): one per constraint

An edge connects variable $i$ to constraint $\alpha$ if variable $i$ appears in constraint $\alpha$. In a $D$-regular $k$-uniform instance, variable nodes have degree $D$ and constraint nodes have degree $k$.

---

## 5. Approximation Ratios vs. Cut Fractions

These are two different ways to measure algorithm performance. **They are not the same thing**, and confusing them is a common trap.

### Approximation ratio

$$r = \frac{\text{value achieved by algorithm}}{\text{optimal value}}$$

The Goemans-Williamson algorithm guarantees $r \geq 0.878$ for MaxCut on any graph.

### Cut fraction

$$f = \frac{\text{number of edges cut}}{\text{total number of edges}}$$

This measures absolute performance, not relative to the optimum. The QAOA papers primarily work with cut fractions (or fraction of constraints satisfied for XORSAT).

**Example:** On a graph where the optimal cut has 90% of edges, achieving a cut fraction of 0.80 gives an approximation ratio of $0.80/0.90 = 0.889$. But on a graph where the optimal cut has 100% of edges (bipartite), the same cut fraction of 0.80 gives an approximation ratio of only 0.80.

**For our project:** We care about the **fraction of constraints satisfied**, which is analogous to cut fraction.

---

## 6. Variational Quantum Algorithms

The QAOA belongs to a family called **variational quantum algorithms** (VQAs). The general pattern:

1. **Parameterised quantum circuit:** Prepare a quantum state $|\psi(\theta)\rangle$ that depends on classical parameters $\theta$.
2. **Measurement:** Measure the cost function to estimate $\langle\psi(\theta)|C|\psi(\theta)\rangle$.
3. **Classical optimisation:** Use a classical optimiser to update $\theta$ to improve the cost.
4. **Repeat** until convergence.

What makes QAOA special within VQAs:
- The circuit structure is **derived from the problem itself** (the cost function determines the gates)
- Performance is **guaranteed to improve** (monotonically) as depth $p$ increases
- At $p \to \infty$, QAOA converges to the optimal solution (it's universal)
- At small fixed $p$, the performance can be analysed exactly on structured graphs

### The critical insight for our project

We are NOT running QAOA on a quantum computer. We are **classically simulating** what QAOA would do, to calculate its expected performance. This is feasible because:
1. On large-girth regular graphs, every constraint sees the same tree neighbourhood
2. We only need to compute the expectation value for ONE constraint (they're all identical)
3. The tree structure allows efficient contraction (we don't need the full exponential state vector)

This classical pre-computation tells us: "If you ran QAOA at depth $p$ with these specific angles, it would satisfy at least this fraction of constraints." It's a mathematical proof of a performance lower bound.

---

## 7. Tensor Networks (Brief Introduction)

Tensor networks are a way to represent and compute with large quantum states efficiently. You'll encounter them in the Farhi et al. 2025 paper.

### What is a tensor?

A tensor is a multi-dimensional array of numbers:
- **Scalar** (0 indices): a single number
- **Vector** (1 index): $v_i$  
- **Matrix** (2 indices): $M_{ij}$
- **3-tensor** (3 indices): $T_{ijk}$

### Tensor contraction

Summing over a shared index between two tensors:

$$C_{ik} = \sum_j A_{ij} B_{jk}$$

This is just matrix multiplication! But it generalises to higher-order tensors.

### Tensor networks for quantum circuits

A quantum circuit can be represented as a network of tensors (one per gate) connected along shared qubit lines. Computing an expectation value $\langle\psi|O|\psi\rangle$ means contracting this entire network.

The order in which you contract matters enormously for efficiency. On a tree-structured network, you can contract from the leaves inward, which is far more efficient than brute-force. This is the key technique in the Farhi et al. 2025 paper that makes large-$p$ calculations feasible.

### The tree contraction trick

On a $D$-regular tree, every branch looks identical. Instead of contracting each branch independently, you:
1. Contract a single branch from its leaves inward to get a tensor $T$
2. Raise each entry of $T$ to the power $(D-1)$ (because there are $D-1$ identical branches)
3. Continue contracting toward the root

This reduces the cost from exponential in the tree size to polynomial in $p$ times $O(4^p)$. It's the reason the Farhi 2025 paper can push to $p=17$.

---

## Next Steps

Now read the paper explainers in order:
1. `01-explainer-farhi2014-original-qaoa.md` — The original QAOA paper
2. `02-explainer-basso2021-high-depth.md` — Scaling to high depth  
3. `03-explainer-farhi2025-maxcut-lower-bound.md` — The tensor network method we'll adapt

Then read `04-our-problem.md` to see how everything connects to our specific task.
