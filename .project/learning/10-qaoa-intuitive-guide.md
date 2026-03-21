# QAOA — Intuitive Guide

> **Purpose:** An equation-free explanation of how QAOA works, why it's related to spin glasses, how it compares to Grover's algorithm, and what our project actually computes. Read this before the paper explainers if the quantum mechanics feels abstract.

---

## The Problem: A Puzzle with Light Switches

You have 100 light switches, each UP or DOWN. There are ~133 rules like "switches 7, 23, and 41 must have an odd number in the UP position." You want to flip switches until as many rules as possible are satisfied.

- A **random guess** satisfies about 50% of rules.
- **Simulated annealing** (try random flips, keep improvements, occasionally accept bad flips to escape dead ends) gets to ~94% at our target problem size.
- **DQI** (Stephen Jordan's quantum algorithm) gets only ~87%.
- **QAOA** at depth $p$: **this is what we're computing**.

---

## What a Quantum Computer Does Differently

A classical computer tries one switch configuration at a time. A quantum computer holds **all $2^{100}$ configurations simultaneously** in superposition — but when you measure, you only see one. The art is biasing *which one* you see.

Each configuration has an "amplitude" (a complex number). The probability of seeing it when you measure is the square of its amplitude. Initially all amplitudes are equal — every configuration is equally likely.

QAOA's job: reshape the amplitudes so that good configurations (many rules satisfied) have large amplitudes, and bad ones have small amplitudes.

---

## The Two QAOA Operations

QAOA uses two knobs, applied in alternation:

### 1. The "Phase Kick" (angle $\gamma$)

This doesn't change any probabilities. It rotates each configuration's amplitude by an angle proportional to **how good that configuration is**. Good ones rotate one way, bad ones the other.

If you measured now, you'd see the same random distribution — rotated phases don't change probabilities. So why bother? Because phases set up **interference**, like light waves through two slits. Phases determine which amplitudes will reinforce and which will cancel in the next step.

### 2. The "Mixer" (angle $\beta$)

Amplitude flows between configurations that differ by a single switch flip — a diffusion process.

The magic: configurations whose phases are **aligned** with their neighbours reinforce each other. Configurations whose phases are **misaligned** cancel out.

The phase kick arranged things so that good configurations have similar phases (they were all rotated by similar amounts, because they have similar scores). The mixer then concentrates amplitude on these good configurations through constructive interference.

---

## Why Multiple Rounds Help

One round of (phase kick → mixer) is a crude filter.

With $p$ rounds, each with its own angles $(\gamma_1, \beta_1, \gamma_2, \beta_2, \ldots)$, the interference pattern becomes more refined:

- $p = 1$: a blurry lens — can tell bright from dark, details lost
- $p = 5$: sharper focus — distinguishes "pretty good" from "very good"
- $p = 15$: high resolution — amplitude concentrates tightly on near-optimal configurations

The angles are tuneable "focal lengths." Choosing them well is an optimisation problem in its own right.

---

## How QAOA Compares to Grover's Algorithm

There's a real family resemblance:

| | Grover | QAOA |
|---|--------|------|
| **Step 1** | Phase-flip the marked items | Phase-rotate all items by their quality score |
| **Step 2** | Diffusion (inversion about mean) | Mixer (amplitude flows between neighbours) |
| **Repeat** | ~$\sqrt{N}$ times, fixed angles | $p$ times, tuneable angles |

Both exploit interference. But:

- **Grover** has a sharp binary oracle: an item is the answer or it isn't. It's searching for a needle.
- **QAOA** has a smooth cost function: every configuration gets a different phase proportional to how good it is. It's sculpting a landscape.

Grover treats the search space as unstructured. QAOA knows about problem structure — the mixer moves amplitude between configurations that are *neighbours* (differ by one bit flip), respecting the geometry of the problem.

If Grover is a **metal detector** (beep / no beep, sweep systematically), then QAOA is a **landscape sculptor**: pour water on the cost landscape, tilt it so good solutions are downhill, let the water flow, repeat, and the water pools in the valleys.

---

## The Connection to Ising Spin Glasses

### The Ising Model

An Ising model is a collection of "spins" ($+1$ or $-1$) on the vertices of a graph. Each edge has a coupling, and the energy is $H = -\sum J_{ij} s_i s_j$. Finding the lowest energy is an optimisation problem.

When couplings are random (some positive, some negative), you get a **spin glass** — no configuration satisfies all couplings simultaneously.

### Our Problem *Is* a Spin Glass

The mapping is direct:
- Boolean variable $x_i \in \{0,1\}$ → spin $s_i = (-1)^{x_i} \in \{+1,-1\}$
- Constraint "$x_a \oplus x_b \oplus x_c = \text{target}$" → energy term $-J \cdot s_a s_b s_c$
- Maximising satisfied constraints = minimising Ising energy

Our problem is literally a spin glass on a 3-uniform hypergraph with uniform couplings.

### QAOA Was Born from Spin Glass Physics

QAOA is the discretised version of **quantum adiabatic computing** on Ising models:

| Adiabatic | QAOA |
|-----------|------|
| Continuous evolution under a slowly-changing Hamiltonian | Discrete rounds of two operations |
| Parameter: total time $T$ | Parameter: depth $p$ and angles |
| Guaranteed optimal as $T \to \infty$ | Guaranteed optimal as $p \to \infty$ |
| Impractical (requires exponential time) | Practical (fixed $p$, optimise angles) |

Each QAOA round is:
1. Evolve under the **Ising Hamiltonian** for "time" $\gamma$ (the phase kick)
2. Evolve under the **transverse field** for "time" $\beta$ (the mixer)

### Why Spin Glass Theory Constrains Our Problem

Results from spin glass theory — particularly the **overlap gap property (OGP)** — constrain what algorithms can achieve. At high constraint density the solution space shatters, blocking local algorithms.

For our specific case (k=3, D=4), density is moderate and we're below the SAT threshold. The glass is "easy" (replica symmetric, no shattering). This is why SA does well and why QAOA has a fighting chance.

---

## Why the Computation Works on a Tree

When you ask "what's the probability that rule #47 is satisfied after QAOA at depth $p$?", the answer depends only on switches and rules within $p$ hops of rule #47. Everything farther away is in uniform superposition and contributes no net bias.

On a regular graph with no short cycles, this neighbourhood is a **tree**. And because the graph is regular, **every rule's tree looks the same**. Compute one tree → done.

---

## Why We Don't Need Exponential Memory

The tree has ~$6^p$ switches, but we never store the full quantum state ($2^{6^p}$ amplitudes).

Instead, we work layer by layer from leaves to root. Each layer's contribution is summarised by a vector of $4^p$ numbers (the "branch tensor"). The $4^p$ comes from tracking both the forward and backward quantum evolution of one qubit through $p$ rounds: $2^p \times 2^p = 4^p$.

The key insight: on a regular tree, every branch at the same depth is **identical**. If one branch contributes a summary $T$, then three identical independent sibling branches contribute $T^3$ — entry by entry.

Instead of contracting each of the $6^p$ branches separately, we contract **one** and raise it to powers at each level.

**Cost**: $O(p \cdot 4^p)$ time, $O(4^p)$ space — independent of the tree size.

| $p$ | $4^p$ entries | Memory | Feasibility |
|-----|--------------|--------|-------------|
| 5 | 1,024 | 8 KB | Trivial |
| 10 | ~1M | 8 MB | Seconds |
| 12 | ~16M | 128 MB | Minutes |
| 15 | ~1B | 8 GB | Hours |

---

## What Our Code Does — In 7 Steps

```
For each candidate set of 2p angles:
  1. Build the summary for one leaf switch         [4^p numbers]
  2. Raise entry-wise to power (D-1) = 3           [sibling branches at variable]
  3. Apply this round's mixing transformation       [small matrix operation]
  4. Raise entry-wise to power (k-1) = 2           [sibling branches at constraint]
  5. Apply this round's phase transformation        [diagonal operation]
  6. Repeat steps 2-5 for rounds p-1, ..., 1
  7. Read off the satisfaction probability          [one number]

Optimise over the 2p angles to maximise that number.
```

This maps directly to our three specs:
- **P1.1** (Factor Tree) = the tree shape that determines step structure
- **P1.2** (Tensors) = the transformations in steps 3 and 5
- **P1.3** (Contraction) = the loop implementing steps 1–7

---

## What Success Looks Like

A table with the `???` filled in:

| $p$ | QAOA fraction | SA (0.9366) | DQI+BP (0.8707) |
|-----|--------------|-------------|-----------------|
| 1 | ??? | | |
| 2 | ??? | | |
| 5 | ??? | | |
| 10 | ??? | | |
| $p_{\max}$ | ??? | | |

If QAOA at some feasible $p$ exceeds 0.9366, a quantum algorithm outperforms the best known classical heuristic at this problem size. If it doesn't, that's equally informative. Stephen gets the precise comparison either way.
