# QAOA-XORSAT Visual Walkthrough

> **Purpose**: Diagrams for explaining the (k=3, D=4) problem to Stephen Jordan.
> Covers the hypergraph structure, light-cone expansion, and the fold contraction.

---

## 1. A Minimal (k=3, D=4) Hypergraph Instance

Each variable participates in exactly D=4 constraints. Each constraint connects
exactly k=3 variables. With n=9 variables and m=12 constraints (since 3m = 4n):

```mermaid
graph LR
    subgraph Variables
        x1((x₁))
        x2((x₂))
        x3((x₃))
        x4((x₄))
        x5((x₅))
        x6((x₆))
        x7((x₇))
        x8((x₈))
        x9((x₉))
    end

    subgraph Constraints
        c1[c₁: x₁⊕x₂⊕x₃=0]
        c2[c₂: x₁⊕x₄⊕x₅=1]
        c3[c₃: x₁⊕x₆⊕x₇=0]
        c4[c₄: x₁⊕x₈⊕x₉=1]
        c5[c₅: x₂⊕x₄⊕x₆=0]
        c6[c₆: x₂⊕x₅⊕x₈=1]
        c7[c₇: x₂⊕x₇⊕x₉=0]
        c8[c₈: x₃⊕x₄⊕x₉=1]
        c9[c₉: x₃⊕x₅⊕x₇=0]
        c10[c₁₀: x₃⊕x₆⊕x₈=1]
        c11[c₁₁: x₄⊕x₇⊕x₈=0]
        c12[c₁₂: x₅⊕x₆⊕x₉=1]
    end

    x1 --- c1
    x1 --- c2
    x1 --- c3
    x1 --- c4
    x2 --- c1
    x2 --- c5
    x2 --- c6
    x2 --- c7
    x3 --- c1
    x3 --- c8
    x3 --- c9
    x3 --- c10
    x4 --- c2
    x4 --- c5
    x4 --- c8
    x4 --- c11
    x5 --- c2
    x5 --- c6
    x5 --- c9
    x5 --- c12
    x6 --- c3
    x6 --- c5
    x6 --- c10
    x6 --- c12
    x7 --- c3
    x7 --- c7
    x7 --- c9
    x7 --- c11
    x8 --- c4
    x8 --- c6
    x8 --- c10
    x8 --- c11
    x9 --- c4
    x9 --- c7
    x9 --- c8
    x9 --- c12
```

**Key observations:**
- Every variable (circle) has exactly **4 edges** → D=4 regularity
- Every constraint (box) has exactly **3 edges** → k=3 uniformity
- Count: 9 variables × 4 = 36 edge-endpoints = 12 constraints × 3 ✓
- The target bits (0 or 1) on each constraint are arbitrary — on a tree they
  can be gauged away (Basso §8.1)
- This graph has **short cycles** (e.g., x₁→c₁→x₂→c₅→x₄→c₂→x₁, length 6).
  For QAOA analysis we need girth > 2p, so at large n the graph would be
  locally tree-like. The diagrams below show the *tree* that would appear
  as the local neighbourhood.

---

## 2. The Light Cone: What One Constraint Sees

### p=1: One round of twist-mix (21 qubits, 10 constraints)

The root constraint c₀ connects 3 variables. Each variable has 3 OTHER
constraints (D-1=3). Each of those constraints connects 2 OTHER variables
(k-1=2). Those are the leaves — the boundary qubits in |+⟩.

```mermaid
graph TD
    c0["🔷 c₀ (ROOT)
    x₁⊕x₂⊕x₃"]

    c0 --> x1(("x₁"))
    c0 --> x2(("x₂"))
    c0 --> x3(("x₃"))

    x1 --> c1["c₁"]
    x1 --> c2["c₂"]
    x1 --> c3["c₃"]

    x2 --> c4["c₄"]
    x2 --> c5["c₅"]
    x2 --> c6["c₆"]

    x3 --> c7["c₇"]
    x3 --> c8["c₈"]
    x3 --> c9["c₉"]

    c1 --> l1(("l₁"))
    c1 --> l2(("l₂"))
    c2 --> l3(("l₃"))
    c2 --> l4(("l₄"))
    c3 --> l5(("l₅"))
    c3 --> l6(("l₆"))

    c4 --> l7(("l₇"))
    c4 --> l8(("l₈"))
    c5 --> l9(("l₉"))
    c5 --> l10(("l₁₀"))
    c6 --> l11(("l₁₁"))
    c6 --> l12(("l₁₂"))

    c7 --> l13(("l₁₃"))
    c7 --> l14(("l₁₄"))
    c8 --> l15(("l₁₅"))
    c8 --> l16(("l₁₆"))
    c9 --> l17(("l₁₇"))
    c9 --> l18(("l₁₈"))

    style c0 fill:#dbeafe,stroke:#2563eb,color:#1e3a5f
    style x1 fill:#dcfce7,stroke:#16a34a,color:#14532d
    style x2 fill:#dcfce7,stroke:#16a34a,color:#14532d
    style x3 fill:#dcfce7,stroke:#16a34a,color:#14532d
    style c1 fill:#fef3c7,stroke:#d97706,color:#78350f
    style c2 fill:#fef3c7,stroke:#d97706,color:#78350f
    style c3 fill:#fef3c7,stroke:#d97706,color:#78350f
    style c4 fill:#fef3c7,stroke:#d97706,color:#78350f
    style c5 fill:#fef3c7,stroke:#d97706,color:#78350f
    style c6 fill:#fef3c7,stroke:#d97706,color:#78350f
    style c7 fill:#fef3c7,stroke:#d97706,color:#78350f
    style c8 fill:#fef3c7,stroke:#d97706,color:#78350f
    style c9 fill:#fef3c7,stroke:#d97706,color:#78350f
    style l1 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style l2 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style l3 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style l4 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style l5 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style l6 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style l7 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style l8 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style l9 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style l10 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style l11 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style l12 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style l13 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style l14 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style l15 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style l16 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style l17 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style l18 fill:#f0fdf4,stroke:#86efac,color:#14532d
```

**Counts at p=1:**
| Level | Type | Count | Colour |
|-------|------|-------|--------|
| 0 | Root constraint | 1 | Light blue |
| 1 | Root variables (x₁,x₂,x₃) | k=3 | Light green |
| 2 | Child constraints | k(D-1) = 9 | Light amber |
| 3 | Leaf variables | k(D-1)(k-1) = 18 | Pale green |
| | **Total qubits** | **21** | |

### Why it's a tree

No variable appears twice. Each branch is **independent** — the subtree below
c₁ (containing l₁, l₂) shares no qubits with the subtree below c₂ (containing
l₃, l₄). This is because girth > 2p = 2 on the original graph.

### p=2: Two rounds (129 qubits, 64 constraints)

At p=2, each leaf variable from p=1 becomes an internal node, sprouting 3 more
child constraints, each with 2 more leaf variables. The tree grows by branching
factor b = (D-1)(k-1) = 6 per two-level step.

```mermaid
graph TD
    c0["🔷 c₀ ROOT"]
    c0 --> x1(("x₁"))
    c0 --> x2(("x₂"))
    c0 --> x3(("x₃"))

    x1 --> c1["c₁"]
    x1 --> c2["c₂"]
    x1 --> c3["c₃"]

    c1 --> v1(("v₁"))
    c1 --> v2(("v₂"))

    v1 --> d1["·"]
    v1 --> d2["·"]
    v1 --> d3["·"]
    v2 --> d4["·"]
    v2 --> d5["·"]
    v2 --> d6["·"]

    d1 --> p1(("·"))
    d1 --> p2(("·"))
    d2 --> p3(("·"))
    d2 --> p4(("·"))
    d3 --> p5(("·"))
    d3 --> p6(("·"))
    d4 --> p7(("·"))
    d4 --> p8(("·"))
    d5 --> p9(("·"))
    d5 --> p10(("·"))
    d6 --> p11(("·"))
    d6 --> p12(("·"))

    c2 --> etc1(("...×2 vars"))
    c3 --> etc2(("...×2 vars"))
    x2 --> etc3["...×3 constraints"]
    x3 --> etc4["...×3 constraints"]

    style c0 fill:#dbeafe,stroke:#2563eb,color:#1e3a5f
    style x1 fill:#dcfce7,stroke:#16a34a,color:#14532d
    style x2 fill:#dcfce7,stroke:#16a34a,color:#14532d
    style x3 fill:#dcfce7,stroke:#16a34a,color:#14532d
    style c1 fill:#fef3c7,stroke:#d97706,color:#78350f
    style c2 fill:#fef3c7,stroke:#d97706,color:#78350f
    style c3 fill:#fef3c7,stroke:#d97706,color:#78350f
    style v1 fill:#dcfce7,stroke:#16a34a,color:#14532d
    style v2 fill:#dcfce7,stroke:#16a34a,color:#14532d
    style d1 fill:#fef3c7,stroke:#d97706,color:#78350f
    style d2 fill:#fef3c7,stroke:#d97706,color:#78350f
    style d3 fill:#fef3c7,stroke:#d97706,color:#78350f
    style d4 fill:#fef3c7,stroke:#d97706,color:#78350f
    style d5 fill:#fef3c7,stroke:#d97706,color:#78350f
    style d6 fill:#fef3c7,stroke:#d97706,color:#78350f
    style p1 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style p2 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style p3 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style p4 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style p5 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style p6 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style p7 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style p8 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style p9 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style p10 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style p11 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style p12 fill:#f0fdf4,stroke:#86efac,color:#14532d
```

**Growth at p=2** (showing only c₁'s subtree; the full tree has 9 such subtrees):
| Level | Type | Count | Running total |
|-------|------|-------|---------------|
| 0 | Root constraint | 1 | 1 constraint |
| 1 | Root variables | 3 | 3 variables |
| 2 | Child constraints | 9 | 10 constraints |
| 3 | Internal variables | 18 | 21 variables |
| 4 | Grandchild constraints | 54 | 64 constraints |
| 5 | **Leaf variables** | **108** | **129 variables** |

**129 qubits at p=2.** Brute-force simulation would need 2¹²⁹ amplitudes — more
than atoms in the observable universe. But the fold needs only a 4² = 16 entry
branch tensor.

---

## 3. The Fold: Contracting from Leaves to Root

This diagram shows the fold for p=1, focusing on the path from leaves through
one child constraint (c₁) up to root variable x₁ and then the root observable.

**The key insight: every dotted box contains the SAME computation** (by symmetry).

### 3a. Zooming in: one constraint neighbourhood

The fold operates locally. Here's the view from inside **one constraint node**
(c₁) looking at its two child variable branches. This is the fundamental unit
of work — repeated at every constraint in the tree, but computed only once.

```mermaid
graph TD
    parent(("x₁ parent
    receives B'"))

    c1["c₁ CONSTRAINT NODE
    combines 2 children
    + applies problem phase"]

    childA(("child A
    branch tensor B"))
    childB(("child B
    branch tensor B
    (identical to A)"))

    leafA1(("🍃"))
    leafA2(("🍃"))
    leafB1(("🍃"))
    leafB2(("🍃"))

    parent --- c1
    c1 --- childA
    c1 --- childB
    childA --- leafA1
    childA --- leafA2
    childB --- leafB1
    childB --- leafB2

    style parent fill:#dcfce7,stroke:#16a34a,color:#14532d
    style c1 fill:#fef3c7,stroke:#d97706,color:#78350f
    style childA fill:#dcfce7,stroke:#16a34a,color:#14532d
    style childB fill:#dcfce7,stroke:#16a34a,color:#14532d
    style leafA1 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style leafA2 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style leafB1 fill:#f0fdf4,stroke:#86efac,color:#14532d
    style leafB2 fill:#f0fdf4,stroke:#86efac,color:#14532d
```

**What happens at c₁:**

Both children have produced the **same** branch tensor B (4ᵖ entries).
The constraint must combine them, weighted by the 3-body problem phase
(which couples parent × child A × child B):

| Step | Operation | What it does | Cost |
|------|-----------|-------------|------|
| 1 | $W = g \star g$ | Self-convolve children (WHT: $\hat{W} = \hat{g}^2$) | O(p · 4ᵖ) |
| 2 | $S = \kappa \star W$ | Convolve with cosine kernel (WHT: $\hat{S} = \hat{\kappa} \cdot \hat{W}$) | O(p · 4ᵖ) |
| 3 | $B'[a] = S[a]^D$ | Raise to Dth power (variable fold above) | O(4ᵖ) |

The combined result B' is passed up to x₁. That's one constraint fold + one
variable fold — one "level" of the tree absorbed into the branch tensor.

### 3b. Zooming in: one variable neighbourhood

Here's the view from inside **one variable node** (x₁) looking at its (D-1)=3
child constraints. Much simpler — no coupling.

```mermaid
graph TD
    root["c₀ ROOT
    receives B''"]

    x1(("x₁ VARIABLE NODE
    absorbs 3 identical constraints"))

    c1["c₁ → B'"]
    c2["c₂ → B'"]
    c3["c₃ → B'"]

    root --- x1
    x1 --- c1
    x1 --- c2
    x1 --- c3

    style root fill:#dbeafe,stroke:#2563eb,color:#1e3a5f
    style x1 fill:#dcfce7,stroke:#16a34a,color:#14532d
    style c1 fill:#fef3c7,stroke:#d97706,color:#78350f
    style c2 fill:#fef3c7,stroke:#d97706,color:#78350f
    style c3 fill:#fef3c7,stroke:#d97706,color:#78350f
```

**What happens at x₁:**

All three child constraints produced the **same** B'. Since they're independent:

| Step | Operation | What it does | Cost |
|------|-----------|-------------|------|
| 1 | $B'' = B'^{D-1}$ | Element-wise cube (3 identical, independent siblings) | O(4ᵖ) |
| 2 | $B'' = M(\beta) \cdot B''$ | Apply mixer for this round | O(4ᵖ) |

That's it. No triple loop, no WHT. Just power and multiply.

### 3c. The full fold pipeline (contracted)

Each round folds one constraint level + one variable level. The branch tensor
B flows upward, absorbing one twist-mix round at each step:

```mermaid
graph BT
    L["🍃 Leaf: B = 2⁻ᵖ everywhere"]

    R2["Round p: constraint fold (WHT) → variable fold (.^D-1) → mixer(βₚ)"]
    R1["Round p-1: constraint fold (WHT) → variable fold (.^D-1) → mixer(βₚ₋₁)"]
    dots["⋮ repeat for each round ⋮"]
    Rf["Round 1: constraint fold (WHT) → variable fold (.^D-1) → mixer(β₁)"]
    ROOT["🎯 Root observable → c̃"]

    L --> R2 --> R1 --> dots --> Rf --> ROOT

    style L fill:#f0fdf4,stroke:#86efac,color:#14532d
    style R2 fill:#fefce8,stroke:#d97706,color:#78350f
    style R1 fill:#fefce8,stroke:#d97706,color:#78350f
    style dots fill:#f9fafb,stroke:#d1d5db,color:#374151,stroke-dasharray: 5 5
    style Rf fill:#fefce8,stroke:#d97706,color:#78350f
    style ROOT fill:#dbeafe,stroke:#2563eb,color:#1e3a5f
```

At every stage, only one vector of 4ᵖ entries exists. The tree (21 qubits at
p=1, 129 at p=2, billions at p=10) never materialises. The fold compresses
it all.

### Variable fold vs. Constraint fold — why one is hard

```mermaid
graph LR
    subgraph VF["Variable Fold (EASY)"]
        direction TB
        vf1["(D-1) identical child CONSTRAINT branches"]
        vf2["All independent, all identical"]
        vf3["B .^= (D-1)
        element-wise power"]
        vf1 --> vf2 --> vf3
    end

    subgraph CF["Constraint Fold (HARD at k>2)"]
        direction TB
        cf1["(k-1) identical child VARIABLE branches"]
        cf2["Interact through k-body problem gate"]
        cf3["NAIVE: sum over all (k-1)-tuples
        O(4^{kp})"]
        cf4["WHT: convolution theorem
        O(p · 4^p)"]
        cf1 --> cf2 --> cf3
        cf2 --> cf4
    end

    style VF fill:#f0fdf4,stroke:#16a34a,color:#14532d
    style CF fill:#fef2f2,stroke:#dc2626,color:#7f1d1d
```

**Why the variable fold is easy:** At a variable node, the (D-1) child constraint
branches are independent and identical. Their combined contribution is just the
product: $B[σ]^{D-1}$ for each hyperindex entry. Element-wise power. Done.

**Why the constraint fold is hard (for k>2):** At a constraint node, the (k-1)
child variable branches interact through the k-body problem gate. The gate
evaluates $\cos(\Gamma \cdot (a \odot b^1 \odot b^2) / \sqrt{D})$, which
couples all children simultaneously. You can't just power one child's tensor.

**Why the WHT saves us:** The coupling has the form of a **convolution on
XOR-space**. The Walsh-Hadamard transform diagonalises this convolution,
reducing the cost from O(4^{kp}) to O(p · 4^p) — the same as the variable fold.

### The complete fold for p=2

At p=2, the fold has **two** rounds. Each round does one variable fold + one
constraint fold. The branch tensor is carried through both rounds:

```mermaid
graph BT
    L["🍃 Leaf: B = [2⁻², ..., 2⁻²]  (16 entries at p=2)"]

    R2V["Round 2 — Variable fold: B .^= (D-1)"]
    R2M["Round 2 — Mixer: B = M(β₂) * B"]
    R2C["Round 2 — Constraint fold: Ŝ = κ̂₂ · ĝ²"]
    R2P["Round 2 — Problem gate: B .*= P(γ₂)"]

    R1V["Round 1 — Variable fold: B .^= (D-1)"]
    R1M["Round 1 — Mixer: B = M(β₁) * B"]
    R1C["Round 1 — Constraint fold: Ŝ = κ̂₁ · ĝ²"]
    R1P["Round 1 — Problem gate: B .*= P(γ₁)"]

    ROOT["🎯 Root observable → c̃ (one number)"]

    L --> R2V --> R2M --> R2C --> R2P
    R2P --> R1V --> R1M --> R1C --> R1P
    R1P --> ROOT

    style L fill:#d4edda
    style R2V fill:#fff3cd
    style R2M fill:#cce5ff
    style R2C fill:#f8d7da
    style R2P fill:#fff3cd
    style R1V fill:#fff3cd
    style R1M fill:#cce5ff
    style R1C fill:#f8d7da
    style R1P fill:#fff3cd
    style ROOT fill:#4a90d9,color:#fff
```

The pattern is clear: **for each round p, p-1, ..., 1: variable fold → mixer →
constraint fold → problem gate. Then apply the root observable.** The branch
tensor B (16 entries at p=2, 4^p entries in general) is the only data structure
that passes between steps. The entire tree — with its 129 qubits and 64
constraints — has been compressed into this single vector.

---

## 4. Cost Summary

| Component | p=1 | p=5 | p=10 | p=15 |
|-----------|-----|-----|------|------|
| Light-cone qubits | 21 | 27,993 | ~6×10⁹ | ~10¹¹ |
| Branch tensor entries | 4 | 1,024 | ~10⁶ | ~10⁹ |
| Memory for branch tensor | 64 B | 16 KB | 16 MB | 16 GB |
| Cost per fold step (WHT) | ~40 ops | ~10⁴ | ~10⁷ | ~10¹⁰ |
| **Full evaluation** | instant | ms | seconds | hours |

The tree size grows as $6^p$ (exponential explosion). The branch tensor grows as
$4^p$ (still exponential, but much slower). The fold never materialises the tree —
it only ever holds the branch tensor.
