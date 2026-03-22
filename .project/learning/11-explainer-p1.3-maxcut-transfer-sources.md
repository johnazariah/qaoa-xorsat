# P1.3 Source Note: MaxCut Transfer-Matrix References

> **Purpose:** Record the external sources used for the recent P1.3 MaxCut
> transfer work, and clarify how they relate to the exact tensor-contraction
> route we ultimately need for Max-k-XORSAT.

---

## Why This Note Exists

Recent P1.3 work touched two different kinds of "compression" for large-girth
QAOA on trees:

1. the **exact finite-D contraction** from Farhi et al. 2025,
2. the **compact MaxCut recursion** associated with Basso et al. and the public
   `benjaminvillalonga/large-girth-maxcut-qaoa` codebase.

Those are related, but they are not the same method and they do not justify the
same claims. This note records the source lineage so later branch work does not
blur them together.

---

## Source Map

| Source | Where it lives in this repo | Role in P1.3 work | What to trust it for |
|--------|------------------------------|-------------------|----------------------|
| Farhi, Gutmann, Ranard, Villalonga (2025), arXiv:2503.12789 | `../papers/farhi2025-maxcut-lower-bound.pdf` and `03-explainer-farhi2025-maxcut-lower-bound.md` | Exact MaxCut tensor-network template | Exact finite-D regular-tree contraction and the `O(p 4^p)` target cost |
| Basso, Farhi, Marwaha, Villalonga, Zhou (2021 preprint; later TQC version) | `../papers/basso2021-qaoa-high-depth.pdf` and `02-explainer-basso2021-high-depth.md` | Compact large-girth MaxCut / Max-q-XORSAT recursion | High-depth transfer-style recursion, especially the `D -> infinity` viewpoint |
| `benjaminvillalonga/large-girth-maxcut-qaoa` | External public GitHub repository; not vendored here | Implementation reference for the compact MaxCut recursion | Practical update ordering, data-shape conventions, and optimisation workflow for the MaxCut-only recurrence |
| This branch's experimental Julia port | `.project/specs/P1.3-contraction.md` and `.project/implementation-notes/P1.3.md` | Local experiment / cross-check scaffold | How the MaxCut transfer ideas were tested against this branch's exact reference machinery |

---

## The Relationship Between the Sources

### 1. Farhi 2025: exact finite-D contraction

Farhi et al. 2025 is the source we should treat as the **ground truth** for the
regular-tree contraction story at fixed finite degree.

For 3-regular MaxCut, the paper shows how to:

- represent the light-cone expectation as a bra-ket tensor network,
- contract one representative branch,
- use entrywise powers only where branch multiplicity is genuinely identical and
  independent,
- obtain an exact cost scaling of `O(p 4^p)` time and `O(4^p)` space.

For this project, that is the key conceptual bridge from a huge tree to a small
branch message. It is also the standard any future k-XORSAT recursion must meet.

### 2. Basso et al.: compact recursion, but not the same exact object

Basso et al. gives a **compact iterative description** of large-girth QAOA,
including the Max-q-XORSAT generalisation that matters for us. But the paper's
main strength is the high-depth large-D regime, especially the SK limit.

The important distinction is:

- Farhi 2025 keeps the finite-D tree contraction exact.
- Basso et al. tracks a more compressed set of correlation objects whose cleanest
  interpretation is in the `D -> infinity` setting.

So Basso is a good source for:

- what a transfer-style recurrence looks like,
- what a compact MaxCut state description can buy you,
- how to organise high-depth angle evaluation.

It is **not** by itself a proof that a compact recursion written for MaxCut is an
exact finite-D evaluator for our `(k, D) = (3, 4)` target.

### 3. The Villalonga repo: implementation reference, not a paper surrogate

The public repository `benjaminvillalonga/large-girth-maxcut-qaoa` was used as a
**code-level reference** for the compact MaxCut recursion.

Its value to this branch is practical rather than foundational:

- what arrays are propagated layer to layer,
- how the MaxCut-only transfer objects are laid out,
- how angle sweeps / optimisation are wired around the recurrence,
- what a working large-girth MaxCut implementation looks like in software.

That makes it useful when porting or sanity-checking a MaxCut transfer matrix.
It does **not** replace the mathematical distinction above: a code path that is
valid for the upstream MaxCut recurrence is not automatically valid for finite-D
k-XORSAT.

### 4. This branch's Julia port: experimental MaxCut scaffold

The Julia work on this branch should be understood as an **experimental port of
the MaxCut transfer viewpoint**, used alongside the exact local-tree evaluator.

The right interpretation is:

- use the exact evaluator as the correctness oracle,
- use the MaxCut transfer port as an implementation experiment,
- do not treat the MaxCut port as a derived proof for the k-XORSAT recursion.

This is exactly why the branch documentation now separates:

- the trusted exact light-cone reference path, and
- the still-open derivation of the effective branch-transfer object.

---

## What This Means for P1.3

For P1.3, the four sources play different roles:

1. **Farhi 2025** tells us what the final exact finite-D contraction should look
   like in the MaxCut case.
2. **Basso et al.** tells us what a compact high-depth recurrence can look like,
   and why such recurrences become especially natural at large `D`.
3. **The upstream MaxCut repo** provides concrete implementation patterns for the
   compact recurrence.
4. **This branch's Julia work** uses those patterns only as a MaxCut transfer
   experiment, while retaining the exact light-cone evaluator as the branch's
   correctness anchor.

The consequence is simple: if a future k-XORSAT contraction is claimed to be the
exact analogue of Farhi 2025, it must be validated against the exact oracle and
not justified only by resemblance to the MaxCut transfer code.

---

## Source Status in This Repo

- The Farhi 2025 PDF is already present.
- The Basso paper is already present as the arXiv preprint PDF used elsewhere in
  this repo. No separate publisher PDF is stored.
- The upstream `large-girth-maxcut-qaoa` code reference is external and is noted
  here as such; it is not a paper and is not stored under `.project/papers`.
- The branch-local documentation of the exact-reference path remains in
  `.project/specs/P1.3-contraction.md` and `.project/implementation-notes/P1.3.md`.

## Practical Takeaway

When continuing P1.3 work, treat the source stack as:

`Farhi 2025 exactness` -> `Basso compact recursion intuition` -> `upstream MaxCut code patterns` -> `branch-local Julia experiment validated against exact reference`.

That ordering is the safe one. Reversing it would risk importing MaxCut-specific
assumptions into the finite-D k-XORSAT derivation without proof.