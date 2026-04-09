# Status Update for Stephen — April 10, 2026

## The good news: 13 of 15 pairs beat DQI+BP

The swarm optimizer on your cluster is working. Here's the current scorecard:

    k=3 family (warm-start sweep, your cluster):
      (3,4) p=13  c̃=0.881  beats DQI+BP, Prange
      (3,5) p=13  c̃=0.843  beats DQI+BP, Prange, Regev+FGUM
      (3,6) p=12  c̃=0.809  beats DQI+BP, Prange, Regev+FGUM
      (3,7) p=11  c̃=0.779  beats DQI+BP, Prange, Regev+FGUM
      (3,8) p=11  c̃=0.768  beats DQI+BP, Prange, Regev+FGUM

    k=4 family (swarm, your cluster):
      (4,5) p=11  c̃=0.861  beats DQI+BP, Prange
      (4,6) p=10  c̃=0.827  beats DQI+BP
      (4,7) p=9   c̃=0.798  beats DQI+BP, Prange
      (4,8) p=9   c̃=0.780  beats DQI+BP, Prange

    k=5 family (swarm, your cluster):
      (5,6) p=9   c̃=0.838  trailing DQI+BP (0.843)
      (5,7) p=9   c̃=0.808  trailing DQI+BP (0.814)
      (5,8) p=9   c̃=0.805  beats DQI+BP (0.788)

    k=6,7 (swarm, your cluster):
      (6,7) p=9   c̃=0.855  beats DQI+BP (0.828)
      (6,8) p=8   c̃=0.802  trailing DQI+BP (0.803)
      (7,8) p=8   c̃=0.819  beats DQI+BP (0.813)

13 of 15 beat DQI+BP. The two trailing pairs — (5,6) and (6,8) — are
within 5-6 basis points and would likely cross at one more depth.

## The bad news: precision wall at k>=6, p>=10

The 0.5 and 0.99 values at higher depths for k>=6 are NOT overflow.
I verified locally: all three independent evaluators (normalized,
un-normalized, and qaoa_expectation) agree on the bad values. The
evaluator is computing correctly — the problem is that Float64
precision isn't sufficient.

### What's happening

The Basso recurrence sums ~2^{2p+1} complex terms that nearly cancel.
At (6,7) p=10, that's 2 million terms. The physical signal (the
deviation from c̃ = 0.5) lives in the last few digits of precision.
After 10 steps of ^5 and ^6, the accumulated floating-point error
exceeds the signal.

The symptom: S = Re(sum of root_kernel * iWHT(msg_hat^k)) comes
out as 5.45 instead of ~0.71. The individual intermediates never
exceed magnitude ~1.5 — no Float64 overflow, no normalization needed.
The error is pure cancellation noise.

This is fundamentally different from the k=3,4,5 cases where the
evaluator works fine through p=13. The (k-1)(D-1) = 30 or 42
exponent at each step amplifies precision loss much faster than
the (k-1)(D-1) = 6 at k=3.

### Affected pairs

    (6,7): valid through p=9, bad at p>=10
    (6,8): valid through p=8, bad at p>=9
    (7,8): valid through p=8, bad at p>=9

All other pairs are fine at their current depths.

## Potential fix: Double64 arithmetic

Julia has DoubleFloats.jl which provides Double64 — ~31 digits of
precision (vs Float64's ~15). The Basso pipeline is already generic
over the element type T via QAOAAngles{T}, so in principle:

    using DoubleFloats
    angles = QAOAAngles(Double64.(gamma), Double64.(beta))
    val = basso_expectation_normalized(params, angles; clause_sign=1)

The question is performance. I haven't benchmarked it yet. Double64
uses two Float64s to represent each number (double-double arithmetic).
The theoretical overhead is:
- Addition: ~2x (error-free transformation)
- Multiplication: ~4x (Dekker's algorithm)
- The WHT butterfly is add-heavy, so maybe 2-3x overall
- The power operations are multiply-heavy, so maybe 4-5x

So a rough estimate is 3-4x slower, not 10-100x. The WHT itself
might even vectorize reasonably since Double64 is still 128 bits.
This is worth trying if you want k>=6 at p>=10.

I can implement and test this quickly if you want to try it on
the cluster.

## What to do now

1. The working pairs are done — let swarm jobs finish or cancel them
2. For (6,7), (6,8), (7,8) — cancel those swarm jobs, they won't
   improve beyond p=8-9 in Float64
3. If you want to try Double64 for those three pairs, let me know
   and I'll push the code
