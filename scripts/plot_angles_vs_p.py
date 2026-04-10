#!/usr/bin/env python3
"""
Plot QAOA angles vs depth p for all 15 (k,D) pairs.
For each pair, we plot how each angle round evolves as p increases.
Two plots: one for γ, one for β. Each (k,D) pair is a separate subplot.

Usage: python3 scripts/plot_angles_vs_p.py
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'results', 'angle-plots')
out_dir = data_dir

# Safe p_max per (k,D) — beyond these, Float64 catastrophic cancellation
# corrupts the objective function and the optimizer finds garbage angles.
# Determined by comparing Float64 vs Double64 evaluations and monitoring
# for sudden c̃ jumps, values > 1, or collapse to 0.5.
SAFE_P_MAX = {
    (3, 4): 13,  (3, 5): 13,  (3, 6): 11,  (3, 7): 11,  (3, 8): 11,
    (4, 5): 11,  (4, 6): 10,  (4, 7): 10,  (4, 8): 9,
    (5, 6): 9,   (5, 7): 9,   (5, 8): 9,
    (6, 7): 8,   (6, 8): 8,
    (7, 8): 7,
}

# Canonicalize γ angles to remove symmetry-equivalent jumps.
# Exact symmetries:
#   All k:   c̃(γ, β) = c̃(γ + 2π, β)    and  c̃(γ, β) = c̃(-γ, -β)
#   Odd k:   c̃(γ, β) = c̃(γ + π, β)     (γ is π-periodic)
#   Even k:  c̃(γ, β) = c̃(2π-γ, π-β)    (reflection)
#   All k:   c̃(γ, β) = c̃(γ, β + π)     (β is π-periodic)
def canonicalize_gamma(v, k):
    """Map γ to canonical range: [0, π) for odd k, [0, π] for even k."""
    import math
    if k % 2 == 1:  # odd k: π-periodic
        return v % math.pi
    else:  # even k: 2π-periodic, reflect to [0, π]
        v = v % (2 * math.pi)
        return v if v <= math.pi else 2 * math.pi - v

def canonicalize_beta(v, k):
    """Map β to canonical range [0, π/2]."""
    import math
    v = v % math.pi
    return v if v <= math.pi / 2 else math.pi - v

# Read gamma_angles.csv: k,D,p,round,gamma
gamma_data = defaultdict(lambda: defaultdict(dict))  # (k,D) -> round -> {p: value}
beta_data = defaultdict(lambda: defaultdict(dict))

for fname, store in [('gamma_angles.csv', gamma_data), ('beta_angles.csv', beta_data)]:
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        print(f"Missing {path} — run: julia --project=. scripts/plot_angles.jl")
        exit(1)
    with open(path) as f:
        reader = csv.DictReader(f)
        angle_col = 'gamma' if 'gamma' in fname else 'beta'
        for row in reader:
            k, D = int(row['k']), int(row['D'])
            p = int(row['p'])
            r = int(row['round'])
            v = float(row[angle_col])
            # Skip data beyond the F64 precision wall
            p_max = SAFE_P_MAX.get((k, D), 8)
            if p > p_max:
                continue
            # Canonicalize to remove symmetry-equivalent jumps
            if angle_col == 'gamma':
                v = canonicalize_gamma(v, k)
            else:
                v = canonicalize_beta(v, k)
            store[(k, D)][r][p] = v

pairs = sorted(set(gamma_data.keys()))
print(f"Loaded {len(pairs)} pairs")

# ── Figure 1: γ angles vs p ──────────────────────────────────────────
n_pairs = len(pairs)
cols = 5
rows = (n_pairs + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), squeeze=False)
fig.suptitle('QAOA γ angles vs depth p (each line = one round r)', fontsize=14, y=1.02)

cmap = cm.viridis

for idx, (k, D) in enumerate(pairs):
    ax = axes[idx // cols][idx % cols]
    rounds = sorted(gamma_data[(k, D)].keys())
    max_round = max(rounds) if rounds else 1

    for r in rounds:
        ps = sorted(gamma_data[(k, D)][r].keys())
        vals = [gamma_data[(k, D)][r][p] for p in ps]
        color = cmap(r / max_round)
        ax.plot(ps, vals, '-o', color=color, markersize=2, linewidth=1, alpha=0.8)

    ax.set_title(f'({k},{D})', fontsize=10)
    ax.set_xlabel('p', fontsize=8)
    ax.set_ylabel('γ_r', fontsize=8)
    ax.tick_params(labelsize=7)

# Hide unused subplots
for idx in range(n_pairs, rows * cols):
    axes[idx // cols][idx % cols].set_visible(False)

plt.tight_layout()
gamma_path = os.path.join(out_dir, 'gamma_vs_p.png')
plt.savefig(gamma_path, dpi=150, bbox_inches='tight')
print(f'Saved: {gamma_path}')
plt.close()

# ── Figure 2: β angles vs p ──────────────────────────────────────────
fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), squeeze=False)
fig.suptitle('QAOA β angles vs depth p (each line = one round r)', fontsize=14, y=1.02)

for idx, (k, D) in enumerate(pairs):
    ax = axes[idx // cols][idx % cols]
    rounds = sorted(beta_data[(k, D)].keys())
    max_round = max(rounds) if rounds else 1

    for r in rounds:
        ps = sorted(beta_data[(k, D)][r].keys())
        vals = [beta_data[(k, D)][r][p] for p in ps]
        color = cmap(r / max_round)
        ax.plot(ps, vals, '-o', color=color, markersize=2, linewidth=1, alpha=0.8)

    ax.set_title(f'({k},{D})', fontsize=10)
    ax.set_xlabel('p', fontsize=8)
    ax.set_ylabel('β_r', fontsize=8)
    ax.tick_params(labelsize=7)

for idx in range(n_pairs, rows * cols):
    axes[idx // cols][idx % cols].set_visible(False)

plt.tight_layout()
beta_path = os.path.join(out_dir, 'beta_vs_p.png')
plt.savefig(beta_path, dpi=150, bbox_inches='tight')
print(f'Saved: {beta_path}')
plt.close()

# ── Figure 3: Combined — one big plot per angle type ─────────────────
# All 15 pairs overlaid, γ_r vs p, colored by (k,D)
pair_colors = cm.tab20(np.linspace(0, 1, len(pairs)))

for angle_name, store, ylabel in [('gamma', gamma_data, 'γ_r'), ('beta', beta_data, 'β_r')]:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title(f'QAOA {ylabel} angles vs depth p (all pairs, all rounds)', fontsize=14)
    ax.set_xlabel('Depth p', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=14)

    for i, (k, D) in enumerate(pairs):
        rounds = sorted(store[(k, D)].keys())
        for r in rounds:
            ps = sorted(store[(k, D)][r].keys())
            vals = [store[(k, D)][r][p] for p in ps]
            label = f'({k},{D})' if r == 1 else None
            ax.plot(ps, vals, '-', color=pair_colors[i], linewidth=0.8, alpha=0.5, label=label)

    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, f'{angle_name}_all_vs_p.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()

print(f'\nAll plots saved to: {out_dir}')
