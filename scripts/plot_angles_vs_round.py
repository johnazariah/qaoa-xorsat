#!/usr/bin/env python3
"""
Plot QAOA angles vs round j for all 15 (k,D) pairs.
Each curve is a different depth p; x-axis is round j (1..p).
Angles normalized to [0, π] via mod π.

Usage: python3 scripts/plot_angles_vs_round.py
"""
import csv
import os
import math
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'results', 'angle-plots')
out_dir = data_dir

# Safe p_max per (k,D) — beyond these, Float64 corrupts the objective.
SAFE_P_MAX = {
    (3, 4): 13,  (3, 5): 13,  (3, 6): 11,  (3, 7): 11,  (3, 8): 11,
    (4, 5): 11,  (4, 6): 10,  (4, 7): 10,  (4, 8): 9,
    (5, 6): 9,   (5, 7): 9,   (5, 8): 9,
    (6, 7): 8,   (6, 8): 8,
    (7, 8): 7,
}

def normalize_to_pm_half_pi(v):
    """Map angle to [-π/2, +π/2] via mod π, then shift."""
    v = v % math.pi          # -> [0, π)
    if v > math.pi / 2:
        v -= math.pi         # -> (-π/2, 0] for the upper half
    return v

# Read data: (k,D) -> p -> [(round, value)]
gamma_data = defaultdict(lambda: defaultdict(list))  # (k,D) -> p -> [(j, val)]
beta_data = defaultdict(lambda: defaultdict(list))

for fname, store, col in [('gamma_angles.csv', gamma_data, 'gamma'),
                           ('beta_angles.csv', beta_data, 'beta')]:
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        print(f"Missing {path} — run: julia --project=. scripts/plot_angles.jl")
        exit(1)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            k, D = int(row['k']), int(row['D'])
            p = int(row['p'])
            j = int(row['round'])
            v = float(row[col])
            p_max = SAFE_P_MAX.get((k, D), 8)
            if p > p_max:
                continue
            v = normalize_to_pm_half_pi(v)
            store[(k, D)][p].append((j, v))

pairs = sorted(set(gamma_data.keys()))
print(f"Loaded {len(pairs)} pairs")

# Sort each p's rounds
for store in [gamma_data, beta_data]:
    for kd in store:
        for p in store[kd]:
            store[kd][p].sort()

# ── Plot function ─────────────────────────────────────────────────
def make_plot(store, angle_name, ylabel, filename):
    n_pairs = len(pairs)
    cols = 5
    rows = (n_pairs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows), squeeze=False)
    fig.suptitle(f'QAOA {ylabel} vs round (each curve = one depth p)',
                 fontsize=14, y=1.02)

    for idx, (k, D) in enumerate(pairs):
        ax = axes[idx // cols][idx % cols]
        p_values = sorted(store[(k, D)].keys())
        if not p_values:
            continue
        p_max = max(p_values)

        cmap = cm.viridis
        for p in p_values:
            rounds_vals = store[(k, D)][p]
            js = [jv[0] for jv in rounds_vals]
            vs = [jv[1] for jv in rounds_vals]
            color = cmap(p / p_max)
            alpha = 0.4 + 0.6 * (p / p_max)  # later p more opaque
            lw = 0.8 + 1.2 * (p / p_max)
            ax.plot(js, vs, '-o', color=color, markersize=2,
                    linewidth=lw, alpha=alpha)

        ax.set_title(f'({k},{D})', fontsize=10)
        ax.set_xlabel('round j', fontsize=8)
        ax.set_ylabel(f'{ylabel} (mod π)', fontsize=8)
        ax.set_ylim(-math.pi/2 - 0.05, math.pi/2 + 0.05)
        ax.set_yticks([-math.pi/2, -math.pi/4, 0, math.pi/4, math.pi/2])
        ax.set_yticklabels(['-π/2', '-π/4', '0', 'π/4', 'π/2'], fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for idx in range(n_pairs, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()


make_plot(gamma_data, 'gamma', 'γ_j', 'gamma_vs_round.png')
make_plot(beta_data, 'beta', 'β_j', 'beta_vs_round.png')

print(f'\nAll plots saved to: {out_dir}')
