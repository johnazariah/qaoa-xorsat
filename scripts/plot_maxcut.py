#!/usr/bin/env python3
"""
Plot MaxCut QAOA results: c̃ vs p, O(1/D) corrections, angle profiles.
Reads from results/maxcut-k2-d*-sweep.csv, auto-updates as data fills in.

Usage: python3 scripts/plot_maxcut.py
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
results_dir = os.path.join(script_dir, '..', 'results')
figures_dir = os.path.join(script_dir, '..', 'results', 'maxcut-plots')
os.makedirs(figures_dir, exist_ok=True)

# Also copy to research paper figures
research_figures = os.path.join(script_dir, '..', '..', 'qaoa-xorsat-research', 'figures')

# ── Read data ─────────────────────────────────────────────────────
# data[D] = [(p, ctilde), ...]
data = {}
for D in [3, 4, 5, 6, 7, 8]:
    fname = os.path.join(results_dir, f'maxcut-k2-d{D}-sweep.csv')
    if not os.path.exists(fname):
        continue
    rows = []
    best = {}  # best c̃ per p
    with open(fname) as f:
        for line in f:
            if line.startswith('#') or line.startswith('k,'):
                continue
            fields = line.strip().split(',')
            if len(fields) < 5:
                continue
            p = int(fields[2])
            v = float(fields[3])
            if p not in best or v > best[p]:
                best[p] = v
    rows = sorted(best.items())
    if rows:
        data[D] = rows

print(f"Loaded data for D = {sorted(data.keys())}")
for D in sorted(data.keys()):
    pmax = data[D][-1][0]
    vmax = data[D][-1][1]
    print(f"  D={D}: p=1..{pmax}, best c̃={vmax:.6f}")

# ── Basso asymptotic ν values for k=2 ────────────────────────────
basso_nu = {
    1: 0.3033, 2: 0.4075, 3: 0.4726, 4: 0.5157, 5: 0.5476,
    6: 0.5721, 7: 0.5915, 8: 0.6073, 9: 0.6203, 10: 0.6314,
    11: 0.6408, 12: 0.649, 13: 0.6561, 14: 0.6623,
    15: 0.6679, 16: 0.6729, 17: 0.6773,
}

def basso_ctilde(p, D):
    """Large-D asymptotic prediction: c̃ = 0.5 + ν_p * sqrt(k/(2*(D-1))) with k=2"""
    if p not in basso_nu:
        return None
    return 0.5 + basso_nu[p] / math.sqrt(D - 1)

# ── Figure 1: c̃ vs p, one curve per D ────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))

colors = {3: '#e41a1c', 4: '#377eb8', 5: '#4daf4a',
          6: '#984ea3', 7: '#ff7f00', 8: '#a65628'}

for D in sorted(data.keys()):
    ps = [r[0] for r in data[D]]
    vs = [r[1] for r in data[D]]
    ax.plot(ps, vs, 'o-', color=colors[D], markersize=4, linewidth=1.5,
            label=f'D={D} (exact)')

# Add Basso asymptotic for D=3 as dashed line for comparison
ps_basso = sorted(basso_nu.keys())
vs_basso_d3 = [basso_ctilde(p, 3) for p in ps_basso]
ax.plot(ps_basso, vs_basso_d3, '--', color=colors[3], alpha=0.4,
        linewidth=1, label=f'D=3 (Basso large-D)')

ax.set_xlabel('Circuit depth p', fontsize=12)
ax.set_ylabel('Satisfaction fraction $\\tilde{c}$', fontsize=12)
ax.set_title('Exact QAOA MaxCut on D-regular graphs', fontsize=13)
ax.legend(fontsize=9, loc='lower right')
ax.set_xlim(0.5, max(max(r[0] for r in data[D]) for D in data) + 0.5)
ax.set_ylim(0.59, 0.92)
ax.grid(True, alpha=0.3)

out1 = os.path.join(figures_dir, 'maxcut_ctilde_vs_p.png')
plt.tight_layout()
plt.savefig(out1, dpi=150, bbox_inches='tight')
print(f'Saved: {out1}')
plt.close()

# ── Figure 2: O(1/D) correction vs D ─────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

# Pick representative depths
p_values = [3, 5, 7, 9]
p_colors = cm.viridis(np.linspace(0.2, 0.9, len(p_values)))

for i, p in enumerate(p_values):
    Ds = []
    deltas = []
    for D in sorted(data.keys()):
        exact = None
        for pp, vv in data[D]:
            if pp == p:
                exact = vv
                break
        if exact is None:
            continue
        basso = basso_ctilde(p, D)
        if basso is None:
            continue
        Ds.append(D)
        deltas.append(exact - basso)

    if Ds:
        ax.plot(Ds, deltas, 'o-', color=p_colors[i], markersize=5,
                linewidth=1.5, label=f'p={p}')

ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.8)
ax.set_xlabel('Degree D', fontsize=12)
ax.set_ylabel('$\\tilde{c}_{\\mathrm{exact}} - \\tilde{c}_{\\mathrm{Basso}}$',
              fontsize=12)
ax.set_title('$O(1/D)$ correction: exact vs large-D asymptotic', fontsize=13)
ax.legend(fontsize=10)
ax.set_xticks([3, 4, 5, 6, 7, 8])
ax.grid(True, alpha=0.3)

out2 = os.path.join(figures_dir, 'maxcut_correction_vs_D.png')
plt.tight_layout()
plt.savefig(out2, dpi=150, bbox_inches='tight')
print(f'Saved: {out2}')
plt.close()

# ── Figure 3: Angle profiles vs round (γ and β) ──────────────────
# Read angles from CSVs
for angle_type, angle_idx_start in [('gamma', 0), ('beta', 1)]:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
    fig.suptitle(f'QAOA MaxCut ${angle_type}$_j vs round j (each curve = one depth p)',
                 fontsize=13, y=1.02)

    for idx, D in enumerate(sorted(data.keys())):
        ax = axes[idx // 3][idx % 3]
        fname = os.path.join(results_dir, f'maxcut-k2-d{D}-sweep.csv')
        if not os.path.exists(fname):
            continue

        # Read all angles per depth
        depth_angles = {}
        with open(fname) as f:
            for line in f:
                if line.startswith('#') or line.startswith('k,'):
                    continue
                fields = line.strip().split(',')
                if len(fields) < 7:
                    continue
                p = int(fields[2])
                gammas = [float(x) for x in fields[5].split(';')]
                betas = [float(x) for x in fields[6].split(';')]
                if angle_type == 'gamma':
                    angles = gammas
                else:
                    angles = betas
                # Normalize to [-π/2, π/2]
                angles = [a % math.pi for a in angles]
                angles = [a - math.pi if a > math.pi/2 else a for a in angles]
                depth_angles[p] = angles

        if not depth_angles:
            continue

        p_max = max(depth_angles.keys())
        cmap = cm.viridis

        for p in sorted(depth_angles.keys()):
            angles = depth_angles[p]
            js = list(range(1, len(angles) + 1))
            color = cmap(p / max(p_max, 1))
            alpha = 0.4 + 0.6 * (p / max(p_max, 1))
            ax.plot(js, angles, '-o', color=color, markersize=2,
                    linewidth=0.8 + 1.2*(p/max(p_max,1)), alpha=alpha)

        ax.set_title(f'D={D}', fontsize=10)
        ax.set_xlabel('round j', fontsize=8)
        sym = 'γ' if angle_type == 'gamma' else 'β'
        ax.set_ylabel(f'{sym}_j (mod π)', fontsize=8)
        ax.set_ylim(-math.pi/2 - 0.05, math.pi/2 + 0.05)
        ax.set_yticks([-math.pi/2, -math.pi/4, 0, math.pi/4, math.pi/2])
        ax.set_yticklabels(['-π/2', '-π/4', '0', 'π/4', 'π/2'], fontsize=7)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    out3 = os.path.join(figures_dir, f'maxcut_{angle_type}_vs_round.png')
    plt.savefig(out3, dpi=150, bbox_inches='tight')
    print(f'Saved: {out3}')
    plt.close()

# ── Copy to research figures if available ─────────────────────────
if os.path.isdir(research_figures):
    import shutil
    for f in os.listdir(figures_dir):
        if f.endswith('.png'):
            shutil.copy2(os.path.join(figures_dir, f),
                        os.path.join(research_figures, f))
    print(f'Copied all plots to {research_figures}')

print(f'\nAll plots saved to: {figures_dir}')
