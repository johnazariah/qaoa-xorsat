#!/usr/bin/env python3
"""
Plot QAOA angles vs round for all 15 (k,D) pairs.
Produces two PNG plots: gamma_angles.png and beta_angles.png

Usage: python3 scripts/plot_angles.py
"""
import csv
import os
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available. Install: pip3 install matplotlib")
    exit(1)

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'results', 'angle-plots')
out_dir = data_dir

# Read best_angles_summary.csv
summary_path = os.path.join(data_dir, 'best_angles_summary.csv')
if not os.path.exists(summary_path):
    print(f"Run the Julia script first: julia --project=. scripts/plot_angles.jl")
    exit(1)

pairs = []
with open(summary_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        k, D = int(row['k']), int(row['D'])
        p = int(row['best_p'])
        gamma = [float(x) for x in row['gamma'].split(';')]
        beta = [float(x) for x in row['beta'].split(';')]
        pairs.append((k, D, p, gamma, beta))

pairs.sort(key=lambda x: (x[0], x[1]))

# Color map
colors = cm.tab20(range(len(pairs)))

# === Gamma plot ===
fig, ax = plt.subplots(figsize=(12, 7))
for i, (k, D, p, gamma, beta) in enumerate(pairs):
    rounds = list(range(1, p + 1))
    ax.plot(rounds, gamma, '-o', color=colors[i], markersize=3,
            linewidth=1.5, label=f'({k},{D}) p={p}')
ax.set_xlabel('Round r', fontsize=12)
ax.set_ylabel('γ_r', fontsize=14)
ax.set_title('QAOA γ angles at best depth', fontsize=14)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.tight_layout()
gamma_path = os.path.join(out_dir, 'gamma_angles.png')
plt.savefig(gamma_path, dpi=150, bbox_inches='tight')
print(f'Saved: {gamma_path}')
plt.close()

# === Beta plot ===
fig, ax = plt.subplots(figsize=(12, 7))
for i, (k, D, p, gamma, beta) in enumerate(pairs):
    rounds = list(range(1, p + 1))
    ax.plot(rounds, beta, '-o', color=colors[i], markersize=3,
            linewidth=1.5, label=f'({k},{D}) p={p}')
ax.set_xlabel('Round r', fontsize=12)
ax.set_ylabel('β_r', fontsize=14)
ax.set_title('QAOA β angles at best depth', fontsize=14)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.tight_layout()
beta_path = os.path.join(out_dir, 'beta_angles.png')
plt.savefig(beta_path, dpi=150, bbox_inches='tight')
print(f'Saved: {beta_path}')
plt.close()

print(f'\nPlots saved to: {out_dir}')
