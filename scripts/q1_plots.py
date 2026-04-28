#!/usr/bin/env python3
"""Q1 plotting: reads the four Q1 CSVs and writes PNGs into figures/.

  q1-angle-schedules.csv      → figures/q1-angle-schedules.png   (E1)
  q1-intermediate-depth.csv   → figures/q1-intermediate-depth.png (E2)
  q1-adiabatic-init.csv       → figures/q1-adiabatic-init.png    (E3)
  q1-angle-curvature.csv      → figures/q1-angle-curvature.png   (E4)

Usage: python3 scripts/q1_plots.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# ── CSV reader (skip lines starting with '#') ─────────────────────────

def load_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open() as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    header = rows[0]
    data   = [dict(zip(header, row)) for row in rows[1:]]
    return header, data


# ── E1: angle schedules ───────────────────────────────────────────────

def plot_angle_schedules() -> None:
    path = RESULTS_DIR / "q1-angle-schedules.csv"
    if not path.exists():
        print(f"skip E1 (no {path})")
        return
    _, rows = load_csv(path)
    Ds = sorted({int(r["D"]) for r in rows})

    fig, (axγ, axβ) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    cmap = plt.get_cmap("viridis")

    for i, D in enumerate(Ds):
        sub = [r for r in rows if int(r["D"]) == D]
        sub.sort(key=lambda r: int(r["j"]))
        p   = int(sub[0]["p"])
        x   = np.array([int(r["j"]) for r in sub]) / p
        γo  = np.array([float(r["gamma_opt"]) for r in sub])
        βo  = np.array([float(r["beta_opt"])  for r in sub])
        γa  = np.array([float(r["gamma_adi"]) for r in sub])
        βa  = np.array([float(r["beta_adi"])  for r in sub])
        c   = cmap(i / max(1, len(Ds) - 1))
        axγ.plot(x, γo, "o-",  color=c, label=f"D={D} opt", lw=2)
        axγ.plot(x, γa, "--",  color=c, alpha=0.6, label=f"D={D} adi")
        axβ.plot(x, βo, "o-",  color=c, label=f"D={D} opt", lw=2)
        axβ.plot(x, βa, "--",  color=c, alpha=0.6, label=f"D={D} adi")

    for ax, title, ylabel in (
        (axγ, "γ schedule: optimal vs linear adiabatic", "γ"),
        (axβ, "β schedule: optimal vs linear adiabatic", "β"),
    ):
        ax.set_xlabel("step j / p")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=2)

    out = FIGURES_DIR / "q1-angle-schedules.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


# ── E2: intermediate-depth performance ────────────────────────────────

def plot_intermediate_depth() -> None:
    path = RESULTS_DIR / "q1-intermediate-depth.csv"
    if not path.exists():
        print(f"skip E2 (no {path})")
        return
    _, rows = load_csv(path)
    Ds = sorted({int(r["D"]) for r in rows})

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("viridis")

    for i, D in enumerate(Ds):
        sub = sorted((r for r in rows if int(r["D"]) == D), key=lambda r: int(r["t"]))
        t   = np.array([int(r["t"])                       for r in sub])
        ct  = np.array([float(r["ctilde_truncated"])      for r in sub])
        co  = np.array([float(r["ctilde_optimal_at_t"])   for r in sub])
        ca  = np.array([float(r["ctilde_adiabatic_at_t"]) for r in sub])
        c   = cmap(i / max(1, len(Ds) - 1))
        ax.plot(t, co, "o-",  color=c, lw=2,  label=f"D={D} optimal@t")
        ax.plot(t, ct, "^--", color=c, lw=1,  alpha=0.8, label=f"D={D} truncated")
        ax.plot(t, ca, "d:",  color=c, lw=1,  alpha=0.7, label=f"D={D} adiabatic")

    ax.axhline(0.5, color="gray", ls=":", label="random (c̃=0.5)")
    ax.set_xlabel("depth t")
    ax.set_ylabel("c̃")
    ax.set_title("QAOA intermediate-depth performance")
    ax.set_ylim(0.45, 0.92)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=2)

    out = FIGURES_DIR / "q1-intermediate-depth.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


# ── E3: adiabatic-init optimization ───────────────────────────────────

def plot_adiabatic_init() -> None:
    path = RESULTS_DIR / "q1-adiabatic-init.csv"
    if not path.exists():
        print(f"skip E3 (no {path})")
        return
    _, rows = load_csv(path)
    if not rows:
        print(f"skip E3 (empty {path})")
        return
    Ds = sorted({int(r["D"]) for r in rows})

    best_adi  = []
    warm      = []
    seed_max  = []
    for D in Ds:
        sub = [r for r in rows if int(r["D"]) == D]
        best_adi.append(max(float(r["ctilde_adi_opt"]) for r in sub))
        warm.append(float(sub[0]["ctilde_warm"]))
        seed_max.append(max(float(r["ctilde_seed"]) for r in sub))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(Ds, warm,     "o-", lw=2, label="warm-start (c̃)")
    ax.plot(Ds, best_adi, "^-", lw=2, label="best adi-init (c̃, after L-BFGS)")
    ax.plot(Ds, seed_max, "d--", lw=1, alpha=0.7, label="best adi-seed (c̃, raw)")
    ax.axhline(0.5, color="gray", ls=":", label="random")

    for D, w, b in zip(Ds, warm, best_adi):
        ax.annotate(f"Δ={w-b:+.3f}", (D, b), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color="firebrick")

    ax.set_xticks(Ds)
    ax.set_xlabel("D")
    ax.set_ylabel("c̃")
    ax.set_title("Adiabatic-initialised QAOA: best-of-grid vs warm-start  (k=2, p=8)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    out = FIGURES_DIR / "q1-adiabatic-init.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


# ── E4: linear-fit r² and cubic improvement ───────────────────────────

def plot_angle_curvature() -> None:
    path = RESULTS_DIR / "q1-angle-curvature.csv"
    if not path.exists():
        print(f"skip E4 (no {path})")
        return
    _, rows = load_csv(path)

    deg1 = [r for r in rows if int(r["deg"]) == 1]
    deg3 = [r for r in rows if int(r["deg"]) == 3]
    Ds   = sorted({int(r["D"]) for r in deg1})

    γ_r2_lin, β_r2_lin = [], []
    γ_r2_cub, β_r2_cub = [], []
    for D in Ds:
        γ_r2_lin.append(float(next(r for r in deg1 if int(r["D"]) == D and r["profile"] == "gamma")["r2"]))
        β_r2_lin.append(float(next(r for r in deg1 if int(r["D"]) == D and r["profile"] == "beta") ["r2"]))
        γ_r2_cub.append(float(next(r for r in deg3 if int(r["D"]) == D and r["profile"] == "gamma")["r2"]))
        β_r2_cub.append(float(next(r for r in deg3 if int(r["D"]) == D and r["profile"] == "beta") ["r2"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(Ds))
    w = 0.35
    ax1.bar(x - w/2, γ_r2_lin, width=w, label="γ", color="C0")
    ax1.bar(x + w/2, β_r2_lin, width=w, label="β", color="C1")
    ax1.axhline(1.0, color="gray", ls=":", label="perfect linear")
    ax1.set_xticks(x); ax1.set_xticklabels(Ds)
    ax1.set_xlabel("D"); ax1.set_ylabel("r² (linear fit)")
    ax1.set_title("Linear adiabatic fit quality (deg=1)")
    ax1.set_ylim(0, 1.05); ax1.grid(alpha=0.3); ax1.legend()

    γ_dr2 = np.array(γ_r2_cub) - np.array(γ_r2_lin)
    β_dr2 = np.array(β_r2_cub) - np.array(β_r2_lin)
    ax2.bar(x - w/2, γ_dr2, width=w, label="γ", color="C0")
    ax2.bar(x + w/2, β_dr2, width=w, label="β", color="C1")
    ax2.set_xticks(x); ax2.set_xticklabels(Ds)
    ax2.set_xlabel("D"); ax2.set_ylabel("Δr² when going linear → cubic")
    ax2.set_title("Variance recovered by adding curvature")
    ax2.grid(alpha=0.3); ax2.legend()

    out = FIGURES_DIR / "q1-angle-curvature.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    plot_angle_schedules()
    plot_intermediate_depth()
    plot_adiabatic_init()
    plot_angle_curvature()


if __name__ == "__main__":
    main()
