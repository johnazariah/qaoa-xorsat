#!/usr/bin/env python3
"""Q1.5 CD-QAOA plots: reverse-induced (λ̄, s̄) protocols + diagnostics.

Reads results/q15-cd-qaoa-reverse.csv and produces:
- figures/q15-induced-protocols.png — λ̄_q and s̄_q vs step q at p=12, all D
- figures/q15-diagnostics.png         — descents-in-λ̄, max|s̄| vs (D, p)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "q15-cd-qaoa-reverse.csv"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)


def load():
    rows = []
    with open(CSV, newline="") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("D,"):
                continue
            f = line.rstrip("\n").split(",")
            if len(f) < 10:
                continue
            rows.append(dict(
                D=int(f[0]), p=int(f[1]), c_warm=float(f[2]),
                n_descent=int(f[3]), n_signflip=int(f[4]),
                s_ampl=float(f[5]), s_mean=float(f[6]),
                tau=np.array([float(x) for x in f[7].split(";")]),
                lam=np.array([float(x) for x in f[8].split(";")]),
                sbar=np.array([float(x) for x in f[9].split(";")]),
            ))
    return rows


def find(rows, D, p):
    for r in rows:
        if r["D"] == D and r["p"] == p:
            return r
    return None


def plot_induced_protocols(rows, p_target=12):
    Ds = [3, 4, 5, 6, 7, 8]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    axes = axes.flatten()
    for ax, D in zip(axes, Ds):
        r = find(rows, D, p_target)
        if r is None:
            ax.set_title(f"D={D} (no data)")
            continue
        steps = np.arange(1, r["p"] + 1)
        ax2 = ax.twinx()
        l1, = ax.plot(steps, r["lam"], "o-", color="C0", lw=1.6, ms=5,
                      label=r"$\bar\lambda_q$")
        l2, = ax2.plot(steps, r["sbar"], "s-", color="C3", lw=1.4, ms=4, alpha=0.85,
                       label=r"$\bar s_q$")
        ax.axhline(0, color="grey", lw=0.5, ls=":")
        ax.axhline(1, color="grey", lw=0.5, ls=":")
        ax2.axhline(0, color="C3", lw=0.5, ls=":", alpha=0.5)
        ax.plot(steps, np.linspace(0, 1, r["p"]), "--", color="C0", alpha=0.3,
                lw=1.0, label="linear ref")
        ax.set_title(f"D={D}, p={r['p']}, c̃={r['c_warm']:.4f}, "
                     f"↓λ̄={r['n_descent']}/{r['p']-1}")
        ax.set_ylabel(r"$\bar\lambda_q$", color="C0")
        ax2.set_ylabel(r"$\bar s_q$", color="C3")
        ax.tick_params(axis="y", labelcolor="C0")
        ax2.tick_params(axis="y", labelcolor="C3")
        ax.set_ylim(-0.1, 1.1)
        if D == 3:
            ax.legend([l1, l2],
                      [r"$\bar\lambda_q$ (induced rate)",
                       r"$\bar s_q$ (induced AGP field)"],
                      loc="lower right", fontsize=8, framealpha=0.85)
    for ax in axes[3:]:
        ax.set_xlabel("step q")
    fig.suptitle("Wurtz–Love induced counterdiabatic protocol from warm-start QAOA "
                 "(p=12, k=2 MaxCut, infinite tree)\n"
                 r"If QAOA were Trotterised CD: $\bar\lambda_q$ monotone $0\to 1$, "
                 r"$\bar s_q\approx 0$. Observed: neither.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIGS / "q15-induced-protocols.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_diagnostics(rows):
    Ds = sorted({r["D"] for r in rows})

    fig, (ax_d, ax_s) = plt.subplots(1, 2, figsize=(12, 4.5))
    cmap = plt.colormaps["viridis"]
    for i, D in enumerate(Ds):
        c = cmap(i / max(1, len(Ds) - 1))
        ps = sorted({r["p"] for r in rows if r["D"] == D})
        descents = [find(rows, D, p)["n_descent"] / max(1, p - 1) for p in ps]
        ampls = [find(rows, D, p)["s_ampl"] for p in ps]
        ax_d.plot(ps, descents, "o-", color=c, lw=1.5, ms=5, label=f"D={D}")
        ax_s.plot(ps, ampls, "s-", color=c, lw=1.5, ms=5, label=f"D={D}")
    ax_d.set_xlabel("p")
    ax_d.set_ylabel(r"fraction of descents in $\bar\lambda_q$")
    ax_d.set_title(r"Non-monotonicity of induced $\bar\lambda$")
    ax_d.set_ylim(-0.02, 1.02)
    ax_d.axhline(0, color="grey", lw=0.5, ls=":")
    ax_d.legend(fontsize=9, ncol=2)
    ax_d.grid(alpha=0.3)

    ax_s.set_xlabel("p")
    ax_s.set_ylabel(r"$\max_q |\bar s_q|$")
    ax_s.set_title(r"Magnitude of required auxiliary field $\bar s$")
    ax_s.axhline(0, color="grey", lw=0.5, ls=":")
    ax_s.legend(fontsize=9, ncol=2)
    ax_s.grid(alpha=0.3)

    fig.suptitle("Q1.5: Wurtz–Love reverse map applied to warm-start optima\n"
                 r"(W–L assume $\bar\lambda$ monotone and $\bar s$ small; "
                 r"observed: both fail across (D, p))",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIGS / "q15-diagnostics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    rows = load()
    print(f"loaded {len(rows)} rows from {CSV}")
    plot_induced_protocols(rows)
    plot_diagnostics(rows)


if __name__ == "__main__":
    main()
