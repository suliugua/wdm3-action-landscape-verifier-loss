#!/usr/bin/env python3
"""Figure: the probe's empirical resolution boundary.

Three perturbations of the SAME 20-evaluation probe, each from a recorded run
(no new FDTD).  Together they carry the third contribution, which until now was
table-only:

  (a) probe LENGTH   k = 5/10/15/20   -> results/probe_cost_curve.json
  (b) probe BUDGET   n = 20/40/60/80  -> results/budget_resolution/*/budget_resolution_check.json
  (c) random SEED    reseeded repeat  -> results/repeat_probe/*/repeat_check.json

Style follows the SPIE "Figure Requirements" verified text: text >= 8 pt, line
weights >= 0.5 pt, no grid background, white background, panel letters "(a)",
parts combined in one file.

Run:  python make_fig_resolution.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("figures")
INK = "#252525"
MUTED = "#6f6f6f"
PAPER = "#ffffff"
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
RED = "#B2182B"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

KG = [5, 10, 15, 20]
NG = ["20", "40", "60", "80"]


def panel_label(ax, label):
    ax.text(-0.30, 1.13, f"({label})", transform=ax.transAxes, fontsize=10,
            fontweight="bold", color=INK, va="top", ha="left")


def load_cost():
    if not os.path.exists("results/probe_cost_curve.json"):
        return {}
    d = json.load(open("results/probe_cost_curve.json", encoding="utf-8"))
    out = {}
    for a, v in d["anchors"].items():
        out[a] = [v["loss_k%d" % k] for k in KG]
    return out


def load_budget():
    out = {}
    for a in ["seed14", "seed18", "seed11", "seed23", "seed16"]:
        p = "results/budget_resolution/%s/budget_resolution_check.json" % a
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        out[a] = [d["gate_curve"][n]["g"] for n in NG]
    return out


def load_repeat():
    out = {}
    for a in ["seed11", "seed16", "seed14", "seed20"]:
        p = "results/repeat_probe/%s/repeat_check.json" % a
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        out[a] = (d["g_pub"], d["g_rep"])
    return out


def main():
    cost, budget, rep = load_cost(), load_budget(), load_repeat()
    fig = plt.figure(figsize=(6.7, 3.30))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.85], wspace=0.42)
    ax0, ax1, ax2 = (fig.add_subplot(gs[0, i]) for i in range(3))

    def band(ax):
        ax.axhspan(0.35, 0.65, color="#f0f0f0", zorder=0)
        ax.axhline(0.5, color=INK, ls="--", lw=0.7, zorder=1)

    # ---- (a) probe length ------------------------------------------------
    panel_label(ax0, "a")
    for a, ys in cost.items():
        if a in ("seed16", "seed23"):
            continue
        ax0.plot(KG, ys, color="#c9c9c9", lw=0.55, marker="o", ms=2.6, zorder=2)
    movers = {"seed16": (BLUE, "^"), "seed23": (ORANGE, "s")}
    for a, (col, mk) in movers.items():
        if a in cost:
            ax0.plot(KG, cost[a], color=col, lw=1.1, marker=mk, ms=5.0,
                     markeredgecolor=INK, markeredgewidth=0.55, label=a, zorder=4)
    band(ax0)
    ax0.set_xticks(KG)
    ax0.set_xlabel("probe length $k$ (evaluations)")
    ax0.set_ylabel("measured gate loss")
    ax0.set_ylim(-0.05, 1.05)
    ax0.set_title("Shorter probe", loc="left")
    for a, (col, mk) in movers.items():
        if a in cost:
            ax0.annotate(a.replace("seed", "s"), (KG[-1], cost[a][-1]),
                         textcoords="offset points", xytext=(-2, 8),
                         fontsize=8, color=col, ha="right", fontweight="bold")
    ax0.text(0.02, 0.02, "26/28 unchanged", transform=ax0.transAxes,
             fontsize=8, color=MUTED, ha="left", va="bottom")

    # ---- (b) probe budget ------------------------------------------------
    panel_label(ax1, "b")
    style = {"seed14": ("#bdbdbd", "o", "clean ctrl"),
             "seed18": ("#7f7f7f", "o", "lossy ctrl"),
             "seed11": (GREEN, "^", "s11 (20)"),
             "seed23": (ORANGE, "s", "s23 (60)"),
             "seed16": (BLUE, "D", "s16 (never)")}
    for a, (col, mk, lab) in style.items():
        if a not in budget:
            continue
        ax1.plot(NG, budget[a], color=col, lw=1.1, marker=mk, ms=4.6,
                 markeredgecolor=INK, markeredgewidth=0.55, label=lab, zorder=4)
    band(ax1)
    ax1.set_xlabel("probe budget $n$ (evaluations)")
    ax1.set_ylabel("measured gate loss")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("More budget?", loc="left")
    ax1.legend(loc="lower left", handlelength=1.1, borderaxespad=0.15,
               fontsize=8, frameon=True, facecolor=PAPER, edgecolor="none",
               framealpha=0.92)

    # ---- (c) random seed -------------------------------------------------
    panel_label(ax2, "c")
    xs = np.arange(len(rep))
    for i, (a, (gp, gr)) in enumerate(sorted(rep.items())):
        col = BLUE if a == "seed16" else (ORANGE if a == "seed11" else "#bdbdbd")
        ax2.plot([i, i], [gp, gr], color=col, lw=1.1, zorder=3)
        ax2.plot(i, gp, marker="o", ms=4.6, color=PAPER, markeredgecolor=col,
                 markeredgewidth=1.1, zorder=4)
        ax2.plot(i, gr, marker="o", ms=4.6, color=col, markeredgecolor=INK,
                 markeredgewidth=0.55, zorder=4)
    band(ax2)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([a.replace("seed", "s") for a in sorted(rep)])
    ax2.set_xlim(-0.55, len(rep) - 0.45)
    ax2.set_xlabel("reseeded probe")
    ax2.set_ylabel("measured gate loss")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Different seed", loc="left")
    ax2.text(0.02, 0.02, "open = published", transform=ax2.transAxes,
             fontsize=8, color=MUTED, ha="left", va="bottom")

    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig6_resolution.{ext}", bbox_inches="tight",
                    facecolor=PAPER)
    plt.close(fig)
    print("[OK] fig6_resolution.pdf/.png")


if __name__ == "__main__":
    main()
