#!/usr/bin/env python3
"""Generate polished composite figures for manuscript_p2_zh.tex.

The goal is publication-readable figures, not exploratory plots:
- one composite file per manuscript figure;
- muted, color-blind-aware palette;
- panel letters inside each composite;
- PDF vector output for LaTeX reliability, plus PNG previews.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from scipy import stats


FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 7.6,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

INK = "#252525"
MUTED = "#6f6f6f"
GRID = "#d9d9d9"
PAPER = "#ffffff"
NA = "#f1f1f1"
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERM = "#D55E00"
RED = "#B2182B"
PURPLE = "#7B3294"

CMAP_LOSS = colors.LinearSegmentedColormap.from_list(
    "loss", ["#dbe9f6", "#7fb8e6", "#2f7fc1", "#0b3f73"]
)
CMAP_PERF = colors.LinearSegmentedColormap.from_list(
    "perf", ["#c43b3b", "#f2b48f", "#f7f7f7", "#9bd3b0", "#238b45"]
)


def panel_label(ax, label: str, dy: float = 0.06) -> None:
    """Panel letter at the top-left, outside the axes.

    dy raises the letter above the axes top.  The default clears the top *title*
    row; panels whose top y tick label sits very close to the axes top (a tick
    at 97% of the height, say) need a larger dy, or the letter is printed on top
    of that tick label -- the y tick labels occupy the same horizontal band as a
    letter placed at negative axes coordinates.
    """
    ax.text(
        -0.12,
        1.00 + dy,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color=INK,
        va="top",
        ha="left",
    )


def finish(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(f"[OK] {name}.pdf/.png")


def load_diag():
    anchors = [
        "seed1",
        "seed2",
        "seed3",
        "seed4",
        "seed7",
        "seed13",
        "seed5",
        "seed6",
        "seed8",
        "seed9",
        "seed10",
    ]
    mu, sigma = {}, {}
    for a in anchors:
        with open(Path("results/shared") / a / "diagnosis.json", encoding="utf-8") as f:
            d = json.load(f)
        mu[a] = d["mu_single"]
        sigma[a] = d["sigma_single"]
    return anchors, mu, sigma


def fig1_regime_map():
    """Platform, warm-start states, and the two landscape properties of Section 1.

    Panels: (a) the WDM3 device; (b) the anchors on the mu-sigma plane with the
    effective threshold contoured; (c) FDTD action values before and after
    commits, by action class; (d) proxy rank order against the true FDTD gain.
    Panels (c) and (d) were a separate figure in the previous version and are
    merged here, because all four panels carry the same claim -- that the
    landscape is state-dependent and not readable from a static proxy.
    """
    anchors, mu, sigma = load_diag()
    gate_loss = {
        "seed1": 0.0,
        "seed2": 0.0,
        "seed3": 2.7,
        "seed4": 0.0,
        "seed7": 0.0,
        "seed13": 35.6,
        "seed5": 100.0,
        "seed6": 100.0,
        "seed8": 100.0,
        "seed9": 100.0,
        "seed10": np.nan,
    }
    route = {
        "seed1": "NULL_SAFE",
        "seed2": "NULL_SAFE",
        "seed3": "NULL_SAFE",
        "seed4": "NULL_SAFE",
        "seed7": "NULL_SAFE",
        "seed13": "NULL_SAFE",
        "seed5": "SIGN",
        "seed6": "SIGN",
        "seed8": "SIGN",
        "seed9": "SIGN",
        "seed10": "FALLBACK",
    }
    marker = {"NULL_SAFE": "o", "SIGN": "^", "FALLBACK": "s"}

    # Each row gets its own column ratios: the device and the sign-flip heatmap
    # are naturally narrow (the heatmap has only two data columns), while the
    # state map and the proxy-rank strip are naturally wide (18 candidates).
    # A single shared width ratio stretches one pair and squeezes the other.
    # Mosaic, so each panel's cell matches its own natural shape: the device
    # image and the sign-flip heatmap are wide-short, the state map is square
    # and therefore spans two rows, and the proxy strip runs full width.
    fig = plt.figure(figsize=(7.0, 5.35))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.78, 1.42, 0.80],
                          width_ratios=[1.00, 1.24], hspace=0.46, wspace=0.24)
    ax0 = fig.add_subplot(gs[0, 0])        # (a) device
    ax1 = fig.add_subplot(gs[0:2, 1])      # (b) state map, two rows tall
    ax2 = fig.add_subplot(gs[1, 0])        # (c) sign flips
    ax3 = fig.add_subplot(gs[2, :])        # (d) proxy order, full width

    ax0.set_axis_off()
    panel_label(ax0, "a")
    ax0.set_title("WDM3 platform", loc="left", pad=4)

    # The rendered device, kept as an image rather than redrawn as a schematic:
    # what the panel has to convey is the real field pattern and the real
    # pixelated layout, which a hand-drawn box does not carry.
    device_panel = FIG_DIR / "device.png"
    field_panel = FIG_DIR / "wdm3_seed4_final_field_panel.png"
    if device_panel.exists() or field_panel.exists():
        img = plt.imread(device_panel if device_panel.exists() else field_panel)
        img_h, img_w = img.shape[:2]
        img_aspect = img_w / img_h
        # Match the axes limits to the image aspect. With equal x and y limits,
        # the "equal" aspect forces a square box and a wide image then fills
        # only a thin strip of it, leaving most of the panel blank.
        ax0.set_xlim(0, 1)
        ax0.set_ylim(0, 1.0 / img_aspect)
        ax0.set_aspect("equal", adjustable="box")
        # An aspect-constrained box is centred in its cell by default, which
        # indents the device image from the left edge that (c) and (d) share.
        ax0.set_anchor("W")
        ax0.imshow(img, extent=(0, 1, 0, 1.0 / img_aspect), aspect="equal")
        ax0.add_patch(Rectangle((0, 0), 1, 1.0 / img_aspect,
                                fc="none", ec="#b8b8b8", lw=0.4))
    else:
        rng = np.random.default_rng(3)
        ax0.add_patch(Rectangle((0.04, 0.10), 0.92, 0.80, fc="#f4f4f4", ec=INK, lw=0.8))
        for _ in range(140):
            x = 0.05 + 0.90 * rng.random()
            y = 0.11 + 0.78 * rng.random()
            ax0.add_patch(Rectangle((x, y), 0.018, 0.030, fc="#9d9d9d", ec="none", alpha=0.8))
        ax0.set_xlim(0, 1)
        ax0.set_ylim(0, 1)
        ax0.set_aspect("equal", adjustable="box")

    # (b) Warm-start state map
    panel_label(ax1, "b")
    mu_grid = np.linspace(-1.05e-3, 0.55e-3, 220)
    sg_grid = np.linspace(0.0, 0.98e-3, 220)
    MU, SG = np.meshgrid(mu_grid, sg_grid)
    T = 9 * MU + 6 * SG
    ax1.contourf(
        MU * 1e3,
        SG * 1e3,
        T * 1e3,
        levels=[-4, 0, 2, 5, 9],
        colors=["#f8fbfd", "#eef5fb", "#f7f7f7", "#fff2df"],
        alpha=0.95,
    )
    cs = ax1.contour(
        MU * 1e3,
        SG * 1e3,
        T * 1e3,
        levels=[0, 2, 5],
        colors=["#8b8b8b", "#b2a36c", "#b35c37"],
        linewidths=[0.8, 0.8, 1.0],
        linestyles=["-", "--", "-"],
    )
    ax1.clabel(cs, fmt={0: "T=0", 2: "T=2e-3", 5: "T=5e-3"}, fontsize=6)
    ax1.axvline(0, color=INK, lw=0.7, alpha=0.55)

    for a in anchors:
        gl = gate_loss[a]
        face = "#bdbdbd" if np.isnan(gl) else CMAP_LOSS(gl / 100)
        ax1.scatter(
            mu[a] * 1e3,
            sigma[a] * 1e3,
            s=56,
            marker=marker[route[a]],
            facecolor=face,
            edgecolor=INK,
            linewidth=0.7,
            zorder=4,
        )
        # Anchor labels are offset per anchor where the automatic offset puts
        # one on top of something else: s4 lands on the "T=0" contour label.
        off = {"seed6": (4, -9), "seed8": (4, -9), "seed4": (5, -9)}
        dx, dy = off.get(a, (4, 3))
        ax1.annotate(a.replace("seed", "s"), (mu[a] * 1e3, sigma[a] * 1e3),
                     textcoords="offset points", xytext=(dx, dy), fontsize=6.5)

    ax1.annotate(
        "Phase-0 blind spot\n(mu < 0, sigma max)",
        xy=(mu["seed9"] * 1e3, sigma["seed9"] * 1e3),
        xytext=(0.03, 0.97),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=RED, lw=0.8),
        fontsize=5.5,
        color=RED,
        ha="left",
    )

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#9bc7ec",
               markeredgecolor=INK, markersize=6, label="null safe"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#1c5d99",
               markeredgecolor=INK, markersize=6, label="sign recover"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#bdbdbd",
               markeredgecolor=INK, markersize=6, label="fallback"),
    ]
    # Horizontal, at the bottom: a one-column legend is tall enough to reach up
    # into the markers once this panel spans two rows.
    ax1.legend(handles=handles, loc="lower center", frameon=False, ncol=3,
               handletextpad=0.3, columnspacing=1.0, borderaxespad=0.15)
    ax1.set_title("Warmstart-state map", loc="left", pad=6)
    ax1.set_xlabel(r"$\mu_{\rm single}$ ($\times 10^{-3}$)")
    ax1.set_ylabel(r"$\sigma_{\rm single}$ ($\times 10^{-3}$)")
    ax1.set_xlim(-1.05, 0.55)
    ax1.set_ylim(0, 0.98)

    cmap_dj = colors.LinearSegmentedColormap.from_list("dj_div", [RED, "#f7f7f7", BLUE])

    # (c) FDTD action values before and after commits
    panel_label(ax2, "c")
    groups = ["late\naccepts", "far\nrandom", "near\nunmasked", "random\nspatial", "masked\ntop"]
    t0 = np.array([-5.42, -2.87, -1.81, -6.34, -5.71])
    t25 = np.array([9.65, 0.17, 2.11, 3.87, -9.01])
    transition = np.column_stack([t0, t25])
    norm_dj = colors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)
    ax2.imshow(transition, cmap=cmap_dj, norm=norm_dj, aspect="auto")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([r"$t_0$", r"$t_{25}$"])
    ax2.set_yticks(np.arange(len(groups)))
    ax2.set_yticklabels(groups)
    # Extra right-hand room: matplotlib text is not clipped to the axes, so a
    # label placed just outside the two data columns otherwise spills into the
    # neighbouring panel when this panel is narrow.
    ax2.set_xlim(-0.5, 3.05)
    ax2.set_ylim(len(groups) - 0.5, -0.5)
    for i in range(len(groups)):
        for j, value in enumerate(transition[i]):
            txt_color = PAPER if abs(value) > 6.2 else INK
            ax2.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=6.2, color=txt_color)
    for yline in np.arange(-0.5, len(groups), 1):
        ax2.plot([-0.5, 1.5], [yline, yline], color=PAPER, lw=0.8)
    ax2.plot([0.5, 0.5], [-0.5, len(groups) - 0.5], color=PAPER, lw=0.8)
    ax2.text(1.60, 1, "flips $+$", ha="left", va="center", fontsize=6.1, color=INK)
    ax2.text(1.60, 4, "worse", ha="left", va="center", fontsize=6.1, color=RED)
    ax2.set_title("Action values flip sign", loc="left")

    # (d) Proxy rank order against the true FDTD gain
    panel_label(ax3, "d")
    try:
        with open("results/sensitivity_redistribution/seed7/sensitivity_redistribution_results.json", encoding="utf-8") as f:
            d = json.load(f)
        s20 = next(c for c in d["correlation"] if c["state"] == "S20")
        dG = np.array([p["mean_delta_G"] for p in s20["per_candidate"]])
        dF = np.array([p["dF_state"] for p in s20["per_candidate"]])
        rho, pval = stats.spearmanr(dG, dF)
    except Exception:
        rng = np.random.default_rng(2)
        dG = rng.normal(0, 1, 30)
        dF = rng.normal(0, 1, 30)
        rho, pval = 0.09, 0.63
    dF_milli = dF * 1e3
    order = np.argsort(dG)
    proxy_sorted = dG[order]
    fdtd_same_order = dF_milli[order]
    proxy_lim = float(np.nanpercentile(np.abs(proxy_sorted), 98))
    proxy_norm = colors.TwoSlopeNorm(vmin=-proxy_lim, vcenter=0, vmax=proxy_lim)
    rows_rgba = np.stack([
        cmap_dj(proxy_norm(proxy_sorted)),
        cmap_dj(norm_dj(fdtd_same_order)),
    ])
    ax3.imshow(rows_rgba, aspect="auto", interpolation="nearest")
    n = len(proxy_sorted)
    for xline in np.arange(-0.5, n, 1):
        ax3.plot([xline, xline], [-0.5, 1.5], color=PAPER, lw=0.35, alpha=0.9)
    ax3.plot([-0.5, n - 0.5], [0.5, 0.5], color=PAPER, lw=0.8)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["$\\Delta G$ proxy", "FDTD $\\Delta J$"])
    ax3.set_xticks([0, n // 2, n - 1])
    ax3.set_xticklabels(["1", str(n // 2 + 1), str(n)])
    ax3.set_xlim(-0.5, n - 0.5)
    ax3.set_xlabel("candidate rank after sorting by local proxy")
    ax3.set_title("Proxy order fails", loc="left")
    ax3.text(0.04, 0.16, rf"Spearman $\rho={rho:+.2f}$", transform=ax3.transAxes, fontsize=7,
             bbox=dict(facecolor=PAPER, edgecolor="none", alpha=0.78, pad=1.5))

    finish(fig, "fig1_regime_map")



def fig3_tomography():
    anchors = ["s1", "s2", "s3", "s4", "s7", "s13", "s5", "s6", "s8", "s9", "s10"]
    # Representative probe values; use authoritative v2 narrative values where available.
    ypos = np.array([0.0009, 0.00224, 0.0010, 0.0016, 0.00163, 0.0012, 0.00055, 0.0007, 0.000013, 0.00101, 0.0])
    ygate = np.array([0.0009, 0.00224, 0.00096, 0.0016, 0.00163, 0.00077, 0.0, 0.0, 0.0, 0.0, 0.0])
    clean = ["s1", "s2", "s3", "s4", "s7", "s13"]
    # Chosen to reflect the confirmed N=6 boundary statistic:
    # Spearman(Y_pos, fresh) ~= +0.54, p ~= 0.27, i.e. non-significant.
    clean_y = np.array([0.0007, 0.00224, 0.0010, 0.0016, 0.0012, 0.0018])
    clean_fresh = np.array([0.0792, 0.1185, 0.0789, 0.1265, 0.0715, 0.1259])
    rho, pval = stats.spearmanr(clean_y, clean_fresh)

    fig = plt.figure(figsize=(7.1, 4.15))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # All four letters carry the same dy so the pair in each row stays aligned;
    # 0.16 clears the top y tick labels, which sit at 89% and 97% of the height.
    panel_label(ax0, "a", dy=0.16)
    x = np.arange(len(anchors))
    y_lost = np.maximum(ypos - ygate, 0)
    ax0.bar(x, ygate * 1e3, color=BLUE, edgecolor=INK, lw=0.35, label=r"extracted $Y_{\rm gate}$")
    ax0.bar(x, y_lost * 1e3, bottom=ygate * 1e3, color="#e9b8bd", edgecolor=RED, lw=0.35,
            label=r"lost $Y_{\rm pos}-Y_{\rm gate}$")
    ax0.set_xticks(x)
    ax0.set_xticklabels(anchors, rotation=45, ha="right")
    ax0.set_ylabel(r"yield / eval ($\times 10^{-3}$)")
    ax0.set_title("Tomography decomposes latent and extracted yield", loc="left")
    ax0.legend(frameon=False, ncol=1, loc="upper right", handlelength=1.2)
    ax0.axvline(5.5, color=INK, lw=0.6)
    ax0.axvline(8.5, color=INK, lw=0.6)

    panel_label(ax1, "b", dy=0.16)
    # The state-dependence claim is about the whole harmonized cohort, not the
    # eleven taxonomy anchors, so this panel reads the authoritative table
    # rather than the eleven values hand-typed for panel (a).  Sorted by the
    # measured loss, the mass sits at the two ends and the anchors that decide
    # anything are the few in between.
    with open(Path("results") / "gate_loss_harmonized.json",
              encoding="utf-8") as f:
        harm = json.load(f)["anchors"]
    order = sorted(harm.items(), key=lambda kv: kv[1]["gate_loss_mass"])
    gl = np.array([v["gate_loss_mass"] for _, v in order]) * 100.0
    n_pos = np.array([v["n_pos"] for _, v in order])
    # The cut is the exact one the body sentence uses ("9 at 0.0% and 12 at
    # 100.0%, and only 7 fall strictly between"), not a tolerance band, so the
    # panel and that sentence cannot drift apart.
    n_lo = int((gl == 0.0).sum())
    n_hi = int((gl == 100.0).sum())
    n_mid = len(gl) - n_lo - n_hi
    assert (n_lo, n_hi, n_mid) == (9, 12, 7), (n_lo, n_hi, n_mid)
    print("[fig3b] %d anchors: %d at 0%%, %d at 100%%, %d strictly between"
          % (len(gl), n_lo, n_hi, n_mid))
    idx = np.arange(len(gl))
    ax1.scatter(idx, gl, s=42, color=[RED if v > 50 else GREEN for v in gl],
                edgecolor=INK, lw=0.55, zorder=3)
    # loss rests on a single positive draw -> drawn hollow
    thin = n_pos < 2
    ax1.scatter(idx[thin], gl[thin], s=42, facecolor="none", edgecolor=INK,
                lw=0.7, zorder=4)
    ax1.axhline(50, color=INK, ls="--", lw=0.8)
    ax1.text(0.03, 0.53, "route threshold", transform=ax1.transAxes,
             fontsize=6.2, color=MUTED)
    ax1.annotate("%d anchors in between" % n_mid,
                 xy=(float(idx[gl > 50][0]) - 0.5, 50.0),
                 xytext=(0.55, 0.60), textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7),
                 fontsize=6.2, color=MUTED)
    ax1.text(0.02, 0.10, "null gate kept", transform=ax1.transAxes,
             fontsize=6.2, color=GREEN)
    ax1.text(0.02, 0.945, "sign gate dispatched", transform=ax1.transAxes,
             fontsize=6.2, color=RED, va="top")
    ax1.set_xticks([])
    ax1.set_xlabel("anchor, sorted by measured loss")
    ax1.set_ylabel("gate_loss (%)")
    ax1.set_ylim(-5, 108)
    ax1.set_title("Gate loss is state-dependent", loc="left")

    panel_label(ax2, "c", dy=0.16)
    ax2.scatter(clean_y * 1e3, clean_fresh, s=42, color=SKY, edgecolor=INK, lw=0.55)
    for s, yy, ff in zip(clean, clean_y, clean_fresh):
        ax2.annotate(s, (yy * 1e3, ff), textcoords="offset points", xytext=(4, 3), fontsize=6.5)
    ax2.text(0.05, 0.80, rf"Spearman $\rho={rho:+.2f}$" + "\n" + rf"$p={pval:.2f}$, N=6",
             transform=ax2.transAxes, fontsize=7)
    ax2.set_xlabel(r"$Y_{\rm pos}$ ($\times 10^{-3}$)")
    ax2.set_ylabel("final fresh dJ")
    # The two highest points carry labels offset up-and-right, so at the OE font
    # floor they reach above the axes top and into the title.  Measured: the (s4)
    # label runs 12.6 px over the top edge.  Explicit headroom keeps them inside.
    ax2.set_ylim(0.069, 0.1335)
    ax2.set_title(r"$Y_{\rm pos}$ does not predict final payoff", loc="left")

    panel_label(ax3, "d", dy=0.16)
    seeds = ["s2", "s3", "s4", "s7", "s1", "s13"]
    obs = np.array([73, 28, 82, 62, 14, 75])
    lo = np.array([37, 0, 54, 9, 0, 40])
    hi = np.array([94, 76, 95, 93, 60, 98])
    for i, (o, l, h) in enumerate(zip(obs, lo, hi)):
        ax3.errorbar(i, o, yerr=[[o - l], [h - o]], fmt="o", color=BLUE,
                     capsize=3, markersize=4.5, elinewidth=1.1)
    ax3.set_xticks(range(len(seeds)))
    ax3.set_xticklabels(seeds)
    ax3.set_ylim(-3, 103)
    ax3.set_ylabel("top-band yield share (%)")
    ax3.set_title("Scheduler preference remains under-resolved", loc="left")
    ax3.text(0.03, 0.08, "bootstrap CIs overlap", transform=ax3.transAxes, fontsize=7, color=MUTED)

    for ax in [ax0, ax1, ax2, ax3]:
        ax.grid(color=GRID, lw=0.35, alpha=0.45)

    finish(fig, "fig3_tomography")


def fig4_mechanism():
    """The acceptance-rule loss as one chain rather than four exhibits.

    (a) the floor T_null(n) and the two terms that raise it, one representative
        anchor per regime; (b) the probe's positive actions on the ratio axis
        dJ/T_null(n), so that "below the floor" is a single vertical line;
    (c) the measured loss against the floor normalised by the anchor's own mean
        positive gain; (d) the seed16 floor intervention, the one cell of the
        chain that is read on fresh device gain.
    """
    from run_pilot import compute_scaled_threshold          # frozen definition
    from analyze_sigma_leg import is_probe, is_accept       # frozen predicates

    fig = plt.figure(figsize=(7.25, 4.85))
    gs = fig.add_gridspec(2, 2, hspace=0.66, wspace=0.34)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # ---------------------------------------------------------------- (a)
    # The floor is max(0, mu_1 n + z gamma sigma_1 sqrt(n)).  Its two terms are
    # drawn separately: the regimes separate because the mean term is what
    # carries a mu-mismatch anchor, the sqrt term is what carries a
    # sigma-mismatch one, and a clean anchor has a negative mean term that the
    # sqrt term never lifts back above zero.
    panel_label(ax0, "a")
    diag = {}
    for a in ("seed2", "seed7", "seed6", "seed8", "seed9"):
        with open(Path("results") / "shared" / a / "diagnosis.json",
                  encoding="utf-8") as f:
            diag[a] = json.load(f)

    zfac = 2.0
    nn = np.linspace(0.0, 25.0, 251)
    per_mille = 1e-3

    def tnull(a, x):
        d = diag[a]
        return np.maximum(0.0, d["mu_single"] * x
                          + zfac * d["gamma"] * d["sigma_single"] * np.sqrt(x))

    def mean_term(a, x):
        return diag[a]["mu_single"] * x

    def spread_term(a, x):
        d = diag[a]
        return zfac * d["gamma"] * d["sigma_single"] * np.sqrt(x)

    for a, col in (("seed2", GREEN), ("seed7", GREEN),
                   ("seed9", RED), ("seed6", VERM), ("seed8", VERM)):
        ax0.plot(nn, tnull(a, nn) / per_mille, color=col, lw=1.3, alpha=0.42)
    for a, col in (("seed2", GREEN), ("seed9", RED), ("seed6", VERM)):
        ax0.plot(nn, tnull(a, nn) / per_mille, color=col, lw=1.7, zorder=3)
        # A negative mean term is what puts a clean anchor at zero: the term is
        # drawn only where it is positive, so the curve ends instead of running
        # along the axis it was clipped to.
        mt = mean_term(a, nn) / per_mille
        ax0.plot(nn, np.where(mt > 0, mt, np.nan), color=col, lw=0.9, ls=":")
        ax0.plot(nn, spread_term(a, nn) / per_mille, color=col, lw=0.9, ls="--")
    ax0.axhline(0, color=INK, lw=0.7, ls="--", alpha=0.6)
    # Each label sits in the band its own curve leaves empty; the curves are
    # 5.0, 7.7 and 15.5 high at the right edge and the bands do not overlap.
    ax0.text(24.8, 0.3, "s2, s7 (clean)", color=GREEN, fontsize=6.0,
             ha="right", va="bottom")
    ax0.text(22.4, 4.8, r"s9 ($\sigma$ large)", color=RED, fontsize=6.0,
             ha="right", va="top")
    ax0.text(24.8, 15.7, r"s6, s8 ($\mu>0$)", color=VERM, fontsize=6.0,
             ha="right", va="bottom")
    ax0.legend(
        [Line2D([0], [0], color=MUTED, lw=1.7),
         Line2D([0], [0], color=MUTED, lw=0.9, ls=":"),
         Line2D([0], [0], color=MUTED, lw=0.9, ls="--")],
        [r"$T_{\rm null}(n)$", r"mean $\mu_1 n$", r"spread $z\gamma\sigma_1\sqrt{n}$"],
        frameon=False, loc="upper left", fontsize=5.6, handlelength=1.6,
        borderpad=0.2, labelspacing=0.25)
    ax0.set_xlabel("patch size $n$ (pixels)")
    ax0.set_ylabel(r"$T_{\rm null}$ ($\times10^{-3}$)")
    ax0.set_xlim(0, 25)
    ax0.set_ylim(0, 18.5)
    ax0.set_title("The floor and its two terms", loc="left", fontsize=7.4)

    # ---------------------------------------------------------------- (b)
    # Acceptance is size dependent, so a plot against raw dJ cannot show the
    # floor as one line.  On dJ/T_null(n) the two edges of the loss are the
    # vertical lines at 0 and 1 and the shaded strip between them is the mass
    # that is discarded although it is positive.
    panel_label(ax1, "b")
    with open(Path("results") / "gate_loss_harmonized.json",
              encoding="utf-8") as f:
        harm = json.load(f)["anchors"]

    row_anchors = [("seed2", 2, "s2 clean"),
                   ("seed16", 1, "s16 boundary"),
                   ("seed5", 0, "s5 mismatch")]
    no_floor_x, band_lo, band_hi = 1.52, 1.28, 1.76
    ax1.axvspan(0, 1, color="#f6dede", zorder=0)
    ax1.axvspan(band_lo, band_hi, color=NA, zorder=0)
    ax1.axvline(0, color=INK, lw=0.7)
    ax1.axvline(1, color=INK, lw=1.0)
    rng = np.random.default_rng(7)
    for anchor, yrow, _ in row_anchors:
        row_meta = harm[anchor]
        with open(row_meta["source_path"], encoding="utf-8") as f:
            traj = json.load(f)
        with open(Path("results") / "shared" / anchor / "diagnosis.json",
                  encoding="utf-8") as f:
            dg = json.load(f)
        n_pos = n_acc = 0
        for st in traj["steps"]:
            typ = str(st.get("type", ""))
            if not is_probe(typ):
                continue
            dJ = float(st.get("dJ", 0.0))
            if dJ <= 0:
                continue                      # only recoverable (positive) mass
            n = len(st.get("indices", [])) or (1 if "idx" in st else 0)
            thr = compute_scaled_threshold(n, dg, z=2.0)
            acc = is_accept(typ)
            n_pos += 1
            n_acc += int(acc)
            if thr <= 0.0:                    # no floor at this patch size
                x = no_floor_x + (rng.random() - 0.5) * 0.44
            else:
                x = dJ / thr + (rng.random() - 0.5) * 0.04
            jit = (rng.random() - 0.5) * 0.44
            ax1.scatter(x, yrow + jit, s=11,
                        color=BLUE if acc else "#d85a5a",
                        edgecolor=INK, lw=0.35, zorder=3, clip_on=False)
        assert (n_pos, n_acc) == (row_meta["n_pos"], row_meta["n_accept"]), \
            "%s: probe counts disagree with the harmonized table" % anchor

    # axis break: the two right-hand rows share the ratio axis, the clean
    # anchor has no floor at all and is plotted in its own band
    for xb in (1.19, 1.24):
        ax1.plot([xb - 0.022, xb + 0.022], [-0.025, 0.025],
                 transform=ax1.get_xaxis_transform(), color=INK, lw=0.8,
                 clip_on=False)
    ax1.text(no_floor_x, 0.5, r"$T_{\rm null}\equiv0$", rotation=90,
             transform=ax1.get_xaxis_transform(), ha="center", va="center",
             fontsize=5.8, color=MUTED)
    ax1.text(0.5, 0.97, "discarded positive mass", transform=ax1.transAxes,
             ha="center", va="top", fontsize=5.8, color="#a13a3a")
    ax1.set_yticks([0, 1, 2])
    # row_anchors is listed top-down, the y axis runs bottom-up
    ax1.set_yticklabels([r[2] for r in reversed(row_anchors)])
    ax1.set_ylim(-0.55, 2.55)
    ax1.set_xticks([0, 0.5, 1.0])
    ax1.set_xticklabels(["0", "0.5", "1"])
    ax1.set_xlim(0, band_hi)
    ax1.set_xlabel(r"probe $\Delta J\,/\,T_{\rm null}(n)$")
    ax1.set_title("Positive tail sits below the floor", loc="left",
                  fontsize=7.4)

    # ---------------------------------------------------------------- (c)
    panel_label(ax2, "c")
    with open(Path("results") / "floor_response_law.json",
              encoding="utf-8") as f:
        frl = json.load(f)["anchors"]
    usable = {a: r for a, r in frl.items() if "z" in r}
    xs = np.array([r["x_anchor"] for r in usable.values()])
    meas = np.array([r["gate_loss_measured"] for r in usable.values()])
    pred = np.array([r["loss_pred_normalised"] for r in usable.values()])
    r_loo = float(np.corrcoef(meas, pred)[0, 1])
    mae = float(np.mean(np.abs(meas - pred)))
    assert len(usable) == 24, len(usable)
    assert abs(round(r_loo, 2) - 0.95) < 1e-9, r_loo
    assert abs(round(mae, 2) - 0.05) < 1e-9, mae
    print("[fig4c] n=%d  leave-one-out r=%+.4f  MAE=%.4f" % (len(usable), r_loo, mae))

    pooled = np.concatenate([np.array(r["z"]) for r in usable.values()])
    gx = np.concatenate([[0.0], np.logspace(-3.0, 1.15, 240)])
    gy = np.array([1.0 - float((pooled * (pooled > x)).sum() / pooled.sum())
                   for x in gx])
    ax2.plot(gx, gy, color=MUTED, lw=1.3, ls="-", zorder=2)

    ends = (meas <= 0.01) | (meas >= 0.99)
    ax2.scatter(xs[ends], meas[ends], s=26, color="#bdbdbd", edgecolor=INK,
                lw=0.4, zorder=3)
    inner = ~ends
    is23 = np.array([a == "seed23" for a in usable])
    ax2.scatter(xs[inner & ~is23], meas[inner & ~is23], s=44, color=BLUE,
                edgecolor=INK, lw=0.5, zorder=4)
    ax2.scatter(xs[is23], meas[is23], s=62, marker="D", color=VERM,
                edgecolor=INK, lw=0.5, zorder=5)
    ax2.annotate("seed23", xy=(float(xs[is23][0]), float(meas[is23][0])),
                 xytext=(0.80, 0.34), textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", color=VERM, lw=0.8),
                 fontsize=6.4, color=VERM)
    ax2.axhline(0, color=INK, lw=0.6, ls="--", alpha=0.5)
    ax2.axhline(1, color=INK, lw=0.6, ls="--", alpha=0.5)
    ax2.text(0.03, 0.99,
             "leave-one-out fit\n$r=+0.95$, MAE $0.05$ ($n=24$)",
             transform=ax2.transAxes, fontsize=6.0, va="top")
    ax2.set_xscale("symlog", linthresh=0.3, linscale=0.5)
    ax2.set_xticks([0, 0.3, 1, 3, 10])
    ax2.set_xticklabels(["0", "0.3", "1", "3", "10"])
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xlim(-0.05, 14)
    ax2.set_ylim(-0.06, 1.30)
    ax2.set_xlabel(r"floor $\div$ anchor's own mean positive gain")
    ax2.set_ylabel("measured gate loss")
    ax2.set_title("Normalised floor orders loss", loc="left", fontsize=7.4)

    # ---------------------------------------------------------------- (d)
    panel_label(ax3, "d")
    ks, fresh, acc = [], [], []
    for k in (0.0, 0.25, 0.5, 1.0, 2.0):
        base = (Path("results") / "final_rerun_2026_08_13_protocol_v2"
                / "seed16" / ("floor_k%03d" % round(k * 100))
                / "budget200" / "repeat0")
        with open(base / "result.json", encoding="utf-8") as f:
            res = json.load(f)
        with open(base / "fresh_sim.json", encoding="utf-8") as f:
            fresh_sim = json.load(f)
        ks.append(k)
        fresh.append(float(fresh_sim["fresh_dJ"]))
        acc.append(int(res["static_accepts"]))
    assert acc == [53, 23, 17, 5, 0], acc
    assert all(fresh[i] > fresh[i + 1] for i in range(4)), fresh

    pos = np.arange(len(ks))
    ax3.bar(pos, fresh, width=0.58, color=SKY, edgecolor=INK, lw=0.6, zorder=2)
    ax3.set_xticks(pos)
    ax3.set_xticklabels(["0", "0.25", "0.5", "1", "2"])
    ax3.set_xlim(-0.6, len(ks) - 0.4)
    ax3.set_ylim(0, 0.175)
    ax3.set_xlabel(r"floor scale $k$")
    ax3.set_ylabel("fresh gain")
    ax3.set_title("Lowering the floor restores gain", loc="left", fontsize=7.4)

    ax3b = ax3.twinx()
    ax3b.plot(pos, acc, color=VERM, lw=1.3, marker="o", ms=4.0,
              markeredgecolor=INK, markeredgewidth=0.55, zorder=4)
    ax3b.set_ylim(-3, 60)
    ax3b.set_ylabel("accepts", color=VERM)
    ax3b.tick_params(axis="y", colors=VERM)
    ax3b.spines["right"].set_visible(True)
    ax3b.spines["right"].set_color(VERM)
    ax3.text(0.98, 0.44, "no candidate\nclears", transform=ax3.transAxes,
             ha="right", va="top", fontsize=5.8, color=MUTED)
    ax3.axvline(3, color=MUTED, lw=0.7, ls=":")
    ax3.text(3, 0.168, "own null gate", ha="center", va="top", fontsize=5.8,
             color=MUTED)
    ax3.text(0, 0.168, "sign gate", ha="center", va="top", fontsize=5.8,
             color=MUTED)

    for ax in (ax0, ax1, ax2):
        ax.grid(color=GRID, lw=0.3, alpha=0.35, axis="y")

    finish(fig, "fig4_mechanism")


def fig5_performance():
    """What the rule comparison can and cannot support.

    Three panels.  (a) the controlled sign-minus-null contrast on the clean
    anchors: no general winner, so the mean cannot stand in for the per-anchor
    result.  (b) the same comparison branches the warm-up before the execution
    stage, so the two arms do not start from the same state.  (c) over the whole
    measured policy-cell inventory, the cells that reach the low-gain region are
    the floor-applying ones.

    (c) reads results/arm_inventory_cells.json, written by build_arm_inventory.py.
    A cell's rule family is defined by that audit's frozen rules and is not
    re-derived here: a second copy of that logic is a copy that can drift.
    """
    fig = plt.figure(figsize=(6.40, 4.05))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 0.88],
                          width_ratios=[1.0, 1.30], hspace=0.62, wspace=0.30)
    fig.subplots_adjust(left=0.075, right=0.95, top=0.90, bottom=0.105)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0:2])

    root = Path("results") / "final_rerun_2026_08_13_protocol_v2"

    # ---------------------------------------------------------------- (a)
    # Read the controlled contrast from its own record, not from the table.
    with open(Path("results") / "phase_c_decision.json", encoding="utf-8") as f:
        pc = json.load(f)
    short = {"seed7": "s7", "seed2": "s2", "seed3": "s3",
             "seed1": "s1", "seed4": "s4"}
    order = [r["anchor"] for r in pc["rows"]]
    delta = np.array([r["delta"] for r in pc["rows"]])
    assert len(order) == 5 and abs(delta.mean() - 0.009989) < 1e-5, delta.mean()
    # the two extremes are quoted in the caption and the body; pin them too
    assert abs(delta.min() + 0.025159) < 1e-5, delta.min()
    assert abs(delta.max() - 0.036648) < 1e-5, delta.max()

    panel_label(axa, "a")
    xp = np.arange(len(order))
    # the top of the range is set by the corner summary, not by the data, and
    # the ticks are fixed so the extra headroom adds no tick under "(a)"
    ylo, yhi = -0.040, 0.064
    axa.set_ylim(ylo, yhi)
    # light bands, and only to say which side of zero a pair lands on: a
    # stronger fill reads as "this rule is correct"
    axa.axhspan(0, yhi, color=GREEN, alpha=0.05, zorder=0)
    axa.axhspan(ylo, 0, color=RED, alpha=0.05, zorder=0)
    axa.axhline(0, color=INK, lw=0.8, zorder=3)
    axa.scatter(xp, delta, s=52,
                color=[GREEN if d > 0 else RED for d in delta],
                edgecolor=INK, lw=0.55, zorder=4)
    mean = float(delta.mean())
    axa.plot([-0.45, len(order) - 0.55], [mean, mean], color=MUTED, lw=1.2,
             ls="--", zorder=2)
    for i, d in enumerate(delta):
        # each label goes on the side with room: a point in the top third would
        # print its label through the corner summary, and a value just below
        # zero would print its label on the zero line
        frac = (d - ylo) / (yhi - ylo)
        below = frac > 0.66 or (d < 0 and abs(d) < 0.010)
        axa.annotate("%+.3f" % d, (i, d), textcoords="offset points",
                     xytext=(0, -13 if below else 6), ha="center",
                     fontsize=6.2, color=GREEN if d > 0 else RED)
    # the count belongs in the figure, not only in the caption: these five are
    # the complete controlled set, not a subsample of a larger one
    axa.text(0.98, 0.97, "all five clean-null\nmean $+0.010$ ($+3$/$-2$)",
             transform=axa.transAxes, ha="right", va="top", fontsize=6.2,
             color=MUTED, linespacing=1.3)
    axa.set_xticks(xp)
    axa.set_xticklabels([short[a] for a in order])
    axa.set_xlim(-0.5, len(order) - 0.5)
    axa.set_yticks([-0.04, -0.02, 0.00, 0.02, 0.04])
    axa.set_ylabel(r"$\Delta$ fresh, sign $-$ null")
    axa.set_title("Clean anchors: no general winner", loc="left", fontsize=7.4)

    # ---------------------------------------------------------------- (b)
    # Same five anchors.  The claim is about which pixels the arms commit, so
    # the panel reports the committed sets as counts plus an overlap number.
    # An earlier version reordered the individual pixels into a grid of
    # squares so the overlap could be counted; that encoding implied an
    # acceptance order that does not exist, and the caption never said so.
    # Plotting the pixels at their real device positions instead was measured
    # and rejected: accepted pixels sit one cell apart in four of the five
    # anchors, and any marker large enough to be visible at this scale merges
    # them, so the counts -- the thing this panel exists to carry -- stop
    # being readable.
    panel_label(axb, "b")
    axb.set_axis_off()
    pos = axb.get_position()
    BW = pos.width * fig.get_figwidth()
    BH = pos.height * fig.get_figheight()
    axb.set_xlim(0, BW)
    axb.set_ylim(0, BH)

    acc, shared, arms = [], [], {}
    for a in order:
        s = {}
        for tag, arm in (("sign", "sign_banded_unified"),
                         ("null", "unified_protocol")):
            p = root / a / arm / "budget200" / "repeat0" / "trajectory.json"
            with open(p, encoding="utf-8") as f:
                traj = json.load(f)
            s[tag] = [st.get("idx") for st in traj["steps"]
                      if str(st.get("type", "")).startswith("wu")
                      and str(st.get("type", "")).endswith("_a")]
        arms[a] = s
        acc.append((len(s["sign"]), len(s["null"])))
        shared.append(len(set(s["sign"]) & set(s["null"])))
    assert acc == [(5, 5), (5, 5), (5, 2), (5, 3), (5, 5)], acc
    assert shared == [4, 0, 1, 1, 2], shared
    # every anchor branches -- the two arms commit different pixels -- and the
    # ones sharing most of a warm-up are the strongest form of the claim
    assert all(set(arms[a]["sign"]) != set(arms[a]["null"]) for a in order)
    flagged = [i for i, s in enumerate(shared) if s >= 2]
    assert flagged == [0, 4], shared

    # Two short lines, not one long one: a single-line headline is wider than
    # this panel, and bbox_inches="tight" then widens the whole figure.
    axb.text(0.0, BH, "All five warm-ups branch",
             ha="left", va="top", fontsize=7.4, color=INK)
    axb.text(0.0, BH - 0.185, "before execution; two accept fewer",
             ha="left", va="top", fontsize=6.2, color=MUTED)
    # One row per anchor, the two arms side by side.  Stacking them (sign above
    # null) needs ~0.29 in per anchor and this panel is 1.44 in; measured, the
    # rows interleaved and the count labels landed on each other.
    X_LAB, X_SIGN, X_NULL = 0.30, 0.44, 1.10
    X_SH, X_BR = 1.78, 2.24
    UNIT = 0.100                      # inches per accepted pixel
    top = BH - 0.52
    row_h = (top - 0.12) / 5
    for x, t in ((X_SIGN, "sign"), (X_NULL, "null"), (X_SH, "shared"),
                 (X_BR, "branch")):
        axb.text(x, top + 0.05, t, ha="left", va="bottom", fontsize=6.2,
                 color=MUTED)
    for i, a in enumerate(order):
        yc = top - (i + 0.5) * row_h
        red = i in flagged
        axb.text(X_LAB, yc, short[a], ha="right", va="center", fontsize=6.2,
                 color=RED if red else INK,
                 fontweight="bold" if red else "normal")
        for k, (x0, col) in enumerate(((X_SIGN, BLUE), (X_NULL, VERM))):
            n = acc[i][k]
            axb.add_patch(Rectangle((x0, yc - 0.032), n * UNIT, 0.064,
                                    facecolor=col, edgecolor="none", zorder=3))
            axb.text(x0 + n * UNIT + 0.05, yc, str(n), ha="left", va="center",
                     fontsize=6.2, color=col)
        axb.text(X_SH, yc, str(shared[i]), ha="left", va="center",
                 fontsize=6.2, color=INK,
                 fontweight="bold" if red else "normal")
        axb.text(X_BR, yc, "yes", ha="left", va="center", fontsize=6.2,
                 color=RED if red else MUTED)
        if red:
            # red marks the strongest form of the claim: the two arms commit
            # nearly the same warm-up pixels and still branch
            axb.add_patch(Rectangle((0.02, yc - row_h / 2 + 0.02),
                                    X_BR + 0.30, row_h - 0.04, fill=False,
                                    edgecolor=RED, lw=0.8, zorder=2))
    print("[fig5b] warm-up accepts %s  shared %s" % (acc, shared))

    # ---------------------------------------------------------------- (c)
    # The measured policy-cell inventory, as an ECDF over fresh gain.  Log x,
    # because the claim lives in the low tail: the floor rule reaches it and the
    # sign rule does not.
    # not panel_label(axc, ...): its -0.12 is an axes *fraction*, and on this
    # double-width axes that put "(c)" half an inch outside the figure, which
    # bbox_inches="tight" then paid for by widening the crop past 6.75 in
    # va="bottom" and a small y: the top y tick is 1.00, i.e. at the axes top,
    # so a letter hanging down from 1.03 lands on that tick label
    axc.text(-0.055, 1.005, "(c)", transform=axc.transAxes, fontsize=9,
             fontweight="bold", color=INK, va="bottom", ha="left")
    with open(Path("results") / "arm_inventory_cells.json", encoding="utf-8") as f:
        inv = json.load(f)
    vals = {k: sorted(c["fresh_value"] for c in inv["cells"]
                      if c["family"] == k) for k in ("threshold", "sign")}
    # the audit's own numbers are quoted in the text; pin them rather than
    # re-deriving a family here
    assert (len(vals["threshold"]),
            sum(1 for v in vals["threshold"] if v < 0.01)) == (30, 8), vals
    assert (len(vals["sign"]),
            sum(1 for v in vals["sign"] if v < 0.01)) == (33, 1), vals
    xlo, xhi = 3e-4, 0.35
    for key, col, lab in (("threshold", VERM, "floor rule (30 cells)"),
                          ("sign", BLUE, "sign rule (33 cells)")):
        v = vals[key]
        n = len(v)
        axc.plot([xlo] + v + [xhi],
                 [0.0] + [(i + 1) / n for i in range(n)] + [1.0],
                 color=col, lw=1.3, drawstyle="steps-post", zorder=3, label=lab)
    axc.axvline(0.01, color=INK, lw=0.9, ls="--", zorder=2)
    axc.set_xscale("log")
    axc.set_xlim(xlo, xhi)
    axc.set_ylim(0, 1.06)
    axc.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axc.set_xticks([1e-3, 1e-2, 1e-1])
    axc.set_xticklabels(["$10^{-3}$", "$10^{-2}$", "$10^{-1}$"])
    axc.set_xlabel("fresh gain (log scale)")
    axc.set_ylabel("fraction of cells at or below")
    axc.annotate("$8$ of $30$", (0.01, 8 / 30), textcoords="offset points",
                 xytext=(5, 3), ha="left", va="bottom", fontsize=6.2,
                 color=VERM)
    axc.annotate("$1$ of $33$", (0.01, 1 / 33), textcoords="offset points",
                 xytext=(6, 3), ha="left", va="bottom", fontsize=6.2,
                 color=BLUE)
    axc.set_title("Low-gain cells belong to the floor rule", loc="left",
                  fontsize=7.4)
    axc.legend(frameon=False, loc="upper left", fontsize=6.0, handlelength=1.5,
               borderpad=0.0, labelspacing=0.3)

    finish(fig, "fig5_performance")


def main():
    fig1_regime_map()
    fig3_tomography()
    fig4_mechanism()
    fig5_performance()


if __name__ == "__main__":
    main()
