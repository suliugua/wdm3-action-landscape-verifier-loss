"""Derive a print-requirements-compliant figure generator from make_figures_polished.py.

Derive : make_figures_polished.py  ->  make_figures_oj.py
(NB: this script must NOT be named make_figures_oj.py -- an earlier version was,
and overwrote itself with its own output.)

WHY
---
SPIE "Figure Requirements" (verified text):
  * Text       : "No smaller than 8 pt."
  * Dimensions : max 3-5/16 in (single column) / 6-3/4 in (two column)
  * Background : "Avoid graphs with shaded, transparent, or grid backgrounds."
  * Line weight: "0.5 points or greater in the final published size."
  * Captions   : must "contain descriptions of all labeled figure parts (a), (b)"
  * Multipart  : all parts in one file, one page.

Measured on the current figures: text runs 4.7-9 pt, 6 ax.grid() calls, 12 line
widths below 0.5, fig5 is 6.824 in wide, and fig1 labels two panels (a)/(b)
without describing them in the caption.

TRANSFORMS (textual, so each is auditable):
  1. rcParams font sizes raised to >= 8
  2. fontsize=X / labelsize=X -> round(max(8.0, 1.3 * X), 1)
  3. panel_label() renders "(a)" instead of "a"
  4. ax.grid(...) removed, together with a `for ...:` header left with no body
  5. lw= / linewidth= / set_linewidth(...) below 0.55 -> 0.55
  6. figsize widths clamped so the crop lands inside 6.75 in

Run:  python derive_figures_oj.py
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "make_figures_polished.py")
OUT = os.path.join(HERE, "make_figures_oj.py")

FLOOR, SCALE = 8.0, 1.3
LW_MIN = 0.55
W_MAX = 6.75


def main():
    assert os.path.basename(__file__) != os.path.basename(OUT), \
        "refusing to run: output would overwrite this script"
    src = open(SRC, encoding="utf-8").read()
    log = []

    # 1. rcParams
    for key in ("axes.titlesize", "axes.labelsize", "xtick.labelsize",
                "ytick.labelsize", "legend.fontsize"):
        src = re.sub(r'"%s": [0-9.]+,' % key, '"%s": 8,' % key, src)
    log.append("rcParams: all font sizes -> 8")

    # 2. explicit sizes
    n_fs = len(re.findall(r"(fontsize|labelsize)=", src))
    src = re.sub(r"(fontsize|labelsize)=([0-9.]+)",
                 lambda m: "%s=%g" % (m.group(1),
                                      round(max(FLOOR, SCALE * float(m.group(2))), 1)),
                 src)
    log.append("fontsize/labelsize raised: %d sites" % n_fs)

    # 3. panel labels get parentheses (SPIE: parts are labelled "(a), (b)"),
    #    and because a parenthesised label at 11.7 pt is wider than the bare
    #    letter it also has to move further left -- at -0.12 it runs into a
    #    loc="left" title.  Both edits live inside panel_label's body, so they
    #    are done as one counted substitution: the previous version anchored the
    #    match on "def panel_label(...)-> None:\n    ax.text(" and the log line
    #    was unconditional, so adding a docstring to the helper stopped the
    #    parenthesisation silently while still printing success.
    n_pl = 0
    marker = "    ax.text(\n        -0.12,\n        1.00 + dy,"
    if marker in src:
        src = src.replace(
            marker,
            '    label = f"({label})"   # SPIE: parts are labelled (a), (b), ...\n'
            "    ax.text(\n        -0.17,\n        1.00 + dy,",
            1)
        n_pl = 1
    log.append("panel_label parenthesised + anchor moved left: %d" % n_pl)

    # 3b. fig2 does not use the helper -- it draws its panel letters inline as
    #     ax.text(..., "a", transform=ax...).  Parenthesise those too, else that
    #     figure keeps bare letters while the others get "(a)".
    n_inline = len(re.findall(r'"[a-d]", transform=ax', src))
    src = re.sub(r'"([a-d])", transform=ax', r'"(\1)", transform=ax', src)
    log.append("inline panel letters parenthesised: %d" % n_inline)

    # 4. drop grid calls; if the call is the sole body of a `for ...:` loop,
    #    drop the header too (otherwise the loop body would be empty)
    n_grid = 0
    out = []
    for line in src.split("\n"):
        if re.match(r"^\s*ax[0-9]*\.grid\(", line):
            n_grid += 1
            if out and re.match(r"^\s*for [^\n]*:\s*$", out[-1]):
                out.pop()
            continue
        out.append(line)
    src = "\n".join(out)
    log.append("ax.grid() removed: %d calls" % n_grid)

    # 5. line weights.  The lookbehind keeps "elinewidth=" from being read as
    #    "linewidth=" and mangled; the "=" must survive the substitution.
    n_lw = len([1 for m in re.finditer(r"(?<![A-Za-z_])(lw|linewidth)=([0-9.]+)", src)
                if float(m.group(2)) < LW_MIN])
    src = re.sub(r"(?<![A-Za-z_])(lw|linewidth)=([0-9.]+)",
                 lambda m: "%s=%g" % (m.group(1), max(LW_MIN, float(m.group(2)))),
                 src)
    n_sw = len([1 for m in re.finditer(r"set_linewidth\(([0-9.]+)\)", src)
                if float(m.group(1)) < LW_MIN])
    src = re.sub(r"set_linewidth\(([0-9.]+)\)",
                 lambda m: "set_linewidth(%g)" % max(LW_MIN, float(m.group(1))), src)
    log.append("line widths raised to >=%g: %d + %d sites" % (LW_MIN, n_lw, n_sw))

    # 5b/5c were retired on 2026-09-20 with the figure they patched.  Both
    #     tuned the old fig5_performance, whose nine policy rows were packed
    #     into one panel; that figure is now a four-panel contrast figure whose
    #     panels are laid out directly, so every one of those strings is gone
    #     and the two rules would log 0/N for a substitution that no longer has
    #     a subject.  Removed rather than left to print a stale success count.

    # 5d. fig1: at 8 pt the "Phase-0 blind spot" annotation runs into the
    #     "Warmstart-state map" title.  It is moved down instead -- but not to
    #     0.30, which lands it on the automatic "T=0" contour label (measured at
    #     axes x 0.46-0.53, y 0.34-0.43).  0.56 clears that label and the legend
    #     while staying below the title row.
    n_f1 = 0
    if 'xytext=(0.03, 0.97),' in src:
        src = src.replace('xytext=(0.03, 0.97),', 'xytext=(0.02, 0.56),', 1)
        n_f1 += 1
    log.append("fig1 annotation cleared of the title: %d" % n_f1)

    # 5e. fig3/fig4: at 8 pt several titles are wider than their panel and run
    #     into the neighbouring panel label.  Shorten the titles (no content
    #     lost -- the captions carry the full description) and drop two text
    #     blocks clear of the data.
    #
    #     Four fig4 entries and the whole of rule 5f were retired on 2026-09-20
    #     when fig4_mechanism was rebuilt as a four-step chain: every one of them
    #     anchored on content that no longer exists (the two deleted panel titles,
    #     the correlation block of the old panel (a), the legend of the old panel
    #     (b) and its synthetic pair of Gaussian curves).  Retired rules would
    #     have gone on printing a success count for a substitution that never
    #     happened, so they are removed rather than left to log 0/N.
    layout_fix = [
        ('set_title("Tomography decomposes latent and extracted yield"',
         'set_title("Latent vs extracted yield"'),
        ('set_title("Scheduler preference remains under-resolved"',
         'set_title("Scheduler still under-resolved"'),
        # fig3(c): the Spearman block sat on the point labels
        ('ax2.text(0.05, 0.80, rf"Spearman', 'ax2.text(0.52, 0.10, rf"Spearman'),
    ]
    # The rule that shortened fig3(b)'s title was retired on 2026-09-20 with
    # that title: the panel was rebuilt around the 28-anchor harmonized cohort
    # and its new title already fits at 8 pt.
    n_lay = 0
    for a, b in layout_fix:
        if a in src:
            src = src.replace(a, b, 1)
            n_lay += 1
    log.append("fig3 layout fixes applied: %d/%d" % (n_lay, len(layout_fix)))

    # 6. figsize width clamp
    n_fg = len([1 for m in re.finditer(r"figsize=\(([0-9.]+),", src)
                if float(m.group(1)) > W_MAX])
    src = re.sub(r"figsize=\(([0-9.]+), ([0-9.]+)\)",
                 lambda m: "figsize=(%g, %g)" % (min(float(m.group(1)), W_MAX),
                                                 float(m.group(2))), src)
    log.append("figsize width clamped to %g: %d sites" % (W_MAX, n_fg))

    open(OUT, "w", encoding="utf-8").write(src)
    print("wrote", os.path.basename(OUT))
    for l in log:
        print("  " + l)


if __name__ == "__main__":
    main()
