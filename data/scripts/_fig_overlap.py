"""Detect overlapping artists inside a figure, by bounding box.

Read-only. Eyeballing a downsampled preview is unreliable: the fig1 alignment
error was 0.029 figure-fraction and read as "looks fine" twice. This draws the
figure, then reports every pair of text/legend artists whose boxes intersect,
and every legend box that contains a data point.

The artist walk is deliberately wider than ax.get_children(): axis labels, tick
labels and the axes titles are children of the Axis objects, not of the Axes,
so a get_children() walk reports no overlap even when a panel letter sits on top
of the y label.  Verified by construction: the panel-letter/y-label collision in
fig3 was invisible to the narrow walk.

Run: python -u _fig_overlap.py [figure_fn_name]
"""
import itertools
import numpy as np
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.text as mtext
from matplotlib.legend import Legend
from matplotlib.transforms import Bbox

import make_figures_oj as M

NAME = sys.argv[1] if len(sys.argv) > 1 else "fig4_mechanism"


def text_box(art, r):
    """Bounding box of the TEXT only.

    Annotation.get_window_extent returns the union of the text and the arrow,
    so for an arrow that runs down-and-left of its label the union rectangle
    covers empty space between the two and reports overlaps that are not
    visible. Use the inherited Text implementation instead.
    """
    if isinstance(art, mtext.Annotation):
        return mtext.Text.get_window_extent(art, renderer=r)
    return art.get_window_extent(renderer=r)


def drawn_tick_labels(ax):
    """Tick labels matplotlib actually draws.

    ax.get_xticklabels() also returns labels for ticks outside the view limits
    (e.g. a 0.14 tick on an axis that stops at 0.134).  Those are never drawn,
    but they keep a stale bounding box that overlaps panel letters and produces
    reported collisions with nothing in them.  Verified by cropping the render:
    the "0.14", "0.06" and "125" reports were blank.
    """
    out = []
    for axis, view in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
        lo, hi = min(view), max(view)
        for tick in axis.get_major_ticks():
            loc = tick.get_loc()
            if loc is None or not (lo - 1e-12 <= loc <= hi + 1e-12):
                continue
            for lab in (tick.label1, tick.label2):
                if lab is not None:
                    out.append(lab)
    return out


def candidates(fig):
    """Every text-bearing artist the renderer will actually draw.

    ax.get_children() alone is NOT enough: the axis labels and the tick labels
    hang off ax.xaxis / ax.yaxis (and the axes title off ax.title), so a walk
    over get_children() silently skips them.  A panel letter placed at negative
    axes coordinates lands exactly on the y label, and that collision went
    unreported until this walk was widened.
    """
    arts = list(fig.texts)
    for ax in fig.axes:
        arts += list(ax.get_children())
        arts += [ax.xaxis.label, ax.yaxis.label, ax.title]
        arts += drawn_tick_labels(ax)
    return arts


def boxes(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for art in candidates(fig):
        if not isinstance(art, (mtext.Text, mtext.Annotation, Legend)):
            continue
        if not art.get_visible():
            continue
        try:
            bb = text_box(art, r)
        except Exception:
            continue
        label = art.get_text() if hasattr(art, "get_text") else "legend"
        if label:
            items.append((type(art).__name__, str(label)[:38], bb))
    return items


def data_points(fig):
    """Every data point that could be covered by a legend.

    Scatter offsets alone are not enough: the threshold panels draw curves with
    plot/fill_between, and a legend sitting on a curve is exactly the overlap a
    reader notices. Lines are therefore sampled and transformed to display
    coordinates so they can be tested point by point.
    """
    pts = []
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        for art in ax.get_children():
            if not art.get_visible():
                continue
            if hasattr(art, "get_offsets"):
                try:
                    xy = np.asarray(art.get_offsets())
                except Exception:
                    continue
                if len(xy):
                    pts.append((ax, xy, "series n=%d" % len(xy)))
        for ln in ax.lines:
            if not ln.get_visible():
                continue
            xd, yd = ln.get_xdata(), ln.get_ydata()
            if xd is None or yd is None or len(xd) == 0:
                continue
            try:
                xy = ax.transData.transform(np.column_stack([xd, yd]))
            except Exception:
                continue
            pts.append((ax, np.asarray(xy), "line n=%d" % len(xd)))
    return pts


def safe(s):
    """Console-safe label text.

    Tick labels and titles routinely contain non-ASCII glyphs (U+2212 minus,
    math italic).  On a GBK console a bare print raises UnicodeEncodeError and
    the run dies part-way through, hiding every overlap after that point.
    """
    return "".join(c if ord(c) < 128 else "?" for c in s.replace("\n", "/"))


def overlaps(fig, min_px=1.5):
    """Pairs of drawn text artists whose boxes intersect by more than min_px."""
    out = []
    for (t1, l1, b1), (t2, l2, b2) in itertools.combinations(boxes(fig), 2):
        if not b1.overlaps(b2):
            continue
        ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
        oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
        if ox < min_px or oy < min_px:
            continue
        out.append((ox, oy, t1, l1, t2, l2, Bbox.union([b1, b2])))
    return out


def report(fig, name):
    items = boxes(fig)
    print("=== %s: %d text/legend artists ===" % (name, len(items)))
    ov = overlaps(fig)
    for ox, oy, t1, l1, t2, l2, _ in ov:
        print("  OVERLAP %5.1f x %5.1f px : %-18s %-38s | %-18s %s"
              % (ox, oy, safe(t1), safe(l1), safe(t2), safe(l2)))
    print("  total text-text overlaps: %d" % len(ov))

    print("=== legends covering data ===")
    m = 0
    for typ, lab, bb in items:
        if typ != "Legend":
            continue
        for ax, xy, cnt in data_points(fig):
            inside = ((xy[:, 0] >= bb.x0) & (xy[:, 0] <= bb.x1) &
                      (xy[:, 1] >= bb.y0) & (xy[:, 1] <= bb.y1))
            k = int(inside.sum())
            if k:
                m += 1
                print("  legend covers %d point(s) of %s" % (k, cnt))
    if not m:
        print("  none")
    raise SystemExit(0)


if __name__ == "__main__":
    M.finish = report
    getattr(M, NAME)()

