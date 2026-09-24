"""Report each panel's box position in figure fraction for fig1.

Read-only. The panels' left edges are what must line up, and they cannot be
judged from a downsampled preview; this prints the numbers that `add_subplot`
actually produced, so the alignment is checked rather than eyeballed.

Run: python -u _fig_positions.py
"""
import make_figures_oj as M


def report(fig, name):
    print(name)
    for i, ax in enumerate(fig.axes):
        p = ax.get_position()
        print("  ax%d  left=%.4f bottom=%.4f width=%.4f height=%.4f"
              % (i, p.x0, p.y0, p.width, p.height))
    raise SystemExit(0)


M.finish = report
M.fig1_regime_map()
