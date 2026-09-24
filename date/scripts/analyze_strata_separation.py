"""Do the adjoint proxy's top and bottom strata separate from each other?

The matched-control experiment (P3/p3_null_control.py) compared the lowest- and
the highest-scoring strata of the adjoint ranking against random actions matched
to the same spatial bin and patch type.  The saved analysis JSON stores the
deep-vs-control paired test but NOT a deep-vs-best test, which is what the
manuscript's sentence about the two strata needs.  This script supplies it.

Reads the raw per-anchor records (never modifies them) and writes
results/null_control_strata.json.

Run:  python -u analyze_strata_separation.py
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "results", "p3_v1_large_sample", "null_control")
OUT = os.path.join(HERE, "results", "null_control_strata.json")

ANCHORS = ("seed2", "seed4", "seed7", "seed13")
N_PERM = 20000
SEED = 20260921
ALPHA = 0.05


def diff_of_means_perm(deep, best, rng, n=N_PERM):
    """Two-sided permutation test on mean(deep) - mean(best), unpaired.

    The two strata are disjoint sets of candidates drawn from the same
    anchor--size cell, so they are not paired by identity.
    """
    obs = sum(deep) / len(deep) - sum(best) / len(best)
    pool = list(deep) + list(best)
    n_a = len(deep)
    n_b = len(pool) - n_a
    cnt = 0
    for _ in range(n):
        rng.shuffle(pool)
        d = sum(pool[:n_a]) / n_a - sum(pool[n_a:]) / n_b
        if abs(d) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (n + 1)


def main():
    rng = random.Random(SEED)
    cells = []
    for a in ANCHORS:
        with open(os.path.join(RAW, a, "null_control.json"), encoding="utf-8") as f:
            rec = json.load(f)
        for sc, cell in sorted(rec["scales"].items(), key=lambda kv: int(kv[0])):
            deep = [s["dF"] for s in cell["deep_score"]["samples"]
                    if s.get("dF") is not None]
            best = [s["dF"] for s in cell["best_score"]["samples"]
                    if s.get("dF") is not None]
            ctrl = cell["location_matched"]["samples"]
            ctrl = [s["dF"] for s in ctrl if s.get("dF") is not None]
            ctrl_mean = sum(ctrl) / len(ctrl)
            md, mb = sum(deep) / len(deep), sum(best) / len(best)
            obs, p = diff_of_means_perm(deep, best, rng)
            cells.append(dict(
                anchor=a, patch_px=int(sc), n_deep=len(deep), n_best=len(best),
                mean_deep=md, mean_best=mb, mean_control=ctrl_mean,
                diff_deep_minus_best=obs, p_two_sided=p,
                strata_separated=(p <= ALPHA),
                top_significantly_above_bottom=(p <= ALPHA and obs < 0),
                bottom_significantly_above_top=(p <= ALPHA and obs > 0),
                # the score "separates actions from the background" when BOTH
                # strata sit above their matched controls
                both_strata_above_control=(md > ctrl_mean and mb > ctrl_mean)))

    n = len(cells)
    summary = dict(
        definition="per anchor--size cell, the lowest-scoring ('deep') and the"
                   " highest-scoring ('best') strata of the adjoint ranking,"
                   " compared with each other and with random actions matched to"
                   " the same spatial bin and patch type",
        n_cells=n,
        both_strata_above_control=sum(c["both_strata_above_control"] for c in cells),
        strata_not_separated=sum(1 for c in cells if not c["strata_separated"]),
        strata_separated=sum(c["strata_separated"] for c in cells),
        top_significantly_above_bottom=sum(
            c["top_significantly_above_bottom"] for c in cells),
        bottom_significantly_above_top=sum(
            c["bottom_significantly_above_top"] for c in cells),
        test="two-sided unpaired permutation on the difference of means",
        permutations=N_PERM, seed=SEED, alpha=ALPHA)

    # The manuscript quotes four of these; fail loudly rather than drift.
    assert n == 8, n
    assert summary["both_strata_above_control"] == 6, summary
    assert summary["strata_not_separated"] == 5, summary
    assert summary["strata_separated"] == 3, summary
    assert summary["top_significantly_above_bottom"] == 2, summary
    assert summary["bottom_significantly_above_top"] == 1, summary

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cells": cells}, f, indent=2)

    print("=== strata separation, per anchor--size cell ===")
    for c in cells:
        print("  %-7s @%2dpx  deep=%+.5f best=%+.5f ctrl=%+.5f  diff=%+.5f"
              "  p=%.3f  %s" % (
                  c["anchor"], c["patch_px"], c["mean_deep"], c["mean_best"],
                  c["mean_control"], c["diff_deep_minus_best"], c["p_two_sided"],
                  "separated" if c["strata_separated"] else "not separated"))
    print("\n  both strata above their controls : %d of %d"
          % (summary["both_strata_above_control"], n))
    print("  strata not separated             : %d of %d"
          % (summary["strata_not_separated"], n))
    print("  top significantly above bottom   : %d of %d"
          % (summary["top_significantly_above_bottom"], n))
    print("  bottom significantly above top   : %d of %d"
          % (summary["bottom_significantly_above_top"], n))
    print("  -> separated cells split in direction: %d vs %d"
          % (summary["top_significantly_above_bottom"],
             summary["bottom_significantly_above_top"]))
    print("\nwrote", os.path.relpath(OUT, HERE))


if __name__ == "__main__":
    main()
