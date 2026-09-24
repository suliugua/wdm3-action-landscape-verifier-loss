"""How well do the phase-0 statistics predict the measured regime?

The manuscript quotes three accuracies.  They come from an analysis note dated
2026-09-06 (results/P0_tnull_predictive.md) that covers the TWELVE anchors which
had a gate loss at that time, not the twenty-eight that carry a harmonized value
now.  This script reproduces the note's convention exactly and then re-derives
the same three accuracies on the current cohort, so the quoted numbers have a
resident home.

The note's convention, read off its own table and verified here:
  * the floor is evaluated at n = 9:  T_null(9) = mu*9 + 2*gamma*sigma*3
    (a check: seed1 -> -2.0167e-3 and seed4 -> +1.3875e-3 reproduce the note)
  * the three-way split uses z = mu/sigma, with z > 0.3 -> mismatch,
    z < -0.4 -> clean, in between -> gray
  * a hit means "predicted mismatch" == "actual mismatch", so predicting
    mismatch for a gray anchor counts as a miss
  * gamma prefers the field the runner used, falling back to sigma_5/(sigma_1*sqrt5)

The regime label is read from the harmonized loss with the cut the note's own
labels imply: mismatch above 0.5, clean at or below 0.1, gray in between.  The
script asserts this rule reproduces all twelve of the note's labels, so the
labelling is not a free choice.

No FDTD.  Read-only over results/; writes results/p0_predictor_accuracy.json.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

HARM = os.path.join(HERE, "results", "gate_loss_harmonized.json")
OUT = os.path.join(HERE, "results", "p0_predictor_accuracy.json")

NOTE_LABELS = {"seed1": "clean", "seed4": "clean", "seed2": "clean",
               "seed7": "clean", "seed3": "clean", "seed9": "mismatch",
               "seed13": "gray", "seed11": "mismatch", "seed0": "mismatch",
               "seed6": "mismatch", "seed5": "mismatch", "seed8": "mismatch"}


def diag(anchor):
    """The anchor's own phase-0 statistics, from the path the runner reads."""
    for p in (os.path.join(HERE, "results", "shared", anchor, "diagnosis.json"),
              os.path.join(HERE, "results", "selector", "selector_raw", anchor,
                           "budget200", "repeat0", "result.json")):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            return d.get("diagnosis", d)
    return None


def gamma_of(d):
    g = d.get("gamma")
    if g is None:
        g = d["sigma_5"] / (d["sigma_single"] * (5 ** 0.5))
    return g


def t_null_9(d):
    return d["mu_single"] * 9 + 2.0 * gamma_of(d) * d["sigma_single"] * 3.0


def label_from_loss(g):
    return "mismatch" if g > 0.5 else ("clean" if g <= 0.1 else "gray")


def score(rows):
    """(mu-sign, ternary, T_null-sign) hits, under the note's convention."""
    mu = sum(1 for r in rows if (r["mu"] > 0) == (r["label"] == "mismatch"))
    tern = sum(1 for r in rows if r["ternary"] == r["label"])
    tn = sum(1 for r in rows
             if (r["t_null_9"] > 0) == (r["label"] == "mismatch"))
    return mu, tern, tn


def build(anchors, harm):
    rows = []
    for a in sorted(anchors):
        d = diag(a)
        assert d is not None, "no phase-0 diagnosis on disk for %s" % a
        z = d["mu_single"] / d["sigma_single"]
        rows.append(dict(
            anchor=a, mu=d["mu_single"], sigma=d["sigma_single"],
            gamma=gamma_of(d), z=z,
            ternary=("mismatch" if z > 0.3 else ("clean" if z < -0.4 else "gray")),
            t_null_9=t_null_9(d), gate_loss=harm[a]["gate_loss_mass"],
            label=label_from_loss(harm[a]["gate_loss_mass"])))
    return rows


def main():
    harm = json.load(open(HARM, encoding="utf-8"))["anchors"]

    note_rows = build(NOTE_LABELS, harm)
    # the label rule must reproduce the note's own labels
    for r in note_rows:
        assert r["label"] == NOTE_LABELS[r["anchor"]], (r["anchor"], r["label"])
    note = score(note_rows)
    assert note == (11, 10, 8), note

    all_rows = build(harm, harm)
    cur = score(all_rows)
    by_label = {}
    for r in all_rows:
        by_label[r["label"]] = by_label.get(r["label"], 0) + 1

    summary = dict(
        cohort=("the anchors with a harmonized gate loss, each read from its own"
                " phase-0 diagnosis"),
        convention=("T_null at n=9; z = mu/sigma with 0.3/-0.4 cuts; a hit means"
                    " predicted-mismatch == actual-mismatch; regime read from the"
                    " harmonized loss (mismatch >0.5, clean <=0.1, gray between)"),
        note_cohort_n=len(note_rows),
        note_mu_sign_hits=note[0],
        note_ternary_hits=note[1],
        note_tnull_sign_hits=note[2],
        cohort_n=len(all_rows),
        n_mismatch=by_label.get("mismatch", 0),
        n_clean=by_label.get("clean", 0),
        n_gray=by_label.get("gray", 0),
        mu_sign_hits=cur[0],
        ternary_hits=cur[1],
        tnull_sign_hits=cur[2],
        note=("the note's twelve-anchor snapshot reproduces exactly under this"
              " convention, which is what licenses re-deriving the same three"
              " accuracies on the current cohort"))
    assert (len(note_rows), len(all_rows)) == (12, 28)
    assert cur == (24, 19, 18), cur
    # consistent with the manuscript's "9 at 0.0% and 12 at 100.0%, only 7
    # strictly between": clean = the 9 zeros + seed3 + seed26, mismatch = the
    # 12 at 100% + seed11/seed16/seed18, gray = seed13 and seed23
    assert by_label == {"mismatch": 15, "clean": 11, "gray": 2}, by_label

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "anchors": all_rows}, f, indent=2)

    print("=== phase-0 predictor accuracy ===")
    print("  note's 12-anchor snapshot, reproduced: mu-sign %d/12, ternary"
          " %d/12, T_null-sign %d/12" % note)
    print("  current %d-anchor cohort: mu-sign %d/%d, ternary %d/%d,"
          " T_null-sign %d/%d"
          % ((len(all_rows),) + (cur[0], len(all_rows), cur[1], len(all_rows),
                                 cur[2], len(all_rows))))
    print("  labels: %d mismatch, %d clean, %d gray"
          % (by_label.get("mismatch", 0), by_label.get("clean", 0),
             by_label.get("gray", 0)))
    print("\nwrote", os.path.relpath(OUT, HERE))


if __name__ == "__main__":
    main()
