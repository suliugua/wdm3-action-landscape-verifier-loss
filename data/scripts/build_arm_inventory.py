"""Build the low-gain arm inventory under the definition frozen on 2026-09-20.

Frozen in p2_arm_inventory_definition_2026-09-20.md.  Supersedes the
name-and-path-offset classification in build_arm_inventory_table.py, which
(S1) took the arm as path.split('/')[-4] although the tree has three path
depths, silently dropping 19 of 135 records, and (S2) classified by arm name.

Rules implemented here:
  R1  cell = (anchor, arm); repeats averaged; n_repeats recorded
  R2  the arm is resolved by walking back from the file, never by a fixed offset
  R3  threshold-rule cells require a data-derived floor that is actually
      positive on a probed patch size for THAT anchor
  R4  sign-gate cells require an acceptance stage that uses dJ > 0, and no
      positive floor on that cell
  R5  an arm whose operator differs between warm-up and main loop is `mixed`
      and enters neither denominator
  R6  every row records generation / operator / evidence / source / fresh /
      n_repeats / exclude reason
  R7  the population is the manuscript roster, listed explicitly; other
      lineages are excluded with a reason rather than silently dropped
  R8  writes arm_inventory.csv and arm_inventory_summary.json into the package

Run: python -u build_arm_inventory.py [--write]
"""
import argparse
import collections
import csv
import glob
import json
import os
import random
import sys

import numpy as np
from scipy import stats as _stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_pilot import compute_scaled_threshold  # noqa: E402

PKG = os.path.join(HERE, "paper2_data_package")
CSV_OUT = os.path.join(PKG, "data_tables", "arm_inventory.csv")
JSON_OUT = os.path.join(PKG, "data_tables", "arm_inventory_summary.json")
# the per-cell values, written under results/ so that the figure generator can
# read the very cells this audit defines.  The figure must not recompute them:
# a cell's family depends on the frozen rules, and a second copy of that logic
# is a copy that can drift from this one.
CELLS_OUT = os.path.join(HERE, "results", "arm_inventory_cells.json")

# ---------------------------------------------------------------------------
# R7 roster: the arms this manuscript's own tables and sections report.
# Everything else in the tree is out of population and is recorded, not dropped.
# ---------------------------------------------------------------------------
ROSTER = {
    # arm name : (nominal operator, evidence for that operator)
    "null_banded":          ("floor", "script: th=compute_scaled_threshold"),
    "null_static":          ("floor", "script: th=compute_scaled_threshold"),
    "null_gate_static":     ("floor", "record protocol=null_gate_static_top_patch"),
    "unified_protocol":     ("floor", "script: th=compute_scaled_threshold"),
    "portfolio":            ("floor", "script: th=compute_scaled_threshold"),
    "policy_switch":        ("floor", "script: if dF > threshold"),
    "positive_mu_boundary": ("floor", "script: if dF > threshold"),
    "dbs_null":             ("floor", "record experiment=dbs_null_gate"),
    "random_dbs":           ("floor", "record: gate_loss>0 on every cell"),
    "raw_static":           ("floor", "record: gate_loss>0 on every cell"),
    "static_top_patch":     ("floor", "record: gate_loss>0 on every cell"),
    "floor_k025":           ("floor", "script: accept iff dF > K*T_ref(n), K=0.25"),
    "floor_k050":           ("floor", "script: accept iff dF > K*T_ref(n), K=0.5"),
    "floor_k100":           ("floor", "script: accept iff dF > K*T_ref(n), K=1"),
    "floor_k200":           ("floor", "script: accept iff dF > K*T_ref(n), K=2"),
    "sign_banded":          ("sign",  "script: sign gate"),
    "sign_banded_unified":  ("sign",  "script: th=0.0, 'SIGN GATE'"),
    "sign_static":          ("sign",  "script: sign gate"),
    "positive_null_sign":   ("sign",  "record protocol=sign_gate_static_top_patch"),
    "dbs_sign":             ("sign",  "record experiment=dbs_sign_gate"),
    "shuffled_pool_sign":   ("sign",  "script: sign gate"),
    "static_top_patch_sign": ("sign", "script: accept if dJ > 0"),
    "verifier_ablation":    ("sign",  "script: Arm B sign gate; gate_loss=0"),
    "frozen_prospective":   ("sign",  "the frozen rule dispatches the sign gate"),
    "floor_k000":           ("sign",  "script: K=0 reproduces the sign gate"),
}

TREES = ("final_rerun_2026_08_13_protocol_v2",)


def resolve(posix):
    """R2: (tree, anchor, arm) by walking back from the file, not by offset."""
    parts = posix.split("/")
    # the glob may hand back an absolute path; anchor on the "results" component
    parts = parts[parts.index("results"):]
    core = parts[1:-1]
    while core and (core[-1].startswith("repeat") or core[-1].startswith("budget")
                    or core[-1] in ("v2", "r0", "r1", "r2")):
        core.pop()
    tree = core[0] if core else "?"
    rest = core[1:]
    if not rest or not rest[0].startswith("seed"):
        return tree, "?", (rest[0] if rest else "?")
    return tree, rest[0], (rest[1] if len(rest) > 1 else "?")


def realized_floor(anchor):
    """R3: is the data-derived floor positive at any probed patch size?"""
    p = os.path.join(HERE, "results", "shared", anchor, "diagnosis.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    return any(compute_scaled_threshold(n, d, z=2.0) > 0 for n in (9, 20, 25))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    meta = {}
    for p in glob.glob(os.path.join(HERE, "results", "**", "fresh_sim.json"),
                       recursive=True):
        posix = p.replace(os.sep, "/")
        if "/P4/" in posix or "_dryrun" in posix or "recon_scratch" in posix:
            continue
        d = json.load(open(p, encoding="utf-8"))
        fj = d.get("fresh_dJ")
        if not isinstance(fj, (int, float)):
            continue
        tree, anchor, arm = resolve(posix)
        if anchor == "?":
            # A cell is (anchor, arm).  A record whose anchor cannot be resolved
            # from its own path belongs to no anchor of this paper, so it must not
            # reach the inventory as a "?" row: the three such rows that shipped
            # in the 2026-09-20 package came from the p3_v1_large_sample tree.
            continue
        cells[(anchor, arm)][tree].append(fj)
        rj = os.path.join(os.path.dirname(p), "result.json")
        if os.path.exists(rj) and (anchor, arm) not in meta:
            meta[(anchor, arm)] = json.load(open(rj, encoding="utf-8"))

    rows = []
    for (anchor, arm), gens in sorted(cells.items()):
        entry = ROSTER.get(arm)
        # v2 record preferred; the legacy record is the fallback, per
        # data_provenance.md section 0 (one cell per anchor+arm, not per generation)
        if TREES[0] in gens:
            vals, gen = gens[TREES[0]], "current"
        elif gens:
            vals = [x for v in gens.values() for x in v]
            gen = "earlier"
        else:
            vals, gen = [], "none"
        if entry is None:
            fam, why = "excluded", "not a manuscript arm (R7)"
        elif not vals:
            fam, why = "excluded", "no fresh record"
        elif entry[0] == "sign":
            fam, why = "sign", ""
        else:
            rf = realized_floor(anchor)
            if rf is None:
                fam, why = "excluded", "no phase-0 diagnosis for this anchor"
            elif rf:
                fam, why = "threshold", ""
            else:
                fam, why = "excluded", "nominal floor, realized floor == 0 on this anchor (R3)"
        fresh = (sum(vals) / len(vals)) if vals else ""
        rj = meta.get((anchor, arm), {})
        rows.append(dict(
            anchor=anchor, arm=arm, family=fam, generation=gen,
            verifier_operator=(entry[0] if entry else ""),
            operator_evidence=(entry[1] if entry else ""),
            scheduler=str(rj.get("scheduler", "") or ""),
            fresh_value=(round(fresh, 6) if vals else ""), n_repeats=len(vals),
            exclude_reason=why))

    print("=== arm inventory, frozen definition 2026-09-20 ===")
    fams = collections.Counter(r["family"] for r in rows)
    print("cells by family:", dict(fams))
    for fam, label in (("threshold", "threshold rule (realized floor > 0)"),
                       ("sign", "sign gate (dJ > 0, no positive floor)")):
        ks = [r for r in rows if r["family"] == fam]
        below = [r for r in ks if r["fresh_value"] < 0.01]
        print("\n%-42s cells=%2d  below 0.01 = %d" % (label, len(ks), len(below)))
        for r in sorted(below, key=lambda z: z["fresh_value"]):
            print("     %-8s %-24s %.5f" % (r["anchor"], r["arm"], r["fresh_value"]))
    print("\n=== excluded ===")
    for r in rows:
        if r["family"] == "excluded":
            print("     %-8s %-24s %s" % (r["anchor"], r["arm"], r["exclude_reason"]))

    if args.write:
        os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
        with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        summary = {
            "definition": "measured policy-cell inventory; the rule family, the"
                          " operator evidence and the exclusion reason are given"
                          " per row in arm_inventory.csv",
            "frozen": "2026-09-20",
            "population": "the manuscript's own measured arms",
            "cells_by_family": dict(fams),
            "threshold_rule": {"cells": sum(1 for r in rows if r["family"] == "threshold"),
                               "below_0.01": sum(1 for r in rows if r["family"] == "threshold"
                                                 and r["fresh_value"] < 0.01)},
            "sign_gate": {"cells": sum(1 for r in rows if r["family"] == "sign"),
                          "below_0.01": sum(1 for r in rows if r["family"] == "sign"
                                            and r["fresh_value"] < 0.01)},
        }
        # The cells are not independent samples: several arms share an anchor,
        # and a reviewer will press exactly there once the contrast is a figure.
        # So the same contrast is reported with the anchor as the unit.
        per = collections.defaultdict(
            lambda: {"threshold": [0, 0], "sign": [0, 0]})
        for r in rows:
            if r["family"] not in ("threshold", "sign"):
                continue
            per[r["anchor"]][r["family"]][1] += 1
            if r["fresh_value"] < 0.01:
                per[r["anchor"]][r["family"]][0] += 1
        tf = [v for v in per.values() if v["threshold"][1]]
        sg = [v for v in per.values() if v["sign"][1]]
        t_low = sum(1 for v in tf if v["threshold"][0] > 0)
        s_low = sum(1 for v in sg if v["sign"][0] > 0)
        p_anchor = float(_stats.fisher_exact(
            [[t_low, len(tf) - t_low], [s_low, len(sg) - s_low]])[1])
        # cluster bootstrap: resample anchors, not cells, with a fixed seed so
        # the interval is a reproducible number and not a line in a work log
        keys = sorted(per)
        rng = random.Random(20260921)
        diffs = []
        for _ in range(20000):
            pick = [per[rng.choice(keys)] for _ in keys]
            T = sum(v["threshold"][1] for v in pick)
            G = sum(v["sign"][1] for v in pick)
            if T and G:
                diffs.append(sum(v["threshold"][0] for v in pick) / T
                             - sum(v["sign"][0] for v in pick) / G)
        lo, hi = (float(x) for x in np.percentile(diffs, [2.5, 97.5]))
        summary["anchor_level"] = {
            "note": "the cells are not independent, several arms share an"
                    " anchor; this block redoes the low-gain contrast with the"
                    " anchor as the unit",
            "floor_anchors": len(tf),
            "floor_anchors_with_a_low_gain_cell": t_low,
            "sign_anchors": len(sg),
            "sign_anchors_with_a_low_gain_cell": s_low,
            "fisher_p": round(p_anchor, 4),
            "cluster_bootstrap": {"resamples": 20000, "seed": 20260921,
                                  "rate_difference_ci95": [round(lo, 3),
                                                           round(hi, 3)]},
        }
        print("\nanchor level: %d of %d floor anchors and %d of %d sign anchors"
              " carry a low-gain cell (Fisher p=%.4f);"
              " cluster bootstrap CI [%+.3f, %+.3f]"
              % (t_low, len(tf), s_low, len(sg), p_anchor, lo, hi))
        json.dump(summary, open(JSON_OUT, "w", encoding="utf-8"), indent=2)
        # the measurable cells only; the figure plots these, not the exclusions
        json.dump(
            {"frozen": summary["frozen"],
             "note": "the cells of arm_inventory.csv that carry a measured"
                     " fresh value, with their rule family",
             "cells": [{"anchor": r["anchor"], "arm": r["arm"],
                        "family": r["family"], "fresh_value": r["fresh_value"]}
                       for r in rows if r["family"] in ("threshold", "sign")]},
            open(CELLS_OUT, "w", encoding="utf-8"), indent=2)
        print("\nwrote", os.path.relpath(CSV_OUT, HERE))
        print("wrote", os.path.relpath(JSON_OUT, HERE))
        print("wrote", os.path.relpath(CELLS_OUT, HERE))


if __name__ == "__main__":
    main()
