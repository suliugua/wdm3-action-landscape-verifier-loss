#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harmonized gate_loss table with a FROZEN source-selection rule (2026-09-17).

Why this exists
---------------
The gate_loss values used in the paper were assembled over many months from
several arms.  A naive scan of `results/**` picks up, for the same anchor,
trajectories from different arms and from the pre-v2 `exp6_*` era.  For seed5
the choice is not innocent: the `verifier_ablation` arm gives 0.000 while the
null-gate arms give 1.000 -- i.e. an undeclared selection rule can flip a
published label.

This script fixes ONE rule and applies it to every anchor.

Frozen selection rule (priority order, first hit wins)
-----------------------------------------------------
  1. results/final_rerun_2026_08_13_protocol_v2/<anchor>/null_gate_probe_v2/trajectory.json
  2. results/final_rerun_2026_08_13_protocol_v2/<anchor>/unified_protocol/budget200/repeat0/trajectory.json
  3. results/final_rerun_2026_08_13_protocol_v2/<anchor>/frozen_prospective/trajectory.json
Pre-v2 trees (`results/exp6*`, `results/exp6_policy*`) are EXCLUDED.

The metric definition is NOT restated here: it is imported verbatim from
`analyze_sigma_leg.gate_loss_from_trajectory` (naming-agnostic; regression 9/9 PASS).

Output: results/gate_loss_harmonized.json  (+ printed SHA-256 of the payload)

Usage: python -u build_harmonized_gate_loss.py
"""
import hashlib
import json
import os
import re
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_sigma_leg import gate_loss_from_trajectory  # noqa: E402

PROJ = os.path.dirname(os.path.abspath(__file__))
V2_REL = os.path.join("results", "final_rerun_2026_08_13_protocol_v2")

# (relative path under V2/<anchor>/, arm label)
PRIORITY = [
    (os.path.join("null_gate_probe_v2", "trajectory.json"), "null_gate_probe_v2"),
    (os.path.join("unified_protocol", "budget200", "repeat0", "trajectory.json"), "unified_protocol"),
    (os.path.join("frozen_prospective", "trajectory.json"), "frozen_prospective"),
]

# Anchors excluded from the harmonized table, with the reason. Explicit so that
# a reader is never left guessing why a published anchor is missing.
EXCLUDED = {
    "seed10": ("no source in the v2 tree; the only available probe trajectory is the "
               "pre-v2 `results/exp6_policy/seed10/null_gate_probe/trajectory.json`, which "
               "is outside the frozen selection rule. Excluded from the harmonized table. "
               "n_pos = 0, so the anchor was already uninformative in every analysis."),
}


def discover_anchors():
    out = []
    for p in glob.glob(os.path.join(PROJ, V2_REL, "seed*")):
        m = re.fullmatch(r"seed(\d+)", os.path.basename(p))
        if m:
            out.append(("seed" + m.group(1), int(m.group(1))))
    return [a for a, _ in sorted(out, key=lambda t: t[1])]


def harmonize():
    rows = {}
    for anchor in discover_anchors():
        for rel, arm in PRIORITY:
            path = os.path.join(PROJ, V2_REL, anchor, rel)
            if not os.path.exists(path):
                continue
            r = gate_loss_from_trajectory(path)
            if r is None:
                continue
            rows[anchor] = {
                "arm": arm,
                "source_path": os.path.relpath(path, PROJ).replace(os.sep, "/"),
                "n_probe": r["n"],
                "n_pos": r["n_pos"],
                "n_accept": r["n_accept"],
                "Y_pos": r["Y_pos"],
                "Y_gate": r["Y_gate"],
                "gate_loss_mass": r["gate_loss_mass"],
                "gate_loss_count": (1.0 - r["n_accept"] / r["n_pos"]) if r["n_pos"] else 0.0,
            }
            break
    return rows


def payload_hash(rows):
    """SHA-256 over the frozen per-anchor row format (anchor-sorted)."""
    parts = []
    for a in sorted(rows):
        r = rows[a]
        parts.append("%s|%s|%d|%d|%.10e|%.10e|%.6f" % (
            a, r["arm"], r["n_pos"], r["n_accept"],
            r["Y_pos"], r["Y_gate"], r["gate_loss_mass"]))
    return hashlib.sha256(";".join(parts).encode()).hexdigest()


def main():
    rows = harmonize()
    h = payload_hash(rows)

    print("=" * 104)
    print("HARMONIZED gate_loss table (frozen source rule)")
    print("=" * 104)
    print(f"{'anchor':<8}{'arm':<22}{'n':>3}{'n_pos':>6}{'n_acc':>6}"
          f"{'Y_pos':>12}{'Y_gate':>12}{'g_mass':>9}{'g_count':>9}")
    for a in sorted(rows, key=lambda s: int(s[4:])):
        r = rows[a]
        print(f"{a:<8}{r['arm']:<22}{r['n_probe']:>3}{r['n_pos']:>6}{r['n_accept']:>6}"
              f"{r['Y_pos']:>12.4e}{r['Y_gate']:>12.4e}"
              f"{r['gate_loss_mass']:>9.3f}{r['gate_loss_count']:>9.3f}")

    print()
    print(f"anchors with a v2-tree source : {len(rows)}")
    print(f"excluded                      : {list(EXCLUDED)}")
    for a, why in EXCLUDED.items():
        print(f"    {a}: {why}")
    print(f"payload SHA-256               : {h}")

    out = {
        "schema": "p2_gate_loss_harmonized",
        "frozen": "2026-09-17",
        "metric_source": "analyze_sigma_leg.gate_loss_from_trajectory (frozen definition)",
        "selection_rule": [
            V2_REL.replace(os.sep, "/") + "/" + PRIORITY[0][0].replace(os.sep, "/"),
            V2_REL.replace(os.sep, "/") + "/" + PRIORITY[1][0].replace(os.sep, "/"),
            V2_REL.replace(os.sep, "/") + "/" + PRIORITY[2][0].replace(os.sep, "/"),
        ],
        "excluded_trees": ["results/exp6*", "results/exp6_policy*"],
        "excluded_anchors": EXCLUDED,
        "payload_sha256": h,
        "anchors": rows,
    }
    out_path = os.path.join(PROJ, "results", "gate_loss_harmonized.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
