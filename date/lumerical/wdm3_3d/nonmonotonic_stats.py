#!/usr/bin/env python
"""Experiment E: Ranking non-monotonicity system statistics.

Zero-cost analysis — uses existing arm data from prospective_series and
prospective directories.

For each 3D anchor with arm data, computes:
  - best accepted rank (where max dF occurs)
  - rank-gain Spearman ρ
  - fraction of anchors where max_dF_rank > 1
  - accepted distribution across ranks
  - comparison to matched random null

Usage:
  python -m wdm3_3d.nonmonotonic_stats
"""

import json, sys
from pathlib import Path
import numpy as np

os_pkg = __import__("os")


def load_all_arm_data():
    """Scan known directories for arm JSON files."""
    base = Path(__file__).resolve().parent / "results_3d" / "landscape"
    arms = []

    for dname in ["prospective_series", "prospective"]:
        d = base / dname
        if not d.exists():
            continue
        for f in sorted(d.glob("arm_*.json")):
            try:
                data = json.load(open(f))
                arms.append(data)
            except Exception:
                pass

    return arms


def analyze(arms):
    """Compute ranking statistics across all arms."""
    results = []

    for arm in arms:
        rounds = arm.get("rounds", arm.get("draws", []))
        if not rounds:
            continue
        dFs = [r.get("dF_meas", r.get("dF", 0)) for r in rounds]
        accepted = [r.get("accepted", False) for r in rounds]
        ranks = list(range(1, len(dFs) + 1))

        max_i = int(np.argmax(dFs))
        max_rank = max_i + 1
        nonmonotonic = max_rank > 1

        # Rank-gain Spearman
        try:
            from scipy.stats import spearmanr
            rho, p = spearmanr(ranks, dFs)
        except Exception:
            rho, p = float("nan"), float("nan")

        results.append({
            "arm": arm.get("arm", "?"),
            "label": arm.get("label", "?"),
            "variant": arm.get("variant", "?"),
            "n_total": len(dFs),
            "n_accepted": sum(1 for a in accepted if a),
            "net_dF": float(np.sum([d for i, d in enumerate(dFs) if accepted[i]])),
            "max_dF": float(np.max(dFs)),
            "max_dF_rank": max_rank,
            "nonmonotonic": nonmonotonic,
            "spearman_rho": float(rho) if not np.isnan(rho) else None,
            "spearman_p": float(p) if not np.isnan(p) else None,
        })

    return results


def summarize(per_arm):
    """Cross-arm aggregation."""
    n_total = len(per_arm)
    n_nonmono = sum(1 for r in per_arm if r["nonmonotonic"])
    n_max_rank1 = n_total - n_nonmono

    # Per anchor: aggregate across arms
    anchors = {}
    for r in per_arm:
        label = r["label"]
        if label not in anchors:
            anchors[label] = {"nonmono_arms": 0, "total_arms": 0}
        anchors[label]["total_arms"] += 1
        if r["nonmonotonic"]:
            anchors[label]["nonmono_arms"] += 1

    n_anchors = len(anchors)
    n_anchors_nonmono = sum(1 for a in anchors.values()
                           if a["nonmono_arms"] > 0)

    summary = {
        "n_arms_total": n_total,
        "n_arms_nonmonotonic": n_nonmono,
        "fraction_nonmonotonic": n_nonmono / n_total if n_total else 0,
        "n_anchors": n_anchors,
        "n_anchors_with_nonmonotonic": n_anchors_nonmono,
        "per_anchor": anchors,
        "per_arm": per_arm,
    }

    print(f"\n{'='*60}")
    print(f"RANKING NON-MONOTONICITY STATISTICS")
    print(f"{'='*60}")
    print(f"  Arms analyzed: {n_total}")
    print(f"  Arms with max_dF at rank>1: {n_nonmono}/{n_total} "
          f"({n_nonmono/n_total*100:.0f}%)")
    print(f"  Anchors with >=1 nonmonotonic arm: "
          f"{n_anchors_nonmono}/{n_anchors}")
    print(f"\n  Per-arm details:")
    for r in per_arm:
        flag = "NONMONO" if r["nonmonotonic"] else "mono"
        print(f"  {r['label']:20s} {r['arm']:25s} "
              f"max_dF_rank={r['max_dF_rank']:2d}  "
              f"rho={r['spearman_rho']:+.3f}  {flag}")

    return summary


if __name__ == "__main__":
    arms = load_all_arm_data()
    per_arm = analyze(arms)
    summary = summarize(per_arm)

    out_dir = Path(__file__).resolve().parent / "results_3d" / "landscape"
    out_path = out_dir / "nonmonotonic_stats.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved → {out_path}")
