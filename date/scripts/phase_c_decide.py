"""Phase C: compute Delta and apply the frozen early-stop rule.

Frozen by the exec doc sec. 2d, before the first run:

    Delta_a = fresh(sign_banded_unified) - fresh(unified_protocol)

    read s7, s2, s3 once, then judge:
      if Delta_s7 >= 0 and Delta_s2 >= 0 and Delta_s3 >= 0
          -> STOP.  The current-generation criterion set contains only the
             observed rows in this script.  Historical s5/s9/s13 rows are not
             silently counted because their arm names/protocol generations differ.
      else
      -> continue with s1, s4; the current-generation criterion set is the
         observed rows after those runs complete.

    The threshold is 0, not a noise band.  It must not be changed afterwards.

Exit code: 0 = early stop hit (do not run s1/s4), 1 = continue, 2 = cannot judge.

Run:  python phase_c_decide.py --stage earlystop
      python phase_c_decide.py --stage final
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "results", "final_rerun_2026_08_13_protocol_v2")
NEW = "sign_banded_unified"
OLD = "unified_protocol"
DECIDE = ["seed7", "seed2", "seed3"]
EXTRA = ["seed1", "seed4"]
OUT = os.path.join(HERE, "results", "phase_c_decision.json")
CRITERION_BASIS = (
    "same-generation paired fresh rows only: "
    "sign_banded_unified minus unified_protocol"
)
EXCLUSIONS = {
    "seed5": "no sign_banded/unified same-generation pair",
    "seed9": "legacy/cross-generation arm names; descriptive only",
    "seed13": "legacy/cross-generation arm names; descriptive only",
}


def arm_fresh(anchor, arm):
    p = os.path.join(BASE, anchor, arm, "budget200", "repeat0", "fresh_sim.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("fresh_dJ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["earlystop", "final"], default="earlystop")
    args = ap.parse_args()

    rows, missing = [], []
    for a in DECIDE + EXTRA:
        n, o = arm_fresh(a, NEW), arm_fresh(a, OLD)
        if n is None or o is None:
            rows.append((a, n, o, None))
            missing.append(a)
        else:
            rows.append((a, n, o, n - o))

    print("%-8s %14s %14s %12s" % ("anchor", "sign+banded", "null+banded", "Delta"))
    for a, n, o, d in rows:
        if d is None:
            print("%-8s %14s %14s %12s" % (a, "--", "--", "not run"))
        else:
            print("%-8s %14.6f %14.6f %+12.6f" % (a, n, o, d))

    have = [r for r in rows if r[3] is not None]
    if have:
        ds = [r[3] for r in have]
        print()
        print("n=%d  mean Delta = %+.6f   >=0: %d   <0: %d"
              % (len(ds), sum(ds) / len(ds), sum(1 for d in ds if d >= 0),
                 sum(1 for d in ds if d < 0)))

    block = {
        "stage": args.stage,
        "rows": [{"anchor": a, "sign": n, "null": o, "delta": d}
                 for a, n, o, d in rows],
        "criterion_basis": CRITERION_BASIS,
        "criterion_anchors": [r[0] for r in have],
        "exclusions": EXCLUSIONS,
    }
    if have:
        ds = [r[3] for r in have]
        mean_delta = sum(ds) / len(ds)
        if len(ds) > 1:
            sd_delta = (
                sum((d - mean_delta) ** 2 for d in ds) / (len(ds) - 1)
            ) ** 0.5
        else:
            sd_delta = None
        block["summary"] = {
            "n": len(ds),
            "mean_delta": mean_delta,
            "sd_delta": sd_delta,
            "n_positive": sum(1 for d in ds if d >= 0),
            "n_negative": sum(1 for d in ds if d < 0),
        }

    if args.stage == "earlystop":
        decide = [r for r in rows if r[0] in DECIDE]
        if any(r[3] is None for r in decide):
            print("\nCANNOT JUDGE: s7/s2/s3 not all freshened yet")
            block["decision"] = "cannot_judge"
            rc = 2
        elif all(r[3] >= 0 for r in decide):
            print("\nEARLY STOP HIT (all of s7/s2/s3 have Delta >= 0)")
            print("=> current-generation criterion set n = %d" % len(decide))
            block["decision"] = "early_stop_hit"
            block["criterion_n"] = len(decide)
            rc = 0
        else:
            print("\nEARLY STOP MISSED (at least one of s7/s2/s3 has Delta < 0)")
            print("=> continue with s1/s4; final n is derived from observed paired rows")
            block["decision"] = "early_stop_missed"
            block["criterion_n"] = len(decide)
            rc = 1
    else:
        block["decision"] = "final"
        block["criterion_n"] = len(have)
        print("\ncurrent-generation n = %d" % block["criterion_n"])
        rc = 0

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(block, f, indent=2)
    print("written: results/phase_c_decision.json")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
