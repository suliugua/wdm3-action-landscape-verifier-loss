"""Emit the SI table comparing the two design axes (verifier vs scheduler) per anchor.

For every anchor that has at least one matched verifier pair AND at least one
matched scheduler pair, report the largest resolvable effect on each axis:

  verifier axis : max over schedulers of |fresh(sign,s) - fresh(null,s)|
  scheduler axis: max over verifiers  of |fresh(v,s1) - fresh(v,s2)|

Records are taken with a v2 preference (a current-protocol value supersedes an
earlier-generation value of the same arm, exactly as in audit_sign_vs_null_arms.py).
Nothing is typed by hand.

Run: python -u build_axis_comparison_table.py
"""
import glob
import json
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ARMS = {
    "positive_null_sign": ("sign", "static"), "sign_gate_replication": ("sign", "static"),
    "sign_static": ("sign", "static"), "static_top_patch_sign": ("sign", "static"),
    "static_top_patch": ("null", "static"), "null_static": ("null", "static"),
    "null_gate_static": ("null", "static"),
    "null_banded": ("null", "banded"), "null_gate_banded": ("null", "banded"),
    "positive_mu_boundary": ("null", "banded"), "unified_protocol": ("null", "banded"),
    "sign_banded": ("sign", "banded"), "verifier_ablation": ("sign", "banded"),
    "dbs_sign": ("sign", "dbs"), "dbs_null": ("null", "dbs"),
}


def collect():
    best = {}
    for path in glob.glob(os.path.join(PROJECT_DIR, "results", "**", "fresh_sim.json"),
                          recursive=True):
        q = path.replace(chr(92), "/")
        if any(s in q for s in ("/P4/", "_dryrun", "recon_scratch")):
            continue
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        fj = d.get("fresh_dJ")
        if not isinstance(fj, (int, float)):
            continue
        parts = q.split("/")
        a = d.get("anchor") or next(
            (x for x in parts if x.startswith("seed") and x[4:].isdigit()), None)
        arm = ""
        for i, x in enumerate(parts):
            if x.startswith("budget"):
                arm = parts[i - 1]
                break
        if not a or arm not in ARMS:
            continue
        v2 = "final_rerun_2026_08_13_protocol_v2" in q
        key = (a,) + ARMS[arm]
        old = best.get(key)
        if old is None or (v2 and not old[1]):
            best[key] = (float(fj), v2, arm)
    return best


def main():
    best = collect()
    by = {}
    for (a, v, s), (fj, _, arm) in best.items():
        by.setdefault(a, {})[(v, s)] = (fj, arm)

    print("%% ===== tab:si_axis_compare =====")
    print("%% rows: anchor & verifier-axis effect (scheduler) & scheduler-axis effect (verifier) & larger")
    for a in sorted(by, key=lambda x: int(x[4:]) if x[4:].isdigit() else 0):
        c = by[a]
        vs = [(abs(c[("sign", s)][0] - c[("null", s)][0]), s) for s in ("static", "banded", "dbs")
              if ("sign", s) in c and ("null", s) in c]
        ss = [(abs(c[(v, s1)][0] - c[(v, s2)][0]), v) for v in ("null", "sign")
              for s1, s2 in (("static", "banded"), ("static", "dbs"), ("banded", "dbs"))
              if (v, s1) in c and (v, s2) in c]
        # Keep seed0 visible because the main text quotes its verifier-only
        # contrast. It has no scheduler axis, so it is reported with N/A rather
        # than silently folded into the two-axis comparison.
        if a == "seed0" and vs and not ss:
            mv, sv = max(vs)
            print("$s$%s & %.4f (%s) & --- & verifier \\\\"
                  % (a[4:], mv, sv))
            continue
        if not vs or not ss:
            continue
        mv, sv = max(vs)
        ms, ssv = max(ss)
        who = "verifier" if mv > ms else "scheduler"
        print("$s$%s & %.4f (%s) & %.4f (%s) & %s \\\\"
              % (a[4:], mv, sv, ms, ssv, who))
    one_axis = []
    for a in sorted(by, key=lambda x: int(x[4:]) if x[4:].isdigit() else 0):
        c = by[a]
        has_v = any(("sign", s) in c and ("null", s) in c for s in ("static", "banded", "dbs"))
        has_s = len({s for _, s in c}) > 1
        if not (has_v and has_s) and a != "seed0":
            one_axis.append("%s (%s only)" % (a, "verifier" if has_v else
                                              "scheduler" if has_s else "neither"))
    print()
    print("%% not in the table (only one axis measurable): " + (", ".join(one_axis) or "none"))


if __name__ == "__main__":
    main()
