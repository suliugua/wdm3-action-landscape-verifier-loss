#!/usr/bin/env python
"""Offline re-adjudication with the FRESH post-round-1 jac (_g2).

Consumes the 29-pixel dF cache from the (stale-jac) adjudication run;
only pixels missing from the cache need new sims — reported, not run here.
Verdict criterion unchanged: top_mean > +2*sigma_5 AND bot_mean < -sigma_5.
Accepted round-1 pixels are excluded from candidate sets (their flip is an
undo, a trivial prediction, and for 4430 contaminated by epistasis).
"""

import json
from pathlib import Path

import numpy as np

from wdm3_3d.wdm3_3d_landscape import apply_col_mask, _col_of
from wdm3_3d.wdm3_3d_jac import GRAD_VARIANTS, _flip_gain

LAND = Path("wdm3_3d/results_3d/landscape")
JAC_DIR = Path("wdm3_3d/results_3d/jac3d")
SIGMA_5 = 1.767e-3
MASK = [(72, 79)]
N = 5
ROUND1_ACCEPTED = {4430, 144, 4530, 341}


def main():
    adj = json.load(open(LAND / "adjudication_seed7_P1@1550nm_r1.json"))
    cache = {int(k): v for k, v in adj["pixel_dF"].items()}
    jac = dict(np.load(JAC_DIR / "jac_seed7_P1@1550nm_g2.npz"))
    params = np.load(LAND / "greedy_seed7_P1@1550nm_r1_params.npz")["params"]
    a = complex(jac["a"])
    excl = apply_col_mask(len(params), MASK)

    print("=== Offline re-adjudication with FRESH jac (_g2) ===")
    print(f"arg(a) fresh = {np.degrees(np.angle(a)):.1f} deg  "
          f"(round-1 jac had -110.7)")
    print(f"cache: {len(cache)} measured pixels\n")

    verdicts, missing_all = {}, set()
    for variant in GRAD_VARIANTS:
        gain = _flip_gain(jac[f"grad_{variant}"], params)
        sel = np.where(excl, -np.inf, gain)
        top, bot = [], []
        for i in np.argsort(sel)[::-1]:
            if int(i) not in ROUND1_ACCEPTED and sel[int(i)] > 0:
                top.append(int(i))
            if len(top) == N:
                break
        sel_b = np.where(excl, np.inf, gain)
        for i in np.argsort(sel_b):
            if int(i) not in ROUND1_ACCEPTED and sel_b[int(i)] < 0:
                bot.append(int(i))
            if len(bot) == N:
                break

        t_meas = [cache.get(i) for i in top]
        b_meas = [cache.get(i) for i in bot]
        missing = [i for i, m in zip(top + bot, t_meas + b_meas) if m is None]
        missing_all |= set(missing)

        row = {"top": top, "bot": bot, "missing": missing}
        if not missing:
            tm, bm = float(np.mean(t_meas)), float(np.mean(b_meas))
            row.update({
                "top_mean": tm, "bot_mean": bm,
                "top_pos": sum(1 for d in t_meas if d > 0),
                "bot_neg": sum(1 for d in b_meas if d < 0),
                "ordinal_pass": bool(tm > 2 * SIGMA_5 and bm < -SIGMA_5),
            })
        verdicts[variant] = row

        cov = len(top + bot) - len(missing)
        print(f"{variant:10s} coverage {cov}/{len(top + bot)}")
        print(f"  top: " + " ".join(
            f"{i}(ix={_col_of(i)}{',?' if cache.get(i) is None else f',{cache[i]:+.1e}'})"
            for i in top))
        print(f"  bot: " + " ".join(
            f"{i}(ix={_col_of(i)}{',?' if cache.get(i) is None else f',{cache[i]:+.1e}'})"
            for i in bot))
        if not missing:
            print(f"  -> top_mean={row['top_mean']:+.3e} ({row['top_pos']}/5 pos)  "
                  f"bot_mean={row['bot_mean']:+.3e} ({row['bot_neg']}/5 neg)  "
                  f"=> {'PASS' if row['ordinal_pass'] else 'FAIL'}")
        else:
            print(f"  -> INCOMPLETE, needs {len(missing)} new sims: {missing}")
        print()

    if missing_all:
        print(f"total new sims needed for full verdict: {len(missing_all)}: "
              f"{sorted(missing_all)}")
    passing = [v for v in GRAD_VARIANTS
               if verdicts[v].get("ordinal_pass")]
    if passing:
        best = max(passing, key=lambda v: verdicts[v]["top_mean"])
        print(f"provisional adjudication: {best} "
              f"(passing: {passing})")
    with open(LAND / "readjudication_g2_offline.json", "w") as f:
        json.dump({"arg_a_deg": float(np.degrees(np.angle(a))),
                   "verdicts": verdicts,
                   "missing": sorted(missing_all)}, f, indent=2)
    print("saved readjudication_g2_offline.json")


if __name__ == "__main__":
    main()
