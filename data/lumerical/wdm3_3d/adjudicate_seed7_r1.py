#!/usr/bin/env python
"""One-off: 4-variant flip-response adjudication at seed7/P1 post-round-1.

Triggered by round-1 phase escalation (hit@5=0.2 < 0.6).  Protocol (user
2026-07-16): same baseline, same mask, same K policy; verdict is ORDINAL
(top mean above the random 2-sigma band AND bottom mean below -1 sigma),
not predicted magnitude.  Hard boundary: this adjudication + at most one
more greedy round, then seed7 stops regardless.
"""

import json
import time
from pathlib import Path

import numpy as np

# Import order matters: wdm3_3d package sets LUMOPT_BACKEND=2026 at import
# time — lum_backend must come AFTER (the 5-min timeout run on 2026-07-16
# imported lum_backend first and silently got the 2024 backend).
from wdm3_3d.wdm3_3d_landscape import (  # noqa: F401 — sets backend env
    IncrementalSession,
    apply_col_mask,
    _col_of,
)
from wdm3_3d.wdm3_3d_jac import GRAD_VARIANTS, _flip_gain, _now_iso, _save_json

import lum_backend  # noqa: F401
import _lumopt_compat  # noqa: F401

LAND = Path("wdm3_3d/results_3d/landscape")
JAC_DIR = Path("wdm3_3d/results_3d/jac3d")

N_TOP, N_BOT = 5, 5
SIGMA_5 = 1.767e-3          # random 5-flip std, 25 draws, mask [(72,79)]
MASK = [(72, 79)]


def main():
    g1 = json.load(open(LAND / "greedy_seed7_P1@1550nm_r1.json"))
    params = np.load(LAND / g1["params_npz"])["params"]
    jac = dict(np.load(JAC_DIR / "jac_seed7_P1@1550nm_g1.npz"))
    a = complex(jac["a"])

    excl = apply_col_mask(len(params), MASK)

    print("=== 4-variant flip-response adjudication (seed7 post-round-1) ===")
    print(f"post-round-1 |a|^2 ~ {g1['end_abs2']:.6f}  "
          f"arg(a)={np.degrees(np.angle(a)):.1f} deg")
    print(f"mask={MASK}  jac=round-1 (4 accepted flips stale — recorded)")
    print(f"random 5-flip 2sig band: +/-{2 * SIGMA_5:.3e}\n")

    # Build each variant's probe list first, then measure with a shared
    # per-pixel cache — variants overlap heavily in their top/bottom sets.
    plans = {}
    for variant in GRAD_VARIANTS:
        gain = _flip_gain(jac[f"grad_{variant}"], params)
        sel = np.where(excl, -np.inf, gain)
        top_idx = [int(i) for i in np.argsort(sel)[::-1][:N_TOP]]
        sel_b = np.where(excl, np.inf, gain)
        bot_idx = [int(i) for i in np.argsort(sel_b)[:N_BOT]]
        plans[variant] = {"gain": gain, "top": top_idx, "bot": bot_idx}
    all_idx = sorted({i for p in plans.values() for i in p["top"] + p["bot"]})
    print(f"unique probe pixels: {len(all_idx)} "
          f"(vs {4 * (N_TOP + N_BOT)} without dedup)\n")

    cache = {}
    t0 = time.time()
    with IncrementalSession(params, 1, tag="adj4v_seed7") as S:
        f0 = S.run_baseline()
        print(f"baseline |a|^2 = {f0:.6f}\n", flush=True)
        for n, idx in enumerate(all_idx, start=1):
            t_e = time.time()
            cache[idx] = float(S.measure([idx]) - f0)
            print(f"  [{n}/{len(all_idx)}] idx={idx} (ix={_col_of(idx)}) "
                  f"dF={cache[idx]:+.3e} ({time.time() - t_e:.0f}s)", flush=True)
    wall_s = time.time() - t0

    print("\n=== Verdict (ordinal: top_mean > +2sig AND bot_mean < -1sig) ===")
    results, adjudicated = {}, None
    for variant in GRAD_VARIANTS:
        p = plans[variant]
        top_dFs = [cache[i] for i in p["top"]]
        bot_dFs = [cache[i] for i in p["bot"]]
        r = {
            "top_idx": p["top"], "bot_idx": p["bot"],
            "top_dFs": top_dFs, "bot_dFs": bot_dFs,
            "top_mean": float(np.mean(top_dFs)),
            "bot_mean": float(np.mean(bot_dFs)),
            "top_pos": int(sum(1 for d in top_dFs if d > 0)),
            "bot_neg": int(sum(1 for d in bot_dFs if d < 0)),
        }
        r["top_ok"] = bool(r["top_mean"] > 2 * SIGMA_5)
        r["bot_ok"] = bool(r["bot_mean"] < -SIGMA_5)
        r["ordinal_pass"] = bool(r["top_ok"] and r["bot_ok"])
        results[variant] = r
        print(f"  {variant:10s}: top={r['top_mean']:+.3e} ({r['top_pos']}/5 pos, "
              f">+2sig? {r['top_ok']})  bot={r['bot_mean']:+.3e} "
              f"({r['bot_neg']}/5 neg, <-1sig? {r['bot_ok']})  "
              f"=> {'PASS' if r['ordinal_pass'] else 'FAIL'}")

    passing = [v for v in GRAD_VARIANTS if results[v]["ordinal_pass"]]
    if passing:  # strongest top separation wins among passers
        adjudicated = max(passing, key=lambda v: results[v]["top_mean"])

    out = {
        "label": "seed7_P1@1550nm", "after_round": 1,
        "post_round_abs2": g1["end_abs2"],
        "baseline_abs2_session": f0,
        "jac_staleness": "round-1 jac, 4 accepted flips stale",
        "timestamp": _now_iso(),
        "mask": [list(m) for m in MASK],
        "sigma_5": SIGMA_5,
        "arg_a_deg": float(np.degrees(np.angle(a))),
        "unique_probes": len(all_idx),
        "pixel_dF": {str(k): v for k, v in cache.items()},
        "variants": results,
        "adjudicated_variant": adjudicated,
        "wall_s": wall_s,
    }
    _save_json(out, LAND / "adjudication_seed7_P1@1550nm_r1.json")
    verdict = adjudicated or "NONE - ranking horizon exhausted at this anchor position"
    print(f"\nadjudicated: {verdict}")
    print(f"wall: {wall_s:.0f}s")


if __name__ == "__main__":
    main()
