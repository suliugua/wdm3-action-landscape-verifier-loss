#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Budget-resolution driver: extend the published 20-step probe to N steps.

Frozen by `preregistration_budget_resolution_2026-09-17.md`.

Intervention = ONE constant: probe budget 20 -> N. Anchor, wavelength, protocol,
z, warmup rule and **the RNG seed** are unchanged, so this is a PURE APPEND to the
published trajectory.

Implementation note (deviation from the prereg's wording, recorded in the prereg
as AMENDMENT #1): the prereg names `repeat_probe_driver.py --probe-budget 80`.
That file is the instrument of the already-published W2 result, so it is left
untouched; this module runs the same published protocol code path
(`probe_only_null_gate_v2`) with a larger target.

Pre-gate (prereg section 3, mandatory)
--------------------------------------
The first 20 probe steps must be bit-identical to the published trajectory:
index-hash sequence AND (dJ, accept) sequence AND the recomputed
gate_loss_mass(20). Any mismatch -> `invalid_provenance`, STOP, no conclusion.

Guards
------
- Refuses to write inside `results/final_rerun_2026_08_13_protocol_v2/**` (the
  provenance source of the published values) or `results/repeat_probe/**` (W2).
- Verifies the harmonized payload SHA-256 before any FDTD starts.

Usage
-----
    python -u budget_resolution_driver.py --anchor seed11 --check-only   # no FDTD
    python -u budget_resolution_driver.py --anchor seed11                # 80 steps
"""
import argparse
import json
import os
import sys
import time

import numpy as np

# Reuse the W2 driver's guards/loaders -- importing it has no side effects
# (its main() is under __main__), and the shared logic must not be copied.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repeat_probe_driver import (  # noqa: E402
    PROJECT_DIR, PARENT_DIR, EXPECTED_PAYLOAD_SHA, PROTECTED_TREE,
    load_frozen_diag, load_harmonized_row,
)

from config import RESULTS_BASE, ANCHORS, RNG_BASE_SEED  # noqa: E402
from budget import BudgetLedger  # noqa: E402
from run_pilot import Session  # noqa: E402
from shared import Trajectory  # noqa: E402

from probe_only_null_gate_v2 import (  # noqa: E402
    _hash, build_strata, run_density_probe, run_warmup,
    WARMUP_MAX_EVAL, WARMUP_MIN_ACCEPTS, PROBE_BUDGET,
)

DEFAULT_PROBE_BUDGET = 80
OUT_ROOT = os.path.join(PROJECT_DIR, "results", "budget_resolution")
WRITE_BAN = [
    os.path.join(PROJECT_DIR, PROTECTED_TREE),
    os.path.join(PROJECT_DIR, "results", "repeat_probe"),
]
MARGIN = 0.15          # prereg section 5.1 / 5.2
CURVE = (20, 40, 60, 80)


def prod_steps(path, n):
    """First n probe-phase steps of a trajectory file."""
    with open(path, encoding="utf-8") as f:
        traj = json.load(f)
    out = []
    for s in traj["steps"]:
        t = s.get("type", "")
        if "probe" in t and "warmup" not in t:
            out.append(s)
            if len(out) >= n:
                break
    return out


def step_hashes(steps):
    return [_hash(s.get("indices", [])) for s in steps]


def is_accept(t):
    return ("accept" in t) or t.endswith("_a")


def band_coverage(steps):
    cov = {}
    for s in steps:
        g = s.get("group", "?")
        cov[g] = cov.get(g, 0) + 1
    return cov


def guard_out_dir(anchor):
    out_dir = os.path.abspath(os.path.join(OUT_ROOT, anchor))
    for banned in WRITE_BAN:
        banned = os.path.abspath(banned)
        if os.path.commonpath([out_dir, banned]) == banned:
            raise SystemExit(f"refusing to write inside {banned}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def pre_gate(pub_path, rep_path):
    """Bit-identical check of the first 20 probe steps (prereg section 3)."""
    from analyze_sigma_leg import gate_loss_from_trajectory

    ps, rs = prod_steps(pub_path, PROBE_BUDGET), prod_steps(rep_path, PROBE_BUDGET)
    detail = {
        "n_pub": len(ps), "n_rep": len(rs),
        "index_hash_identical": step_hashes(ps) == step_hashes(rs),
        "dJ_max_abs_diff": None,
        "accept_seq_identical": None,
        "gate_loss_20_equal": None,
        "pub_gate_loss_20": None, "rep_gate_loss_20": None,
    }
    if len(ps) != len(rs) or not ps:
        return False, detail

    d = [abs(float(a.get("dJ", 0.0)) - float(b.get("dJ", 0.0))) for a, b in zip(ps, rs)]
    detail["dJ_max_abs_diff"] = max(d)
    detail["accept_seq_identical"] = (
        [is_accept(s.get("type", "")) for s in ps]
        == [is_accept(s.get("type", "")) for s in rs])
    gp = gate_loss_from_trajectory(pub_path, n_probe=PROBE_BUDGET)
    gr = gate_loss_from_trajectory(rep_path, n_probe=PROBE_BUDGET)
    detail["pub_gate_loss_20"] = gp["gate_loss_mass"]
    detail["rep_gate_loss_20"] = gr["gate_loss_mass"]
    detail["gate_loss_20_equal"] = abs(gp["gate_loss_mass"] - gr["gate_loss_mass"]) < 1e-12

    ok = (detail["index_hash_identical"]
          and detail["accept_seq_identical"]
          and max(d) < 1e-12
          and detail["gate_loss_20_equal"])
    return ok, detail


def curve(rep_path):
    from analyze_sigma_leg import gate_loss_from_trajectory
    out = {}
    for n in CURVE:
        r = gate_loss_from_trajectory(rep_path, n_probe=n)
        if r is None:
            continue
        # Key by the n ACTUALLY evaluated, not by the n requested.  If a run is
        # short (ledger exhausted), asking for n=80 returns whatever steps exist,
        # and keying that as "80" would label a shorter curve point as the full
        # budget.  verdict_of() already falls back to the largest present key.
        out[str(r["n"])] = {"g": r["gate_loss_mass"], "n_probe": r["n"],
                       "n_pos": r["n_pos"], "n_accept": r["n_accept"],
                       "Y_pos": r["Y_pos"], "Y_gate": r["Y_gate"]}
    return out


def verdict_of(cv, is_control_clean, is_control_lossy):
    g80 = cv.get("80", cv.get(str(max(int(k) for k in cv)))).get("g")
    allg = [(int(n), v["g"]) for n, v in cv.items()]
    ns = sorted(n for n, _ in allg)
    g = dict(allg)

    label = "clean" if g80 < 0.35 else ("lossy" if g80 > 0.65 else "unresolved")
    if g80 > 0.65 or g80 < 0.35:
        v = "RESOLVED"
    else:
        v = "PERSISTENTLY-UNRESOLVED"

    # section 5.2: resolution cost n* (only for RESOLVED)
    n_star, inconsistent = None, False
    if v == "RESOLVED":
        for start in ns:
            tail = [m for m in ns if m >= start]
            side_ok = all((g[m] - 0.5) * (g80 - 0.5) > 0 and abs(g[m] - 0.5) > MARGIN
                          for m in tail)
            if side_ok:
                n_star = start
                break
        if n_star == 20 and abs(g[20] - 0.5) <= MARGIN:
            inconsistent, n_star = True, None

    # section 5.3 controls
    control = None
    if is_control_clean:
        control = "ok" if all(abs(x) < 1e-12 for _, x in allg) else "control_failure"
    if is_control_lossy:
        control = "ok" if all(x > 0.5 for _, x in allg) else "control_failure"

    return {"verdict": v, "label_at_80": label, "n_star": n_star,
            "n_star_inconsistent": inconsistent, "control": control}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--probe-budget", type=int, default=DEFAULT_PROBE_BUDGET)
    ap.add_argument("--check-only", action="store_true",
                    help="guards + pool feasibility + dry curve check, no FDTD")
    args = ap.parse_args()
    anchor = args.anchor

    row = load_harmonized_row(anchor)
    pub_path = os.path.join(PROJECT_DIR, row["source_path"])
    fd = load_frozen_diag(anchor)

    print("=" * 78)
    print(f"BUDGET-RESOLUTION PROBE: {anchor}   (probe budget {PROBE_BUDGET} -> {args.probe_budget})")
    print("=" * 78)
    print(f"  RNG seed (published, unchanged) = {RNG_BASE_SEED}")
    print(f"  published source                = {row['source_path']}")
    print(f"  published g                     = {row['gate_loss_mass']:.3f}  "
          f"(n_pos={row['n_pos']}, n_accept={row['n_accept']})")
    pub_steps = prod_steps(pub_path, PROBE_BUDGET)
    print(f"  published first-{PROBE_BUDGET} band coverage = {band_coverage(pub_steps)}")

    strata = build_strata(anchor, fd)
    print("  band pool sizes (zero FDTD):")
    need = -(-args.probe_budget // len(strata))
    for k, v in strata.items():
        if v:
            print(f"    {k:<22} {len(v):>6}   need ~{need}")

    if args.check_only:
        print(f"\n[check-only] no FDTD started. payload SHA = {EXPECTED_PAYLOAD_SHA[:12]}...")
        return 0

    out_dir = guard_out_dir(anchor)
    port = ANCHORS[anchor]["port"]
    jac_dir = os.path.join(PARENT_DIR, "wdm3_3d", "results_3d", "jac3d")
    jac_path = os.path.join(jac_dir, f"jac_{anchor}_P{port}@1550nm_g0.npz")
    if not os.path.exists(jac_path):
        jac_path = os.path.join(jac_dir, f"jac_{anchor}_P{port}@1550nm.npz")

    rng = np.random.default_rng(RNG_BASE_SEED)      # SAME seed as published
    t0 = time.time()
    import lum_backend, _lumopt_compat  # noqa: F401

    # Each probe evaluation charges 1 unit, and an ACCEPT charges a second unit
    # for the commit (try_and_accept_null).  So the worst case is 2 units per
    # requested valid eval; a fixed "+10" slack only covers accept rates up to
    # 10/probe_budget and silently truncates high-accept anchors.
    total_budget = WARMUP_MAX_EVAL + 2 * args.probe_budget + 10
    ledger = BudgetLedger(total_budget=total_budget)
    trajectory = Trajectory("budget_resolution", anchor, total_budget, 0)

    with Session(anchor, tag="null_probe") as session:      # SAME tag as published
        run_warmup(session, jac_path, fd, ledger, trajectory, rng)
        warmup_evals = ledger.eval_count          # read here, BEFORE the probe charges
        committed = {s["idx"] for s in trajectory.steps if s.get("type") == "warmup_accept"}
        tested = set()
        for step in trajectory.steps:
            il = [step["idx"]] if "idx" in step else step.get("indices", [])
            if il:
                tested.add(_hash(il))
        target = min(args.probe_budget, ledger.remaining())
        rho, group_stats, probe_log = run_density_probe(
            session, strata, fd, ledger, trajectory,
            tested, committed, rng, target_evals=target)
        # NOTE: do NOT re-derive warmup_evals from ledger.eval_count here -- the
        # probe charges a second unit per accept, so that difference is not the
        # warmup cost (it inflated the field by the accept surcharge).

    traj_path = os.path.join(out_dir, "trajectory.json")
    trajectory.to_json(traj_path)

    ok, gate = pre_gate(pub_path, traj_path)
    reps = prod_steps(traj_path, args.probe_budget)
    cv = curve(traj_path)
    summary = {
        "anchor": anchor,
        "probe_budget_requested": args.probe_budget,
        "probe_budget_valid": len(reps),
        "budget_shortfall": len(reps) < args.probe_budget,
        "rng_seed": RNG_BASE_SEED,
        "published_source": row["source_path"],
        "published_g": row["gate_loss_mass"],
        "pre_gate_ok": ok,
        "pre_gate_detail": gate,
        "gate_curve": cv,
        "band_coverage": band_coverage(reps),
        "strata_sizes": {k: len(v) for k, v in strata.items()},
        "warmup_evals": warmup_evals,
        "probe_log": probe_log,
        "group_stats": group_stats,
        "wall_clock_s": time.time() - t0,
        "harmonized_payload_sha256": EXPECTED_PAYLOAD_SHA,
    }
    if not ok:
        summary["verdict"] = "INVALID_PROVENANCE"
        summary["control"] = None
    else:
        summary.update(verdict_of(cv,
                                  is_control_clean=(anchor == "seed14"),
                                  is_control_lossy=(anchor == "seed18")))

    with open(os.path.join(out_dir, "budget_resolution_check.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    for n in sorted(cv, key=int):
        v = cv[n]
        print(f"  n={n:>3}  g={v['g']:.3f}  n_pos={v['n_pos']:>3}  n_acc={v['n_accept']:>3}")
    print(f"  pre-gate OK = {ok}   (index hashes / dJ / accept seq / g(20))")
    print(f"  VERDICT: {summary['verdict']}"
          + (f"   n* = {summary.get('n_star')}" if summary.get('n_star') else "")
          + (f"   control = {summary.get('control')}" if summary.get('control') else ""))
    print(f"  written: {os.path.relpath(out_dir, PROJECT_DIR)}")
    return 0 if summary["verdict"] != "INVALID_PROVENANCE" else 3


if __name__ == "__main__":
    sys.exit(main())
