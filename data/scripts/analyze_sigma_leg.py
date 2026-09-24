#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sigma-leg (B1) analysis: naming-agnostic gate loss + regression self-check + 2x2 readout.

Frozen definition (preregistration_sigma_leg_seed20plus_2026-09-17.md section 5):

    probe steps  = steps whose type contains "probe" and not "warmup", first 20
    accepted     = type contains "accept"  OR  type endswith "_a"
    gate_loss_mass = 1 - Y_gate / Y_pos      (0 when Y_pos <= 1e-15)

WARNING: do NOT reuse frozen_routing_v1.self_test()'s `"accept" in type` test.
probe_only_null_gate_v2 writes `probe_accept`/`probe_reject`, but
unified_protocol writes `probe_a`/`probe_r`; under that test the unified
trajectories report gate_loss = 1.000 for every anchor (including seed1/2/3/4/7/13).

Usage:
  python analyze_sigma_leg.py regress     # zero-FDTD regression against known values
  python analyze_sigma_leg.py readout     # B1 anchors: predictions + 2x2 + Fisher
"""
import json
import os
import sys
from math import comb

import numpy as np

PROJ = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(PROJ, "results", "final_rerun_2026_08_13_protocol_v2")

N_PROBE = 20
N_CROSS_THRESHOLD = 25.0
GATE_LOSS_LO = 0.2

# B1 cohort, frozen in the preregistration (section 4). Do not edit after probing.
B1_COHORT = ["seed20", "seed21", "seed22", "seed23", "seed26", "seed27"]

# Regression targets, frozen. (anchor, run, expected gate_loss_mass)
REGRESSION = [
    ("seed16", "null_gate_probe_v2", 0.541),
    ("seed9",  "null_gate_probe_v2", 1.000),
    ("seed12", "null_gate_probe_v2", 0.000),
    ("seed15", "null_gate_probe_v2", 0.000),
    ("seed14", "null_gate_probe_v2", 0.000),
    ("seed13", "unified_protocol",   0.356),
    ("seed3",  "unified_protocol",   0.027),
    ("seed2",  "unified_protocol",   0.000),
    ("seed4",  "unified_protocol",   0.000),
]


def is_probe(step_type):
    return "probe" in step_type and "warmup" not in step_type


def is_accept(step_type):
    return ("accept" in step_type) or step_type.endswith("_a")


def trajectory_path(anchor, run):
    if run == "unified_protocol":
        return os.path.join(V2, anchor, run, "budget200", "repeat0", "trajectory.json")
    return os.path.join(V2, anchor, run, "trajectory.json")


def gate_loss_from_trajectory(path, n_probe=N_PROBE):
    """Frozen gate-loss definition. Returns None if the trajectory has no probe steps."""
    with open(path) as f:
        traj = json.load(f)

    steps = [s for s in traj["steps"] if is_probe(s.get("type", ""))][:n_probe]
    n = len(steps)
    if n == 0:
        return None

    dJ = np.array([float(s.get("dJ", 0.0)) for s in steps])
    accepted = np.array([float(s.get("dJ", 0.0)) for s in steps if is_accept(s.get("type", ""))])

    n_pos = int((dJ > 0).sum())
    Y_pos = float(np.maximum(dJ, 0.0).sum() / n)
    Y_gate = float(accepted.sum() / n) if accepted.size else 0.0
    gate_loss = (1.0 - Y_gate / Y_pos) if Y_pos > 1e-15 else 0.0

    return dict(n=n, n_pos=n_pos, n_accept=int(accepted.size),
                Y_pos=Y_pos, Y_gate=Y_gate, gate_loss_mass=gate_loss)


def load_n_cross(anchors):
    """n_cross = (2*gamma*sigma/mu)^2 for mu<0, else inf.

    Read from results/shared/<anchor>/diagnosis.json -- the authoritative source
    named in the preregistration. (results/n_cross_all_anchors.json only covers seed0-16.)
    """
    out = {}
    for anchor in anchors:
        path = os.path.join(PROJ, "results", "shared", anchor, "diagnosis.json")
        with open(path) as f:
            d = json.load(f)
        mu = d.get("mu_single")
        sigma = d.get("sigma_single")
        gamma = d.get("gamma") or 1.0
        if mu is None:
            out[anchor] = float("inf")
        elif mu >= 0:
            out[anchor] = float("inf")
        else:
            out[anchor] = (2.0 * gamma * sigma / mu) ** 2
    return out


def fisher_one_sided(a, b, c, d):
    """One-sided Fisher exact p = P(X >= a) for table [[a, b], [c, d]]."""
    r1, r2 = a + b, c + d
    c1 = a + c
    n = r1 + r2
    hi = min(r1, c1)
    if comb(n, c1) == 0:
        return float("nan")
    return sum(comb(r1, x) * comb(r2, c1 - x) for x in range(a, hi + 1)) / comb(n, c1)


def cmd_regress():
    print("=" * 78)
    print("REGRESSION: frozen gate-loss definition vs known values")
    print("=" * 78)
    ok = True
    for anchor, run, expected in REGRESSION:
        path = trajectory_path(anchor, run)
        if not os.path.exists(path):
            print(f"  {anchor:<7} {run:<20} MISSING -> {path}")
            ok = False
            continue
        r = gate_loss_from_trajectory(path)
        if r is None:
            print(f"  {anchor:<7} {run:<20} no probe steps")
            ok = False
            continue
        match = abs(r["gate_loss_mass"] - expected) < 1e-3
        ok &= match
        print(f"  {anchor:<7} {run:<20} got={r['gate_loss_mass']:.3f} "
              f"expected={expected:.3f} n_pos={r['n_pos']:>2} "
              f"{'OK' if match else 'XX MISMATCH'}")
    print(f"\n  Result: {'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


def cmd_readout():
    n_cross = load_n_cross(B1_COHORT)
    print("=" * 92)
    print("B1 READOUT: n_cross prediction vs measured gate loss")
    print("=" * 92)
    print(f"{'anchor':<8}{'n_cross':>10}{'predicted':>12}{'measured':>11}"
          f"{'n_pos':>7}{'verdict':>12}  H1/H3")

    rows = []
    for anchor in B1_COHORT:
        nc = n_cross.get(anchor)
        path = os.path.join(V2, anchor, "null_gate_probe_v2", "trajectory.json")
        if not os.path.exists(path):
            print(f"{anchor:<8}{nc if nc is not None else -1:>10.1f}{'':>12}{'PENDING':>11}")
            continue
        r = gate_loss_from_trajectory(path)
        if r is None:
            print(f"{anchor:<8}{nc if nc is not None else -1:>10.1f}{'':>12}{'NO PROBE':>11}")
            continue

        predicted_flagged = nc is not None and nc > N_CROSS_THRESHOLD
        if r["n_pos"] == 0:
            verdict, hyp = "clean", "H0 uninformative"
        else:
            observed_flagged = r["gate_loss_mass"] > GATE_LOSS_LO
            if predicted_flagged:
                verdict = "flagged" if observed_flagged else "clean"
                hyp = "H1 ok" if observed_flagged else "H1 FAIL"
            else:
                verdict = "clean" if not observed_flagged else "flagged"
                hyp = "H3 ok" if not observed_flagged else "H3 FAIL"
        rows.append((anchor, nc, predicted_flagged, verdict, r["n_pos"], r["gate_loss_mass"]))
        nc_str = "inf" if nc == float("inf") else f"{nc:.1f}"
        print(f"{anchor:<8}{nc_str:>10}"
              f"{'flagged' if predicted_flagged else 'clean':>12}"
              f"{r['gate_loss_mass']:>11.3f}{r['n_pos']:>7}{verdict:>12}  {hyp}")

    if not rows:
        print("\nNo probe results yet. Run probe_only_null_gate_v2.py for the cohort first.")
        return

    print("\n" + "=" * 92)
    print("2x2 contingency (predicted x observed), excluding H0-uninformative rows")
    print("=" * 92)
    tp = sum(1 for _, _, pf, v, np_, _ in rows if pf and v == "flagged")
    fn = sum(1 for _, _, pf, v, np_, _ in rows if pf and v == "clean" and np_ > 0)
    fp = sum(1 for _, _, pf, v, np_, _ in rows if not pf and v == "flagged")
    tn = sum(1 for _, _, pf, v, np_, _ in rows if not pf and v == "clean")
    uninf = sum(1 for *_, np_, _ in rows if np_ == 0)
    print(f"  predicted flagged / observed flagged : {tp}")
    print(f"  predicted flagged / observed clean   : {fn}")
    print(f"  predicted clean   / observed flagged : {fp}")
    print(f"  predicted clean   / observed clean   : {tn}")
    print(f"  H0 uninformative (n_pos = 0)         : {uninf}")
    if tp + fn > 0 and tp + fp > 0:
        p = fisher_one_sided(tp, fn, fp, tn)
        print(f"\n  Fisher exact (one-sided) p = {p:.4f}")
    n_eff = tp + fn + fp + tn
    print(f"  NOTE: with n={n_eff} the maximum achievable one-sided p is "
          f"1/C({n_eff},{tp + fp}) = {1.0 / comb(n_eff, tp + fp):.4f} even under perfect separation.")
    print("  Confirmatory weight requires pooling with the seed29-36 virgin cohort (B2).")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "regress"
    if mode == "regress":
        sys.exit(0 if cmd_regress() else 1)
    elif mode == "readout":
        cmd_readout()
    else:
        print(__doc__)
        sys.exit(2)
