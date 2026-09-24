"""Is the measured gate loss exactly the positive-tail / floor overlap?

The rule the probe applies is  accept iff dJ > theta(n),  with

    theta(n) = max(0, mu_1*n + z*gamma*sigma_1*sqrt(n))

evaluated on the anchor's own phase-0 statistics -- the same T_null(n) the
manuscript writes in Eq. (Tnull).  If the recorded verdicts reproduce
"dJ > theta(n)" at every probe step, then

    L_a = sum(0 < dJ <= theta(n)) / sum(dJ > 0)

is an identity rather than an approximation, and it may be written as one.

This script reads the released trajectories and each anchor's own phase-0
diagnosis, recomputes theta per step, and reports:
  * the number of probe steps whose recorded verdict matches the rule;
  * the rejected positive mass computed from the floor, against the measured
    loss, so the identity is checked rather than assumed;
  * how many steps sit at or above the anchor's crossing size, where the
    applied floor is identically zero and the null gate is the sign gate.

No FDTD.  Read-only over results/; writes results/floor_identity_audit.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_sigma_leg as asl  # noqa: E402  (frozen predicates + metric source)

HARM = os.path.join(HERE, "results", "gate_loss_harmonized.json")
OUT = os.path.join(HERE, "results", "floor_identity_audit.json")
Z = 2.0


def load_diag(anchor):
    """The anchor's own phase-0 statistics, from the path the runner reads."""
    for p in (os.path.join(HERE, "results", "shared", anchor, "diagnosis.json"),
              os.path.join(HERE, "results", "selector", "selector_raw", anchor,
                           "budget200", "repeat0", "result.json")):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            return d.get("diagnosis", d)
    return None


def theta(n, d):
    """The floor the rule applies at patch size n, on this anchor's statistics."""
    mu1, sig1 = d["mu_single"], d["sigma_single"]
    g = d.get("gamma")
    if g is None:
        g = d["sigma_5"] / (sig1 * np.sqrt(5))
    return max(0.0, mu1 * n + Z * g * sig1 * np.sqrt(n))


def main():
    harm = json.load(open(HARM, encoding="utf-8"))
    rows = []
    n_steps = n_match = n_zero = n_zero_25 = 0
    n_false_reject = n_false_accept = 0
    all_zero_anchors = []
    floor_class = {"both": [], "only9": [], "neither": []}
    for anchor, rec in sorted(harm["anchors"].items()):
        d = load_diag(anchor)
        assert d is not None, "no phase-0 diagnosis on disk for %s" % anchor
        traj = json.load(open(os.path.join(HERE, rec["source_path"]),
                              encoding="utf-8"))
        steps = [s for s in traj["steps"] if asl.is_probe(s.get("type", ""))][:20]

        y_pos = 0.0
        y_rej_meas = 0.0
        y_rej_floor = 0.0
        z_here = 0
        n_pos_here = 0
        n_acc_here = 0
        for s in steps:
            n_steps += 1
            n = len(s["indices"])
            t = theta(n, d)
            v = float(s.get("dJ", 0.0))
            accepted = asl.is_accept(s.get("type", ""))
            if t == 0.0:
                n_zero += 1
                z_here += 1
                if n == 25:
                    n_zero_25 += 1
            # the recorded verdict must be exactly the rule
            if accepted == (v > t):
                n_match += 1
            elif accepted:
                n_false_accept += 1
            else:
                n_false_reject += 1
            if v > 0.0:
                y_pos += v
                n_pos_here += 1
            if accepted:
                n_acc_here += 1
            if (not accepted) and v > 0.0:
                y_rej_meas += v
            if (not accepted) and 0.0 < v <= t:
                y_rej_floor += v

        # The operating envelope: is the floor alive at the sizes the probe
        # samples?  Both sampled extremes are tested, 9 px and 25 px.
        t9, t25 = theta(9, d), theta(25, d)
        cls = ("both" if (t9 > 0.0 and t25 > 0.0)
               else ("only9" if t9 > 0.0 else "neither"))
        floor_class[cls].append(anchor)
        if z_here == len(steps) and steps:
            all_zero_anchors.append(anchor)

        rows.append(dict(
            anchor=anchor, arm=rec["arm"], n_probe=len(steps),
            loss_measured=rec["gate_loss_mass"],
            loss_from_floor=((y_rej_floor / y_pos) if y_pos > 1e-15 else None),
            identical=bool(abs(y_rej_floor - y_rej_meas) < 1e-12),
            n_steps_at_zero_floor=z_here,
            n_pos=n_pos_here, n_accept=n_acc_here,
            theta_9=t9, theta_25=t25, floor_class=cls,
            mu_single=d["mu_single"], sigma_single=d["sigma_single"]))

    defined = [r for r in rows if r["loss_from_floor"] is not None]
    worst = max(abs(r["loss_from_floor"] - r["loss_measured"]) for r in defined)
    summary = dict(
        cohort=("the anchors with a harmonized gate-loss value, each read from"
                " its own declared probe trajectory and its own phase-0"
                " diagnosis"),
        probe_steps_checked=n_steps,
        probe_steps_matching_the_rule=n_match,
        false_reject=n_false_reject,
        false_accept=n_false_accept,
        n_anchors=len(rows),
        n_anchors_with_positive_probe_mass=len(defined),
        max_abs_loss_difference=max(abs(r["loss_from_floor"] - r["loss_measured"])
                                    for r in defined),
        probe_steps_at_zero_floor=n_zero,
        probe_steps_at_zero_floor_fraction=round(n_zero / n_steps, 4),
        probe_steps_at_zero_floor_at_25px=n_zero_25,
        anchors_with_zero_floor_at_every_step=all_zero_anchors,
        envelope_note=("floor_class says whether the applied floor is positive"
                       " at the probe's two sampled extremes: both = alive at 9"
                       " and 25 px, only9 = alive at 9 px only, neither = the"
                       " rule is a sign gate at every sampled size, so a zero"
                       " reading there is structural rather than measured"),
        envelope_floor_alive_at_both_sizes=len(floor_class["both"]),
        envelope_floor_alive_at_smaller_size_only=len(floor_class["only9"]),
        envelope_floor_dead_at_every_sampled_size=len(floor_class["neither"]),
        envelope_floor_dead_at_largest_size=sorted(floor_class["only9"]
                                                   + floor_class["neither"]),
        note=("the floor is theta(n)=max(0,mu1*n+z*gamma*sigma1*sqrt(n)) on the"
              " anchor's own phase-0 statistics, the threshold the rule"
              " applied; the identity L = sum(rejected positives)/sum(positives)"
              " is therefore exact on the recorded actions and does not depend"
              " on how well theta tracks the null statistics measured directly"
              " at each patch size"))

    assert n_steps == 560, n_steps
    assert n_match == n_steps and n_false_reject == 0 and n_false_accept == 0
    assert len(defined) == 26 and worst < 1e-12, worst
    assert n_zero == 175 and n_zero_25 == 156, (n_zero, n_zero_25)
    assert all_zero_anchors == ["seed1", "seed15", "seed2"], all_zero_anchors
    # the envelope, and the cross-check that a dead floor is what the recorded
    # data already show: where the rule is a sign gate at every sampled size,
    # every positive draw was accepted.
    assert (len(floor_class["both"]), len(floor_class["only9"]),
            len(floor_class["neither"])) == (18, 7, 3), floor_class
    for r in rows:
        if r["floor_class"] == "neither":
            assert r["n_pos"] == r["n_accept"], r
            assert r["loss_measured"] == 0.0, r

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "anchors": rows}, f, indent=2)

    print("=== floor/positive-tail identity audit (%d anchors) ===" % len(rows))
    print("  recorded verdict == (dJ > theta): %d of %d steps  (false reject %d,"
          " false accept %d)"
          % (n_match, n_steps, n_false_reject, n_false_accept))
    print("  floor-derived overlap loss vs measured loss, %d anchors with"
          " positive probe mass: max |diff| %.6f"
          % (len(defined), summary["max_abs_loss_difference"]))
    print("  steps whose applied floor is identically zero: %d of %d (%.0f%%),"
          " %d of them at 25 px"
          % (n_zero, n_steps, 100 * n_zero / n_steps, n_zero_25))
    print("  anchors where this holds at every probed size: %s"
          % ", ".join(all_zero_anchors))
    print("  operating envelope -- floor alive at both sampled sizes %d, at the"
          " smaller size only %d, dead at every sampled size %d"
          % (len(floor_class["both"]), len(floor_class["only9"]),
             len(floor_class["neither"])))
    print("\nwrote", os.path.relpath(OUT, HERE))


if __name__ == "__main__":
    main()
