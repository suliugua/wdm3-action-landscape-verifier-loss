"""Is the measured gate loss a property of the candidate pool's composition?

The manuscript declares the probe pool (rank >= 51, dominated by the sampled
5x5 actions) and says the measured loss is "in effect, the loss at the largest
probed size".  This script checks that statement against the recorded probe
trajectories: it recomputes the same frozen loss over (a) the modal patch-size
sub-pool and (b) each rank tier left out, and reports how far the loss moves.

No FDTD.  Read-only over results/; writes results/probe_pool_size_audit.json.

The frozen loss is a ratio of sums -- 1 - sum(accepted dJ)/sum(positive dJ) --
so the probe count cancels and any sub-pool restriction is well defined.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_sigma_leg as asl  # noqa: E402  (frozen predicates + metric source)

HARM = os.path.join(HERE, "results", "gate_loss_harmonized.json")
OUT = os.path.join(HERE, "results", "probe_pool_size_audit.json")


def loss_of(steps):
    """The frozen loss, restricted to the given probe steps.

    Returns None when the restriction leaves no positive mass: the frozen
    loss is a ratio, and with a zero denominator it is undefined rather than
    zero, so a shift must not be computed against it.
    """
    dJ = [float(s.get("dJ", 0.0)) for s in steps]
    acc = [float(s.get("dJ", 0.0)) for s in steps if asl.is_accept(s.get("type", ""))]
    Y_pos = sum(max(v, 0.0) for v in dJ)
    Y_gate = sum(acc) if acc else 0.0
    return (1.0 - Y_gate / Y_pos) if Y_pos > 1e-15 else None


def size_of(s):
    idx = s.get("indices")
    return len(idx) if isinstance(idx, list) else s.get("n")


def main():
    harm = json.load(open(HARM, encoding="utf-8"))
    rows = []
    pooled_mass = {}
    pooled_count = {}
    counts = {"total": 0, "size_25": 0, "size_9": 0}
    for anchor, rec in sorted(harm["anchors"].items()):
        path = os.path.join(HERE, rec["source_path"])
        traj = json.load(open(path, encoding="utf-8"))
        steps = [s for s in traj["steps"] if asl.is_probe(s.get("type", ""))][:20]

        full = asl.gate_loss_from_trajectory(path)["gate_loss_mass"]
        assert abs(full - rec["gate_loss_mass"]) < 1e-9, (anchor, full)

        sizes = {}
        for s in steps:
            sizes.setdefault(size_of(s), []).append(s)
            d = float(s.get("dJ", 0.0))
            if d > 0:                      # pooled shares over positive actions
                k = size_of(s)
                pooled_mass[k] = pooled_mass.get(k, 0.0) + d
                pooled_count[k] = pooled_count.get(k, 0) + 1

        modal = max(sizes, key=lambda k: len(sizes[k]))
        pos_all = sum(max(float(s.get("dJ", 0.0)), 0.0) for s in steps)
        pos_modal = sum(max(float(s.get("dJ", 0.0)), 0.0) for s in sizes[modal])
        share = (pos_modal / pos_all) if pos_all > 1e-15 else None

        sub = loss_of(sizes[modal]) if len(sizes) > 1 else full

        tiers = {}
        for s in steps:
            tiers.setdefault(s.get("group", "?"), []).append(s)
        loo = [loss_of([s for s in steps if s.get("group") != t])
               for t in tiers]
        loo = [v for v in loo if v is not None]   # drop undefined restrictions

        rows.append(dict(
            anchor=anchor, arm=rec["arm"], n_probe=len(steps),
            modal_size=modal, modal_share_of_positive_mass=share,
            loss_full=full, loss_modal_only=sub,
            d_loss_modal=(sub - full) if sub is not None else None,
            loo_min=min(loo) if loo else None, loo_max=max(loo) if loo else None,
            d_loss_loo_max=max(abs(v - full) for v in loo) if loo else None,
            sizes={str(k): len(v) for k, v in sorted(sizes.items())}))

    # ---- the pooled composition the manuscript quotes ("76% ... 10% ... 14%")
    mass_tot = sum(pooled_mass.values())
    count_tot = sum(pooled_count.values())
    for k in (25, 9):
        counts["size_%d" % k] = pooled_count[k]
    counts["total"] = count_tot
    c25 = 100.0 * pooled_count[25] / count_tot
    c9 = 100.0 * pooled_count[9] / count_tot
    assert abs(c25 - 76.0) < 1.0 and abs(c9 - 10.0) < 1.0, (c25, c9)

    n = len(rows)
    # shifts are only defined where the restriction keeps positive mass
    defined = [r for r in rows if r["d_loss_modal"] is not None]
    d_modal = [abs(r["d_loss_modal"]) for r in defined]
    d_loo = [r["d_loss_loo_max"] for r in rows if r["d_loss_loo_max"] is not None]
    shares = [r["modal_share_of_positive_mass"] for r in rows
              if r["modal_share_of_positive_mass"] is not None]
    med = lambda v: sorted(v)[len(v) // 2]
    summary = dict(
        cohort=("the anchors with a harmonized gate-loss value, each read from"
                " its own declared probe trajectory"),
        n_anchors=n,
        modal_size_is_25=sum(1 for r in rows if r["modal_size"] == 25),
        n_anchors_with_positive_probe_mass=len(defined),
        anchors_without_positive_probe_mass=[r["anchor"] for r in rows
                                             if r["modal_share_of_positive_mass"] is None],
        pooled_positive_actions_total=count_tot,
        pooled_positive_action_share_size_25=round(c25, 1),
        pooled_positive_action_share_size_9=round(c9, 1),
        pooled_positive_mass_share_size_25=round(100.0 * pooled_mass[25] / mass_tot, 1),
        positive_mass_share_modal_size_median=med(shares),
        d_loss_modal_only_median=med(d_modal),
        d_loss_modal_only_max=max(d_modal),
        n_modal_shift_above_003=sum(1 for v in d_modal if v > 0.03),
        d_loss_modal_only_worst=sorted(
            [dict(anchor=r["anchor"], loss_full=r["loss_full"],
                  loss_modal_only=r["loss_modal_only"],
                  d_loss_modal=r["d_loss_modal"]) for r in defined],
            key=lambda r: -abs(r["d_loss_modal"]))[:3],
        d_loss_leave_one_tier_out_median=med(d_loo),
        d_loss_leave_one_tier_out_max=max(d_loo),
        n_loo_shift_above_003=sum(1 for v in d_loo if v > 0.03),
        note=("d_loss_* are absolute shifts in the frozen loss when the pool is"
              " restricted to the modal patch size, or when one rank tier is"
              " left out; anchors whose restriction leaves no positive mass are"
              " excluded from the shift statistics, because the loss is then"
              " undefined rather than zero"))
    assert summary["modal_size_is_25"] == n == 28
    assert summary["pooled_positive_actions_total"] == 200
    assert summary["n_anchors_with_positive_probe_mass"] == 26
    assert summary["d_loss_modal_only_median"] == 0.0
    assert summary["n_modal_shift_above_003"] == 2

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "anchors": rows}, f, indent=2)

    print("=== probe-pool composition audit (%d anchors) ===" % n)
    print("  pooled positive actions: %d; %s at 25 px, %s at 9 px, %s between"
          % (count_tot, summary["pooled_positive_action_share_size_25"],
             summary["pooled_positive_action_share_size_9"],
             round(100.0 - c25 - c9, 1)))
    print("  modal patch size is 25 px in %d of %d anchors; it carries a median"
          " %.2f of the anchor's positive mass"
          % (summary["modal_size_is_25"], n,
             summary["positive_mass_share_modal_size_median"]))
    print("  |loss shift| restricted to the modal size: median %.3f  max %.3f"
          "  (>0.03 in %d of %d)"
          % (summary["d_loss_modal_only_median"],
             summary["d_loss_modal_only_max"],
             summary["n_modal_shift_above_003"],
             summary["n_anchors_with_positive_probe_mass"]))
    print("  |loss shift| leaving one rank tier out:    median %.3f  max %.3f"
          "  (>0.03 in %d of %d)"
          % (summary["d_loss_leave_one_tier_out_median"],
             summary["d_loss_leave_one_tier_out_max"],
             summary["n_loo_shift_above_003"], len(d_loo)))
    print("  anchors with no positive probe mass (excluded): %s"
          % ", ".join(summary["anchors_without_positive_probe_mass"]))
    print("\n  worst 3 anchors by modal-size shift:")
    for r in summary["d_loss_modal_only_worst"]:
        print("    %-7s loss %.3f -> %.3f (shift %+.3f)"
              % (r["anchor"], r["loss_full"], r["loss_modal_only"],
                 r["d_loss_modal"]))
    print("\nwrote", os.path.relpath(OUT, HERE))


if __name__ == "__main__":
    main()
