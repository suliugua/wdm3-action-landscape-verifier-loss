#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe-label precision: stratified bootstrap (frozen protocol, 2026-09-17).

Implements `preregistration_probe_label_precision_2026-09-17.md`:
  §4.1 primary  = within-band (stratified) bootstrap, 5 bands x 4 replicates,
                  B = 10000, rng = np.random.default_rng(20260917)
  §4.2 secondary= whole-cycle block bootstrap (4 cycles x 5 steps)
  §4.3 NOT_BOOTSTRAPPABLE_PROTOCOL = the unified_protocol family (no bands)
  §3            = structural classification (n_acc = 0 -> g==1 ; n_acc = n_pos -> g==0)
  §5 quantities = g_hat, g_lo/g_hi (2.5/97.5 pct), P_flip, n_pos, n_accept, Y_pos, Y_gate
  §6 verdicts   = IDENTITY > INDETERMINATE (CI contains 0.5) > STABLE (P_flip<0.05) > UNSTABLE
  §9 I4         = flag bands whose 4 replicates span more than a factor of 2 in pixel count

Input  : results/gate_loss_harmonized.json  (payload SHA verified; §13 AMENDMENT #1)
Output : results/probe_label_precision/probe_label_precision.json

Zero FDTD.
"""
import hashlib
import json
import os
import sys

import numpy as np

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
from analyze_sigma_leg import is_accept, is_probe  # noqa: E402

HARMONIZED = os.path.join(PROJ, "results", "gate_loss_harmonized.json")
EXPECTED_PAYLOAD_SHA = "4aaa2aef19dcf1b5c07852023afc415684a775761e82b8c051dabce0bba93943"
COHORT_SHA = "12571a31994da04349a8b87923d429afdb1da3a3618d925526b5d124a345a9ae"

B = 10000
RNG_SEED = 20260917
N_PROBE = 20
N_BANDS = 5           # 4 adjoint-rank bands + spatial


# -- definition (identical to analyze_sigma_leg.gate_loss_from_trajectory) --
def gate_loss(dj, acc):
    n = len(dj)
    if n == 0:
        return 0.0
    Y_pos = float(np.maximum(dj, 0.0).sum() / n)
    Y_gate = float(dj[acc].sum() / n) if acc.any() else 0.0
    return (1.0 - Y_gate / Y_pos) if Y_pos > 1e-15 else 0.0


def read_probe_steps(path, n=N_PROBE):
    with open(path) as f:
        traj = json.load(f)
    steps = [s for s in traj["steps"] if is_probe(s.get("type", ""))][:n]
    dj = np.array([float(s.get("dJ", 0.0)) for s in steps])
    acc = np.array([is_accept(s.get("type", "")) for s in steps])
    grp = [s.get("group") for s in steps]
    npx = np.array([len(s.get("indices", [])) for s in steps], dtype=float)
    return dj, acc, grp, npx


def verify_inputs():
    with open(HARMONIZED, encoding="utf-8") as f:
        d = json.load(f)
    if d["payload_sha256"] != EXPECTED_PAYLOAD_SHA:
        raise SystemExit("payload SHA mismatch vs prereg §13 -- STOP")
    rows = d["anchors"]
    inf = [a for a, v in rows.items() if v["n_pos"] >= 3]
    lossy = {a for a in inf if rows[a]["gate_loss_mass"] > 0.5}
    h = hashlib.sha256(";".join(
        a + "|" + ("1" if a in lossy else "0") for a in sorted(inf)).encode()).hexdigest()
    if h != COHORT_SHA:
        raise SystemExit(f"cohort SHA mismatch -- STOP (got {h})")
    return d


def stratified_bootstrap(dj, acc, grp, rng, b=B):
    """§4.1: resample each band's replicates with replacement."""
    order, groups = [], {}
    for i, g in enumerate(grp):
        if g not in groups:
            groups[g] = []
            order.append(g)
        groups[g].append(i)
    idx_by_group = [np.array(groups[g]) for g in order]
    out = np.empty(b)
    for k in range(b):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True)
                              for ix in idx_by_group])
        out[k] = gate_loss(dj[idx], acc[idx])
    return out


def block_bootstrap(dj, acc, rng, b=B):
    """§4.2: resample whole 5-step cycles (4 cycles of 5 steps)."""
    n = len(dj)
    cycles = [np.arange(i, min(i + N_BANDS, n)) for i in range(0, n, N_BANDS)]
    out = np.empty(b)
    for k in range(b):
        pick = rng.choice(len(cycles), size=len(cycles), replace=True)
        idx = np.concatenate([cycles[p] for p in pick])
        out[k] = gate_loss(dj[idx], acc[idx])
    return out


def strata_heterogeneous(grp, npx):
    """§9 I4 operationalisation: a band's 4 replicates span > 2x in pixel count."""
    groups = {}
    for g, m in zip(grp, npx):
        groups.setdefault(g, []).append(m)
    bad = {}
    for g, vals in groups.items():
        vals = [v for v in vals if v > 0]
        if len(vals) >= 2 and max(vals) / max(min(vals), 1.0) > 2.0:
            bad[g] = (min(vals), max(vals))
    return bad


def main():
    harm = verify_inputs()
    rows = harm["anchors"]
    cohort = sorted([a for a, v in rows.items() if v["n_pos"] >= 3],
                    key=lambda s: int(s[4:]))

    out_dir = os.path.join(PROJ, "results", "probe_label_precision")
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    print("=" * 118)
    print("PROBE-LABEL PRECISION  (frozen: preregistration_probe_label_precision_2026-09-17.md)")
    print("=" * 118)
    print(f"{'anchor':<8}{'arm':<22}{'n_pos':>6}{'n_acc':>6}{'g_hat':>8}"
          f"{'g_lo':>8}{'g_hi':>8}{'P_flip':>8}  {'P_flip_blk':>10}  verdict")

    for a in cohort:
        r = rows[a]
        arm = r["arm"]
        n_pos, n_acc, g_hat = r["n_pos"], r["n_accept"], r["gate_loss_mass"]

        # §3 structural classification (assumption-free) takes precedence
        if n_acc == 0 or n_acc == n_pos:
            klass = "n_acc=0" if n_acc == 0 else "n_acc=n_pos"
            results[a] = dict(arm=arm, n_pos=n_pos, n_accept=n_acc, g_hat=g_hat,
                              structural="IDENTITY", identity_reason=klass,
                              verdict="IDENTITY")
            print(f"{a:<8}{arm:<22}{n_pos:>6}{n_acc:>6}{g_hat:>8.3f}"
                  f"{'-':>8}{'-':>8}{'-':>8}  {'-':>10}  IDENTITY ({klass})")
            continue

        if arm != "null_gate_probe_v2":
            results[a] = dict(arm=arm, n_pos=n_pos, n_accept=n_acc, g_hat=g_hat,
                              structural="STOCHASTIC", verdict="NOT_BOOTSTRAPPABLE_PROTOCOL",
                              note="no band structure in schema; excluded by prereg 4.3")
            print(f"{a:<8}{arm:<22}{n_pos:>6}{n_acc:>6}{g_hat:>8.3f}"
                  f"{'-':>8}{'-':>8}{'-':>8}  {'-':>10}  NOT_BOOTSTRAPPABLE_PROTOCOL")
            continue

        dj, acc, grp, npx = read_probe_steps(os.path.join(PROJ, rows[a]["source_path"]))
        rng = np.random.default_rng(RNG_SEED)
        bs = stratified_bootstrap(dj, acc, grp, rng)
        rng2 = np.random.default_rng(RNG_SEED)
        blk = block_bootstrap(dj, acc, rng2)

        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        p_flip = float(np.mean((bs > 0.5) != (g_hat > 0.5)))
        p_flip_blk = float(np.mean((blk > 0.5) != (g_hat > 0.5)))
        het = strata_heterogeneous(grp, npx)

        if lo <= 0.5 <= hi:
            verdict = "INDETERMINATE"
        elif p_flip < 0.05:
            verdict = "STABLE"
        else:
            verdict = "UNSTABLE"

        results[a] = dict(
            arm=arm, source_path=rows[a]["source_path"],
            n_pos=n_pos, n_accept=n_acc, g_hat=g_hat,
            Y_pos=r["Y_pos"], Y_gate=r["Y_gate"],
            structural="STOCHASTIC",
            g_lo=lo, g_hi=hi, p_flip=p_flip, p_flip_block=p_flip_blk,
            strata_heterogeneous=({k: list(v) for k, v in het.items()} or None),
            verdict=verdict,
            block_verdict=("INDETERMINATE" if float(np.percentile(blk, 2.5)) <= 0.5
                           <= float(np.percentile(blk, 97.5))
                           else ("STABLE" if p_flip_blk < 0.05 else "UNSTABLE")),
        )
        flag = "  [strata_heterogeneous]" if het else ""
        print(f"{a:<8}{arm:<22}{n_pos:>6}{n_acc:>6}{g_hat:>8.3f}"
              f"{lo:>8.3f}{hi:>8.3f}{p_flip:>8.3f}  {p_flip_blk:>10.3f}  {verdict}{flag}")

    # ── summary ─────────────────────────────────────────────────────────────
    tally = {}
    for v in results.values():
        tally[v["verdict"]] = tally.get(v["verdict"], 0) + 1
    print()
    print("verdict tally:", dict(sorted(tally.items())))
    stoch = [a for a, v in results.items() if v["structural"] == "STOCHASTIC"]
    print("stochastic anchors:", sorted(stoch, key=lambda s: int(s[4:])),
          f"({len(stoch)} of {len(cohort)})")
    print("NOT_BOOTSTRAPPABLE :", sorted(
        [a for a, v in results.items() if v["verdict"] == "NOT_BOOTSTRAPPABLE_PROTOCOL"],
        key=lambda s: int(s[4:])))
    print("INDETERMINATE      :", sorted(
        [a for a, v in results.items() if v["verdict"] == "INDETERMINATE"],
        key=lambda s: int(s[4:])))
    print("UNSTABLE           :", sorted(
        [a for a, v in results.items() if v["verdict"] == "UNSTABLE"],
        key=lambda s: int(s[4:])))

    payload = {
        "schema": "p2_probe_label_precision",
        "frozen": "2026-09-17",
        "prereg": "preregistration_probe_label_precision_2026-09-17.md",
        "input": "results/gate_loss_harmonized.json",
        "input_payload_sha256": harm["payload_sha256"],
        "cohort_sha256": COHORT_SHA,
        "B": B, "rng_seed": RNG_SEED,
        "cohort_size": len(cohort),
        "verdict_tally": dict(sorted(tally.items())),
        "anchors": results,
    }
    out_path = os.path.join(out_dir, "probe_label_precision.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nwritten: {os.path.relpath(out_path, PROJ)}")


if __name__ == "__main__":
    main()
