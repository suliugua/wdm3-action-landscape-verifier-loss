#!/usr/bin/env python
"""Experiment A: Train/test split prospective role assignment.

Phase 1 (TRAIN — N_PROBE pixels × 4 variants):
  Each variant selects its own top-ranked probe pixels by its own gradient
  ranking, then measures their actual dF via FDTD.  The variant with the
  largest net dF (magnitude-weighted) is ADOPTED.  (Prior to 2026-07-24
  all variants shared a common |O_pix|-based probe set, which made them
  indistinguishable because the measurement is variant-agnostic.)

Phase 2 (TEST — withheld top-K candidates):
  Exclude ALL Phase 1 probe pixels (union across variants).  Run greedy
  accept/reject using ONLY the ADOPTED variant's gradient ranking on the
  remaining top-K candidates.  No REJECTED arms — they would measure the
  same physical pixel flips (try_flip is variant-agnostic).

Primary endpoint:
  net dF > 0 OR at least one test candidate exceeds the noise-calibrated
  accept threshold.  A FAIL means the train-selected variant does not
  generalise to unseen pixels — itself a valid diagnostic finding.

Anchors: seed9 P1, seed0 P1 (jacs already computed).

Usage:
  python -m wdm3_3d.prospective_train_test --seed 9 --port 1
  python -m wdm3_3d.prospective_train_test --seed 0 --port 1
"""

import json, os, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault("LUMOPT_BACKEND", "2026")
import lum_backend  # noqa

from .wdm3_3d_device import WL_NM, load_warmstart_params
from .wdm3_3d_jac import RESULTS_DIR, _flip_gain, _get_jac, _label, _now_iso, _save_json
from .wdm3_3d_landscape import (
    RULES, LANDSCAPE_DIR, IncrementalSession, apply_col_mask,
    variant_phi_deg, phase_scan, phi_in_plateaus,
)

EXP_DIR = LANDSCAPE_DIR / "train_test"
N_PROBE = 10  # Phase 1 probe pixels
K_MAX = 10    # Phase 2 test candidates
ACCEPT_SIGMA = 2.0  # accept threshold multiplier
VARIANTS = ["raw", "i_conj_a", "conj_a", "i"]


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Train — probe 10 pixels × 4 variants
# ═══════════════════════════════════════════════════════════════════════════

def phase1_train(seed, target_port, wl_nm=WL_NM, rules=RULES):
    """Measure 10 probe pixels × 4 variants. Freeze variant choice."""
    label = _label(seed, target_port, wl_nm)
    train_path = EXP_DIR / f"TRAIN_{label}.json"

    if train_path.exists():
        print(f"[Phase 1] Train already frozen: {train_path}")
        return json.load(open(train_path))

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    jac = _get_jac(seed, target_port, wl_nm, RESULTS_DIR, reuse=True)

    print(f"\n{'='*60}")
    print(f"PHASE 1 — TRAIN: {label}")
    print(f"  {N_PROBE} probe pixels × {len(VARIANTS)} variants = "
          f"{N_PROBE * len(VARIANTS)} FDTD evals")
    print(f"{'='*60}")

    params = np.asarray(jac["params"], dtype=float).reshape(-1)
    a = jac["a"]

    # ── Phase 1: Per-variant probe pixel selection ──
    # Each variant selects its own top-ranked pixels by its own gradient,
    # then measures their actual dF.  The variant whose ranking best
    # predicts positive dF is adopted.  (Bug fix 2026-07-24: was selecting
    # a single common |O_pix|-based probe set for all variants — all got
    # identical results because the measurement is variant-agnostic.)
    results = {v: [] for v in VARIANTS}
    variant_probe_idx = {}  # per-variant probe pixels for Phase 2 exclusion

    for variant in VARIANTS:
        gain = _flip_gain(jac["grads"][variant], params)
        phi = variant_phi_deg(variant, a)

        # Select N_PROBE pixels: top-ranked by this variant's gain,
        # skip the very top (often edge artifacts), then stratified.
        # Guard: if a variant has fewer than N_PROBE positive-gain
        # pixels, take all of them (the variant is likely weak).
        pos_rank = np.argsort(gain)[::-1]
        pos_rank = pos_rank[gain[pos_rank] > 0]  # only positive-gain pixels
        if len(pos_rank) < N_PROBE + 5:
            candidates = pos_rank  # too few candidates — don't skip top-5
        else:
            candidates = pos_rank[5:]  # skip top-5 (edge-artifact guard)
        n_take = min(N_PROBE, len(candidates))
        step = max(1, len(candidates) // n_take) if n_take else 1
        probe_idx = [int(candidates[i * step]) for i in range(n_take)]
        if n_take < N_PROBE:
            print(f"  {variant:12s}: WARNING only {n_take}/{N_PROBE} "
                  f"positive-gain pixels available")
        variant_probe_idx[variant] = probe_idx
        print(f"  {variant:12s} (phi={phi:6.1f}): probe pixels={probe_idx}")

        with IncrementalSession(params.copy(), target_port, wl_nm,
                                tag=f"train_{label}_{variant}") as S:
            f0 = S.run_baseline()
            for idx in probe_idx:
                f = S.measure([int(idx)])
                dF = float(f - f0)
                results[variant].append({
                    "idx": int(idx),
                    "dF": dF,
                    "gain_pred": float(gain[idx]),
                    "accept": dF > 0,
                })

        n_pos = sum(1 for r in results[variant] if r["accept"])
        net = sum(r["dF"] for r in results[variant])
        print(f"  {variant:12s}: {n_pos}/{N_PROBE} positive, net={net:+.4e}")

    # ── Freeze: select variant with largest net dF (magnitude-weighted) ──
    # n_positive alone favours variants that produce many tiny-positive flips
    # that won't survive a noise-calibrated accept threshold; net dF weights
    # by effect size — the variant whose flips deliver the most total improvement.
    nets = {v: sum(r["dF"] for r in results[v]) for v in VARIANTS}
    n_pos = {v: sum(1 for r in results[v] if r["accept"]) for v in VARIANTS}
    best_variant = max(nets, key=nets.get)
    adopted = best_variant if nets[best_variant] > 0 else None

    # ── Zero-cost cross-check: phase scan vs adopted variant ──
    # Does the S(φ) peak (the diagnosis-based recommendation) agree with
    # the train-based adoption?  Mismatch is itself informative — it means
    # the per-pixel dF measurements and the phase-scan aggregate disagree
    # on which variant is best.
    scan = phase_scan(jac["O_pix"], params)
    peak_phi = scan["S_peak_deg"]
    adopted_phi = variant_phi_deg(adopted, a) if adopted else None
    in_band = (phi_in_plateaus(adopted_phi, scan["gain_bands_deg"])
               if adopted_phi is not None else False)
    nearest_variant = min(VARIANTS,
                          key=lambda v: abs(variant_phi_deg(v, a) - peak_phi))
    agreement = (adopted == nearest_variant) if adopted else False

    print(f"\n  Phase scan: S(φ) peak at {peak_phi:.1f}° → "
          f"nearest variant = {nearest_variant} "
          f"(phi={variant_phi_deg(nearest_variant, a):.1f}°)")
    print(f"  Adopted: {adopted} (phi={adopted_phi:.1f}°)" if adopted
          else "  Adopted: NONE")
    print(f"  Adopted in gain band: {in_band}  |  "
          f"Agreement with phase scan: {agreement}")

    # Also record which to reject
    rejected = [v for v in VARIANTS if v != adopted]

    train_record = {
        "label": label, "seed": seed, "target_port": target_port,
        "timestamp": _now_iso(),
        "phase": "TRAIN",
        "probe_idx_per_variant": variant_probe_idx,
        "n_probe": N_PROBE,
        "variants_tested": VARIANTS,
        "results": results,
        "n_positive": n_pos,
        "adopted_variant": adopted,
        "rejected_variants": rejected,
        "phase_scan_crosscheck": {
            "S_peak_deg": peak_phi,
            "nearest_variant": nearest_variant,
            "adopted_phi": adopted_phi,
            "in_gain_band": bool(in_band),
            "agreement_with_scan": bool(agreement),
        },
        "status": "FROZEN — do not edit before Phase 2",
    }
    _save_json(train_record, train_path)
    print(f"\n  Adopted: {adopted}  Rejected: {rejected}")
    print(f"  Train saved → {train_path}")
    return train_record


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Test — withheld top-K candidates, adopted variant only
# ═══════════════════════════════════════════════════════════════════════════

def phase2_test(seed, target_port, wl_nm=WL_NM, rules=RULES):
    """Run adopted variant greedy on withheld top-K candidates.

    Phase 1 froze a variant based on per-variant probe-pixel measurements.
    Phase 2 tests whether that variant's gradient ranking generalises to
    unseen pixels — greedy accept/reject with the adopted variant's ranking.
    No REJECTED arms: the measurement is variant-agnostic (try_flip just
    flips a binary pixel), so REJECTED arms on the same candidates would
    be bit-identical — a waste of FDTD time.
    """
    label = _label(seed, target_port, wl_nm)
    test_path = EXP_DIR / f"TEST_{label}.json"

    train = json.load(open(EXP_DIR / f"TRAIN_{label}.json"))
    # Union of all variant probe pixels (each variant selected its own)
    all_probes = train.get("probe_idx_per_variant", train.get("probe_idx"))
    if isinstance(all_probes, dict):
        probe_set = set()
        for v_probes in all_probes.values():
            probe_set.update(v_probes)
    else:
        probe_set = set(all_probes)  # backward compat with old format
    adopted = train["adopted_variant"]

    jac = _get_jac(seed, target_port, wl_nm, RESULTS_DIR, reuse=True)
    params = np.asarray(jac["params"], dtype=float).reshape(-1)

    # ── Measure sigma_single for accept threshold ──
    rng = np.random.RandomState(seed * 200 + target_port)
    all_pool = np.arange(len(params))
    safe_pool = np.array([i for i in all_pool if i not in probe_set])
    sigma_idx = rng.choice(safe_pool, size=min(15, len(safe_pool)), replace=False)
    single_dFs = []
    with IncrementalSession(params.copy(), target_port, wl_nm,
                            tag=f"sigma_{label}") as S:
        f0 = S.run_baseline()
        for idx in sigma_idx:
            f = S.measure([int(idx)])
            single_dFs.append(float(f - f0))
    sigma = float(np.std(single_dFs))
    accept_thr = ACCEPT_SIGMA * sigma

    print(f"\n{'='*60}")
    print(f"PHASE 2 — TEST: {label}")
    print(f"  Adopted variant: {adopted}")
    print(f"  sigma={sigma:.3e}  accept_thr={accept_thr:.3e}")
    print(f"{'='*60}")

    # ── Test: adopted variant greedy on top-K (excluding all probes) ──
    excl = np.zeros(len(params), dtype=bool)
    for px in probe_set:
        excl[int(px)] = True

    gain_adopted = _flip_gain(jac["grads"][adopted], params)
    sel = np.where(excl, -np.inf, gain_adopted)
    nom_idx = [int(i) for i in np.argsort(sel)[::-1]
               if sel[int(i)] > 0][:K_MAX]

    print(f"  Test candidates: {len(nom_idx)} "
          f"(excluded {len(probe_set)} probe pixels)")

    rounds = []
    total_dF = 0.0
    with IncrementalSession(params.copy(), target_port, wl_nm,
                            tag=f"test_{label}") as S:
        f0 = S.run_baseline()
        for rank, idx in enumerate(nom_idx, start=1):
            dF, accept = S.try_flip(idx, accept_thr)
            rounds.append({
                "rank": rank, "idx": int(idx),
                "dF": float(dF), "accepted": bool(accept),
                "gain_pred": float(gain_adopted[idx]),
            })
            if accept:
                total_dF += dF
            print(f"  [{rank}/{len(nom_idx)}] idx={idx} "
                  f"pred={gain_adopted[idx]:+.4f} dF={dF:+.4e} "
                  f"{'ACCEPT' if accept else 'reject'}")

    accepted = [r for r in rounds if r["accepted"]]
    all_dFs = [r["dF"] for r in rounds]
    max_i = int(np.argmax(all_dFs)) if all_dFs else 0
    n_pos = sum(1 for d in all_dFs if d > 0)

    # ── Primary endpoint: at least one test candidate improves ──
    primary = total_dF > 0 or len(accepted) > 0

    test_record = {
        "label": label, "timestamp": _now_iso(),
        "phase": "TEST",
        "adopted_variant": adopted,
        "sigma_single": sigma,
        "accept_threshold": accept_thr,
        "probe_pixels_excluded": sorted(probe_set),
        "train_summary": {
            "n_positive": train["n_positive"],
            "adopted_net": sum(r["dF"] for r in train["results"][adopted]),
        },
        "rounds": rounds,
        "n_tested": len(rounds),
        "n_accepted": len(accepted),
        "n_positive_dF": n_pos,
        "net_dF": float(total_dF),
        "max_dF": max(all_dFs) if all_dFs else 0.0,
        "max_dF_rank": max_i + 1 if all_dFs else 0,
        "primary_endpoint": {
            "criterion": "net_dF > 0 OR n_accepted > 0",
            "PASS": bool(primary),
        },
    }
    _save_json(test_record, test_path)
    print(f"\n  VERDICT: {'PASS' if primary else 'FAIL'}")
    print(f"  ADOPTED({adopted}): net={total_dF:+.4e}  "
          f"accept={len(accepted)}/{len(rounds)}  "
          f"n_pos={n_pos}  max_dF_rank={test_record['max_dF_rank']}")
    return test_record


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train/test split role assignment")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--port", type=int, default=1)
    args = ap.parse_args()

    train = phase1_train(args.seed, args.port)
    if train["adopted_variant"] is None:
        print("No variant passed Phase 1 — all rejected. Stopping.")
        sys.exit(0)
    test = phase2_test(args.seed, args.port)
    sys.exit(0 if test["primary_endpoint"]["PASS"] else 1)
