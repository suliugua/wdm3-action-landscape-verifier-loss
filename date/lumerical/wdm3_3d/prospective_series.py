#!/usr/bin/env python
"""3-anchor prospective role-assignment series.

Anchors: seed9 P1, seed11 P1, seed12 P1 (fresh 3D, unused in seed7/10 experiments).

Per anchor, 4 arms run with FROZEN diagnosis:

  RAW_DEFAULT       – raw |a|² variant, no mask, full K_max
  PHASE_DIAGNOSED   – Rule-P-selected phase only, no mask, full K_max
  PHASE_MASKED      – selected phase + Rule-M mask, full K_max
  PHASE_MASKED_ACCEPT – phase + mask + FDTD accept/reject (ΔF > +2σ),
                         no K_eff hard stop; report effective ranking depth

Primary endpoint (per anchor):
  PHASE_DIAGNOSED > RAW_DEFAULT  (diagnostic activation: does Rule P activate
                                  an otherwise unusable proposal signal?)

Secondary (per anchor):
  - Acceptance necessity: rejected high-rank vs accepted low-rank candidates
  - Rule M contribution: PHASE_MASKED vs PHASE_DIAGNOSED
  - Effective depth: where do accepted flips concentrate?
  - FDTD acceptor: does accept/reject filter outperform raw ranking?

Cross-anchor endpoints:
  - Phase diagnostic activation rate (N_anchors with PHASE > RAW)
  - Mask contribution variability
  - Ranking non-monotonicity rate (fraction of anchors where max dF rank > 1)
  - Sigma_single variability

Usage:
  python -m wdm3_3d.prospective_series --seeds 9,11,12 --port 1
"""

import json
import time
import sys
from pathlib import Path

import numpy as np

# ── Backend ──
import os
os.environ.setdefault("LUMOPT_BACKEND", "2026")
import lum_backend  # noqa: F401

from .wdm3_3d_device import WL_NM, load_warmstart_params
from .wdm3_3d_jac import (
    RESULTS_DIR,
    _flip_gain,
    _get_jac,
    _label,
    _now_iso,
    _save_json,
)
from .wdm3_3d_landscape import (
    RULES,
    LANDSCAPE_DIR,
    IncrementalSession,
    apply_col_mask,
    k_eff_from_rank_curve,
    diagnose_anchor,
)

SERIES_DIR = LANDSCAPE_DIR / "prospective_series"
ACCEPT_SIGMA_MULT = 2.0
K_MAX = 10


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Diagnosis (frozen per anchor)
# ═══════════════════════════════════════════════════════════════════════════

def diagnose_anchor_frozen(seed, target_port, wl_nm=WL_NM, rules=RULES):
    """Run Rule M/P/K diagnosis, freeze prescription, return it + probe pixels."""
    label = _label(seed, target_port, wl_nm)
    rx_path = SERIES_DIR / f"PRESCRIPTION_{label}.json"

    if rx_path.exists():
        rx = json.load(open(rx_path))
        return rx, set(rx.get("probe_pixels", []))

    SERIES_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    diag = diagnose_anchor(seed=seed, target_port=target_port, wl_nm=wl_nm,
                           variant="auto", replay=False, rules=rules,
                           out_dir=SERIES_DIR, jac_dir=RESULTS_DIR)

    # Collect probe pixels
    probe_pixels = set()
    for z in diag.get("zones", []):
        for v in z.get("verdicts", []):
            probe_pixels.add(int(v["idx"]))

    # Measure sigma_single
    params = np.asarray(load_warmstart_params(seed), dtype=float).reshape(-1)
    mask = [tuple(m) for m in diag["mask"]]
    excl = apply_col_mask(len(params), mask)
    unmasked_pool = np.where(~excl)[0]
    sigma_pool = np.array([i for i in unmasked_pool if i not in probe_pixels])
    rng = np.random.RandomState(seed * 100 + target_port)
    k_single = min(15, len(sigma_pool))
    draw_idx = rng.choice(sigma_pool, size=k_single, replace=False)

    single_dFs = []
    with IncrementalSession(params, target_port, wl_nm,
                            tag=f"sigma_{label}") as S:
        S.run_baseline()
        for idx in draw_idx:
            f = S.measure([int(idx)])
            single_dFs.append(float(f - S.fom))
    sigma_measured = float(np.std(single_dFs))
    accept_threshold = ACCEPT_SIGMA_MULT * sigma_measured

    prescription = {
        "label": label, "seed": seed, "target_port": target_port,
        "wl_nm": wl_nm, "timestamp": _now_iso(),
        "status": "FROZEN",
        "diagnosis": diag,
        "probe_pixels": sorted(probe_pixels),
        "prescription": {
            "variant": diag["variant"],
            "mask": diag["mask"],
            "sigma_single_measured": sigma_measured,
            "accept_threshold": accept_threshold,
            "k_max": K_MAX,
        },
        "wall_s": time.time() - t0,
    }
    _save_json(prescription, rx_path)
    return prescription, probe_pixels


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Single-arm runner
# ═══════════════════════════════════════════════════════════════════════════

def run_arm(seed, target_port, wl_nm, arm_name, variant, mask,
            accept_threshold, shared_jac, probe_pixels, rules=RULES):
    """Run one arm: rank candidates by |a|² gain, sequential FDTD try_flip."""
    label = _label(seed, target_port, wl_nm)
    arm_label = f"{arm_name}_{label}"

    jac = shared_jac
    params = np.asarray(jac["params"], dtype=float).reshape(-1)
    gain = _flip_gain(jac["grads"][variant], params)

    # Exclusion
    excl = apply_col_mask(len(params), mask)
    if probe_pixels:
        for px in probe_pixels:
            excl[int(px)] = True

    sel = np.where(excl, -np.inf, gain)
    nom_idx = [int(i) for i in np.argsort(sel)[::-1]
              if sel[int(i)] > 0][:K_MAX]

    print(f"  [{arm_name}] nominated={len(nom_idx)} "
          f"(masked={np.sum(apply_col_mask(len(params),mask))} "
          f"probe={len(probe_pixels)})")

    rounds = []
    total_dF = 0.0
    with IncrementalSession(params.copy(), target_port, wl_nm,
                            tag=f"series_{arm_label}") as S:
        f0 = S.run_baseline()
        for rank, idx in enumerate(nom_idx, start=1):
            dF, accept = S.try_flip(idx, accept_threshold)
            rounds.append({
                "rank": rank, "idx": int(idx),
                "ix": int(idx) // 100, "iy": int(idx) % 100,
                "gain_pred": float(gain[idx]),
                "dF_meas": float(dF), "accepted": bool(accept),
            })
            if accept:
                total_dF += dF

    accepted = [r for r in rounds if r["accepted"]]
    all_dFs = [r["dF_meas"] for r in rounds]
    summary = {
        "net_dF": float(total_dF),
        "n_nominated": len(nom_idx),
        "n_accepted": len(accepted),
        "hit5": sum(1 for r in rounds[:5] if r["dF_meas"] > accept_threshold)
                / min(5, len(rounds)),
        "max_dF": float(max(all_dFs)) if all_dFs else 0.0,
        "max_dF_rank": int(np.argmax(all_dFs) + 1) if all_dFs else 0,
        "rank_of_last_accept": int(accepted[-1]["rank"]) if accepted else 0,
        "k_eff": k_eff_from_rank_curve(all_dFs, rules),
    }

    print(f"  [{arm_name}] net={total_dF:+.4e} accept={len(accepted)}/"
          f"{len(rounds)} max_dF_rank={summary['max_dF_rank']}")

    record = {"arm": arm_name, "label": label, "variant": variant,
              "mask": [list(m) for m in mask], "summary": summary,
              "rounds": rounds}
    _save_json(record, SERIES_DIR / f"arm_{arm_label}.json")
    return record


# ═══════════════════════════════════════════════════════════════════════════
# Per-anchor orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def run_anchor(seed, target_port=1, wl_nm=WL_NM, rules=RULES):
    """Run the full 4-arm protocol on one anchor."""
    label = _label(seed, target_port, wl_nm)

    print(f"\n{'#'*60}")
    print(f"ANCHOR: {label}")
    print(f"{'#'*60}")

    # Phase 1
    rx, probe_pixels = diagnose_anchor_frozen(seed, target_port, wl_nm, rules)
    p = rx["prescription"]

    print(f"  variant={p['variant']}  mask={p['mask']}  "
          f"sigma={p['sigma_single_measured']:.3e}  "
          f"thr={p['accept_threshold']:.3e}")

    # Shared jac
    jac = _get_jac(seed, target_port, wl_nm, RESULTS_DIR, reuse=True)

    # 4 arms
    arms_config = [
        ("RAW_DEFAULT",       "raw",           []),
        ("PHASE_DIAGNOSED",   p["variant"],    []),
        ("PHASE_MASKED",      p["variant"],    [tuple(m) for m in p["mask"]]),
        ("PHASE_MASKED_ACCEPT", p["variant"],  [tuple(m) for m in p["mask"]]),
    ]

    results = {}
    for arm_name, variant, mask in arms_config:
        print(f"\n  --- {arm_name} (variant={variant}, mask={mask}) ---")
        results[arm_name] = run_arm(
            seed, target_port, wl_nm, arm_name, variant, mask,
            p["accept_threshold"], jac, probe_pixels, rules)

    # Per-anchor verdict
    raw_net = results["RAW_DEFAULT"]["summary"]["net_dF"]
    phase_net = results["PHASE_DIAGNOSED"]["summary"]["net_dF"]
    masked_net = results["PHASE_MASKED"]["summary"]["net_dF"]
    accept_net = results["PHASE_MASKED_ACCEPT"]["summary"]["net_dF"]

    diag_activation = phase_net > raw_net
    max_rank_raw = results["RAW_DEFAULT"]["summary"]["max_dF_rank"]
    max_rank_phase = results["PHASE_DIAGNOSED"]["summary"]["max_dF_rank"]
    nonmonotonic = any(
        r["summary"]["max_dF_rank"] > 1 for r in results.values()
    )

    anchor_verdict = {
        "label": label,
        "diagnostic_activation": bool(diag_activation),
        "activation_margin": phase_net - raw_net,
        "mask_contribution": masked_net - phase_net,
        "accept_filter_contribution": accept_net - masked_net,
        "ranking_nonmonotonic": bool(nonmonotonic),
        "results": {arm: r["summary"] for arm, r in results.items()},
    }
    _save_json(anchor_verdict, SERIES_DIR / f"VERDICT_{label}.json")

    print(f"\n  VERDICT {label}: activation={'YES' if diag_activation else 'NO'} "
          f"(margin={diag_activation and phase_net - raw_net:+.4e})  "
          f"nonmonotonic={'YES' if nonmonotonic else 'NO'}")
    return anchor_verdict


# ═══════════════════════════════════════════════════════════════════════════
# Cross-anchor synthesis
# ═══════════════════════════════════════════════════════════════════════════

def synthesize(verdicts):
    n = len(verdicts)
    n_activated = sum(1 for v in verdicts if v["diagnostic_activation"])
    n_nonmono = sum(1 for v in verdicts if v["ranking_nonmonotonic"])

    synthesis = {
        "n_anchors": n,
        "anchors": [v["label"] for v in verdicts],
        "diagnostic_activation_rate": f"{n_activated}/{n}",
        "ranking_nonmonotonic_rate": f"{n_nonmono}/{n}",
        "per_anchor": {v["label"]: {
            "activation": v["diagnostic_activation"],
            "mask_contrib": v["mask_contribution"],
            "nonmonotonic": v["ranking_nonmonotonic"],
        } for v in verdicts},
    }

    print(f"\n{'#'*60}")
    print(f"CROSS-ANCHOR SYNTHESIS")
    print(f"  Diagnostic activation: {n_activated}/{n}")
    print(f"  Ranking non-monotonic: {n_nonmono}/{n}")
    print(f"{'#'*60}")

    _save_json(synthesis, SERIES_DIR / "SYNTHESIS.json")
    return synthesis


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="3-anchor prospective role-assignment series")
    ap.add_argument("--seeds", type=str, default="9,11,12",
                    help="comma-separated seed list")
    ap.add_argument("--port", type=int, default=1)
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    verdicts = []
    for seed in seeds:
        v = run_anchor(seed=seed, target_port=args.port)
        verdicts.append(v)

    synth = synthesize(verdicts)
    sys.exit(0 if synth["diagnostic_activation_rate"] == f"{len(seeds)}/{len(seeds)}" else 0)
