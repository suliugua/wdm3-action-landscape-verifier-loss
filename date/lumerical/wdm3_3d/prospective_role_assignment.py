#!/usr/bin/env python
"""Prospective role-assignment experiment — revised design.

Anchor: seed9 P1@1550nm (fresh 3D anchor, not used in seed7/seed10 masked-greedy).

Three arms, pre-registered before data collection:

  ARM_DIAGNOSED   – Rule M/P/K diagnosis → frozen prescription → masked greedy
                    (selected-phase variant, mask, K_eff horizon)
  ARM_RAW_PHASED  – Same selected-phase variant, NO mask, NO K_eff stop,
                    same sequential accept/reject budget.
                    Isolates mask+K_eff contribution.
  ARM_RANDOM_MASKED – Matched random single-flip draws within the unmasked
                    (diagnosis-eligible) pixel pool. Fair baseline.

Discipline:
  - Phase 1 probe pixels are tracked and EXCLUDED from Phase 3 nomination pools.
  - Accept/reject threshold: ΔF > +2*sigma_single (not > 0).
  - K_eff stop: 3 consecutive candidates with ΔF ≤ +2*sigma_single.
  - Primary endpoint: DIAGNOSED > max(RAW_PHASED, RANDOM_MASKED)
    by at least 2*sigma_single.
  - Secondary: full ordering, hit@5, accepted count, K_eff, per-arm breakdown.

Resource budget: ~50 FDTD evaluations (~30-40 min at 30-40s/eval)
  Phase 1: 2 (adjoint) + ~4 (probes) + 15 (sigma) ≈ 21 evals
  Phase 3: ~10 (DIAGNOSED) + ~10 (RAW_PHASED) + ~10 (RANDOM_MASKED) ≈ 30 evals

Usage:
  python -m wdm3_3d.prospective_role_assignment --seed 9 --port 1
"""

import os
# ── MUST be set and imported before ANY wdm3_3d import ──
os.environ.setdefault("LUMOPT_BACKEND", "2026")
import lum_backend  # noqa: F401 — side-effect: adds Lumerical API to sys.path

import json
import time
import sys
from pathlib import Path

import numpy as np

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
    propose_suspicion_zones,
    zone_probe_plan,
    mask_from_zones,
    apply_col_mask,
    phase_scan,
    phi_in_plateaus,
    variant_phi_deg,
    k_eff_from_rank_curve,
    gain_stats,
    diagnose_anchor,
)

EXP_DIR = LANDSCAPE_DIR / "prospective"

# ── Frozen accept/reject rule (2026-07-23, pre-registered) ──
# Use 2*sigma_single as acceptance threshold. This is stricter than the
# original accept_floor=0.0 and ensures accepted flips exceed the local
# random noise floor.
ACCEPT_SIGMA_MULT = 2.0
K_EFF_CONSECUTIVE = 3  # consecutive sub-threshold steps trigger K_eff stop


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Diagnosis (prospective, frozen before greedy)
# ═══════════════════════════════════════════════════════════════════════════

def phase1_diagnose(seed, target_port, wl_nm=WL_NM, rules=RULES):
    """Run Rule M/P/K diagnosis and FREEZE the prescription.

    RETURNS the prescription dict AND the set of probe pixel indices (to be
    excluded from Phase 3 nomination pools — prevents information leak).
    """
    label = _label(seed, target_port, wl_nm)
    diag_path = EXP_DIR / f"PRESCRIPTION_{label}.json"

    if diag_path.exists():
        print(f"\n[Phase 1] Prescription already frozen: {diag_path}")
        rx = json.load(open(diag_path))
        return rx, set(rx.get("probe_pixels", []))

    EXP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"PHASE 1 — DIAGNOSIS (prospective, frozen before greedy)")
    print(f"  Anchor: {label}")
    print(f"{'='*70}")

    t0 = time.time()
    diag = diagnose_anchor(
        seed=seed, target_port=target_port, wl_nm=wl_nm,
        variant="auto",
        replay=False,
        rules=rules,
        out_dir=EXP_DIR,
        jac_dir=RESULTS_DIR,
    )

    # ── Collect probe pixel indices (exclude from Phase 3) ──
    probe_pixels = set()
    for z in diag.get("zones", []):
        for v in z.get("verdicts", []):
            probe_pixels.add(int(v["idx"]))
    print(f"\n  Probe pixels (excluded from Phase 3): "
          f"{len(probe_pixels)} pixels {sorted(probe_pixels)}")

    # ── Rule K: sigma_single from live random single-flips ──
    print("\n  [Rule K] Measuring sigma_single (local random null)...")
    params = np.asarray(load_warmstart_params(seed), dtype=float).reshape(-1)
    mask = [tuple(m) for m in diag["mask"]]
    excl = apply_col_mask(len(params), mask)

    unmasked_pool = np.where(~excl)[0]
    # Exclude probe pixels from sigma measurement too
    sigma_pool = np.array([i for i in unmasked_pool if i not in probe_pixels])
    rng = np.random.RandomState(seed * 100 + target_port)
    k_single = min(15, len(sigma_pool))
    draw_idx = rng.choice(sigma_pool, size=k_single, replace=False)

    single_dFs = []
    with IncrementalSession(params, target_port, wl_nm,
                            tag=f"sigma_{label}") as S:
        f0 = S.run_baseline()
        for idx in draw_idx:
            f = S.measure([int(idx)])
            single_dFs.append(float(f - f0))
    sigma_measured = float(np.std(single_dFs))
    accept_threshold = ACCEPT_SIGMA_MULT * sigma_measured
    print(f"  sigma_single = {sigma_measured:.3e} (n={k_single})")
    print(f"  accept threshold = {ACCEPT_SIGMA_MULT}*sigma "
          f"= {accept_threshold:.3e}")

    # ── Freeze prescription ──
    prescription = {
        "label": label,
        "seed": seed,
        "target_port": target_port,
        "wl_nm": wl_nm,
        "timestamp_diagnosis": _now_iso(),
        "status": "FROZEN — do not edit based on greedy outcomes",
        "diagnosis": diag,
        "probe_pixels": sorted(probe_pixels),
        "prescription": {
            "variant": diag["variant"],
            "mask": diag["mask"],
            "sigma_single_measured": sigma_measured,
            "accept_threshold": accept_threshold,
            "k_max_budget": 10,
            "k_eff_consecutive": K_EFF_CONSECUTIVE,
        },
        "accept_rule": {
            "criterion": f"ΔF > +{ACCEPT_SIGMA_MULT}*sigma_single",
            "sigma_single": sigma_measured,
            "threshold": accept_threshold,
        },
        "k_eff_rule": {
            "criterion": (f"{K_EFF_CONSECUTIVE} consecutive candidates "
                          f"with ΔF ≤ {ACCEPT_SIGMA_MULT}*sigma"),
        },
        "rules_snapshot": rules,
        "wall_s": time.time() - t0,
    }
    _save_json(prescription, diag_path)
    print(f"\n  Prescription frozen → {diag_path}")
    return prescription, probe_pixels


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Role assignment (deterministic from diagnosis)
# ═══════════════════════════════════════════════════════════════════════════

def phase2_assign_roles(prescription):
    """Read frozen prescription → assign roles.

    Key design choice: RAW_PHASED uses the SAME selected-phase variant as
    DIAGNOSED, so any DIAGNOSED advantage comes from mask+K_eff, not from
    an unfair phase mismatch.
    """
    p = prescription["prescription"]

    roles = {
        "ARM_DIAGNOSED": {
            "description": "Masked proposal + K_eff horizon",
            "variant": p["variant"],
            "mask": [tuple(m) for m in p["mask"]],
            "k_max": p["k_max_budget"],
            "use_mask": True,
            "use_k_eff_stop": True,
        },
        "ARM_RAW_PHASED": {
            "description": (f"No mask, same phase ({p['variant']}), "
                            f"no K_eff stop — isolates mask+K_eff"),
            "variant": p["variant"],
            "mask": [],
            "k_max": p["k_max_budget"],
            "use_mask": False,
            "use_k_eff_stop": False,
        },
        "ARM_RANDOM_MASKED": {
            "description": "Matched random draws from unmasked (eligible) pool",
            "n_draws": p["k_max_budget"],
            "seed_offset": 999,
        },
    }

    print(f"\n{'='*70}")
    print(f"PHASE 2 — ROLE ASSIGNMENT (deterministic from diagnosis)")
    for arm, role in roles.items():
        print(f"  {arm}: {role['description']}")
    print(f"{'='*70}")

    return roles


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: FDTD acceptance test
# ═══════════════════════════════════════════════════════════════════════════

def _greedy_arm(seed, target_port, wl_nm, arm_name, arm_config, prescription,
                rules, shared_jac=None, probe_pixels=None, out_dir=None):
    """One greedy arm: rank candidates, sequential try_flip with FDTD verdict.

    Parameters
    ----------
    probe_pixels : set of int
        Pixel indices consumed by Phase 1 diagnostic probes. EXCLUDED from
        nomination pool to prevent information leak.
    """
    out_dir = Path(out_dir) if out_dir else EXP_DIR
    label = _label(seed, target_port, wl_nm)
    arm_label = f"{label}_{arm_name}"

    k_max = arm_config["k_max"]
    variant = arm_config["variant"]
    mask = arm_config["mask"]
    use_mask = arm_config["use_mask"]
    use_k_eff = arm_config["use_k_eff_stop"]
    accept_threshold = prescription["prescription"]["accept_threshold"]
    k_eff_n = prescription["prescription"]["k_eff_consecutive"]

    print(f"\n{'='*70}")
    print(f"PHASE 3 — {arm_name}")
    print(f"  variant={variant}  mask={mask}  k_max={k_max}  "
          f"k_eff_stop={use_k_eff}  accept_thr={accept_threshold:.3e}")
    print(f"{'='*70}")

    # ── Use shared jac ──
    if shared_jac is not None:
        jac = shared_jac
        print(f"  [shared jac]")
    else:
        jac = _get_jac(seed, target_port, wl_nm, RESULTS_DIR, reuse=True,
                       params=None, label_suffix=f"_{arm_name}")

    params = np.asarray(jac["params"], dtype=float).reshape(-1)
    gain = _flip_gain(jac["grads"][variant], params)

    # ── Build exclusion mask ──
    excl = apply_col_mask(len(params), mask)
    if probe_pixels:
        for px in probe_pixels:
            excl[int(px)] = True

    sel = np.where(excl, -np.inf, gain)
    nom_idx_all = [int(i) for i in np.argsort(sel)[::-1] if sel[int(i)] > 0]
    nom_idx = nom_idx_all[:k_max]

    n_masked = np.sum(apply_col_mask(len(params), mask))
    n_probe = len(probe_pixels) if probe_pixels else 0
    print(f"  nominated pixels: {len(nom_idx)} (of {len(nom_idx_all)} "
          f"positive-gain unmasked; {n_masked} masked + {n_probe} probe "
          f"= {n_masked + n_probe} excluded)")

    arm_record = {
        "arm": arm_name,
        "arm_config": arm_config,
        "label": label,
        "variant": variant,
        "mask": [list(m) for m in mask],
        "probe_pixels_excluded": sorted(probe_pixels) if probe_pixels else [],
        "accept_threshold": accept_threshold,
        "timestamp_start": _now_iso(),
        "rounds": [],
    }

    total_dF = 0.0
    history = []

    with IncrementalSession(params.copy(), target_port, wl_nm,
                            tag=f"greedy_{arm_label}") as S:
        f0 = S.run_baseline()
        print(f"  baseline |a|^2 = {f0:.6f}")

        consecutive_below = 0
        for rank, idx in enumerate(nom_idx, start=1):
            t_e = time.time()
            # ── FDTD verdict ──
            dF, accept = S.try_flip(idx, accept_threshold)
            entry = {
                "rank": rank,
                "idx": int(idx),
                "ix": int(idx) // 100,
                "iy": int(idx) % 100,
                "gain_pred": float(gain[idx]),
                "dF_meas": float(dF),
                "accepted": bool(accept),
                "wall_s": time.time() - t_e,
            }
            arm_record["rounds"].append(entry)

            status = "ACCEPT" if accept else "reject"
            print(f"  [{rank}/{len(nom_idx)}] idx={idx} "
                  f"pred={gain[idx]:+.4f} dF={dF:+.4e} {status} "
                  f"({entry['wall_s']:.0f}s)")

            if accept:
                total_dF += dF
                consecutive_below = 0
            else:
                consecutive_below += 1
            history.append(float(total_dF))

            # ── K_eff early stop ──
            if use_k_eff and consecutive_below >= k_eff_n:
                print(f"  [K_eff STOP] {k_eff_n} consecutive ΔF ≤ "
                      f"{accept_threshold:.3e} — horizon reached at rank {rank}")
                break

        f_end = S.fom

    # ── Summary ──
    accepted = [r for r in arm_record["rounds"] if r["accepted"]]
    all_dFs = [r["dF_meas"] for r in arm_record["rounds"]]
    arm_record["summary"] = {
        "baseline_abs2": float(f0),
        "end_abs2": float(f_end),
        "n_nominated": len(nom_idx),
        "n_accepted": len(accepted),
        "n_rejected": len(arm_record["rounds"]) - len(accepted),
        "net_dF": float(total_dF),
        "history": history,
        "hit_rate_top5": (sum(1 for r in arm_record["rounds"][:5]
                            if r["dF_meas"] > accept_threshold)
                          / min(5, len(arm_record["rounds"]))),
        "k_eff_observed": k_eff_from_rank_curve(all_dFs, rules),
        "consecutive_reject_at_end": consecutive_below,
    }

    print(f"\n  {arm_name} SUMMARY: "
          f"net_dF={total_dF:+.4e}  accepted={len(accepted)}/"
          f"{len(arm_record['rounds'])}  "
          f"hit5={arm_record['summary']['hit_rate_top5']:.1f}  "
          f"K_eff={arm_record['summary']['k_eff_observed']}")

    _save_json(arm_record, out_dir / f"arm_{arm_label}.json")
    return arm_record


def _random_masked_arm(seed, target_port, wl_nm, arm_config, prescription,
                       rules, probe_pixels=None, out_dir=None):
    """Matched random baseline: draws from unmasked (diagnosis-eligible) pool.

    Samples ONLY from pixels that survive the diagnosis mask — this ensures
    DIAGNOSED > RANDOM_MASKED reflects genuine proposal quality, not just
    regional prior.
    """
    out_dir = Path(out_dir) if out_dir else EXP_DIR
    label = _label(seed, target_port, wl_nm)
    arm_label = f"{label}_ARM_RANDOM_MASKED"

    n_draws = arm_config["n_draws"]
    seed_offset = arm_config["seed_offset"]
    mask = prescription["prescription"]["mask"]
    accept_threshold = prescription["prescription"]["accept_threshold"]

    print(f"\n{'='*70}")
    print(f"PHASE 3 — ARM_RANDOM_MASKED "
          f"(n={n_draws}, unmasked pool only)")
    print(f"{'='*70}")

    params = np.asarray(load_warmstart_params(seed), dtype=float).reshape(-1)
    rng = np.random.RandomState(seed * 1000 + target_port * 100 + seed_offset)

    # ── Eligible pool: unmasked, non-probe pixels ──
    excl = apply_col_mask(len(params), mask)
    if probe_pixels:
        for px in probe_pixels:
            excl[int(px)] = True
    eligible_pool = np.where(~excl)[0]
    print(f"  eligible pool: {len(eligible_pool)}/{len(params)} pixels")

    draws = []
    total_dF = 0.0

    with IncrementalSession(params, target_port, wl_nm,
                            tag=f"rand_{arm_label}") as S:
        f0 = S.run_baseline()
        print(f"  baseline |a|^2 = {f0:.6f}")

        for i in range(n_draws):
            idx = int(rng.choice(eligible_pool))
            f = S.measure([idx])
            dF = float(f - f0)
            draws.append({"draw": i + 1, "idx": idx, "dF": dF})
            total_dF += dF

    accepted = [d for d in draws if d["dF"] > accept_threshold]
    hit5 = (sum(1 for d in draws[:5] if d["dF"] > accept_threshold)
            / min(5, len(draws)))

    rand_record = {
        "arm": "ARM_RANDOM_MASKED",
        "arm_config": arm_config,
        "label": label,
        "eligible_pool_size": len(eligible_pool),
        "accept_threshold": accept_threshold,
        "timestamp_start": _now_iso(),
        "draws": draws,
        "summary": {
            "baseline_abs2": float(f0),
            "n_draws": n_draws,
            "n_accepted": len(accepted),
            "net_dF": float(total_dF),
            "mean_dF": float(np.mean([d["dF"] for d in draws])),
            "std_dF": float(np.std([d["dF"] for d in draws])),
            "hit_rate_top5": hit5,
        },
    }

    print(f"\n  RANDOM_MASKED SUMMARY: net_dF={total_dF:+.4e}  "
          f"mean={rand_record['summary']['mean_dF']:+.4e}  "
          f"hit5={hit5:.1f}")

    _save_json(rand_record, out_dir / f"arm_{arm_label}.json")
    return rand_record


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(seed=9, target_port=1, wl_nm=WL_NM, rules=RULES):
    """Full prospective role-assignment experiment."""
    label = _label(seed, target_port, wl_nm)
    verdict_path = EXP_DIR / f"VERDICT_{label}.json"

    print(f"\n{'#'*70}")
    print(f"PROSPECTIVE ROLE-ASSIGNMENT EXPERIMENT")
    print(f"  Anchor: {label}")
    print(f"  Design: 3-arm (DIAGNOSED vs RAW_PHASED vs RANDOM_MASKED)")
    print(f"  Accept rule: ΔF > +{ACCEPT_SIGMA_MULT}*sigma_single")
    print(f"  K_eff stop: {K_EFF_CONSECUTIVE} consecutive ΔF ≤ threshold")
    print(f"  Status: pre-registered — roles frozen before greedy")
    print(f"{'#'*70}")

    # ── Phase 1: Diagnose & freeze ──
    prescription, probe_pixels = phase1_diagnose(seed, target_port, wl_nm, rules)

    # ── Phase 2: Assign roles ──
    roles = phase2_assign_roles(prescription)

    # ── Compute shared base jac (2 sims) ──
    print("\n" + "="*70)
    print("Computing shared base jac for greedy arms...")
    print("="*70)
    shared_jac = _get_jac(seed, target_port, wl_nm, RESULTS_DIR, reuse=True,
                          params=None, label_suffix="")

    # ── Phase 3: Three-arm FDTD acceptance test ──
    results = {}

    print("\n" + "="*70)
    print("ARM 1/3 — DIAGNOSED (mask + selected phase + K_eff)")
    print("="*70)
    results["ARM_DIAGNOSED"] = _greedy_arm(
        seed, target_port, wl_nm, "ARM_DIAGNOSED",
        roles["ARM_DIAGNOSED"], prescription, rules,
        shared_jac=shared_jac, probe_pixels=probe_pixels, out_dir=EXP_DIR)

    print("\n" + "="*70)
    print("ARM 2/3 — RAW_PHASED (no mask, same phase, no K_eff)")
    print("="*70)
    results["ARM_RAW_PHASED"] = _greedy_arm(
        seed, target_port, wl_nm, "ARM_RAW_PHASED",
        roles["ARM_RAW_PHASED"], prescription, rules,
        shared_jac=shared_jac, probe_pixels=probe_pixels, out_dir=EXP_DIR)

    print("\n" + "="*70)
    print("ARM 3/3 — RANDOM_MASKED (unmasked pool only)")
    print("="*70)
    results["ARM_RANDOM_MASKED"] = _random_masked_arm(
        seed, target_port, wl_nm,
        roles["ARM_RANDOM_MASKED"], prescription, rules,
        probe_pixels=probe_pixels, out_dir=EXP_DIR)

    # ── Verdict ──
    diag_net  = results["ARM_DIAGNOSED"]["summary"]["net_dF"]
    raw_net   = results["ARM_RAW_PHASED"]["summary"]["net_dF"]
    rand_net  = results["ARM_RANDOM_MASKED"]["summary"]["net_dF"]
    sigma     = prescription["prescription"]["sigma_single_measured"]
    threshold = prescription["prescription"]["accept_threshold"]
    diag_hit5 = results["ARM_DIAGNOSED"]["summary"]["hit_rate_top5"]
    raw_hit5  = results["ARM_RAW_PHASED"]["summary"]["hit_rate_top5"]
    rand_hit5 = results["ARM_RANDOM_MASKED"]["summary"]["hit_rate_top5"]
    diag_acc  = results["ARM_DIAGNOSED"]["summary"]["n_accepted"]
    raw_acc   = results["ARM_RAW_PHASED"]["summary"]["n_accepted"]
    diag_keff = results["ARM_DIAGNOSED"]["summary"]["k_eff_observed"]

    max_control = max(raw_net, rand_net)
    primary_pass = (diag_net > max_control + 2 * sigma)

    # ── Decomposition: mask contribution, phase contribution, K_eff ──
    # These are secondary, mechanistic readings
    mask_contrib = diag_net - raw_net  # mask + K_eff combined
    ordering = " > ".join(
        f"{name}({val:+.4e})" for name, val in
        sorted([("DIAGNOSED", diag_net), ("RAW_PHASED", raw_net),
                ("RANDOM_MASKED", rand_net)], key=lambda x: -x[1])
    )

    verdict = {
        "label": label,
        "timestamp_verdict": _now_iso(),
        "prescription": str(EXP_DIR / f"PRESCRIPTION_{label}.json"),
        "rules": {
            "accept_criterion": f"ΔF > +{ACCEPT_SIGMA_MULT}*sigma",
            "sigma_single": sigma,
            "accept_threshold": threshold,
            "k_eff_consecutive": K_EFF_CONSECUTIVE,
        },
        "results": {
            arm: {
                "net_dF": r["summary"]["net_dF"],
                "hit_rate_top5": r["summary"]["hit_rate_top5"],
                "n_accepted": r["summary"]["n_accepted"],
            }
            for arm, r in results.items()
        },
        "ordering": ordering,
        "primary_endpoint": {
            "criterion": ("DIAGNOSED > max(RAW_PHASED, RANDOM_MASKED) "
                          f"+ 2*sigma ({2*sigma:.3e})"),
            "diagnosed": diag_net,
            "max_control": max_control,
            "margin": diag_net - max_control,
            "PASS": bool(primary_pass),
        },
        "secondary": {
            "mask_k_eff_contribution": f"{mask_contrib:+.4e}",
            "hit5": f"DIAGNOSED={diag_hit5:.1f} RAW={raw_hit5:.1f} RANDOM={rand_hit5:.1f}",
            "accepted_count": f"DIAGNOSED={diag_acc} RAW={raw_acc}",
            "k_eff": diag_keff,
        },
        "claim": (
            f"Diagnosis-prescribed role (masked+gated) "
            f"{'OUTPERFORMS' if primary_pass else 'does NOT outperform'} "
            f"both raw-phased and random-masked controls by >2*sigma. "
            f"DIAGNOSED={diag_net:+.4e} vs "
            f"RAW_PHASED={raw_net:+.4e} vs "
            f"RANDOM_MASKED={rand_net:+.4e}. "
            f"sigma={sigma:.3e}, threshold={threshold:.3e}."
        ),
    }

    _save_json(verdict, verdict_path)

    print(f"\n{'#'*70}")
    print(f"VERDICT: {'PASS' if primary_pass else 'FAIL'}")
    print(f"  {verdict['claim']}")
    print(f"  Ordering: {ordering}")
    print(f"  Margin over max(control): {diag_net - max_control:+.4e} "
          f"({'PASS' if primary_pass else 'BELOW'} 2*sigma={2*sigma:.3e})")
    print(f"{'#'*70}")

    return verdict


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Prospective role-assignment experiment (revised)")
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--port", type=int, default=1)
    ap.add_argument("--wl", type=float, default=1550.0)
    args = ap.parse_args()

    verdict = run_experiment(seed=args.seed, target_port=args.port,
                             wl_nm=args.wl)

    primary_pass = verdict["primary_endpoint"]["PASS"]
    sys.exit(0 if primary_pass else 1)
