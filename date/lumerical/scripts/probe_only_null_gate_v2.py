#!/usr/bin/env python3
"""
Probe-only: run null-gate warmup + 30-eval yield probe, save trajectory, STOP.

Fills the 3 data gaps from the gate-loss replay:
  seed7  — has diagnosis but no exp6 probe trajectory
  seed8  — null gate never directly tested (only sign gate)
  seed10 — has diagnosis but no exp6 probe trajectory

Usage: python -u probe_only_null_gate.py --anchor seed7
       python -u probe_only_null_gate.py --anchor seed8
       python -u probe_only_null_gate.py --anchor seed10
"""
import sys, os, json, time, hashlib, argparse
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, PARENT_DIR)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["LUMOPT_BACKEND"] = "2026"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from config import (RESULTS_BASE, ANCHORS, DEFAULT_K,
                    RNG_BASE_SEED, PATCH_SIZES, PATCH_TYPES)
from shared import load_jac_npz
from shared.index_utils import flat_to_col, flat_to_row, N_COLS, N_ROWS
from strategies import _global_patch_scan
from run_pilot import Session, compute_scaled_threshold
from budget import BudgetLedger

WARMUP_MAX_EVAL = 50
WARMUP_MIN_ACCEPTS = 5
PROBE_BUDGET = 20

BANDS = [(51, 200), (201, 500), (501, 1000), (1000, 99999)]
SPATIAL_GROUP = "spatial"


def _hash(indices):
    return hashlib.md5(",".join(str(i) for i in sorted(indices)).encode()).hexdigest()[:12]


def _any_masked(indices, mask_zones):
    for idx in indices:
        col = flat_to_col(idx)
        for lo, hi in mask_zones:
            if lo <= col <= hi:
                return True
    return False


def build_strata(anchor, diag):
    mask = diag["mask"]
    variant = diag["variant"]
    port = ANCHORS[anchor]["port"]
    jac_dir = os.path.join(PARENT_DIR, "wdm3_3d", "results_3d", "jac3d")
    jac_path = os.path.join(jac_dir, f"jac_{anchor}_P{port}@1550nm_g0.npz")
    if not os.path.exists(jac_path):
        jac_path = os.path.join(jac_dir, f"jac_{anchor}_P{port}@1550nm.npz")
    jac = load_jac_npz(jac_path)
    gain_gated = jac.get(f"gain_{variant}", jac["gain_raw"])

    import config as _cfg
    _cfg.PATCH_SIZES = PATCH_SIZES
    _cfg.PATCH_TYPES = PATCH_TYPES
    all_patches = _global_patch_scan(np.abs(gain_gated), gain_raw=gain_gated)

    scored = []
    for p in all_patches:
        if _any_masked(p["indices"], mask):
            continue
        gs = 0.0
        for idx in p["indices"]:
            ix, iy = flat_to_col(idx), flat_to_row(idx)
            if 0 <= ix < N_COLS and 0 <= iy < N_ROWS:
                gs += abs(float(gain_gated[ix, iy]))
        if gs <= 0:
            continue
        scored.append({
            "hash": _hash(p["indices"]),
            "indices": p["indices"],
            "n": len(p["indices"]),
            "size": p["size"],
            "score": gs,
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    for i, s in enumerate(scored):
        s["rank"] = i + 1

    strata = {}
    for lo, hi in BANDS:
        key = f"band_{lo}_{hi}"
        strata[key] = [s for s in scored if lo <= s.get("rank", 0) <= hi]
    strata[SPATIAL_GROUP] = []
    return strata


def generate_spatial_patches(n, committed_pixels, rng):
    quadrants = [
        (0, 40, 0, 50), (40, 80, 0, 50),
        (0, 40, 50, 100), (40, 80, 50, 100),
    ]
    n_per_q = max(1, n // len(quadrants))
    seen = set()
    patches = []
    for ci_lo, ci_hi, rj_lo, rj_hi in quadrants:
        count = 0
        attempts = 0
        while count < n_per_q and attempts < n_per_q * 30:
            attempts += 1
            sz = int(rng.choice([3, 4, 5]))
            ci_max = max(ci_lo, min(ci_hi - sz, N_COLS - sz))
            rj_max = max(rj_lo, min(rj_hi - sz, N_ROWS - sz))
            if ci_max <= ci_lo or rj_max <= rj_lo:
                continue
            ci = int(rng.integers(ci_lo, ci_max + 1))
            rj = int(rng.integers(rj_lo, rj_max + 1))
            indices = []
            overlap = False
            for di in range(sz):
                for dj in range(sz):
                    idx = (ci + di) * N_ROWS + (rj + dj)
                    if idx in committed_pixels:
                        overlap = True
                        break
                    indices.append(idx)
                if overlap:
                    break
            if overlap or not indices:
                continue
            h = _hash(indices)
            if h in seen:
                continue
            seen.add(h)
            patches.append({
                "hash": h, "indices": sorted(indices),
                "size": int(sz), "n": len(indices),
                "score": float('nan'),
            })
            count += 1
    return patches[:n]


def try_and_accept_null(session, indices, frozen_diag, ledger, trajectory, z=2.0):
    """Null-calibrated verifier: accept if dF > threshold = max(0, mu*n + z*gamma*sigma*sqrt(n))."""
    n = len(indices)
    threshold = compute_scaled_threshold(n, frozen_diag, z=z)
    dF = session.try_patch(indices)
    ledger.charge("candidate_reject", n=1)
    trajectory.eval_count = ledger.eval_count
    if dF > threshold:
        if not ledger.is_exhausted():
            session.commit_patch(indices)
            ledger.charge("patch_accept", n=1)
            trajectory.eval_count = ledger.eval_count
            trajectory.cumulative_dJ += dF
            return dF, True, threshold
    return dF, False, threshold


def run_warmup(session, jac_path, frozen_diag, ledger, trajectory, rng):
    """Single-pixel null-gate warmup (same as exp6_policy_switch.py)."""
    from shared.index_utils import get_top_k_indices, apply_mask_to_field
    jac = load_jac_npz(jac_path)
    gain_raw = jac["gain_raw"]
    pool = set()
    accept_count = 0

    while not ledger.is_exhausted():
        if accept_count >= WARMUP_MIN_ACCEPTS:
            break
        if ledger.eval_count >= WARMUP_MAX_EVAL:
            break
        masked = apply_mask_to_field(gain_raw.copy(), [])
        for idx in pool:
            ix, iy = flat_to_col(idx), flat_to_row(idx)
            if 0 <= ix < N_COLS and 0 <= iy < N_ROWS:
                masked[ix, iy] = -float("inf")
        candidates = get_top_k_indices(masked, DEFAULT_K)
        if not candidates:
            break
        for idx in candidates:
            if ledger.is_exhausted() or accept_count >= WARMUP_MIN_ACCEPTS:
                break
            if idx in pool:
                continue
            dF, accepted = session.try_single(idx, threshold=frozen_diag["threshold_single"])
            pool.add(idx)
            ledger.charge("candidate_accept" if accepted else "candidate_reject", n=1)
            trajectory.eval_count = ledger.eval_count
            if accepted:
                accept_count += 1
                trajectory.cumulative_dJ += dF
                trajectory.record_step({
                    "type": "warmup_accept", "idx": idx, "dJ": dF,
                })
            else:
                trajectory.record_step({
                    "type": "warmup_reject", "idx": idx, "dJ": dF,
                })
    return list(pool)


def run_density_probe(session, strata, frozen_diag, ledger, trajectory,
                       tested_hashes, committed_pixels, rng,
                       target_evals=PROBE_BUDGET):
    """Null-gate multi-scale probe (same as exp6_policy_switch.py)."""
    group_stats = {}
    total_accepts = 0
    total_valid = 0
    total_attempted = 0
    skipped_overlap = 0
    skipped_duplicate = 0

    keys = list(strata.keys())
    key_idx = 0
    cursor = {k: 0 for k in keys}
    available_pool = {}
    for k in keys:
        if k == SPATIAL_GROUP:
            available_pool[k] = None
        else:
            available_pool[k] = [c for c in strata[k]
                                 if c["hash"] not in tested_hashes]

    while total_valid < target_evals and not ledger.is_exhausted():
        k = keys[key_idx % len(keys)]
        key_idx += 1

        if k == SPATIAL_GROUP:
            patches = generate_spatial_patches(1, committed_pixels, rng)
            if not patches:
                continue
            c = patches[0]
        else:
            pool = available_pool[k]
            found = False
            while cursor[k] < len(pool):
                c = pool[cursor[k]]
                cursor[k] += 1
                total_attempted += 1
                h = c["hash"]
                if h in tested_hashes:
                    skipped_duplicate += 1
                    continue
                if any(p in committed_pixels for p in c["indices"]):
                    skipped_overlap += 1
                    continue
                found = True
                break
            if not found:
                continue

        h = c["hash"]
        if h in tested_hashes:
            skipped_duplicate += 1
            continue
        if any(p in committed_pixels for p in c["indices"]):
            skipped_overlap += 1
            continue

        tested_hashes.add(h)
        dF, accepted, _ = try_and_accept_null(
            session, c["indices"], frozen_diag, ledger, trajectory)
        total_valid += 1
        if accepted:
            total_accepts += 1
            committed_pixels.update(c["indices"])
            trajectory.record_step({
                "type": "probe_accept",
                "indices": c["indices"], "dJ": dF, "group": k,
            })
            if k not in group_stats:
                group_stats[k] = {"sampled": 0, "accepts": 0}
            group_stats[k]["sampled"] += 1
            group_stats[k]["accepts"] += 1
        else:
            trajectory.record_step({
                "type": "probe_reject",
                "indices": c["indices"], "dJ": dF, "group": k,
            })
            if k not in group_stats:
                group_stats[k] = {"sampled": 0, "accepts": 0}
            group_stats[k]["sampled"] += 1

    for k in group_stats:
        gs = group_stats[k]
        gs["rho"] = gs["accepts"] / gs["sampled"] if gs["sampled"] > 0 else 0.0

    rho_aggregate = total_accepts / total_valid if total_valid > 0 else 0.0

    probe_log = {
        "attempted": total_attempted + total_valid,
        "valid": total_valid,
        "skipped_overlap": skipped_overlap,
        "skipped_duplicate": skipped_duplicate,
    }

    return rho_aggregate, group_stats, probe_log


def compute_yield_probe(trajectory):
    """Extract yield metrics from probe-phase steps."""
    probe_dJs = []
    n_accepts = 0
    for step in trajectory.steps:
        if 'probe' in step.get('type', ''):
            dJ = step.get('dJ', 0.0)
            probe_dJs.append(dJ)
            if 'accept' in step['type']:
                n_accepts += 1

    n = len(probe_dJs)
    if n == 0:
        return dict(rho_protocol=0.0, rho_pos=0.0, mu_plus=0.0, Y=0.0,
                    max_dJ=0.0, n_probe=0)

    dJs = np.array(probe_dJs)
    pos = dJs[dJs > 0]
    rho_protocol = n_accepts / n
    rho_pos = len(pos) / n
    mu_plus = float(np.mean(pos)) if len(pos) > 0 else 0.0
    Y = float(np.sum(np.maximum(dJs, 0)) / n)
    max_dJ = float(np.max(dJs))

    return dict(rho_protocol=rho_protocol, rho_pos=rho_pos,
                mu_plus=mu_plus, Y=Y, max_dJ=max_dJ, n_probe=n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", required=True)
    args = parser.parse_args()
    anchor = args.anchor

    t0 = time.time()
    rng = np.random.default_rng(RNG_BASE_SEED)
    out_dir = os.path.join(RESULTS_BASE, "final_rerun_2026_08_13_protocol_v2", anchor, "null_gate_probe_v2")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"NULL-GATE PROBE ONLY: {anchor}")
    print(f"Warmup: max {WARMUP_MAX_EVAL} eval, min {WARMUP_MIN_ACCEPTS} accepts")
    print(f"Probe:  {PROBE_BUDGET} valid patch evals")
    print("=" * 60)

    # -- Load diagnosis --
    cache_path = os.path.join(RESULTS_BASE, "shared", anchor, "diagnosis.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            frozen_diag = json.load(f)
    else:
        raw_path = os.path.join(RESULTS_BASE, "selector", "selector_raw", anchor,
                                "budget200", "repeat0", "result.json")
        with open(raw_path) as f:
            frozen_diag = json.load(f)["diagnosis"]
    frozen_diag["threshold_single"] = frozen_diag.get(
        "threshold_single", frozen_diag["sigma_single"] * 2)

    # C
    if "C" not in frozen_diag:
        mask_ok = bool(frozen_diag.get("mask", []))
        mu_ok = frozen_diag.get("mu_single", 0) <= 0
        frozen_diag["C"] = 1.0 if (mask_ok and mu_ok) else (0.5 if mu_ok else 0.0)

    C = frozen_diag["C"]
    mu_single = frozen_diag.get("mu_single", 0)
    sigma_single = frozen_diag.get("sigma_single", 0)
    print(f"Diagnosis: C={C:.2f}  mu_single={mu_single:+.2e}  sigma_single={sigma_single:.2e}  "
          f"mask={frozen_diag.get('mask','?')}")

    # -- Build strata --
    print("\nBuilding strata...")
    strata = build_strata(anchor, frozen_diag)

    port = ANCHORS[anchor]["port"]
    jac_dir = os.path.join(PARENT_DIR, "wdm3_3d", "results_3d", "jac3d")
    jac_path = os.path.join(jac_dir, f"jac_{anchor}_P{port}@1550nm_g0.npz")
    if not os.path.exists(jac_path):
        jac_path = os.path.join(jac_dir, f"jac_{anchor}_P{port}@1550nm.npz")

    import lum_backend, _lumopt_compat  # noqa
    from shared import Trajectory

    total_budget = WARMUP_MAX_EVAL + PROBE_BUDGET + 10  # generous cap
    ledger = BudgetLedger(total_budget=total_budget)
    trajectory = Trajectory("null_gate_probe", anchor, total_budget, 0)

    with Session(anchor, tag="null_probe") as session:
        # Phase 1: Warmup
        print(f"\n{'--'*20}\nWarmup (null gate, single-pixel)\n{'--'*20}")
        warmup_pool = run_warmup(session, jac_path, frozen_diag,
                                  ledger, trajectory, rng)
        warmup_evals = ledger.eval_count
        warmup_dJ = trajectory.cumulative_dJ
        committed_pixels = {s['idx'] for s in trajectory.steps
                            if s.get('type') == 'warmup_accept'}
        print(f"Warmup done: {warmup_evals} eval, {len(committed_pixels)} pixels accepted, "
              f"{len(warmup_pool)} pixels tested, dJ={warmup_dJ:+.4e}")

        # Track hashes
        tested_hashes = set()
        for step in trajectory.steps:
            idx_list = [step["idx"]] if "idx" in step else step.get("indices", [])
            if idx_list:
                tested_hashes.add(_hash(idx_list))

        # Phase 2: Probe
        remaining = ledger.remaining()
        target_probe = min(PROBE_BUDGET, remaining)
        print(f"\n{'--'*20}\nProbe (null gate, {target_probe} valid eval)\n{'--'*20}")
        rho_probe, group_stats, probe_log = run_density_probe(
            session, strata, frozen_diag, ledger, trajectory,
            tested_hashes, committed_pixels, rng,
            target_evals=target_probe)

    # -- Results --
    yp = compute_yield_probe(trajectory)
    probe_evals = probe_log["valid"]
    probe_accepts = int(rho_probe * probe_evals)

    # Also compute Y_gate and gate_loss_mass from trajectory
    probe_dJs_all = []
    accepted_dJs = []
    for step in trajectory.steps:
        if 'probe' in step.get('type', ''):
            dJ = step.get('dJ', 0.0)
            probe_dJs_all.append(dJ)
            if 'accept' in step['type']:
                accepted_dJs.append(dJ)

    n_probe = len(probe_dJs_all)
    Y_pos_val = float(np.sum(np.maximum(probe_dJs_all, 0)) / n_probe) if n_probe > 0 else 0.0
    Y_gate_val = float(np.sum(accepted_dJs) / n_probe) if n_probe > 0 else 0.0
    gate_loss_mass = 1.0 - Y_gate_val / Y_pos_val if Y_pos_val > 1e-15 else 0.0
    rho_pos_val = float(np.sum(np.array(probe_dJs_all) > 0) / n_probe) if n_probe > 0 else 0.0
    gate_loss_count = 1.0 - (probe_accepts / n_probe) / rho_pos_val if rho_pos_val > 0 else 0.0

    # Rejected positive actions
    rej_pos_dJs = []
    for step in trajectory.steps:
        if 'probe_reject' in step.get('type', '') and step.get('dJ', 0) > 0:
            rej_pos_dJs.append(step['dJ'])
    n_rej_pos = len(rej_pos_dJs)
    rej_pos_mass = float(np.sum(rej_pos_dJs)) if rej_pos_dJs else 0.0

    result = {
        "anchor": anchor,
        "experiment": "null_gate_probe_v2",
        "C": C,
        "mu_single": mu_single,
        "sigma_single": sigma_single,
        "rho_protocol": float(rho_probe),
        "probe_evals": probe_evals,
        "probe_accepts": probe_accepts,
        "probe_budget": PROBE_BUDGET,
        "verifier": "null_gate",
        "warmup_evals": warmup_evals,
        "warmup_dJ": float(warmup_dJ),
        "total_evals": ledger.eval_count,
        "yield_probe": {
            "rho_protocol": yp["rho_protocol"],
            "rho_pos": yp["rho_pos"],
            "mu_plus": yp["mu_plus"],
            "Y_pos": Y_pos_val,
            "Y_gate": Y_gate_val,
            "gate_loss_mass": gate_loss_mass,
            "gate_loss_count": gate_loss_count,
            "max_dJ": yp["max_dJ"],
            "n_probe": yp["n_probe"],
        },
        "rejected_positive": {
            "n_rejected_pos": n_rej_pos,
            "rejected_pos_mass": float(rej_pos_mass),
            "rejected_pos_dJs": [float(x) for x in rej_pos_dJs],
        },
        "gate_loss_mass": gate_loss_mass,
        "gate_loss_count": gate_loss_count,
        "Y_pos": Y_pos_val,
        "Y_gate": Y_gate_val,
        "group_stats": {k: v for k, v in group_stats.items()},
        "probe_log": probe_log,
        "wall_clock_s": time.time() - t0,
        "metadata": {
            "protocol_version": "v2-compatible",
            "probe_budget": PROBE_BUDGET,
            "verifier_type": "null_gate",
            "purpose": "tomography probe only (NOT full optimization)",
            "bugfix_flags": ["reject_not_committed", "resume_ec_bi_restore", "sign_threshold_zero"],
            "rerun_dir": "final_rerun_2026_08_13_protocol_v2",
        },
    }

    traj_path = os.path.join(out_dir, "trajectory.json")
    result_path = os.path.join(out_dir, "result.json")
    trajectory.to_json(traj_path)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"PROBE COMPLETE: {anchor}")
    print(f"{'='*60}")
    print(f"  C={C:.2f}  mu_single={mu_single:+.2e}  sigma_single={sigma_single:.2e}")
    print(f"  rho_protocol={rho_probe:.3f}  rho_pos={rho_pos_val:.3f}  mu_plus={yp['mu_plus']:+.2e}")
    print(f"  Y_pos={Y_pos_val:+.2e}  Y_gate={Y_gate_val:+.2e}")
    print(f"  gate_loss_mass={gate_loss_mass:.1%}  gate_loss_count={gate_loss_count:.1%}")
    print(f"  n_rejected_pos={n_rej_pos}  rejected_pos_mass={rej_pos_mass:+.2e}")
    print(f"  warmup: {warmup_evals}e  probe: {probe_evals}e  total: {ledger.eval_count}e")
    print(f"  wall: {result['wall_clock_s']:.0f}s")
    print(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()
