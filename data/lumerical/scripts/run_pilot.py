#!/usr/bin/env python3
"""
Pilot v2: Phase-0 diagnosis (separate session) + try_flip optimization (clean session)

Core fixes:
  1. Phase-0 diagnosis runs in a separate session and produces a frozen diagnosis
  2. The optimization session starts from a clean warmstart and loads the frozen diagnosis
  3. Every single-pixel evaluation goes through IncrementalSession.try_flip (one run, atomic accept/revert)
  4. No manual toggle -> run -> toggle_back remains

Usage:
  python run_pilot.py --strategies naive_single,diag_single
"""

import sys, os, json, time
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, PARENT_DIR)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Force unbuffered output so stdout is visible immediately in the background
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

from config import (
    RESULTS_BASE, get_results_dir, ANCHORS, BUDGET_LEVELS,
    STRATEGY_BUDGET_MAP, N_REPEATS, RNG_BASE_SEED,
    NULL_DRAWS_K1, NULL_DRAWS_K5,
    DEFAULT_K, MAX_CONSECUTIVE_MISS, L4_MAX_EVAL_CAP,
    PATCH_SIZES, PATCH_TYPES,
    PROTOCOL_DEFAULTS, PERFORMANCE_DEFAULTS, SELECTOR_DEFAULTS,
)
from budget import BudgetLedger
from shared import Trajectory


PILOT_ANCHOR = "seed10"
PILOT_BUDGET = 500
PILOT_REPEAT = 0
strategies_DEFAULT = STRATEGY_BUDGET_MAP[PILOT_BUDGET]

# Module-level run config — set by main(), read by strategy functions
_run_config: dict = {}


# ═══════════════════════════════════════════════════════════════════════════
# Session wrapper: thin layer over Paper 1 IncrementalSession
# ═══════════════════════════════════════════════════════════════════════════

class Session:
    """Thin IncrementalSession wrapper exposing only try_flip / measure / baseline."""

    def __init__(self, anchor_name: str, tag: str = "opt", params=None):
        port = ANCHORS[anchor_name]["port"]
        from wdm3_3d.wdm3_3d_device import load_warmstart_params
        from wdm3_3d.wdm3_3d_landscape import IncrementalSession
        if params is None:
            seed_num = int(anchor_name.replace("seed", ""))
            params = np.asarray(load_warmstart_params(seed_num), dtype=float).reshape(-1)
        self._incr = IncrementalSession(params, target_port=port, tag=tag)
        self._params = params

    def __enter__(self):
        self._incr.__enter__()
        self._incr.run_baseline()
        return self

    def __exit__(self, *exc):
        self._incr.close()

    @property
    def fom(self) -> float:
        return self._incr.fom

    @property
    def params(self) -> "np.ndarray":
        """Current geometry as flat float array (0/1)."""
        return self._incr.state.astype(float)

    def try_single(self, idx: int, threshold: float = 0.0) -> tuple[float, bool]:
        """Atomic single-pixel evaluation: toggle -> run -> accept if dF > threshold else revert. One run."""
        return self._incr.try_flip(idx, accept_floor=threshold)

    def try_patch(self, indices: list[int]) -> float:
        """Test a patch: flip -> run -> revert. One run. Returns dF."""
        f_before = self._incr.fom
        f_after = self._incr.measure(indices, keep=False)
        return f_after - f_before

    def commit_patch(self, indices: list[int]):
        """Commit a patch (flip again and keep). One run."""
        self._incr.measure(indices, keep=True)

    def snapshot_fom(self) -> float:
        return self._incr.fom


# ═══════════════════════════════════════════════════════════════════════════
# Phase-0: diagnosis (separate session)
# ═══════════════════════════════════════════════════════════════════════════

def run_diagnosis_phase0(anchor_name: str, rng: np.random.Generator) -> dict:
    """Run every diagnostic in a separate session, producing frozen values. Closes it before returning."""
    import lum_backend, _lumopt_compat  # noqa
    from wdm3_3d.wdm3_3d_device import load_warmstart_params

    print("  [Phase-0] Starting diagnosis session ...")
    with Session(anchor_name, tag="diag") as diag_session:
        incr = diag_session._incr

        # Null calibration
        nulls_single = []
        for _ in range(NULL_DRAWS_K1):
            idx = int(rng.integers(0, 8000))
            dF, _ = incr.try_flip(idx, accept_floor=float("inf"))  # always revert
            nulls_single.append(dF)

        nulls_batch = []
        nan_retries = 0
        for _ in range(NULL_DRAWS_K5):
            # choice(replace=False) guarantees no duplicate indices
            indices = rng.choice(8000, size=5, replace=False).tolist()
            f_before = incr.fom
            f_after = incr.measure(indices, keep=False)
            dF = f_after - f_before
            if np.isnan(dF):
                nan_retries += 1
                indices = rng.choice(8000, size=5, replace=False).tolist()
                f_after = incr.measure(indices, keep=False)
                dF = f_after - f_before
            nulls_batch.append(dF)

        sigma_single = float(np.std(nulls_single, ddof=1))
        mu_single = float(np.mean(nulls_single))
        sigma_5 = float(np.std(nulls_batch, ddof=1))
        mu_5 = float(np.mean(nulls_batch))
        # NaN guards: fall back to defaults if calibration fails
        if np.isnan(sigma_single) or sigma_single < 1e-15:
            sigma_single = 1e-4
        if np.isnan(sigma_5) or sigma_5 < 1e-15:
            sigma_5 = sigma_single * np.sqrt(5)
        threshold_single = mu_single + 2 * sigma_single
        threshold_5 = mu_5 + 2 * sigma_5
        gamma_raw = sigma_5 / (sigma_single * np.sqrt(5))
        gamma = max(1.0, gamma_raw) if not np.isnan(gamma_raw) else 1.0

        # Rule M/P/K -- per-anchor frozen values read from config
        anchor_cfg = ANCHORS.get(anchor_name, {})
        mask = anchor_cfg.get("mask_paper1", [(0, 2)])
        variant = anchor_cfg.get("variant_paper1", "conj_a")
        K_eff = 5

    diagnosis = {
        "mask": mask,
        "variant": variant,
        "K_eff": K_eff,
        "sigma_single": sigma_single,
        "mu_single": mu_single,
        "sigma_5": sigma_5,
        "mu_5": mu_5,
        "gamma": gamma,
        "threshold_single": threshold_single,
        "threshold_5": threshold_5,
        "null_single_draws": len(nulls_single),
        "null_batch_draws": len(nulls_batch),
    }
    print(f"  [Phase-0] Done. sigma1={sigma_single:.6e} sigma5={sigma_5:.6e} "
          f"gamma={gamma:.4f} thr_single={threshold_single:.4e} thr5={threshold_5:.4e}")
    return diagnosis


def compute_scaled_threshold(n_flipped: int, diag: dict, z: float = 2.0) -> float:
    """Globally scaled null model: infer the threshold at any n from the Phase-0

    threshold = max(0, μ₁·n + z·γ·σ₁·√n)
    Here gamma = sigma_5 / (sigma_1 * sqrt(5)), so the batch-5 measurement calibrates
    """
    mu_1 = diag["mu_single"]
    sigma_1 = diag["sigma_single"]
    gamma = diag["gamma"]
    return max(0.0, mu_1 * n_flipped + z * gamma * sigma_1 * np.sqrt(n_flipped))


# ═══════════════════════════════════════════════════════════════════════════
# Progressive relaxation helpers (for diag patch/trajectory)
# ═══════════════════════════════════════════════════════════════════════════

def _spatial_diverse_seeds(gain_field: "np.ndarray", mask: list,
                           exclude_seeds: set, n_cells: int = 16,
                           k_per_cell: int = 3) -> set:
    """Grid-based diverse seed sampling OUTSIDE mask zones.

    mask = exclusion zones (pollution).  We divide the CLEAN area (outside mask)
    into a grid and pick top-k_per_cell gated pixels from each cell.
    Returns set of global flat indices.
    """
    from shared.index_utils import N_COLS, N_ROWS, get_top_k_indices, apply_mask_to_field

    # Work on a field with mask zones already excluded
    clean_field = apply_mask_to_field(gain_field, mask)
    grid_size = int(np.sqrt(n_cells))
    cell_h = N_COLS // grid_size
    cell_w = N_ROWS // grid_size
    seeds = set()

    for gi in range(grid_size):
        for gj in range(grid_size):
            i0, i1 = gi * cell_h, min((gi + 1) * cell_h, N_COLS)
            j0, j1 = gj * cell_w, min((gj + 1) * cell_w, N_ROWS)

            # Skip cells entirely inside mask
            cell_fully_masked = all(
                i0 >= lo and i1 <= hi + 1 for lo, hi in mask)
            if cell_fully_masked:
                continue

            cell = clean_field[i0:i1, j0:j1].copy()
            for idx in exclude_seeds:
                ix, iy = idx // N_ROWS, idx % N_ROWS
                if i0 <= ix < i1 and j0 <= iy < j1:
                    cell[ix - i0, iy - j0] = -float("inf")

            top_local = get_top_k_indices(cell, k_per_cell)
            for local_idx in top_local:
                li = local_idx // cell.shape[1]
                lj = local_idx % cell.shape[1]
                global_idx = (i0 + li) * N_ROWS + (j0 + lj)
                if 0 <= (i0 + li) < N_COLS and 0 <= (j0 + lj) < N_ROWS:
                    seeds.add(global_idx)

    return seeds


def _build_seed_pool(level: int, M_gated: int, C_explore: int,
                     accepted_seeds: list, mask: list,
                     gain_gated: "np.ndarray", gain_raw: "np.ndarray") -> tuple:
    """Build seed pixel pool according to relaxation level.

    Returns (seeds: set, level_label: str).
    """
    from shared.index_utils import N_COLS, N_ROWS, get_top_k_indices, apply_mask_to_field

    seeds = set(accepted_seeds)
    label = f"L{level}"

    if level >= 1:
        # B: gated top-M OUTSIDE mask (mask = exclusion zone)
        outside_mask = apply_mask_to_field(gain_gated, mask)
        for idx in seeds:
            ix, iy = idx // N_ROWS, idx % N_ROWS
            if 0 <= ix < N_COLS and 0 <= iy < N_ROWS:
                outside_mask[ix, iy] = -float("inf")
        top_gated = get_top_k_indices(outside_mask, M_gated)
        seeds.update(top_gated)
        label += f"_M{M_gated}"

    if level >= 2:
        # Spatial-diverse OUTSIDE mask
        grid_seeds = _spatial_diverse_seeds(gain_gated, mask, seeds)
        seeds.update(grid_seeds)
        label += "_spdiv"

    if level >= 3:
        # C: exploration outside the mask (wider aperture)
        outside = apply_mask_to_field(gain_raw, mask)
        for idx in seeds:
            ix, iy = idx // N_ROWS, idx % N_ROWS
            if 0 <= ix < N_COLS and 0 <= iy < N_ROWS:
                outside[ix, iy] = -float("inf")
        top_explore = get_top_k_indices(outside, C_explore)
        seeds.update(top_explore)
        label += f"_C{C_explore}"

    return seeds, label


def _relax_level(level: int, M_gated: int, C_explore: int) -> tuple:
    """Advance to next relaxation tier.

    Level 0→1→ (M expands 15→50→100) →2→3→4.
    Returns (new_level, new_M, new_C).
    """
    if level == 0:
        return 1, 15, 5
    if level == 1 and M_gated < 100:
        return 1, min(M_gated * 2, 100), C_explore
    if level == 1:
        return 2, M_gated, C_explore
    if level == 2:
        return 3, M_gated, 10  # expand exploration
    if level == 3:
        return 4, M_gated, C_explore  # rescue: global scan
    return 4, M_gated, C_explore  # fully exhausted


# ═══════════════════════════════════════════════════════════════════════════
# strategy functions (rewritten for the try_flip pattern)
# ═══════════════════════════════════════════════════════════════════════════

def run_naive_single_v2(
    session: Session,
    jac_path: str,
    ledger: BudgetLedger,
    trajectory: Trajectory,
    rng: np.random.Generator,
    frozen_diag: dict,
    line_dir: str = None,
    resume_state: dict = None,
) -> dict:
    """Naive single-pixel greedy: rank all pixels by raw gain, evaluate atomically with try_flip."""
    from shared import load_jac_npz
    from shared.index_utils import get_top_k_indices, apply_mask_to_field

    jac = load_jac_npz(jac_path)
    gain_raw = jac["gain_raw"]
    threshold = frozen_diag["threshold_single"]  # single-pixel threshold, NOT batch-5

    # ── Resume: restore tested pool ──
    pool = set(resume_state.get("pool", [])) if resume_state else set()
    accepted_count = resume_state.get("accepted_count", 0) if resume_state else 0
    consecutive_miss = resume_state.get("consecutive_miss", 0) if resume_state else 0
    if pool:
        print(f"  [naive_single] resume: {len(pool)} tested, {accepted_count} accepts",
              flush=True)
    print(f"  [naive_single] start, threshold={threshold:.4e}, budget_left={ledger.remaining()}", flush=True)

    while not ledger.is_exhausted():
        masked = apply_mask_to_field(gain_raw.copy(), [])  # no mask for naive
        for idx in pool:
            ix, iy = idx // 100, idx % 100
            if 0 <= ix < 80 and 0 <= iy < 100:
                masked[ix, iy] = -float("inf")
        candidates = get_top_k_indices(masked, DEFAULT_K)
        if not candidates:
            print(f"  [naive_single] no candidates left, break", flush=True)
            break

        accepted_this_round = 0
        for idx in candidates:
            if ledger.is_exhausted():
                break
            if idx in pool:
                continue

            dF, accepted = session.try_single(idx, threshold=threshold)
            pool.add(idx)
            ledger.charge("candidate_accept" if accepted else "candidate_reject", n=1)
            trajectory.eval_count = ledger.eval_count
            print(f"  [naive_single] idx={idx} dF={dF:+.4e} {'ACCEPT' if accepted else 'reject'}  "
                  f"ΣdJ={trajectory.cumulative_dJ + (dF if accepted else 0):+.4e}  "
                  f"evals={ledger.eval_count}/{ledger.total_budget}", flush=True)

            if accepted:
                accepted_count += 1
                trajectory.cumulative_dJ += dF
                trajectory.record_step({
                    "type": "single_accept", "idx": idx, "dJ": dF,
                })
                accepted_this_round += 1
                # ── Checkpoint every 5 accepts ──
                if line_dir and accepted_count % 5 == 0:
                    _dump_checkpoint(line_dir, "naive_single",
                                     session.params.copy(), ledger, trajectory,
                                     {"pool": list(pool), "accepted_count": accepted_count,
                                      "consecutive_miss": consecutive_miss})
            else:
                trajectory.record_step({
                    "type": "single_reject", "idx": idx, "dJ": dF,
                })

        trajectory.eval_count = ledger.eval_count

        if accepted_this_round == 0:
            consecutive_miss += 1
            if consecutive_miss >= MAX_CONSECUTIVE_MISS:
                break
        else:
            consecutive_miss = 0

    return {
        "final_dJ": trajectory.cumulative_dJ,
        "eval_spent": ledger.eval_count,
        "n_accepted": accepted_count,
    }


def run_diag_single_v2(
    session: Session,
    jac_path: str,
    ledger: BudgetLedger,
    trajectory: Trajectory,
    rng: np.random.Generator,
    frozen_diag: dict,
    line_dir: str = None,
    resume_state: dict = None,
) -> dict:
    """Diag-gated single-pixel greedy -- try_flip + frozen diagnosis."""
    from shared import load_jac_npz
    from shared.index_utils import get_top_k_indices, apply_mask_to_field

    mask = frozen_diag["mask"]
    variant = frozen_diag["variant"]
    K_eff = frozen_diag.get("K_eff", DEFAULT_K)
    threshold = frozen_diag["threshold_single"]  # single-pixel threshold, NOT batch-5

    jac = load_jac_npz(jac_path)
    gain_gated = jac.get(f"gain_{variant}", jac["gain_raw"])

    # ── Resume: restore tested pool ──
    pool = set(resume_state.get("pool", [])) if resume_state else set()
    accepted_count = resume_state.get("accepted_count", 0) if resume_state else 0
    consecutive_miss = resume_state.get("consecutive_miss", 0) if resume_state else 0
    if pool:
        print(f"  [diag_single] resume: {len(pool)} tested, {accepted_count} accepts",
              flush=True)
    print(f"  [diag_single] start, mask={mask} variant={variant} "
          f"threshold={threshold:.4e} budget_left={ledger.remaining()}", flush=True)

    while not ledger.is_exhausted():
        masked = apply_mask_to_field(gain_gated.copy(), mask)
        for idx in pool:
            ix, iy = idx // 100, idx % 100
            if 0 <= ix < 80 and 0 <= iy < 100:
                masked[ix, iy] = -float("inf")
        candidates = get_top_k_indices(masked, DEFAULT_K)
        if not candidates:
            print(f"  [diag_single] no candidates left, break", flush=True)
            break

        accepted_this_round = 0
        for idx in candidates:
            if ledger.is_exhausted():
                break
            if idx in pool:
                continue

            dF, accepted = session.try_single(idx, threshold=threshold)
            pool.add(idx)
            ledger.charge("candidate_accept" if accepted else "candidate_reject", n=1)
            trajectory.eval_count = ledger.eval_count
            print(f"  [diag_single] idx={idx} dF={dF:+.4e} {'ACCEPT' if accepted else 'reject'}  "
                  f"ΣdJ={trajectory.cumulative_dJ + (dF if accepted else 0):+.4e}  "
                  f"evals={ledger.eval_count}/{ledger.total_budget}", flush=True)

            if accepted:
                accepted_count += 1
                trajectory.cumulative_dJ += dF
                trajectory.record_step({
                    "type": "single_accept_gated", "idx": idx, "dJ": dF,
                })
                accepted_this_round += 1
                # ── Checkpoint every 5 accepts ──
                if line_dir and accepted_count % 5 == 0:
                    _dump_checkpoint(line_dir, "diag_single",
                                     session.params.copy(), ledger, trajectory,
                                     {"pool": list(pool), "accepted_count": accepted_count,
                                      "consecutive_miss": consecutive_miss})
            else:
                trajectory.record_step({
                    "type": "single_reject_gated", "idx": idx, "dJ": dF,
                })

        trajectory.eval_count = ledger.eval_count

        if accepted_this_round == 0:
            consecutive_miss += 1
            if consecutive_miss >= MAX_CONSECUTIVE_MISS:
                break
        else:
            consecutive_miss = 0

    return {
        "final_dJ": trajectory.cumulative_dJ,
        "eval_spent": ledger.eval_count,
        "n_accepted": accepted_count,
        "accepted_indices": sorted([s["idx"] for s in trajectory.steps
                                    if "accept" in s.get("type", "")]),
    }


def run_naive_patch_v2(
    session: Session,
    jac_path: str,
    ledger: BudgetLedger,
    trajectory: Trajectory,
    frozen_diag: dict,
    line_dir: str = None,
    resume_state: dict = None,
) -> dict:
    """Naive patch search over a global sliding window, with measure for test and commit."""
    from shared import load_jac_npz
    from strategies import _global_patch_scan

    jac = load_jac_npz(jac_path)
    gain_raw = jac["gain_raw"]
    gain_abs = np.abs(gain_raw)
    print(f"  [naive_patch] scanning all patches ...", flush=True)
    all_patches = _global_patch_scan(gain_abs, gain_raw=gain_raw)
    ranked = sorted(all_patches, key=lambda p: p["score"], reverse=True)
    print(f"  [naive_patch] {len(ranked)} candidates, budget_left={ledger.remaining()}", flush=True)

    # ── Resume: restore tested set and skip already-tested ──
    tested_patches = set()
    if resume_state:
        for key_list in resume_state.get("tested_keys", []):
            tested_patches.add(frozenset(key_list))
        print(f"  [naive_patch] resume: {len(tested_patches)} already tested",
              flush=True)

    ckpt_counter = 0
    for i, patch_info in enumerate(ranked):
        if ledger.is_exhausted():
            break

        key = frozenset(patch_info["indices"])
        if key in tested_patches:
            continue
        tested_patches.add(key)

        n_flipped = patch_info.get("n_pos") if patch_info["type"] == "sparse" else patch_info["size"] * patch_info["size"]
        threshold = compute_scaled_threshold(n_flipped, frozen_diag,
                                             z=_run_config.get("z_threshold", 2.0))
        dF = session.try_patch(patch_info["indices"])
        ledger.charge("candidate_reject", n=1)
        trajectory.eval_count = ledger.eval_count

        if dF > threshold:
            if ledger.is_exhausted():
                trajectory.record_step({
                    "type": "patch_accept_naive_skipped",
                    "indices": patch_info["indices"], "dJ": dF,
                    "reason": "budget_exhausted_before_commit",
                })
                break
            session.commit_patch(patch_info["indices"])
            ledger.charge("patch_accept", n=1)
            trajectory.eval_count = ledger.eval_count
            trajectory.cumulative_dJ += dF
            trajectory.record_step({
                "type": "patch_accept_naive",
                "indices": patch_info["indices"], "dJ": dF,
                "size": patch_info["size"], "patch_type": patch_info["type"],
            })
            print(f"  [naive_patch] #{i} ACCEPT sz={patch_info['size']} {patch_info['type']} "
                  f"dF={dF:+.4e} ΣdJ={trajectory.cumulative_dJ:+.4e}", flush=True)
            # ── Checkpoint after accept ──
            if line_dir:
                ckpt_counter += 1
                if ckpt_counter % 5 == 0:
                    _dump_checkpoint(line_dir, "naive_single+patch",
                                     session.params.copy(), ledger, trajectory,
                                     {"phase": "patch",
                                      "tested_keys": [list(k) for k in tested_patches]})
        else:
            trajectory.record_step({
                "type": "patch_reject_naive",
                "indices": patch_info["indices"], "dJ": dF,
            })
            if i % 20 == 0:
                print(f"  [naive_patch] #{i}/{len(ranked)} reject sz={patch_info['size']} "
                      f"dF={dF:+.4e} evals={ledger.eval_count}", flush=True)

        trajectory.eval_count = ledger.eval_count

    print(f"  [naive_patch] done, evals={ledger.eval_count}", flush=True)
    return {"final_dJ": trajectory.cumulative_dJ, "eval_spent": ledger.eval_count}


def run_diag_patch_v2(
    session: Session,
    jac_path: str,
    ledger: BudgetLedger,
    trajectory: Trajectory,
    frozen_diag: dict,
    accepted_single: list[int],
    line_dir: str = None,
    resume_state: dict = None,
) -> dict:
    """Diag-gated patch search with progressive relaxation (Level 0–4).

    Level 0: accepted clusters only
    Level 1: + gated top-M (M expanding 15→50→100)
    Level 2: + spatial-diverse gated seeds
    Level 3: + exploration outside the mask
    Level 4: rescue — naive-like global scan (marked)

    Any accept resets to Level 0 with expanded accepted set.
    All levels use scaled-null acceptance: max(0, μ₁n + 2γσ₁√n).
    """
    from shared import load_jac_npz, spatial_cluster
    from strategies import _cluster_patch_scan, _global_patch_scan

    mask = frozen_diag["mask"]
    variant = frozen_diag["variant"]

    jac = load_jac_npz(jac_path)
    gain_gated = jac.get(f"gain_{variant}", jac["gain_raw"])
    gain_raw = jac["gain_raw"]

    # ── Resume: restore full internal state ──
    if resume_state:
        accepted_all = list(resume_state.get("accepted_all", accepted_single))
        tested_keys_raw = resume_state.get("tested_keys", [])
        tested_patches = set()
        for k in tested_keys_raw:
            tested_patches.add(frozenset(k))
        total_accepts = resume_state.get("total_accepts", 0)
        level = resume_state.get("level", 0)
        M_gated = resume_state.get("M_gated", 15)
        C_explore = resume_state.get("C_explore", 5)
        consecutive_miss = resume_state.get("consecutive_miss", 0)
        l4_count = resume_state.get("l4_count", 0)
        print(f"  [diag_patch] resume: {len(tested_patches)} tested, "
              f"{total_accepts} accepts, L{level} M={M_gated} C={C_explore}, "
              f"l4_count={l4_count}", flush=True)
    else:
        accepted_all = list(accepted_single)
        tested_patches: set = set()
        total_accepts = 0
        level = 0
        M_gated = 15
        C_explore = 5
        consecutive_miss = 0
        l4_count = 0

    # ── Non-progressive fallback (ablation: no_progressive) ──
    if not _run_config.get("progressive_relaxation", True):
        seeds, _ = _build_seed_pool(1, 15, 5, accepted_all, mask, gain_gated, gain_raw)
        clusters = spatial_cluster(list(seeds))
        if clusters:
            all_patches = _cluster_patch_scan(gain_gated, clusters, mask=mask)
            ranked = sorted(all_patches, key=lambda p: p["score"], reverse=True)
            print(f"  [diag_patch] non-progressive: {len(clusters)} clusters → "
                  f"{len(ranked)} candidates, budget_left={ledger.remaining()}", flush=True)
            for patch_info in ranked:
                if ledger.is_exhausted():
                    break
                key = frozenset(patch_info["indices"])
                if key in tested_patches:
                    continue
                tested_patches.add(key)
                n_flipped = (patch_info.get("n_pos") if patch_info["type"] == "sparse"
                             else patch_info["size"] * patch_info["size"])
                threshold = compute_scaled_threshold(n_flipped, frozen_diag,
                                                     z=_run_config.get("z_threshold", 2.0))
                dF = session.try_patch(patch_info["indices"])
                ledger.charge("candidate_reject", n=1)
                trajectory.eval_count = ledger.eval_count
                if dF > threshold:
                    if ledger.is_exhausted():
                        trajectory.record_step({
                            "type": "patch_accept_gated_skipped",
                            "indices": patch_info["indices"], "dJ": dF,
                            "reason": "budget_exhausted_before_commit",
                        })
                        break
                    session.commit_patch(patch_info["indices"])
                    ledger.charge("patch_accept", n=1)
                    trajectory.eval_count = ledger.eval_count
                    trajectory.cumulative_dJ += dF
                    trajectory.record_step({
                        "type": "patch_accept_gated",
                        "indices": patch_info["indices"], "dJ": dF,
                        "size": patch_info["size"], "patch_type": patch_info["type"],
                    })
                    total_accepts += 1
                else:
                    trajectory.record_step({
                        "type": "patch_reject_gated",
                        "indices": patch_info["indices"], "dJ": dF,
                    })
        print(f"  [diag_patch] non-progressive done, {total_accepts} accepts, "
              f"evals={ledger.eval_count}", flush=True)
        return {"final_dJ": trajectory.cumulative_dJ, "eval_spent": ledger.eval_count,
                "n_patches_accepted": total_accepts, "peak_relaxation_level": 0}

    # ── Progressive relaxation loop ──

    print(f"  [diag_patch] start progressive, A={len(accepted_all)} accepted singles, "
          f"budget_left={ledger.remaining()}", flush=True)

    while not ledger.is_exhausted():
        # Build seed pool for current level
        seeds, level_label = _build_seed_pool(
            level, M_gated, C_explore, accepted_all, mask, gain_gated, gain_raw)

        # Generate candidates
        if level >= 4:
            gain_abs = np.abs(gain_raw)
            all_patches = _global_patch_scan(gain_abs, gain_raw=gain_raw)
            level_label = "L4_rescue"
        else:
            clusters = spatial_cluster(list(seeds))
            if not clusters:
                level, M_gated, C_explore = _relax_level(level, M_gated, C_explore)
                consecutive_miss = 0
                print(f"  [diag_patch] no clusters → relax to L{level} M={M_gated} C={C_explore}",
                      flush=True)
                continue
            all_patches = _cluster_patch_scan(gain_gated, clusters, mask=mask)

        ranked = sorted(all_patches, key=lambda p: p["score"], reverse=True)
        # Filter tested
        ranked = [p for p in ranked if frozenset(p["indices"]) not in tested_patches]

        if not ranked:
            level, M_gated, C_explore = _relax_level(level, M_gated, C_explore)
            consecutive_miss = 0
            print(f"  [diag_patch] {len(all_patches)} candidates all tested → relax to "
                  f"L{level} M={M_gated} C={C_explore}", flush=True)
            continue

        n_clusters = len(clusters) if level < 4 else 0
        print(f"  [diag_patch] {level_label}: {n_clusters} clusters → {len(ranked)} new candidates, "
              f"budget_left={ledger.remaining()}", flush=True)

        accepted_this_round = 0
        test_limit = DEFAULT_K if level < 4 else DEFAULT_K * 4  # test more in rescue
        l4_cap = _run_config.get("l4_cap", L4_MAX_EVAL_CAP) if level >= 4 else None

        for patch_info in ranked[:test_limit]:
            if ledger.is_exhausted():
                break
            if level >= 4 and l4_count >= l4_cap:
                break  # L4 cap reached, save budget

            key = frozenset(patch_info["indices"])
            if key in tested_patches:
                continue
            tested_patches.add(key)

            n_flipped = (patch_info.get("n_pos") if patch_info["type"] == "sparse"
                         else patch_info["size"] * patch_info["size"])
            threshold = compute_scaled_threshold(n_flipped, frozen_diag,
                                                 z=_run_config.get("z_threshold", 2.0))

            dF = session.try_patch(patch_info["indices"])
            ledger.charge("candidate_reject", n=1)
            trajectory.eval_count = ledger.eval_count
            if level >= 4:
                l4_count += 1

            if dF > threshold:
                if ledger.is_exhausted():
                    trajectory.record_step({
                        "type": f"patch_accept_gated_{level_label}_skipped",
                        "indices": patch_info["indices"], "dJ": dF,
                        "reason": "budget_exhausted_before_commit",
                        "relaxation_level": level,
                    })
                    break
                session.commit_patch(patch_info["indices"])
                ledger.charge("patch_accept", n=1)
                trajectory.eval_count = ledger.eval_count
                trajectory.cumulative_dJ += dF
                trajectory.record_step({
                    "type": f"patch_accept_gated_{level_label}",
                    "indices": patch_info["indices"], "dJ": dF,
                    "size": patch_info["size"], "patch_type": patch_info["type"],
                    "relaxation_level": level,
                })
                accepted_all.extend(patch_info["indices"])
                accepted_this_round += 1
                total_accepts += 1
                print(f"  [diag_patch] {level_label} ACCEPT sz={patch_info['size']} "
                      f"{patch_info['type']} dF={dF:+.4e} ΣdJ={trajectory.cumulative_dJ:+.4e}",
                      flush=True)
                # ── Checkpoint after accept ──
                if line_dir:
                    _dump_checkpoint(line_dir, "diag_single+patch",
                                     session.params.copy(), ledger, trajectory,
                                     {"phase": "patch",
                                      "accepted_single": accepted_single,
                                      "accepted_all": accepted_all,
                                      "tested_keys": [list(k) for k in tested_patches],
                                      "total_accepts": total_accepts,
                                      "level": level,
                                      "M_gated": M_gated,
                                      "C_explore": C_explore,
                                      "consecutive_miss": consecutive_miss,
                                      "l4_count": l4_count})
                # Reset on success
                level = 0
                M_gated = 15
                C_explore = 5
                break
            else:
                trajectory.record_step({
                    "type": f"patch_reject_gated_{level_label}",
                    "indices": patch_info["indices"], "dJ": dF,
                    "relaxation_level": level,
                })

        if accepted_this_round == 0:
            if level >= 4 and l4_count >= l4_cap:
                print(f"  [diag_patch] L4 cap ({l4_cap}) exhausted → stop rescue",
                      flush=True)
                # Checkpoint before stopping
                if line_dir:
                    _dump_checkpoint(line_dir, "diag_single+patch",
                                     session.params.copy(), ledger, trajectory,
                                     {"phase": "patch",
                                      "accepted_single": accepted_single,
                                      "accepted_all": accepted_all,
                                      "tested_keys": [list(k) for k in tested_patches],
                                      "total_accepts": total_accepts,
                                      "level": level, "M_gated": M_gated,
                                      "C_explore": C_explore,
                                      "consecutive_miss": consecutive_miss,
                                      "l4_count": l4_count})
                break  # exit while, stop burning budget on rescue
            consecutive_miss += 1
            if consecutive_miss >= MAX_CONSECUTIVE_MISS:
                level, M_gated, C_explore = _relax_level(level, M_gated, C_explore)
                consecutive_miss = 0
                print(f"  [diag_patch] {consecutive_miss} consecutive misses → relax to "
                      f"L{level} M={M_gated} C={C_explore}", flush=True)

    print(f"  [diag_patch] done, {total_accepts} accepts, "
          f"peak_level=L{level}, evals={ledger.eval_count}", flush=True)
    return {"final_dJ": trajectory.cumulative_dJ, "eval_spent": ledger.eval_count,
            "n_patches_accepted": total_accepts, "peak_relaxation_level": level}


# ═══════════════════════════════════════════════════════════════════════════
# Main: line runner
# ═══════════════════════════════════════════════════════════════════════════

def run_naive_trajectory_v2(
    session: Session,
    jac_path: str,
    ledger: BudgetLedger,
    trajectory: Trajectory,
    rng: np.random.Generator,
    frozen_diag: dict,
) -> dict:
    """Naive trajectory loop — iterative patch search with gradient refresh.

    Per round: refresh jac -> global scan -> test top-5 patches -> accept above threshold.
    Loop until the budget is exhausted or two consecutive rounds gain nothing.
    """
    from shared import load_jac_npz
    from strategies import _global_patch_scan

    consecutive_miss = 0
    tested_patches: set[frozenset] = set()
    total_accepted = 0

    round_num = 0
    current_jac_path = jac_path
    anchor_name = trajectory.anchor
    target_port = ANCHORS[anchor_name]["port"]
    while not ledger.is_exhausted():
        round_num += 1
        print(f"  [naive_traj] round {round_num} start, budget_left={ledger.remaining()}", flush=True)
        # Gradient refresh — real recompute after round 1 (initial jac is precomputed)
        if round_num > 1:
            ledger.charge("grad_refresh", n=1, wall_clock_s=360)
            try:
                current_jac_path = _refresh_jac_inplace(
                    session, anchor_name, target_port, round_num)
            except Exception as e:
                print(f"  [naive_traj] jac refresh failed: {e} — reusing stale jac", flush=True)

        jac = load_jac_npz(current_jac_path)
        gain_raw = jac["gain_raw"]
        gain_abs = np.abs(gain_raw)
        all_patches = _global_patch_scan(gain_abs, gain_raw=gain_raw)
        ranked = sorted(all_patches, key=lambda p: p["score"], reverse=True)

        accepted_this_round = 0
        tested_this_round = 0

        for patch_info in ranked:
            if ledger.is_exhausted():
                break
            if tested_this_round >= DEFAULT_K:
                break

            # Skip already-tested patches
            key_frozen = frozenset(patch_info["indices"])
            if key_frozen in tested_patches:
                continue

            n_flipped = patch_info.get("n_pos") if patch_info["type"] == "sparse" else patch_info["size"] * patch_info["size"]
            threshold = compute_scaled_threshold(n_flipped, frozen_diag,
                                                 z=_run_config.get("z_threshold", 2.0))

            if ledger.is_exhausted():
                break

            # Mark as tested ONLY after confirming budget is available
            tested_this_round += 1
            tested_patches.add(key_frozen)

            dF = session.try_patch(patch_info["indices"])
            ledger.charge("candidate_reject", n=1)
            trajectory.eval_count = ledger.eval_count

            if dF > threshold:
                if ledger.is_exhausted():
                    trajectory.record_step({
                        "type": "patch_accept_naive_traj_skipped",
                        "indices": patch_info["indices"], "dJ": dF,
                        "reason": "budget_exhausted_before_commit",
                    })
                    break
                session.commit_patch(patch_info["indices"])
                ledger.charge("patch_accept", n=1)
                trajectory.eval_count = ledger.eval_count
                total_accepted += 1
                trajectory.cumulative_dJ += dF
                trajectory.record_step({
                    "type": "patch_accept_naive_traj",
                    "indices": patch_info["indices"], "dJ": dF,
                    "size": patch_info["size"], "patch_type": patch_info["type"],
                })
                accepted_this_round += 1
            else:
                trajectory.record_step({
                    "type": "patch_reject_naive_traj",
                    "indices": patch_info["indices"], "dJ": dF,
                })

        trajectory.eval_count = ledger.eval_count

        if accepted_this_round == 0:
            consecutive_miss += 1
            if consecutive_miss >= MAX_CONSECUTIVE_MISS:
                break
        else:
            consecutive_miss = 0

    return {
        "final_dJ": trajectory.cumulative_dJ,
        "eval_spent": ledger.eval_count,
        "n_patches_accepted": total_accepted,
    }


def run_diag_trajectory_v2(
    session: Session,
    jac_path: str,
    ledger: BudgetLedger,
    trajectory: Trajectory,
    rng: np.random.Generator,
    frozen_diag: dict,
    accepted_history: list[int],
) -> dict:
    """Diag-gated trajectory loop with progressive relaxation + real jac refresh.

    Each round: refresh jac from current geometry → progressive relaxation
    (Level 0–4) over re-clustered patches → accept best above scaled null.
    Real jac refresh uses compute_jac_3d(params=current_params) after geometry changes.
    """
    from shared import load_jac_npz, spatial_cluster
    from strategies import _cluster_patch_scan, _global_patch_scan

    mask = frozen_diag["mask"]
    variant = frozen_diag["variant"]

    tested_patches: set = set()
    total_accepted = 0
    round_num = 0
    current_jac_path = jac_path  # starts with precomputed; refreshed after accepts

    # Load initial jac
    jac = load_jac_npz(current_jac_path)
    gain_gated = jac.get(f"gain_{variant}", jac["gain_raw"])
    gain_raw = jac["gain_raw"]
    anchor_name = trajectory.anchor
    target_port = ANCHORS[anchor_name]["port"]

    while not ledger.is_exhausted():
        round_num += 1
        print(f"  [diag_traj] round {round_num} start, "
              f"accepted_history={len(accepted_history)}px, budget_left={ledger.remaining()}",
              flush=True)

        # ── Non-progressive fallback (ablation: no_progressive) ──
        if not _run_config.get("progressive_relaxation", True):
            seeds, _ = _build_seed_pool(1, 15, 5, accepted_history,
                                        mask, gain_gated, gain_raw)
            clusters = spatial_cluster(list(seeds))
            round_accepted = 0
            if clusters:
                all_patches = _cluster_patch_scan(gain_gated, clusters, mask=mask)
                ranked = sorted(all_patches, key=lambda p: p["score"], reverse=True)
                for patch_info in ranked[:DEFAULT_K]:
                    if ledger.is_exhausted():
                        break
                    key = frozenset(patch_info["indices"])
                    if key in tested_patches:
                        continue
                    tested_patches.add(key)
                    n_flipped = (patch_info.get("n_pos") if patch_info["type"] == "sparse"
                                 else patch_info["size"] * patch_info["size"])
                    threshold = compute_scaled_threshold(n_flipped, frozen_diag,
                                                         z=_run_config.get("z_threshold", 2.0))
                    dF = session.try_patch(patch_info["indices"])
                    ledger.charge("candidate_reject", n=1)
                    trajectory.eval_count = ledger.eval_count
                    if dF > threshold:
                        if ledger.is_exhausted():
                            break
                        session.commit_patch(patch_info["indices"])
                        ledger.charge("patch_accept", n=1)
                        trajectory.eval_count = ledger.eval_count
                        total_accepted += 1
                        round_accepted += 1
                        accepted_history.extend(patch_info["indices"])
                        trajectory.cumulative_dJ += dF
                        trajectory.record_step({
                            "type": "patch_accept_gated_traj",
                            "indices": patch_info["indices"], "dJ": dF,
                            "size": patch_info["size"], "patch_type": patch_info["type"],
                            "round": round_num,
                        })
                    else:
                        trajectory.record_step({
                            "type": "patch_reject_gated_traj",
                            "indices": patch_info["indices"], "dJ": dF,
                            "round": round_num,
                        })
            if round_accepted == 0:
                print(f"  [diag_traj] non-progressive round {round_num} zero accept → stop",
                      flush=True)
                break
            # Refresh jac for next round
            if round_num > 1:
                ledger.charge("grad_refresh", n=1, wall_clock_s=360)
                try:
                    current_jac_path = _refresh_jac_inplace(
                        session, anchor_name, target_port, round_num)
                    jac = load_jac_npz(current_jac_path)
                    gain_gated = jac.get(f"gain_{variant}", jac["gain_raw"])
                    gain_raw = jac["gain_raw"]
                except Exception as e:
                    print(f"  [diag_traj] jac refresh failed: {e}", flush=True)
            continue  # next round

        # ── Progressive relaxation within this round ──
        level = 0
        M_gated = 15
        C_explore = 5
        round_accepted = 0
        l4_round_count = 0  # track L4 rescue evals per round

        while not ledger.is_exhausted():
            seeds, level_label = _build_seed_pool(
                level, M_gated, C_explore, accepted_history,
                mask, gain_gated, gain_raw)

            # Generate candidates
            if level >= 4:
                gain_abs = np.abs(gain_raw)
                all_patches = _global_patch_scan(gain_abs, gain_raw=gain_raw)
                level_label = "L4_rescue"
            else:
                clusters = spatial_cluster(list(seeds))
                if not clusters:
                    level, M_gated, C_explore = _relax_level(level, M_gated, C_explore)
                    continue
                all_patches = _cluster_patch_scan(gain_gated, clusters, mask=mask)

            ranked = sorted(all_patches, key=lambda p: p["score"], reverse=True)
            ranked = [p for p in ranked
                      if frozenset(p["indices"]) not in tested_patches]

            if not ranked:
                level, M_gated, C_explore = _relax_level(level, M_gated, C_explore)
                if level >= 4 and not ranked:
                    break  # truly exhausted
                continue

            # Test top candidates at this level
            l4_cap = _run_config.get("l4_cap", L4_MAX_EVAL_CAP) if level >= 4 else None
            test_limit = DEFAULT_K if level < 4 else DEFAULT_K * 2
            for patch_info in ranked[:test_limit]:
                if ledger.is_exhausted():
                    break
                if level >= 4 and l4_round_count >= l4_cap:
                    break  # L4 cap reached, save budget for next round

                key = frozenset(patch_info["indices"])
                if key in tested_patches:
                    continue
                tested_patches.add(key)

                n_flipped = (patch_info.get("n_pos") if patch_info["type"] == "sparse"
                             else patch_info["size"] * patch_info["size"])
                threshold = compute_scaled_threshold(n_flipped, frozen_diag,
                                                     z=_run_config.get("z_threshold", 2.0))

                dF = session.try_patch(patch_info["indices"])
                ledger.charge("candidate_reject", n=1)
                trajectory.eval_count = ledger.eval_count
                if level >= 4:
                    l4_round_count += 1

                if dF > threshold:
                    if ledger.is_exhausted():
                        trajectory.record_step({
                            "type": f"patch_accept_gated_traj_{level_label}_skipped",
                            "indices": patch_info["indices"], "dJ": dF,
                            "reason": "budget_exhausted_before_commit",
                            "relaxation_level": level, "round": round_num,
                        })
                        break
                    session.commit_patch(patch_info["indices"])
                    ledger.charge("patch_accept", n=1)
                    trajectory.eval_count = ledger.eval_count
                    total_accepted += 1
                    round_accepted += 1
                    accepted_history.extend(patch_info["indices"])
                    trajectory.cumulative_dJ += dF
                    trajectory.record_step({
                        "type": f"patch_accept_gated_traj_{level_label}",
                        "indices": patch_info["indices"], "dJ": dF,
                        "size": patch_info["size"], "patch_type": patch_info["type"],
                        "relaxation_level": level,
                        "round": round_num,
                    })
                    print(f"  [diag_traj] R{round_num} {level_label} ACCEPT "
                          f"sz={patch_info['size']} {patch_info['type']} "
                          f"dF={dF:+.4e} ΣdJ={trajectory.cumulative_dJ:+.4e}",
                          flush=True)
                    # Refresh jac after geometry change
                    ledger.charge("grad_refresh", n=1, wall_clock_s=360)
                    try:
                        current_jac_path = _refresh_jac_inplace(
                            session, anchor_name, target_port, round_num)
                        # Reload gains from refreshed jac
                        jac = load_jac_npz(current_jac_path)
                        gain_gated = jac.get(f"gain_{variant}", jac["gain_raw"])
                        gain_raw = jac["gain_raw"]
                    except Exception as e:
                        print(f"  [diag_traj] jac refresh failed: {e} — reusing stale jac",
                              flush=True)
                    # Reset level for next probe
                    level = 0
                    M_gated = 15
                    break  # re-cluster with new jac + expanded seeds
                else:
                    trajectory.record_step({
                        "type": f"patch_reject_gated_traj_{level_label}",
                        "indices": patch_info["indices"], "dJ": dF,
                        "relaxation_level": level,
                        "round": round_num,
                    })

            if round_accepted > 0 and level == 0:
                break  # go to next round (jac already refreshed)
            if level >= 4:
                break  # rescue exhausted, go to next round
            level, M_gated, C_explore = _relax_level(level, M_gated, C_explore)

        if round_accepted == 0:
            print(f"  [diag_traj] round {round_num} zero accept → stop", flush=True)
            break

    return {
        "final_dJ": trajectory.cumulative_dJ,
        "eval_spent": ledger.eval_count,
        "n_patches_accepted": total_accepted,
    }


def _refresh_jac_inplace(session: Session, anchor_name: str,
                         target_port: int, round_num: int) -> str:
    """Recompute adjoint jacobian at current geometry, save to disk.

    Returns path to the newly saved .npz file so the caller can reload.
    Raises on failure (caller should catch and fall back to stale jac).
    """
    from wdm3_3d.wdm3_3d_jac import compute_jac_3d

    seed_num = int(anchor_name.replace("seed", ""))
    label_suffix = f"_g{round_num}"
    compute_jac_3d(
        seed_num, target_port, params=session.params.copy(),
        label_suffix=label_suffix)

    # compute_jac_3d saves to results_3d/jac3d/jac_{label}_g{round_num}.npz
    new_path = os.path.join(
        PARENT_DIR, "wdm3_3d", "results_3d", "jac3d",
        f"jac_{anchor_name}_P{target_port}@1550nm_g{round_num}.npz")
    if not os.path.exists(new_path):
        raise FileNotFoundError(f"jac refresh did not produce: {new_path}")
    return new_path


def get_jac_path(anchor_name: str, round_num: int = 0) -> str:
    port = ANCHORS[anchor_name]["port"]
    return os.path.join(
        PARENT_DIR, "wdm3_3d", "results_3d", "jac3d",
        f"jac_{anchor_name}_P{port}@1550nm_g{round_num}.npz"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Selector mode — unified candidate universe, different ranking/filtering
# ═══════════════════════════════════════════════════════════════════════════

ATLAS_DRIFT_CHECK_INTERVAL = 25  # baseline re-check every N evals in atlas mode

def _patch_hash(indices: list[int]) -> str:
    """Deterministic 12-char hex hash for a patch (list of flat indices)."""
    import hashlib
    key = ",".join(str(i) for i in sorted(indices))
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _score_patch_selector(patch_info: dict, gain_field: "np.ndarray",
                          selector: str, mask: list = None) -> float:
    """Re-score a patch according to selector type on the given gain_field.

    gain_field should be the signed gain (raw or conj_a) — we take abs()
    for scoring magnitude, and sign for sparse positive-position filtering.

    Returns new score (higher = better).  Returns -inf if excluded by mask.
    """
    from shared.index_utils import is_masked, N_COLS, N_ROWS
    from shared.index_utils import flat_to_col, flat_to_row

    indices = patch_info["indices"]

    # Mask check for masked/full selectors
    if selector in ("masked", "full") and mask:
        for idx in indices:
            if is_masked(idx, mask):
                return -float("inf")

    sz = patch_info["size"]
    ptype = patch_info["type"]

    # Collect per-pixel signed gain
    signed_vals = []
    for idx in indices:
        ix = flat_to_col(idx)
        iy = flat_to_row(idx)
        if 0 <= ix < N_COLS and 0 <= iy < N_ROWS:
            signed_vals.append(float(gain_field[ix, iy]))

    if not signed_vals:
        return -float("inf")

    if ptype == "sparse":
        # Only positive-gain positions contribute (matches _global_patch_scan)
        pos_vals = [v for v in signed_vals if v > 0]
        n_pos = len(pos_vals)
        if n_pos == 0:
            return -float("inf")
        return float(sum(pos_vals) / np.sqrt(n_pos))
    else:
        # dense: all positions, absolute value
        return float(sum(abs(v) for v in signed_vals) / sz)


def run_selector_eval(
    session: Session,
    jac_path: str,
    ledger: BudgetLedger,
    trajectory: Trajectory,
    frozen_diag: dict,
    selector: str,
    line_dir: str = None,
    resume_state: dict = None,
) -> dict:
    """Selector mode B: unified global-patch universe, selector-specific ranking.

    All selectors share the same candidate pool (_global_patch_scan over
    abs(gain_raw)).  Only the score/filter function changes:

      raw     — score = |raw_gain|, no mask
      variant — score = |conj_a gain|, no mask
      masked  — score = |conj_a gain|, exclude mask-overlapping patches
      full    — masked + keep only top-M (M from SELECTOR_DEFAULTS)
    """
    from shared import load_jac_npz
    from strategies import _global_patch_scan

    jac = load_jac_npz(jac_path)
    gain_raw = jac["gain_raw"]
    variant = frozen_diag["variant"]
    gain_conj_a = jac.get(f"gain_{variant}", gain_raw)
    mask = frozen_diag["mask"]
    top_m = _run_config.get("selector_top_m", 100)

    # Select gain field for scoring
    if selector == "raw":
        gain_field = gain_raw
        active_mask = None
    else:
        gain_field = gain_conj_a
        active_mask = mask if selector in ("masked", "full") else None

    # ── Unified candidate universe ──
    gain_abs = np.abs(gain_raw)
    all_patches = _global_patch_scan(gain_abs, gain_raw=gain_raw)
    print(f"  [selector:{selector}] unified universe: {len(all_patches)} patches, "
          f"budget_left={ledger.remaining()}", flush=True)

    # ── Re-score & filter ──
    for p in all_patches:
        p["selector_score"] = _score_patch_selector(
            p, gain_field, selector, active_mask)

    # Filter out excluded patches
    kept = [p for p in all_patches if p["selector_score"] > -float("inf") / 2]
    n_excluded = len(all_patches) - len(kept)
    if n_excluded:
        print(f"  [selector:{selector}] {n_excluded} patches excluded by mask/filter",
              flush=True)

    # Sort by selector score descending
    ranked = sorted(kept, key=lambda p: p["selector_score"], reverse=True)

    # Full selector: keep top-M only
    if selector == "full":
        ranked = ranked[:top_m]
        print(f"  [selector:{selector}] top-M={top_m} → {len(ranked)} candidates",
              flush=True)
    else:
        print(f"  [selector:{selector}] {len(ranked)} candidates (no top-M limit)",
              flush=True)

    # ── Resume: restore tested set ──
    tested = set()
    if resume_state:
        for key_list in resume_state.get("tested_keys", []):
            tested.add(frozenset(key_list))
        print(f"  [selector:{selector}] resume: {len(tested)} already tested, "
              f"skipping", flush=True)

    total_accepts = resume_state.get("total_accepts", 0) if resume_state else 0
    ckpt_counter = 0

    for patch_info in ranked:
        if ledger.is_exhausted():
            break

        key = frozenset(patch_info["indices"])
        if key in tested:
            continue
        tested.add(key)

        n_flipped = (patch_info.get("n_pos") if patch_info["type"] == "sparse"
                     else patch_info["size"] * patch_info["size"])
        threshold = compute_scaled_threshold(n_flipped, frozen_diag,
                                             z=_run_config.get("z_threshold", 2.0))

        dF = session.try_patch(patch_info["indices"])
        ledger.charge("candidate_reject", n=1)
        trajectory.eval_count = ledger.eval_count

        if dF > threshold:
            if ledger.is_exhausted():
                trajectory.record_step({
                    "type": f"patch_accept_selector_{selector}_skipped",
                    "indices": patch_info["indices"], "dJ": dF,
                    "reason": "budget_exhausted_before_commit",
                    "selector": selector,
                })
                break
            session.commit_patch(patch_info["indices"])
            ledger.charge("patch_accept", n=1)
            trajectory.eval_count = ledger.eval_count
            trajectory.cumulative_dJ += dF
            trajectory.record_step({
                "type": f"patch_accept_selector_{selector}",
                "indices": patch_info["indices"], "dJ": dF,
                "size": patch_info["size"], "patch_type": patch_info["type"],
                "selector_score": patch_info["selector_score"],
                "selector": selector,
            })
            total_accepts += 1
            print(f"  [selector:{selector}] ACCEPT sz={patch_info['size']} "
                  f"{patch_info['type']} dF={dF:+.4e} ΣdJ={trajectory.cumulative_dJ:+.4e}",
                  flush=True)
            # ── Checkpoint after each accept ──
            if line_dir:
                ckpt_counter += 1
                if ckpt_counter % 5 == 0:  # every 5 accepts (balance I/O vs safety)
                    _dump_checkpoint(line_dir,
                                     f"selector:{selector}",
                                     session.params.copy(),
                                     ledger, trajectory,
                                     {"tested_keys": [list(k) for k in tested],
                                      "total_accepts": total_accepts})
        else:
            trajectory.record_step({
                "type": f"patch_reject_selector_{selector}",
                "indices": patch_info["indices"], "dJ": dF,
                "selector_score": patch_info["selector_score"],
                "selector": selector,
            })

    # Final checkpoint
    if line_dir and total_accepts > 0:
        _dump_checkpoint(line_dir, f"selector:{selector}",
                         session.params.copy(), ledger, trajectory,
                         {"tested_keys": [list(k) for k in tested],
                          "total_accepts": total_accepts})

    print(f"  [selector:{selector}] done, {total_accepts} accepts, "
          f"evals={ledger.eval_count}", flush=True)
    return {"final_dJ": trajectory.cumulative_dJ, "eval_spent": ledger.eval_count,
            "n_patches_accepted": total_accepts, "candidates_generated": len(all_patches),
            "candidates_kept": len(kept), "candidates_excluded": n_excluded}


# ═══════════════════════════════════════════════════════════════════════════
# Fresh-sim final verification
# ═══════════════════════════════════════════════════════════════════════════

def _fresh_sim_verify(anchor_name: str, final_params: "np.ndarray",
                      trajectory_cumulative_dJ: float,
                      online_baseline_abs2: float = None) -> dict:
    """Open a CLEAN independent session with final geometry, run FDTD, report drift.

    Uses online_baseline_abs2 (actual measured warmstart |a|^2 from the Phase-1
    session) rather than a config constant, so drift reflects real session-to-session
    variation rather than config-vs-measurement mismatch.
    """
    import lum_backend, _lumopt_compat  # noqa
    from wdm3_3d.wdm3_3d_device import load_warmstart_params

    seed_num = int(anchor_name.replace("seed", ""))
    warmstart_raw = np.asarray(load_warmstart_params(seed_num), dtype=float).reshape(-1)
    warmstart_binary = (warmstart_raw > 0.5).astype(float)

    n_changed = int(np.sum(final_params != warmstart_binary))

    print(f"  [fresh-sim] opening independent session with {n_changed} flipped pixels ...",
          flush=True)
    t0 = time.time()
    try:
        with Session(anchor_name, tag="fresh_sim", params=final_params) as fresh:
            fresh_abs2 = fresh.fom
    except Exception as e:
        print(f"  [fresh-sim] FAILED: {e}", flush=True)
        return {
            "fresh_abs2": None, "fresh_dJ": None,
            "reported_dJ": trajectory_cumulative_dJ,
            "drift_abs": None, "drift_rel": None,
            "error": str(e),
            "n_flipped": n_changed,
            "wall_clock_s": time.time() - t0,
        }

    wall = time.time() - t0
    baseline = online_baseline_abs2  # actual measured warmstart from Phase-1 session
    fresh_dJ = fresh_abs2 - baseline if baseline is not None else None
    drift_abs = (trajectory_cumulative_dJ - fresh_dJ) if fresh_dJ is not None else None
    drift_rel = (drift_abs / abs(fresh_dJ)) if (fresh_dJ is not None and abs(fresh_dJ) > 1e-15) else None

    print(f"  [fresh-sim] fresh_abs2={fresh_abs2:.6e}  fresh_dJ={fresh_dJ:+.6e}  "
          f"reported_dJ={trajectory_cumulative_dJ:+.6e}  "
          f"drift={drift_abs:+.2e}" + (f" ({drift_rel*100:+.2f}%)" if drift_rel is not None else ""),
          flush=True)

    return {
        "fresh_abs2": fresh_abs2,
        "online_baseline_abs2": baseline,
        "fresh_dJ": fresh_dJ,
        "reported_dJ": trajectory_cumulative_dJ,
        "drift_abs": drift_abs,
        "drift_rel": drift_rel,
        "n_flipped": n_changed,
        "wall_clock_s": wall,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint / Resume helpers
# ═══════════════════════════════════════════════════════════════════════════

def _checkpoint_dir_from_config(strategy: str, anchor: str, budget: int,
                                 repeat: int, tag: str = "") -> str:
    """Compute output dir for a run line (same as run_single_line uses at end)."""
    safe = strategy.replace(":", "_")
    if tag:
        safe = f"{safe}__{tag}"
    return os.path.join(get_results_dir(_run_config.get("mode", "protocol")),
                        safe, anchor, f"budget{budget}", f"repeat{repeat}")


def _dump_checkpoint(line_dir: str, strategy: str, session_params: "np.ndarray",
                     ledger: "BudgetLedger", trajectory: "Trajectory",
                     resume_state: dict = None):
    """Write intermediate checkpoint files to line_dir."""
    os.makedirs(line_dir, exist_ok=True)
    # Also write standalone files for crash safety
    trajectory.to_json(os.path.join(line_dir, "trajectory.json"))
    ledger.to_json(os.path.join(line_dir, "budget_ledger.json"))
    ckpt = {
        "strategy": strategy,
        "params": session_params.tolist(),
        "ledger": ledger.to_dict(),
        "trajectory": trajectory.to_dict(),
        "resume_state": resume_state or {},
    }
    with open(os.path.join(line_dir, "_checkpoint.json"), "w") as f:
        json.dump(ckpt, f)


def _load_checkpoint(line_dir: str) -> dict | None:
    """Load checkpoint if it exists."""
    ckpt_path = os.path.join(line_dir, "_checkpoint.json")
    if not os.path.exists(ckpt_path):
        return None
    try:
        with open(ckpt_path) as f:
            return json.load(f)
    except Exception:
        return None


def _replay_geometry(session: "Session", trajectory_dict: dict) -> int:
    """Replay all accept steps from a saved trajectory to reconstruct geometry.

    Returns number of patches replayed.
    """
    steps = trajectory_dict.get("steps", [])
    n = 0
    for step in steps:
        if "accept" not in step.get("type", ""):
            continue
        if "skipped" in step.get("type", ""):
            continue
        indices = step.get("indices") or [step.get("idx")]
        if indices and len(indices) > 0:
            session.commit_patch(list(indices))
            n += 1
    return n


def run_single_line(strategy: str, anchor: str, budget: int, repeat: int,
                    tag: str = "") -> dict:
    import config as _cfg
    rng = np.random.default_rng(RNG_BASE_SEED + repeat * 100)
    t0 = time.time()

    # ── Compute output dir early (needed for resume check + checkpoint writes) ──
    line_dir = _checkpoint_dir_from_config(strategy, anchor, budget, repeat, tag)
    do_resume = _run_config.get("resume", False)

    # ── Check for existing checkpoint ──
    ckpt = _load_checkpoint(line_dir) if do_resume else None
    if ckpt:
        print(f"  [resume] found checkpoint: {line_dir}", flush=True)
        print(f"  [resume] evals={ckpt['ledger']['eval_count']}/{budget}", flush=True)
    else:
        if do_resume:
            print(f"  [resume] no checkpoint found at {line_dir}, starting fresh",
                  flush=True)

    # ── Phase 0 cache: shared across strategies/modes for same anchor ──
    cache_dir = os.path.join(_cfg.RESULTS_BASE, "shared", anchor)
    cache_path = os.path.join(cache_dir, "diagnosis.json")
    no_cache = _run_config.get("no_cache_diag", False)

    if os.path.exists(cache_path) and not no_cache:
        with open(cache_path) as f:
            frozen_diag = json.load(f)
        diag_cost = frozen_diag["null_single_draws"] + frozen_diag["null_batch_draws"]
        print(f"  [Phase-0] CACHED from {cache_path} (null_cost={diag_cost})", flush=True)
    else:
        frozen_diag = run_diagnosis_phase0(anchor, rng)
        diag_cost = frozen_diag["null_single_draws"] + frozen_diag["null_batch_draws"]
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(frozen_diag, f, indent=2)
        print(f"  [Phase-0] saved to {cache_path} (null_cost={diag_cost})", flush=True)

    # Apply ablation BEFORE passing to strategies
    frozen_diag = _apply_ablation(frozen_diag, _run_config.get("ablate", "none"))

    # Phase 1: optimization (clean session)
    import lum_backend, _lumopt_compat  # noqa

    # ── Resume: restore ledger + trajectory from checkpoint ──
    if ckpt:
        ledger = BudgetLedger.from_dict(ckpt["ledger"])
        traj_dict = ckpt["trajectory"]
        trajectory = Trajectory.from_dict(traj_dict)
        resume_state = ckpt.get("resume_state", {})
    else:
        ledger = BudgetLedger(total_budget=budget)
        trajectory = Trajectory(strategy, anchor, budget, repeat)
        resume_state = None

    try:
        with Session(anchor, tag=f"opt_{strategy.replace(':', '_')}") as session:
            # ── Replay geometry if resuming ──
            if ckpt:
                n_replay = _replay_geometry(session, ckpt["trajectory"])
                # FOM already up-to-date: commit_patch → measure(keep=True) updates fom
                print(f"  [resume] replayed {n_replay} accepts, "
                      f"abs2={session.fom:.6e}", flush=True)

            online_baseline_abs2 = session.fom
            jac_path = get_jac_path(anchor, 0)

            if strategy == "naive_single":
                result = run_naive_single_v2(session, jac_path, ledger, trajectory,
                                             rng, frozen_diag, line_dir=line_dir,
                                             resume_state=resume_state)

            elif strategy == "diag_single":
                result = run_diag_single_v2(session, jac_path, ledger, trajectory,
                                            rng, frozen_diag, line_dir=line_dir,
                                            resume_state=resume_state)

            elif strategy == "naive_single+patch":
                if ckpt and resume_state.get("phase") == "patch":
                    # Single phase already done — restore and skip to patch
                    s_traj = Trajectory("naive_single", anchor, budget, repeat)
                    # Reconstruct from trajectory steps (single-phase steps come first)
                    single_steps = [s for s in trajectory.steps
                                    if "patch" not in s.get("type", "")]
                    s_traj.steps = single_steps
                    s_traj.eval_count = sum(
                        1 for s in single_steps if "accept" in s.get("type", "")
                        or "reject" in s.get("type", ""))
                    s_traj.cumulative_dJ = sum(
                        s.get("dJ", 0) for s in single_steps
                        if "accept" in s.get("type", "")
                        and "skipped" not in s.get("type", ""))
                    # Patch phase with resume
                    p_traj = Trajectory(strategy, anchor, budget, repeat)
                    patch_steps = [s for s in trajectory.steps
                                   if "patch" in s.get("type", "")]
                    p_traj.steps = patch_steps
                    p_traj.cumulative_dJ = sum(
                        s.get("dJ", 0) for s in patch_steps
                        if "accept" in s.get("type", "")
                        and "skipped" not in s.get("type", ""))
                    p_res = run_naive_patch_v2(session, jac_path, ledger, p_traj,
                                               frozen_diag, line_dir=line_dir,
                                               resume_state=resume_state)
                    trajectory.steps = single_steps + p_traj.steps
                    trajectory.eval_count = ledger.eval_count
                    trajectory.cumulative_dJ = s_traj.cumulative_dJ + p_traj.cumulative_dJ
                    result = {"final_dJ": trajectory.cumulative_dJ,
                              "eval_spent": ledger.eval_count}
                else:
                    s_traj = Trajectory("naive_single", anchor, budget, repeat)
                    s_res = run_naive_single_v2(session, jac_path, ledger, s_traj,
                                                rng, frozen_diag, line_dir=line_dir,
                                                resume_state=resume_state)
                    trajectory.extend_steps(s_traj.steps)
                    # Checkpoint after single phase
                    _dump_checkpoint(line_dir, strategy, session.params.copy(),
                                     ledger, trajectory,
                                     {"phase": "patch", "tested_keys": []})
                    p_traj = Trajectory(strategy, anchor, budget, repeat)
                    p_res = run_naive_patch_v2(session, jac_path, ledger, p_traj,
                                               frozen_diag, line_dir=line_dir)
                    trajectory.extend_steps(p_traj.steps)
                    trajectory.cumulative_dJ = s_traj.cumulative_dJ + p_traj.cumulative_dJ
                    result = {"final_dJ": trajectory.cumulative_dJ,
                              "eval_spent": ledger.eval_count}

            elif strategy == "diag_single+patch":
                if ckpt and resume_state.get("phase") == "patch":
                    # Single phase done — restore accepted list and skip to patch
                    accepted = resume_state.get("accepted_single", [])
                    single_steps = [s for s in trajectory.steps
                                    if "patch" not in s.get("type", "")]
                    p_traj = Trajectory(strategy, anchor, budget, repeat)
                    patch_steps = [s for s in trajectory.steps
                                   if "patch" in s.get("type", "")]
                    p_traj.steps = patch_steps
                    p_traj.cumulative_dJ = sum(
                        s.get("dJ", 0) for s in patch_steps
                        if "accept" in s.get("type", "")
                        and "skipped" not in s.get("type", ""))
                    p_res = run_diag_patch_v2(session, jac_path, ledger, p_traj,
                                              frozen_diag, accepted,
                                              line_dir=line_dir,
                                              resume_state=resume_state)
                    trajectory.steps = single_steps + p_traj.steps
                    trajectory.eval_count = ledger.eval_count
                    trajectory.cumulative_dJ = (sum(
                        s.get("dJ", 0) for s in single_steps
                        if "accept" in s.get("type", "")
                        and "skipped" not in s.get("type", ""))
                        + p_traj.cumulative_dJ)
                    result = {"final_dJ": trajectory.cumulative_dJ,
                              "eval_spent": ledger.eval_count,
                              "n_patches": p_res.get("n_patches_accepted", 0),
                              "peak_relaxation_level": p_res.get("peak_relaxation_level", 0)}
                else:
                    s_traj = Trajectory("diag_single", anchor, budget, repeat)
                    s_res = run_diag_single_v2(session, jac_path, ledger, s_traj,
                                               rng, frozen_diag, line_dir=line_dir,
                                               resume_state=resume_state)
                    accepted = s_res.get("accepted_indices", [])
                    trajectory.extend_steps(s_traj.steps)
                    _dump_checkpoint(line_dir, strategy, session.params.copy(),
                                     ledger, trajectory,
                                     {"phase": "patch", "accepted_single": accepted,
                                      "tested_keys": [], "accepted_all": list(accepted)})
                    p_traj = Trajectory(strategy, anchor, budget, repeat)
                    p_res = run_diag_patch_v2(session, jac_path, ledger, p_traj,
                                              frozen_diag, accepted, line_dir=line_dir)
                    trajectory.extend_steps(p_traj.steps)
                    trajectory.cumulative_dJ = s_traj.cumulative_dJ + p_traj.cumulative_dJ
                    result = {"final_dJ": trajectory.cumulative_dJ,
                              "eval_spent": ledger.eval_count,
                              "n_patches": p_res.get("n_patches_accepted", 0),
                              "peak_relaxation_level": p_res.get("peak_relaxation_level", 0)}

            elif strategy == "naive_single+patch+trajectory":
                # single
                s_traj = Trajectory("naive_single", anchor, budget, repeat)
                s_res = run_naive_single_v2(session, jac_path, ledger, s_traj,
                                            rng, frozen_diag, line_dir=line_dir)
                trajectory.extend_steps(s_traj.steps)
                # patch
                p_traj = Trajectory("naive_single+patch", anchor, budget, repeat)
                p_res = run_naive_patch_v2(session, jac_path, ledger, p_traj,
                                           frozen_diag, line_dir=line_dir)
                trajectory.extend_steps(p_traj.steps)
                # trajectory loop
                t_traj = Trajectory(strategy, anchor, budget, repeat)
                t_res = run_naive_trajectory_v2(session, jac_path, ledger, t_traj,
                                                rng, frozen_diag)
                trajectory.extend_steps(t_traj.steps)
                trajectory.cumulative_dJ = (s_traj.cumulative_dJ +
                                            p_traj.cumulative_dJ +
                                            t_traj.cumulative_dJ)
                t_res["accepted_single"] = s_res.get("accepted_indices", [])
                result = {"final_dJ": trajectory.cumulative_dJ,
                          "eval_spent": ledger.eval_count,
                          "n_patches": t_res.get("n_patches_accepted", 0)}

            elif strategy == "diag_single+patch+trajectory":
                # single
                s_traj = Trajectory("diag_single", anchor, budget, repeat)
                s_res = run_diag_single_v2(session, jac_path, ledger, s_traj,
                                           rng, frozen_diag, line_dir=line_dir)
                accepted_single = s_res.get("accepted_indices", [])
                trajectory.extend_steps(s_traj.steps)
                # patch
                p_traj = Trajectory("diag_single+patch", anchor, budget, repeat)
                p_res = run_diag_patch_v2(session, jac_path, ledger, p_traj,
                                          frozen_diag, accepted_single,
                                          line_dir=line_dir)
                trajectory.extend_steps(p_traj.steps)
                # trajectory loop — accumulate all accepted pixels as cluster seeds
                all_accepted = list(accepted_single)
                for step in p_traj.steps:
                    if "accept" in step.get("type", ""):
                        all_accepted.extend(step.get("indices", []))
                t_traj = Trajectory(strategy, anchor, budget, repeat)
                t_res = run_diag_trajectory_v2(session, jac_path, ledger, t_traj,
                                               rng, frozen_diag, all_accepted)
                trajectory.extend_steps(t_traj.steps)
                trajectory.cumulative_dJ = (s_traj.cumulative_dJ +
                                            p_traj.cumulative_dJ +
                                            t_traj.cumulative_dJ)
                t_res["accepted_single"] = accepted_single
                result = {"final_dJ": trajectory.cumulative_dJ,
                          "eval_spent": ledger.eval_count,
                          "n_patches": t_res.get("n_patches_accepted", 0)}

            elif strategy.startswith("selector:"):
                sel = strategy.split(":", 1)[1]
                result = run_selector_eval(session, jac_path, ledger, trajectory,
                                           frozen_diag, sel, line_dir=line_dir,
                                           resume_state=resume_state)

            else:
                raise ValueError(f"Unknown: {strategy}")

            # Snapshot final geometry BEFORE session closes (for fresh-sim)
            final_params = session.params.copy()

    except Exception as e:
        import traceback
        traceback.print_exc()
        result = {"error": str(e), "final_dJ": trajectory.cumulative_dJ}
        final_params = None
        # Save what we have on crash (including checkpoint for --resume)
        os.makedirs(line_dir, exist_ok=True)
        trajectory.to_json(os.path.join(line_dir, "trajectory.json"))
        ledger.to_json(os.path.join(line_dir, "budget_ledger.json"))
        try:
            # Embed to_dict() format so _load_checkpoint can restore
            ckpt_data = {
                "strategy": strategy,
                "params": None,  # can't snapshot params from crashed session
                "ledger": ledger.to_dict(),
                "trajectory": trajectory.to_dict(),
                "resume_state": resume_state if resume_state else {},
                "crash_saved": True,  # marker: params not available, needs replay
            }
            with open(os.path.join(line_dir, "_checkpoint.json"), "w") as f:
                json.dump(ckpt_data, f)
        except Exception:
            pass

    # ── Fresh-sim final verification ──
    if final_params is not None:
        try:
            fresh_result = _fresh_sim_verify(
                anchor, final_params, trajectory.cumulative_dJ,
                online_baseline_abs2=online_baseline_abs2)
            result["fresh_sim"] = fresh_result
            ledger.charge("fresh_sim", n=1,
                          wall_clock_s=fresh_result.get("wall_clock_s", 0))
        except Exception as e:
            result["fresh_sim"] = {"error": str(e)}
    else:
        result["fresh_sim"] = {"error": "session crashed, no params to verify"}

    result["wall_clock_s"] = time.time() - t0
    result["strategy"] = strategy
    result["anchor"] = anchor
    result["budget"] = budget
    result["repeat"] = repeat
    result["diagnosis_cost"] = diag_cost
    result["gate_cost"] = 0
    result["diagnosis"] = frozen_diag
    result["metrics"] = _compute_metrics(result, trajectory, ledger)

    # -- write out to the mode-specific directory --
    os.makedirs(line_dir, exist_ok=True)
    trajectory.to_json(os.path.join(line_dir, "trajectory.json"))
    ledger.to_json(os.path.join(line_dir, "budget_ledger.json"))
    with open(os.path.join(line_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(line_dir, "diagnosis.json"), "w") as f:
        json.dump(frozen_diag, f, indent=2)
    with open(os.path.join(line_dir, "run_config.json"), "w") as f:
        json.dump(_run_config, f, indent=2, default=str)
    # Clean up checkpoint on successful completion
    ckpt_path = os.path.join(line_dir, "_checkpoint.json")
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return result


def _compute_metrics(result: dict, trajectory: "Trajectory",
                     ledger: "BudgetLedger") -> dict:
    """Extract standardized metrics from a completed run.

    Returns a flat dict ready for the paper's results table.
    """
    s = ledger.summary()
    bd = s["breakdown"]

    # ── core outcomes ──
    final_dJ = trajectory.cumulative_dJ

    # Cost accounting: split by source
    null_cost = result.get("diagnosis_cost", 0)   # Phase-0 σ₁/σ₅ calibration draws
    gate_cost = result.get("gate_cost", 0)         # 0 for frozen-gate transfer
    online_action_eval = ledger.eval_count          # FDTD evals during optimization
    total_online_cost = null_cost + gate_cost + online_action_eval

    fs = result.get("fresh_sim", {})
    fresh_dJ = fs.get("fresh_dJ")
    retention = (fresh_dJ / final_dJ) if (final_dJ and fresh_dJ and abs(final_dJ) > 1e-15) else None
    drift_abs = fs.get("drift_abs")

    # ── acceptance quality ──
    candidate_tested = bd["candidate_accept"] + bd["candidate_reject"]
    total_accepts = bd["candidate_accept"] + bd["patch_accept"]
    pos_accept_rate = (total_accepts / candidate_tested) if candidate_tested > 0 else 0.0

    # negative accepts: accepted steps whose recorded dJ < 0
    negative_accept_count = sum(
        1 for step in trajectory.steps
        if "accept" in step.get("type", "") and step.get("dJ", 0) < 0
    )

    # ── patch-level (from trajectory steps, not ledger which mixes single+patch) ──
    patch_tested = sum(1 for step in trajectory.steps if "patch" in step.get("type", ""))
    patch_accept_count = sum(1 for step in trajectory.steps
                             if "patch" in step.get("type", "")
                             and "accept" in step.get("type", "")
                             and "skipped" not in step.get("type", ""))

    # ── relaxation level breakdown (diag only) ──
    level_counts = {}
    for step in trajectory.steps:
        lv = step.get("relaxation_level")
        if lv is not None:
            level_counts[str(lv)] = level_counts.get(str(lv), 0) + 1

    # ── efficiency ──
    dJ_per_action_eval = final_dJ / online_action_eval if online_action_eval > 0 else 0.0
    dJ_per_online_eval = final_dJ / total_online_cost if total_online_cost > 0 else 0.0

    return {
        "final_dJ": final_dJ,
        "fresh_dJ": fresh_dJ,
        "retention": retention,
        "drift_abs": drift_abs,
        "null_cost": null_cost,
        "gate_cost": gate_cost,
        "online_action_eval": online_action_eval,
        "total_online_cost": total_online_cost,
        "candidate_tested": candidate_tested,
        "total_accepts": total_accepts,
        "positive_accept_rate": round(pos_accept_rate, 6),
        "negative_accept_count": negative_accept_count,
        "patch_accept_count": patch_accept_count,
        "patch_tested_approx": patch_tested,
        "relaxation_level_breakdown": level_counts,
        "dJ_per_action_eval": round(dJ_per_action_eval, 10),
        "dJ_per_online_eval": round(dJ_per_online_eval, 10),
    }


def _apply_ablation(diag: dict, ablate: str) -> dict:
    """Modify frozen diagnosis dict according to ablation mode(s).

    Supports comma-separated combinations: --ablate no_mask,no_keff
    Returns a (possibly modified) copy.  Original is unchanged.
    """
    import copy
    d = copy.deepcopy(diag)

    if ablate == "none" or not ablate:
        return d

    components = [a.strip() for a in ablate.split(",")]
    for comp in components:
        if comp == "no_mask":
            d["mask"] = []
        elif comp == "no_variant":
            d["variant"] = "raw"
        elif comp == "no_keff":
            d["K_eff"] = DEFAULT_K * 10
        elif comp == "no_progressive":
            pass  # handled by _run_config flag, not diag dict
        elif comp == "z_sweep":
            pass  # handled by --z
    return d


def _aggregate_pilot_summary(results_dir: str):
    """Merge all pilot_*.json files into PILOT_SUMMARY.json.

    Sources: (1) pilot_*.json flat files in results_dir,
             (2) result.json in subdirectories (strategy/anchor/budgetX/repeatY/).
    Excludes PILOT_SUMMARY.json itself.
    """
    import glob as glob_mod
    all_results = {}

    # (1) Flat per-run files
    for path in sorted(glob_mod.glob(os.path.join(results_dir, "pilot_*.json"))):
        fname = os.path.basename(path)
        if fname == "PILOT_SUMMARY.json":
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            tag = fname.replace("pilot_", "").replace(".json", "")
            all_results[tag] = data
        except Exception:
            pass

    # (2) Subdirectory results from earlier runs
    for path in sorted(glob_mod.glob(
            os.path.join(results_dir, "*", "*", "budget*", "repeat*", "result.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
            # Derive key from path relative to results_dir:
            #   strategy/anchor/budgetX/repeatY  (4 segments)
            rel_path = os.path.relpath(path, results_dir)
            key = rel_path.replace(os.sep, "/").replace("/result.json", "")
            all_results[key] = data
        except Exception:
            pass

    summary_path = os.path.join(results_dir, "PILOT_SUMMARY.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Paper 2 Pilot v2 — dual-track")
    parser.add_argument("--strategies", type=str, default=None)
    parser.add_argument("--anchor", default=PILOT_ANCHOR)
    parser.add_argument("--budget", type=int, default=PILOT_BUDGET)
    parser.add_argument("--repeat", type=int, default=PILOT_REPEAT)
    parser.add_argument("--backend", default="2026")
    parser.add_argument("--l4-cap", type=int, default=None,
                        help=f"L4 rescue max evals per round (default: {L4_MAX_EVAL_CAP})")
    parser.add_argument("--no-cache-diag", action="store_true",
                        help="Force recompute Phase-0 diagnosis (skip shared cache)")
    parser.add_argument("--mode", default="protocol",
                        choices=["protocol", "performance", "selector"],
                        help="protocol=fixed paper settings; performance=aggressive; selector=unified candidate universe")
    parser.add_argument("--selector", type=str, default=None,
                        choices=["raw", "variant", "masked", "full"],
                        help="selector type for --mode selector")
    parser.add_argument("--z", type=float, default=None,
                        help="threshold z-multiplier (default: mode-dependent)")
    parser.add_argument("--patch-sizes", type=str, default=None,
                        help="comma-separated patch sizes, e.g. '2,3,5,7'")
    parser.add_argument("--ablate", type=str, default="none",
                        help="comma-separated ablation components: no_mask,no_variant,no_keff,no_progressive,z_sweep")
    parser.add_argument("--z-sweep", action="store_true", default=False,
                        help="run z=[2.0, 1.5, 1.0] sweep and compare")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="resume from checkpoint if exists")
    parser.add_argument("--tag", type=str, default="",
                        help="optional tag to disambiguate output paths (appended to strategy dir name)")
    args = parser.parse_args()

    os.environ["LUMOPT_BACKEND"] = args.backend

    # ── Build run_config ──
    if args.mode == "protocol":
        mode_defaults = PROTOCOL_DEFAULTS
    elif args.mode == "performance":
        mode_defaults = PERFORMANCE_DEFAULTS
    else:
        mode_defaults = SELECTOR_DEFAULTS
    patch_sizes = (
        [int(x) for x in args.patch_sizes.split(",")] if args.patch_sizes
        else mode_defaults["patch_sizes"]
    )
    z_val = args.z if args.z is not None else mode_defaults["z_threshold"]

    global _run_config
    _run_config = {
        "mode": args.mode,
        "z_threshold": z_val,
        "patch_sizes": patch_sizes,
        "patch_types": mode_defaults["patch_types"],
        "progressive_relaxation": mode_defaults["progressive_relaxation"],
        "fresh_sim": mode_defaults["fresh_sim"],
        "refresh_policy": mode_defaults["refresh_policy"],
        "refresh_every_N": mode_defaults["refresh_every_N"],
        "anchor": args.anchor,
        "budget": args.budget,
        "repeat": args.repeat,
        "backend": args.backend,
        # Cost accounting policy
        "gate_source": "paper1_frozen",
        "gate_cost_policy": "zero_online_cost",
        "null_policy": "phase0_scaled",
        # L4 rescue cap
        "l4_cap": args.l4_cap if args.l4_cap is not None else L4_MAX_EVAL_CAP,
        # Diagnosis cache
        "no_cache_diag": args.no_cache_diag,
        # Ablation
        "ablate": args.ablate,
        # Selector
        "selector": args.selector,
        "selector_top_m": mode_defaults.get("selector_top_m", 100),
        # Checkpoint / Resume
        "resume": args.resume,
        # Tag for output path disambiguation
        "tag": args.tag,
    }
    if "no_progressive" in (args.ablate or "").split(","):
        _run_config["progressive_relaxation"] = False

    # Update config PATCH_SIZES so scan functions pick up the right sizes
    import config as _cfg
    _cfg.PATCH_SIZES = patch_sizes
    # Also update the already-imported reference in strategies module
    import strategies as _strat
    _strat.PATCH_SIZES = patch_sizes
    _strat.PATCH_TYPES = mode_defaults["patch_types"]

    results_dir = get_results_dir(args.mode)

    if args.mode == "selector":
        if not args.selector:
            print("ERROR: --mode selector requires --selector raw|variant|masked|full")
            sys.exit(1)
        strategies = [f"selector:{args.selector}"]
    elif args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",")]
    else:
        strategies = strategies_DEFAULT

    # ── z sweep or single z ──
    z_values = [2.0, 1.5, 1.0] if args.z_sweep else [z_val]
    z_labels = [f"z{zv}" for zv in z_values]

    print("=" * 70)
    if args.z_sweep:
        print(f"PAPER 2 — Z-SWEEP  MODE={args.mode}  z={z_values}  patches={patch_sizes}")
    else:
        print(f"PAPER 2 — PILOT v2  MODE={args.mode}  z={z_val}  patches={patch_sizes}")
    print(f"Anchor: {args.anchor}  Budget: {args.budget}  Repeat: {args.repeat}")
    print(f"Strategies: {strategies}")
    print("=" * 70)

    all_z_results = {}
    for z_idx, zv in enumerate(z_values):
        _run_config["z_threshold"] = zv
        _run_config["z_sweep_index"] = z_idx

        if args.z_sweep:
            print(f"\n{'#'*70}")
            print(f"#  Z-SWEEP: z={zv}  ({z_idx+1}/{len(z_values)})")
            print(f"{'#'*70}")

        results = {}
        for strategy in strategies:
            print(f"\n{'─'*50}")
            print(f"Running: {strategy}" +
                  (f"  z={zv}" if args.z_sweep else ""))
            print(f"{'─'*50}")
            result = run_single_line(strategy, args.anchor, args.budget, args.repeat,
                                     tag=args.tag)
            results[strategy] = result

            if "error" in result:
                print(f"  ERROR: {result['error']}")
            else:
                dJ = result.get("final_dJ", None)
                m = result.get("metrics", {})
                print(f"  final_dJ: {dJ:+.6e}" if isinstance(dJ, float)
                      else f"  final_dJ: {dJ}")
                print(f"  eval_spent: {result.get('eval_spent')}"
                      f"  diag_cost: {result.get('diagnosis_cost')}"
                      f"  wall_clock: {result.get('wall_clock_s', 0):.0f}s")
                if m:
                    print(f"  metrics: fresh_dJ={m.get('fresh_dJ')}  "
                          f"retention={m.get('retention')}  "
                          f"pos_rate={m.get('positive_accept_rate')}  "
                          f"neg_accepts={m.get('negative_accept_count')}  "
                          f"dJ/act={m.get('dJ_per_action_eval')}  "
                          f"dJ/online={m.get('dJ_per_online_eval')}")

        if args.z_sweep:
            all_z_results[f"z{zv}"] = results
            # Per-z summary
            label = f"z{zv}"
            strategy_tag = "+".join(strategies).replace(":", "_")
            if args.tag:
                strategy_tag = f"{strategy_tag}__{args.tag}"
            run_tag = f"{args.anchor}_{strategy_tag}_b{args.budget}_r{args.repeat}_{label}"
            per_run_path = os.path.join(results_dir, f"pilot_{run_tag}.json")
            with open(per_run_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Saved: {per_run_path}")

    if args.z_sweep:
        # ── Z-sweep comparison table ──
        print(f"\n{'='*70}")
        print("Z-SWEEP COMPARISON")
        print(f"{'='*70}")
        header = f"{'Strategy':<28}"
        for zl in z_labels:
            header += f" {zl:>12}"
        print(header)
        print(f"{'-'*len(header)}")
        for s in strategies:
            row = f"{s:<28}"
            for zl in z_labels:
                r = all_z_results.get(zl, {}).get(s, {})
                dJ = r.get("final_dJ")
                row += f" {dJ:>+12.6e}" if isinstance(dJ, float) else f" {'—':>12}"
            print(row)
        print()
        # Also save sweep summary
        sweep_path = os.path.join(results_dir,
            f"zsweep_{args.anchor}_{'+'.join(strategies)}_b{args.budget}_r{args.repeat}.json")
        with open(sweep_path, "w") as f:
            json.dump(all_z_results, f, indent=2, default=str)
        print(f"Sweep saved: {sweep_path}")
    else:
        # Single-z: per-run file + aggregate (original behavior)
        print(f"\n{'='*70}")
        print("PILOT SUMMARY")
        print(f"{'='*70}")
        print(f"{'Strategy':<30} {'final_dJ':>12} {'evals':>8} {'diag':>6} {'wc':>8}")
        print(f"{'-'*62}")
        for s in strategies:
            r = results.get(s, {})
            dJ = r.get("final_dJ", None)
            ev = r.get("eval_spent", "—")
            dc = r.get("diagnosis_cost", "—")
            wc = r.get("wall_clock_s", 0)
            dJ_str = f"{dJ:>+12.6e}" if isinstance(dJ, float) else f"{'ERROR':>12}"
            print(f"{s:<30} {dJ_str} {str(ev):>8} {str(dc):>6} {wc:>7.0f}s")

        strategy_tag = "+".join(strategies).replace(":", "_")
        if args.tag:
            strategy_tag = f"{strategy_tag}__{args.tag}"
        run_tag = f"{args.anchor}_{strategy_tag}_b{args.budget}_r{args.repeat}"
        per_run_path = os.path.join(results_dir, f"pilot_{run_tag}.json")
        with open(per_run_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved: {per_run_path}")

    # Aggregate: merge all per-run files into PILOT_SUMMARY.json
    _aggregate_pilot_summary(results_dir)
    print(f"Aggregate: {os.path.join(results_dir, 'PILOT_SUMMARY.json')}")


if __name__ == "__main__":
    main()
