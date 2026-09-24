#!/usr/bin/env python
"""Type B cluster-patch actionability audit for WDM3 3D.

Leverages spatial clustering discovered in the single-pixel audit
(accepted pixels clustered around ix=43-49, iy=26-29) to test whether
patch-scale actions improve device-level outcomes.

Patch types:
  B-sparse : only positive-gain pixels inside the cluster bbox
  B-dense  : all pixels inside the cluster bbox (contiguous block)

Scoring (per user spec 2026-07-26):
  S_patch = sum(g_i for g_i > 0) / sqrt(|P|)
  with positive_fraction > 0.6 required for ranking.

Usage:
  python -m wdm3_3d.cluster_patch_audit --seed 7 --port 1
  python -m wdm3_3d.cluster_patch_audit --seed 7 --port 1 --sizes 2,3,5 --null 6
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# -- scipy fallback --
try:
    from scipy.stats import spearmanr as _spearmanr
except ImportError:
    _spearmanr = None

# -- backend --
from . import wdm3_3d_landscape  # noqa: F401
import lum_backend  # noqa: F401
import _lumopt_compat  # noqa: F401

from .wdm3_3d_device import (
    NX_2D,
    NY_2D,
    WL_NM,
    _build_3d_sim,
    _close_sim,
)
from .wdm3_3d_smoke import _apply_params_to_design
from .wdm3_3d_audit import _read_expansion_coefficient_3d
from .wdm3_3d_jac import (
    GRAD_VARIANTS,
    RESULTS_DIR,
    _flip_gain,
    _label,
    _now_iso,
    _save_json,
    _sweep_stray_engines,
    load_jac_3d,
)
from .single_pixel_audit import (
    AUDIT_DIR,
    DeviceAuditSession,
    compute_j_wdm,
)

CLUSTER_DIR = Path(__file__).resolve().parent / "results_3d" / "cluster_patch"

# -- seed7 discovered active region centers (from single-pixel audit) --
DEFAULT_CENTERS = [(43, 26), (46, 29)]


# =====================================================================
# Patch definition & scoring
# =====================================================================

def _bbox_pixels(cx, cy, size, nx=NX_2D, ny=NY_2D):
    """Return flat indices of all pixels in a size*size box centred at (cx, cy)."""
    half = size // 2
    x0, x1 = max(0, cx - half), min(nx, cx + half + size % 2)
    y0, y1 = max(0, cy - half), min(ny, cy + half + size % 2)
    indices = []
    for ix in range(x0, x1):
        for iy in range(y0, y1):
            indices.append(int(ix * ny + iy))
    return indices


def _score_patch(indices, gain):
    """Return (S_patch, pos_frac, sum_pos, n_total) for a set of pixel indices."""
    g = np.asarray([gain[i] for i in indices])
    n_total = len(g)
    pos_mask = g > 0
    n_pos = int(np.sum(pos_mask))
    pos_frac = n_pos / n_total if n_total > 0 else 0.0
    sum_pos = float(np.sum(g[pos_mask])) if n_pos > 0 else 0.0
    score = sum_pos / np.sqrt(n_total) if n_total > 0 else 0.0
    return score, pos_frac, sum_pos, n_total, n_pos


def _build_patches(centers, sizes, gain, params,
                   variant="sparse", pos_frac_min=0.6):
    """Build all cluster-patch candidates.

    Returns list of dicts: {tag, center, size, variant, indices, score, ...}
    """
    patches = []
    for cx, cy in centers:
        for sz in sizes:
            all_idx = _bbox_pixels(cx, cy, sz)
            if variant == "sparse":
                idx = [i for i in all_idx if gain[i] > 0]
            else:
                idx = all_idx

            if len(idx) == 0:
                continue

            score, pos_frac, sum_pos, n_total, n_pos = _score_patch(idx, gain)

            # determine if this patch meets the ranking criterion
            rankable = pos_frac >= pos_frac_min

            tag = (f"B-{variant}_c({cx},{cy})_s{sz}"
                   f"_{'pos' if variant == 'sparse' else 'all'}")
            patches.append({
                "tag": tag,
                "center": (cx, cy),
                "size": sz,
                "variant": variant,
                "indices": idx,
                "n_pixels": len(idx),
                "n_total_bbox": n_total,
                "n_pos": n_pos,
                "score": score,
                "pos_frac": pos_frac,
                "sum_pos_gain": sum_pos,
                "rankable": rankable,
            })
    return patches


def _random_patch(size, gain, params, exclude_sets, variant="sparse",
                  rng=None):
    """Generate one random patch matching *size* and *variant*.

    *exclude_sets*: list of sets of indices to avoid (true clusters).
    """
    if rng is None:
        rng = np.random.RandomState()
    nx, ny = NX_2D, NY_2D
    half = size // 2

    # sample random centre, avoiding edges
    for _ in range(200):
        cx = rng.randint(half, nx - half)
        cy = rng.randint(half, ny - half)
        all_idx = set(_bbox_pixels(cx, cy, size))

        # check overlap with any excluded set
        ok = True
        for es in exclude_sets:
            if all_idx & es:
                ok = False
                break
        if not ok:
            continue

        idx_list = sorted(all_idx)
        if variant == "sparse":
            idx_list = [i for i in idx_list if gain[i] > 0]
        if len(idx_list) < 2:  # need at least 2 pixels for meaningful patch
            continue

        score, pos_frac, sum_pos, n_total, n_pos = _score_patch(idx_list, gain)
        return {
            "tag": f"RANDOM-B-{variant}_c({cx},{cy})_s{size}",
            "center": (cx, cy),
            "size": size,
            "variant": variant,
            "indices": idx_list,
            "n_pixels": len(idx_list),
            "n_total_bbox": n_total,
            "n_pos": n_pos,
            "score": score,
            "pos_frac": pos_frac,
            "sum_pos_gain": sum_pos,
            "rankable": False,  # random is not ranked
        }
    return None


# =====================================================================
# Main audit
# =====================================================================

def run_cluster_audit(seed=7, target_port=1, wl_nm=WL_NM,
                      variant="conj_a", mask_cols=None,
                      centers=None, sizes=(2, 3, 5),
                      patch_variants=("sparse", "dense"),
                      null_per_config=2,
                      replay_null_per_type=15,
                      lam_R=1.0, lam_X=0.5,
                      jac_suffix=None, out_dir=None):
    """Run the Type B cluster-patch actionability audit.

    Parameters
    ----------
    centers : list[(int,int)]
        Cluster centers from single-pixel audit discovery.
    sizes : tuple[int]
        Bbox sizes to test.
    patch_variants : tuple[str]
        "sparse" (positive-gain only) and/or "dense" (all pixels).
    null_per_config : int
        Random patches per (size, variant) combination.
    """
    out_dir = Path(out_dir) if out_dir else CLUSTER_DIR
    label = _label(seed, target_port, wl_nm)
    out_dir.mkdir(parents=True, exist_ok=True)

    if centers is None:
        centers = DEFAULT_CENTERS

    # -- 1. Load jacobian --------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"CLUSTER PATCH AUDIT (Type B): {label}")
    print(f"{'=' * 60}")

    if jac_suffix is not None:
        jac_npz = Path(RESULTS_DIR) / f"jac_{label}{jac_suffix}.npz"
    else:
        jac_npz = Path(RESULTS_DIR) / f"jac_{label}.npz"
        if not jac_npz.exists():
            candidates = sorted(Path(RESULTS_DIR).glob(f"jac_{label}_g*.npz"))
            jac_npz = candidates[-1] if candidates else None
    if jac_npz is None or not jac_npz.exists():
        raise FileNotFoundError(f"No jac for {label}")
    print(f"  jac: {jac_npz.name}")

    jac = load_jac_3d(seed, target_port, wl_nm, RESULTS_DIR,
                      label_suffix=jac_npz.stem.replace(f"jac_{label}", ""))
    params = np.asarray(jac["params"], dtype=float).reshape(-1)
    grad = jac["grads"][variant]
    gain = _flip_gain(grad, params)
    print(f"  variant: {variant}  baseline |a|^2={jac['baseline']['abs2']:.6f}")

    # -- 2. Apply mask -----------------------------------------------------
    excluded = np.zeros(len(params), dtype=bool)
    if mask_cols:
        ix_of = np.arange(len(params)) // NY_2D
        for lo, hi in mask_cols:
            excluded |= (ix_of >= lo) & (ix_of <= hi)
        print(f"  mask: {mask_cols} ({int(np.sum(excluded))} px excluded)")

    # -- 3. Build cluster patches ------------------------------------------
    all_patches = []
    for pv in patch_variants:
        patches = _build_patches(centers, sizes, gain, params, variant=pv)
        # filter rankable
        rankable = [p for p in patches if p["rankable"]]
        unrankable = [p for p in patches if not p["rankable"]]
        if rankable:
            rankable.sort(key=lambda p: p["score"], reverse=True)
        all_patches.extend(rankable)
        all_patches.extend(unrankable)
        print(f"  B-{pv}: {len(rankable)} rankable, {len(unrankable)} below "
              f"pos-frac threshold")

    # -- 4. Build random control patches -----------------------------------
    rng = np.random.RandomState(seed * 100 + target_port + 42)
    exclude_sets = [set(p["indices"]) for p in all_patches]
    random_patches = []
    for pv in patch_variants:
        for sz in sizes:
            for _ in range(null_per_config):
                rp = _random_patch(sz, gain, params, exclude_sets,
                                   variant=pv, rng=rng)
                if rp:
                    random_patches.append(rp)
                    exclude_sets.append(set(rp["indices"]))
    print(f"  random controls: {len(random_patches)}")

    # -- 5. Print patch roster ---------------------------------------------
    print(f"\n  --- Patch Roster ---")
    print(f"  {'type':<6s} {'tag':<40s} {'n_px':>5s} {'score':>10s} "
          f"{'pos_frac':>8s}")
    print(f"  {'-'*72}")
    for p in all_patches + random_patches:
        ptype = "CLUSTER" if not p["tag"].startswith("RANDOM") else "RANDOM"
        print(f"  {ptype:<6s} {p['tag']:<40s} {p['n_pixels']:>5d} "
              f"{p['score']:>10.3e} {p['pos_frac']:>8.2f}")

    # -- 6. Independent device-level measurement ---------------------------
    t0 = time.time()
    candidates = all_patches + random_patches

    results = []
    with DeviceAuditSession(params, target_port, wl_nm,
                            tag=f"cpatch_{label}") as S:
        bl = S.read_baseline()
        J_bl = compute_j_wdm(bl["T_out"], bl["R_in"], target_port, lam_R, lam_X)
        print(f"\n  baseline J_WDM = {J_bl:.4e}  "
              f"T_target={bl['T_out'][target_port-1]:.4e}  "
              f"R_in={bl['R_in']:.4e}")

        # -- null phase: measure random patches first --
        null_dJs = []
        for p in random_patches:
            t_c = time.time()
            m = S.measure(p["indices"])
            J_m = compute_j_wdm(m["T_out"], m["R_in"], target_port, lam_R, lam_X)
            dJ = J_m - J_bl
            null_dJs.append(dJ)
            p["result"] = {
                "T_out": [float(v) for v in m["T_out"]],
                "R_in": float(m["R_in"]),
                "abs2": float(m["abs2"]),
                "J_WDM": float(J_m),
                "dJ_WDM": float(dJ),
                "wall_s": time.time() - t_c,
            }
            results.append({**p, "accepted": None})
            print(f"  [null] {p['tag']} n={p['n_pixels']} "
                  f"dJ={dJ:+.3e} ({time.time()-t_c:.0f}s)")

        # null statistics -> calibrated threshold
        if null_dJs:
            null_mean = float(np.mean(null_dJs))
            null_std = float(np.std(null_dJs, ddof=1)) if len(null_dJs) >= 2 else 0.0
            threshold = max(1e-5, null_mean + 2.0 * null_std)
        else:
            null_mean, null_std = 0.0, 0.0
            threshold = 1e-4
        print(f"  -- null: mean={null_mean:+.3e} sigma={null_std:.3e} "
              f"threshold={threshold:.3e}")

        # -- cluster patch phase --
        n_total = len(all_patches)
        for n, p in enumerate(all_patches, start=1):
            t_c = time.time()
            m = S.measure(p["indices"])
            J_m = compute_j_wdm(m["T_out"], m["R_in"], target_port, lam_R, lam_X)
            dJ = J_m - J_bl
            accepted = bool(dJ > threshold)

            dT_target = m["T_out"][target_port-1] - bl["T_out"][target_port-1]
            dR_in = m["R_in"] - bl["R_in"]

            p["result"] = {
                "T_out": [float(v) for v in m["T_out"]],
                "R_in": float(m["R_in"]),
                "abs2": float(m["abs2"]),
                "J_WDM": float(J_m),
                "dJ_WDM": float(dJ),
                "dT_target": float(dT_target),
                "dR_in": float(dR_in),
                "wall_s": time.time() - t_c,
            }
            results.append({**p, "accepted": accepted})

            verdict = "ACCEPT" if accepted else "reject"
            print(f"  [{n}/{n_total}] {p['tag']} n={p['n_pixels']} "
                  f"dJ={dJ:+.3e} dT={dT_target:+.3e} dR={dR_in:+.3e} "
                  f"-> {verdict} ({time.time()-t_c:.0f}s)")
            sys.stdout.flush()

    # -- 7. Sequential replay with matched null calibration -----------------
    accepted_patches = [r for r in results
                        if not r["tag"].startswith("RANDOM") and r.get("accepted")]
    accepted_patches.sort(key=lambda r: r.get("score", 0), reverse=True)

    replay_results = None
    if accepted_patches:
        # Build matched random patches per accepted candidate
        # (same type, same size, same n_px approx, random non-overlapping centers)
        rng2 = np.random.RandomState(seed * 100 + target_port + 99)
        all_exclude = set()
        for p in accepted_patches:
            all_exclude.update(p["indices"])

        matched_nulls = []
        for p in accepted_patches:
            sz = p["size"]
            pv = p["variant"]
            n_target = p["n_pixels"]
            for _ in range(replay_null_per_type):
                rp = _random_patch(sz, gain, params,
                                   [all_exclude, set(p["indices"])],
                                   variant=pv, rng=rng2)
                if rp and rp["n_pixels"] >= max(2, n_target - 2):
                    matched_nulls.append(rp)
                    all_exclude.update(rp["indices"])
        # If not enough matched, generate extra with relaxed overlap
        while len(matched_nulls) < len(accepted_patches) * 10:
            for p in accepted_patches:
                rp = _random_patch(p["size"], gain, params, [all_exclude],
                                   variant=p["variant"], rng=rng2)
                if rp:
                    matched_nulls.append(rp)
                    all_exclude.update(rp["indices"])
                    if len(matched_nulls) >= len(accepted_patches) * 15:
                        break

        print(f"\n  -- sequential replay with matched null --")
        print(f"  accepted: {len(accepted_patches)}  "
              f"matched null: {len(matched_nulls)}")

        kept = []
        t_r0 = time.time()
        with DeviceAuditSession(params, target_port, wl_nm,
                                tag=f"cpatch_replay_{label}") as S2:
            bl2 = S2.read_baseline()
            J_base2 = compute_j_wdm(bl2["T_out"], bl2["R_in"],
                                    target_port, lam_R, lam_X)
            print(f"  replay baseline J={J_base2:.4e}")

            # -- within-session matched null calibration --
            matched_by_size = {}
            for rp in matched_nulls:
                sz = rp["size"]
                matched_by_size.setdefault(sz, []).append(rp)

            thresholds = {}
            for sz, patches in matched_by_size.items():
                dJs = []
                for rp in patches:
                    t_c = time.time()
                    m = S2.measure(rp["indices"])
                    Jm = compute_j_wdm(m["T_out"], m["R_in"],
                                       target_port, lam_R, lam_X)
                    dJ = Jm - J_base2
                    dJs.append(dJ)
                    print(f"  [matched_null] {rp['tag']} n={rp['n_pixels']} "
                          f"dJ={dJ:+.3e} ({time.time()-t_c:.0f}s)")
                mu = float(np.mean(dJs))
                sd = float(np.std(dJs, ddof=1)) if len(dJs) >= 2 else 0.0
                t_sz = max(1e-4, mu + 2.0 * sd)
                thresholds[sz] = t_sz
                print(f"  matched null size={sz}: n={len(dJs)} "
                      f"mu={mu:+.3e} sigma={sd:.3e} threshold={t_sz:.3e}")

            # -- sequential replay against per-size matched threshold --
            J_cur = J_base2
            for p in accepted_patches:
                t_c = time.time()
                m = S2.measure(p["indices"])
                J_m = compute_j_wdm(m["T_out"], m["R_in"],
                                    target_port, lam_R, lam_X)
                dJ = J_m - J_cur
                t_sz = thresholds.get(p["size"],
                                      max(1e-4, np.mean(list(thresholds.values()))
                                          if thresholds else 1e-4))
                if dJ > t_sz:
                    for idx in p["indices"]:
                        S2._toggle(idx)
                    kept.append({**p, "replay_dJ": float(dJ),
                                 "matched_threshold": float(t_sz)})
                    J_cur = J_m
                    tag = "KEPT"
                else:
                    tag = "undone"
                print(f"  [replay] {p['tag']} n={p['n_pixels']} "
                      f"dJ={dJ:+.3e} thresh={t_sz:.3e} -> {tag} "
                      f"({time.time()-t_c:.0f}s)")
                sys.stdout.flush()

            final_params = S2.get_params()

        replay_wall = time.time() - t_r0

        # -- 8. Fresh-sim verification (only if any kept) -----------------
        fresh_result = None
        if kept:
            print(f"\n  -- fresh-sim verification ({len(kept)} kept) --")
            t_fs = time.time()
            sim_v = _build_3d_sim(tag=f"cpatch_fresh_{label}")
            try:
                _apply_params_to_design(sim_v.fdtd,
                                        np.asarray(final_params, dtype=float).reshape(-1))
                sim_v.fdtd.run()
                T_out_v = []
                for port_i in [1, 2, 3]:
                    coeff = _read_expansion_coefficient_3d(
                        sim_v.fdtd, f"P{port_i}_prof_me")
                    T_out_v.append(coeff.get("T_forward", float("nan")))
                in_c = _read_expansion_coefficient_3d(sim_v.fdtd, "P_in_prof_me")
                R_in_v = abs(in_c.get("T_backward", float("nan")))
                abs2_c = _read_expansion_coefficient_3d(
                    sim_v.fdtd, f"P{target_port}_prof_me")
                abs2_v = abs2_c.get("abs2", float("nan"))
            finally:
                _close_sim(sim_v)
                _sweep_stray_engines()

            J_v = compute_j_wdm(T_out_v, R_in_v, target_port, lam_R, lam_X)
            dJ_fresh = J_v - J_base2
            fresh_wall = time.time() - t_fs
            fresh_result = {
                "J_WDM": float(J_v),
                "dJ_vs_baseline": float(dJ_fresh),
                "T_out": [float(v) for v in T_out_v],
                "R_in": float(R_in_v),
                "abs2": float(abs2_v),
                "dT_target": float(T_out_v[target_port-1] - bl2["T_out"][target_port-1]),
                "dR_in": float(R_in_v - bl2["R_in"]),
                "dAbs2": float(abs2_v - bl2["abs2"]),
            }
            print(f"  fresh-sim: J={J_v:.4e} DeltaJ={dJ_fresh:+.3e} "
                  f"kept={len(kept)}/{len(accepted_patches)}")
        else:
            print(f"\n  -- no patches kept, skipping fresh-sim --")

        replay_results = {
            "n_accepted": len(accepted_patches),
            "n_kept": len(kept),
            "retention_rate": len(kept) / max(len(accepted_patches), 1),
            "matched_thresholds": {str(k): float(v) for k, v in thresholds.items()},
            "kept": kept,
            "incremental_final_J": float(J_cur),
            "fresh_sim": fresh_result,
            "wall_s": {"replay": replay_wall,
                       "fresh_sim": fresh_result.get("wall_s", 0) if fresh_result else 0},
        }
    else:
        print(f"\n  -- no accepted patches for replay --")

    wall_s = time.time() - t0

    # -- 9. Summary --------------------------------------------------------
    cluster_results = [r for r in results
                       if not r["tag"].startswith("RANDOM")]
    n_cluster_acc = sum(1 for r in cluster_results if r.get("accepted"))
    cluster_dJs = [r["result"]["dJ_WDM"] for r in cluster_results]

    # compare to single-pixel baseline
    sp_summary = {
        "n_cluster_patches": len(cluster_results),
        "n_accepted": n_cluster_acc,
        "accept_rate": n_cluster_acc / max(len(cluster_results), 1),
        "null_mean": null_mean,
        "null_std": null_std,
        "threshold": threshold,
        "accepted_patches": [r["tag"] for r in cluster_results if r.get("accepted")],
        "dJ_range": [float(np.min(cluster_dJs)) if cluster_dJs else 0,
                     float(np.max(cluster_dJs)) if cluster_dJs else 0],
        "dJ_mean": float(np.mean(cluster_dJs)) if cluster_dJs else 0,
    }

    summary = {
        "label": label,
        "timestamp": _now_iso(),
        "experiment": "Type B cluster-patch audit",
        "seed": seed, "target_port": target_port, "wl_nm": wl_nm,
        "variant": variant, "jac_file": str(jac_npz.name),
        "mask_cols": mask_cols,
        "centers": [list(c) for c in centers],
        "sizes": list(sizes),
        "patch_variants": list(patch_variants),
        "lam_R": lam_R, "lam_X": lam_X,
        "baseline": {
            "T_out": [float(v) for v in bl["T_out"]],
            "R_in": float(bl["R_in"]),
            "abs2": float(bl["abs2"]),
            "J_WDM": float(J_bl),
        },
        "cluster_summary": sp_summary,
        "replay": replay_results,
        "wall_s": wall_s,
        "entries": results,
    }

    # -- 10. Output --------------------------------------------------------
    mask_tag = ""
    if mask_cols:
        mask_tag = "_mask" + "+".join(f"{lo}-{hi}" for lo, hi in mask_cols)
    size_tag = "s" + "s".join(str(s) for s in sizes)
    pv_tag = "-".join(patch_variants)
    out_name = (f"cluster_{label}_{variant}{mask_tag}_{size_tag}_"
                f"{pv_tag}.json")
    out_json = out_dir / out_name
    _save_json(summary, out_json)

    print(f"\n{'=' * 60}")
    print("CLUSTER AUDIT SUMMARY")
    print(f"{'=' * 60}")
    print(f"  cluster patches: {len(cluster_results)} total, "
          f"{n_cluster_acc} accepted ({sp_summary['accept_rate']:.0%})")
    print(f"  accepted: {sp_summary['accepted_patches']}")
    if replay_results:
        print(f"  replay: {replay_results['n_kept']}/{replay_results['n_accepted']} "
              f"kept, fresh-sim DeltaJ={replay_results['fresh_sim']['dJ_vs_baseline']:+.3e}")
    print(f"  wall: {wall_s:.0f}s  ({wall_s/60:.1f}min)")
    print(f"  -> {out_json}")
    return summary


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Type B cluster-patch actionability audit (WDM3 3D)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--port", type=int, default=1)
    ap.add_argument("--wl", type=float, default=WL_NM)
    ap.add_argument("--variant", default="conj_a",
                    choices=list(GRAD_VARIANTS))
    ap.add_argument("--mask", type=str, default="72-79")
    ap.add_argument("--centers", type=str, default="43,26;46,29",
                    help="Cluster centers, e.g. '43,26;46,29'")
    ap.add_argument("--sizes", type=str, default="2,3,5",
                    help="Bbox sizes, e.g. '2,3,5'")
    ap.add_argument("--patch-variants", type=str, default="sparse,dense",
                    help="'sparse,dense' or 'sparse' or 'dense'")
    ap.add_argument("--null", type=int, default=2,
                    help="Random patches per (size,variant) combo")
    ap.add_argument("--lam-R", type=float, default=1.0)
    ap.add_argument("--lam-X", type=float, default=0.5)
    ap.add_argument("--jac-suffix", type=str, default=None)
    ap.add_argument("--replay-from", type=str, default=None,
                    metavar="AUDIT_JSON",
                    help="Replay-only: load audit JSON, matched-null replay")
    ap.add_argument("--replay-null", type=int, default=15,
                    help="Matched random patches per accepted candidate")
    args = ap.parse_args()

    # --- Replay-only mode ---
    if args.replay_from:
        audit_path = Path(args.replay_from)
        if not audit_path.exists():
            audit_path = CLUSTER_DIR / args.replay_from
        if not audit_path.exists():
            raise FileNotFoundError(f"Audit not found: {args.replay_from}")
        with open(audit_path) as f:
            audit = json.load(f)

        # extract accepted cluster patches
        accepted = [e for e in audit["entries"]
                    if not e["tag"].startswith("RANDOM") and e.get("accepted")]
        accepted.sort(key=lambda e: e.get("score", 0), reverse=True)
        print(f"Loaded {len(accepted)} accepted patches from {audit_path.name}")

        # load jac
        label = _label(args.seed, args.port, args.wl)
        jac_file = audit.get("jac_file", f"jac_{label}.npz")
        jac_sfx = jac_file.replace(f"jac_{label}", "").replace(".npz", "")
        jac = load_jac_3d(args.seed, args.port, args.wl, RESULTS_DIR,
                          label_suffix=jac_sfx)
        params = np.asarray(jac["params"], dtype=float).reshape(-1)
        gain = _flip_gain(jac["grads"][args.variant], params)

        mask_cols = None
        if args.mask and args.mask.lower() != "none":
            mask_cols = []
            for part in args.mask.split(","):
                lo, hi = part.strip().split("-")
                mask_cols.append((int(lo), int(hi)))

        # Generate matched null random patches
        rng = np.random.RandomState(args.seed * 100 + args.port + 99)
        all_exclude = set()
        for p in accepted:
            all_exclude.update(p["indices"])
        matched_nulls = []
        for p in accepted:
            sz = p["size"]
            pv = p["variant"]
            for _ in range(args.replay_null):
                rp = _random_patch(sz, gain, params,
                                   [all_exclude], variant=pv, rng=rng)
                if rp:
                    matched_nulls.append(rp)
                    all_exclude.update(rp["indices"])

        print(f"Matched null: {len(matched_nulls)} generated")

        # Run replay with matched null
        thresholds = {}
        kept = []
        t_r0 = time.time()
        with DeviceAuditSession(params, args.port, args.wl,
                                tag=f"cpatch_rf_{label}") as S:
            bl = S.read_baseline()
            J_bl = compute_j_wdm(bl["T_out"], bl["R_in"], args.port,
                                 args.lam_R, args.lam_X)
            print(f"Replay baseline J={J_bl:.4e}")

            # Within-session null calibration per size
            null_by_size = {}
            for rp in matched_nulls:
                null_by_size.setdefault(rp["size"], []).append(rp)

            for sz, patches in null_by_size.items():
                dJs = []
                for rp in patches:
                    m = S.measure(rp["indices"])
                    Jm = compute_j_wdm(m["T_out"], m["R_in"],
                                       args.port, args.lam_R, args.lam_X)
                    dJ = Jm - J_bl
                    dJs.append(dJ)
                    print(f"  [null] {rp['tag']} n={rp['n_pixels']} dJ={dJ:+.3e}")
                mu = float(np.mean(dJs))
                sd = float(np.std(dJs, ddof=1)) if len(dJs) >= 2 else 0.0
                thresholds[sz] = max(1e-4, mu + 2.0 * sd)
                print(f"  size={sz}: n={len(dJs)} mu={mu:+.3e} sigma={sd:.3e} "
                      f"thresh={thresholds[sz]:.3e}")

            # Sequential replay
            J_cur = J_bl
            for p in accepted:
                m = S.measure(p["indices"])
                Jm = compute_j_wdm(m["T_out"], m["R_in"],
                                   args.port, args.lam_R, args.lam_X)
                dJ = Jm - J_cur
                t_sz = thresholds.get(p["size"], max(thresholds.values()) if thresholds else 1e-4)
                if dJ > t_sz:
                    for idx in p["indices"]:
                        S._toggle(idx)
                    kept.append({**p, "replay_dJ": float(dJ), "matched_threshold": float(t_sz)})
                    J_cur = Jm
                    tag = "KEPT"
                else:
                    tag = "undone"
                print(f"  [replay] {p['tag']} dJ={dJ:+.3e} thresh={t_sz:.3e} -> {tag}")
                sys.stdout.flush()

            final_params = S.get_params()

        # Fresh-sim if any kept
        fresh_result = None
        if kept:
            print(f"\n-- fresh-sim ({len(kept)} kept) --")
            sim_v = _build_3d_sim(tag=f"cpatch_rf_fresh_{label}")
            try:
                _apply_params_to_design(sim_v.fdtd,
                                        np.asarray(final_params, dtype=float).reshape(-1))
                sim_v.fdtd.run()
                T_out_v = []
                for pi in [1, 2, 3]:
                    c = _read_expansion_coefficient_3d(sim_v.fdtd, f"P{pi}_prof_me")
                    T_out_v.append(c.get("T_forward", float("nan")))
                in_c = _read_expansion_coefficient_3d(sim_v.fdtd, "P_in_prof_me")
                R_in_v = abs(in_c.get("T_backward", float("nan")))
                abs2_c = _read_expansion_coefficient_3d(sim_v.fdtd, f"P{args.port}_prof_me")
                abs2_v = abs2_c.get("abs2", float("nan"))
            finally:
                _close_sim(sim_v)
                _sweep_stray_engines()
            J_v = compute_j_wdm(T_out_v, R_in_v, args.port, args.lam_R, args.lam_X)
            fresh_result = {
                "J_WDM": float(J_v), "dJ_vs_baseline": float(J_v - J_bl),
                "T_out": [float(v) for v in T_out_v], "R_in": float(R_in_v),
                "abs2": float(abs2_v),
            }
            print(f"  fresh-sim J={J_v:.4e} DeltaJ={J_v-J_bl:+.3e}")
        else:
            print("  no patches kept, skip fresh-sim")

        # Save replay result
        out_name = audit_path.stem.replace("cluster_", "replay_cluster_") + ".json"
        out_json = CLUSTER_DIR / out_name
        replay_out = {
            "timestamp": _now_iso(),
            "audit_source": str(audit_path.name),
            "accepted": accepted,
            "matched_nulls": len(matched_nulls),
            "thresholds": {str(k): float(v) for k, v in thresholds.items()},
            "kept": kept,
            "n_kept": len(kept), "n_accepted": len(accepted),
            "fresh_sim": fresh_result,
            "wall_s": time.time() - t_r0,
        }
        _save_json(replay_out, out_json)
        print(f"-> {out_json}")
        sys.exit(0)

    # --- Full audit mode (unchanged) ---

    # parse
    mask_cols = None
    if args.mask and args.mask.lower() != "none":
        mask_cols = []
        for part in args.mask.split(","):
            lo, hi = part.strip().split("-")
            mask_cols.append((int(lo), int(hi)))

    centers = []
    for pair in args.centers.split(";"):
        cx, cy = pair.strip().split(",")
        centers.append((int(cx), int(cy)))

    sizes = tuple(int(s.strip()) for s in args.sizes.split(","))
    patch_variants = tuple(args.patch_variants.split(","))

    run_cluster_audit(
        seed=args.seed,
        target_port=args.port,
        wl_nm=args.wl,
        variant=args.variant,
        mask_cols=mask_cols,
        centers=centers,
        sizes=sizes,
        patch_variants=patch_variants,
        null_per_config=args.null,
        lam_R=args.lam_R,
        lam_X=args.lam_X,
        jac_suffix=args.jac_suffix,
    )
