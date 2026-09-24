#!/usr/bin/env python
"""Single-pixel actionability audit for WDM3 3D warm-start anchors.

After phase/mask diagnostics identify the nearest actionable formal variant,
this script audits whether ranked single-pixel proposals survive device-level
acceptance.  The answer — positive or negative — is itself the result.

Framing (2026-07-26):
  formal–device gap       : continuous proposal signal ≠ device metric
  diagnostic framework    : Rule M/P/K identifies actionable formal variants
  single-pixel audit      : even diagnostically selected single-pixel proposals
                            may fail at device level — not because the effect
                            is too small to measure, but because the ranking
                            criterion (mode overlap) and the acceptance
                            criterion (port power distribution) are
                            structurally decoupled at single-pixel action
                            granularity.

Usage:
  # independent audit
  python -m wdm3_3d.single_pixel_audit --seed 7 --port 1 --k 20 --null 25
  python -m wdm3_3d.single_pixel_audit --seed 7 --port 1 --k 20 --mask none
  python -m wdm3_3d.single_pixel_audit --seed 7 --port 1 --jac-suffix _g1
  # sequential replay (after audit)
  python -m wdm3_3d.single_pixel_audit --seed 7 --port 1 ^
      --replay results_3d/single_pixel_audit/audit_seed7_P1@1550nm_...json
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── scipy: try import early; fall back to numpy-only Spearman if absent ──
try:
    from scipy.stats import spearmanr as _spearmanr
except ImportError:
    _spearmanr = None

# ── package init sets LUMOPT_BACKEND=2026; load module to trigger env ──
from . import wdm3_3d_landscape  # noqa: F401 — sets backend env
import lum_backend  # noqa: F401
import _lumopt_compat  # noqa: F401

from .wdm3_3d_device import (
    DESIGN_X_NM,
    DESIGN_Y_NM,
    NX_2D,
    NY_2D,
    WL_NM,
    _build_3d_sim,
    _close_sim,
    load_warmstart_params,
)
from .wdm3_3d_smoke import _apply_params_to_design
from .wdm3_3d_audit import (
    _ensure_mode_expansion_3d,
    _read_expansion_coefficient_3d,
)
from .wdm3_3d_jac import (
    GRAD_VARIANTS,
    RESULTS_DIR,
    _flip_gain,
    _flip_single_pixel,
    _label,
    _now_iso,
    _save_json,
    _sweep_stray_engines,
    load_jac_3d,
)

AUDIT_DIR = Path(__file__).resolve().parent / "results_3d" / "single_pixel_audit"

# Mode expansions that must exist before device readout
_EXPANSIONS = {
    "output": [("P1_prof", "P1_prof_me"), ("P2_prof", "P2_prof_me"),
               ("P3_prof", "P3_prof_me")],
    "input": [("P_in_prof", "P_in_prof_me")],
}


# ═══════════════════════════════════════════════════════════════════════════
# Spearman rank correlation (numpy fallback)
# ═══════════════════════════════════════════════════════════════════════════

def _spearman_r(x, y):
    """Spearman rank correlation; uses scipy if available, else numpy."""
    if _spearmanr is not None:
        rho, p = _spearmanr(x, y)
        return float(rho), float(p)
    # numpy fallback: compute rank correlation manually
    from numpy import argsort, mean, sqrt, sum as npsum
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    rx = argsort(argsort(x)).astype(float)
    ry = argsort(argsort(y)).astype(float)
    # Pearson on ranks
    mx, my = mean(rx), mean(ry)
    num = npsum((rx - mx) * (ry - my))
    den = sqrt(npsum((rx - mx) ** 2) * npsum((ry - my) ** 2))
    if den == 0:
        return float("nan"), float("nan")
    return float(num / den), float("nan")  # no p-value in fallback


# ═══════════════════════════════════════════════════════════════════════════
# Device-metric session
# ═══════════════════════════════════════════════════════════════════════════

class DeviceAuditSession:
    """One CAD session; per-pixel toggle + device-level metric readout.

    Keeps the FDTD alive across flips (≈30–40 s per candidate vs ≈5 min
    fresh-sim).  Explicitly ensures all mode expansions exist before the
    baseline run, and reads T_forward (P1/P2/P3) + T_backward (P_in) after
    each run.
    """

    def __init__(self, params, target_port, wl_nm=WL_NM, tag="dev_audit"):
        self.target_port = target_port
        self.state = np.asarray(params, dtype=float).reshape(-1) > 0.5
        self.sim = _build_3d_sim(tag=tag, wl_nm=wl_nm)
        _apply_params_to_design(self.sim.fdtd,
                                np.asarray(params, dtype=float).reshape(-1))
        self._expansions_ok = False
        # baseline device metrics (populated by read_baseline)
        self.baseline_T_out = None
        self.baseline_R_in = None
        self.baseline_abs2 = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        _close_sim(self.sim)
        _sweep_stray_engines()

    # ── internal ──────────────────────────────────────────────────────

    def _ensure_expansions(self):
        """Create mode expansions if missing; return True if a rerun is needed."""
        if self._expansions_ok:
            return False
        needs_rerun = False
        fdtd = self.sim.fdtd
        for _, pairs in _EXPANSIONS.items():
            for prof_name, me_name in pairs:
                if _ensure_mode_expansion_3d(fdtd, prof_name, me_name,
                                             mode_number=1):
                    needs_rerun = True
        self._expansions_ok = True
        return needs_rerun

    def _run_and_read_all(self):
        """Run FDTD, then read T_forward (P1/P2/P3) + T_backward (P_in) + |a|^2.

        R_in is taken as abs(T_backward) to guard against sign conventions.
        """
        self.sim.fdtd.run()

        T_out = []
        for port_i in [1, 2, 3]:
            coeff = _read_expansion_coefficient_3d(
                self.sim.fdtd, f"P{port_i}_prof_me")
            T_out.append(coeff.get("T_forward", float("nan")))

        in_coeff = _read_expansion_coefficient_3d(self.sim.fdtd, "P_in_prof_me")
        R_in = abs(in_coeff.get("T_backward", float("nan")))

        abs2_coeff = _read_expansion_coefficient_3d(
            self.sim.fdtd, f"P{self.target_port}_prof_me")
        abs2 = abs2_coeff.get("abs2", float("nan"))

        return {"T_out": T_out, "R_in": R_in, "abs2": abs2}

    def _toggle(self, idx):
        """Flip pixel *idx* Si↔SiO2 (single rect toggle in layout)."""
        self.sim.fdtd.switchtolayout()
        _flip_single_pixel(self.sim.fdtd, int(idx), to_si=not self.state[idx])
        self.state[idx] = not self.state[idx]

    # ── public ────────────────────────────────────────────────────────

    def read_baseline(self):
        """Ensure expansions exist, run, and establish baseline device metrics."""
        if self._ensure_expansions():
            # expansions were missing — one initial run to populate them
            self.sim.fdtd.run()
        result = self._run_and_read_all()
        self.baseline_T_out = result["T_out"]
        self.baseline_R_in = result["R_in"]
        self.baseline_abs2 = result["abs2"]
        print(f"  baseline |a|^2={self.baseline_abs2:.6f}  "
              f"T_out={[f'{t:.4e}' for t in self.baseline_T_out]}  "
              f"R_in={self.baseline_R_in:.4e}")
        return result

    def measure(self, idx_list, keep=False):
        """Flip pixels, run, read all metrics; revert unless *keep*.

        Returns dict: T_out (list[3]), R_in (float), abs2 (float).
        """
        for i in idx_list:
            self._toggle(i)
        result = self._run_and_read_all()
        if not keep:
            for i in reversed(idx_list):
                self._toggle(i)
        return result

    def get_params(self):
        """Return current binary state as float (0.0/1.0) array."""
        return self.state.astype(float)


# ═══════════════════════════════════════════════════════════════════════════
# J_WDM computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_j_wdm(T_out, R_in, target_port, lam_R=1.0, lam_X=0.5):
    """Device-level scalar FOM for WDM3 single-pixel audit.

    J_WDM = T_target - lam_R * R_in - lam_X * T_wrong_mean

    target_port is 1-indexed.  R_in should already be abs(T_backward).
    """
    t_idx = target_port - 1
    T_target = T_out[t_idx]
    T_wrong_mean = float(np.mean(
        [T_out[i] for i in range(3) if i != t_idx]))
    return float(T_target - lam_R * R_in - lam_X * T_wrong_mean)


# ═══════════════════════════════════════════════════════════════════════════
# Output filename builder
# ═══════════════════════════════════════════════════════════════════════════

def _make_out_name(label, variant, k, mask_cols, lam_R, lam_X, null_draws, bottom=False):
    """Build a unique output filename that encodes all key parameters."""
    mask_tag = ""
    if mask_cols:
        mask_tag = "_mask" + "+".join(f"{lo}-{hi}" for lo, hi in mask_cols)
    else:
        mask_tag = "_unmasked"
    direction = "bottom" if bottom else "top"
    return (f"audit_{label}_{variant}_{direction}{k}{mask_tag}"
            f"_lamR{lam_R}_lamX{lam_X}_null{null_draws}.json")


# ═══════════════════════════════════════════════════════════════════════════
# Main: single-pixel actionability audit
# ═══════════════════════════════════════════════════════════════════════════

def run_audit(seed=7, target_port=1, wl_nm=WL_NM, k=20,
              variant="conj_a", mask_cols=None, null_draws=25,
              lam_R=1.0, lam_X=0.5, accept_threshold_user=None,
              jac_suffix=None, out_dir=None, bottom=False):
    """Run the single-pixel actionability audit.

    Parameters
    ----------
    seed, target_port, wl_nm : int
        Anchor spec.
    k : int
        Number of top-ranked (or bottom-ranked) pixels to audit.
    variant : str
        Adjoint gradient variant for ranking (default "conj_a").
    mask_cols : list[tuple] | None
        Column ranges to exclude from ranking, e.g. [(72, 79)].
    null_draws : int
        Number of random pixels to measure for null distribution (≥25 for
        publication-grade evidence).
    lam_R, lam_X : float
        Penalty weights for input reflection and crosstalk.
    accept_threshold_user : float | None
        User-specified minimum DeltaJ_WDM. If None, threshold is calibrated from
        the null distribution: max(1e-5, null_mean + 2·null_std).
    jac_suffix : str | None
        Explicit jac generation suffix, e.g. "_g1". If None, auto-selects
        the latest available .npz.
    out_dir : Path | None
        Output directory (default: results_3d/single_pixel_audit/).
    bottom : bool
        If True, audit the *least* promising pixels (bottom-K by gain) instead
        of top-K.  Used to verify ranking directionality: bottom-ranked pixels
        should show worse FDTD outcomes than random null.
    """
    out_dir = Path(out_dir) if out_dir else AUDIT_DIR
    label = _label(seed, target_port, wl_nm)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load jacobian ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"SINGLE-PIXEL ACTIONABILITY AUDIT: {label}")
    print(f"{'=' * 60}")

    if jac_suffix is not None:
        jac_npz = Path(RESULTS_DIR) / f"jac_{label}{jac_suffix}.npz"
        if not jac_npz.exists():
            raise FileNotFoundError(f"jac not found: {jac_npz}")
        print(f"  jac: {jac_npz.name} (explicit)")
    else:
        jac_npz = Path(RESULTS_DIR) / f"jac_{label}.npz"
        if not jac_npz.exists():
            candidates = sorted(Path(RESULTS_DIR).glob(f"jac_{label}_g*.npz"))
            if candidates:
                jac_npz = candidates[-1]
                print(f"  jac: {jac_npz.name} (auto, latest of {len(candidates)})")
            else:
                raise FileNotFoundError(
                    f"No jac found for {label} in {RESULTS_DIR}. "
                    f"Use --jac-suffix to specify explicitly.")
        else:
            print(f"  jac: {jac_npz.name}")

    jac = load_jac_3d(seed, target_port, wl_nm, RESULTS_DIR,
                      label_suffix=jac_npz.stem.replace(f"jac_{label}", ""))
    if jac is None:
        raise RuntimeError(f"Failed to load jac from {jac_npz}")

    params = np.asarray(jac["params"], dtype=float).reshape(-1)
    baseline_abs2_jac = jac["baseline"]["abs2"]
    grad = jac["grads"].get(variant)
    if grad is None:
        raise KeyError(f"Variant {variant!r} not in jac; "
                       f"available: {list(jac['grads'].keys())}")
    print(f"  variant: {variant}  baseline |a|^2 = {baseline_abs2_jac:.6f}")

    # ── 2. Rank pixels by flip-gain (masked) ──────────────────────────
    gain = _flip_gain(grad, params)
    excluded = np.zeros(len(params), dtype=bool)

    if mask_cols:
        ix_of = np.arange(len(params)) // NY_2D
        for lo, hi in mask_cols:
            excluded |= (ix_of >= lo) & (ix_of <= hi)
        n_masked = int(np.sum(excluded))
        print(f"  mask: {mask_cols} ({n_masked} pixels excluded)")

    sel_gain = np.where(excluded, -np.inf, gain)
    if bottom:
        order = np.argsort(sel_gain)       # ascending: most negative first
        n_candidates = int(np.sum(sel_gain < 0))
        direction = "bottom"
        tag_prefix = "bottom"
    else:
        order = np.argsort(sel_gain)[::-1]  # descending: most positive first
        n_candidates = int(np.sum(sel_gain > 0))
        direction = "top"
        tag_prefix = "top"
    top_k = [int(order[i]) for i in range(min(k, n_candidates))]
    top_k_set = set(top_k)
    print(f"  negative-gain pool: {n_candidates}  {direction}-{k} effective: {len(top_k)}"
          if bottom else
          f"  positive-gain pool: {n_candidates}  {direction}-{k} effective: {len(top_k)}")
    if top_k:
        print(f"  {direction}-5 predicted gains: "
              f"{[f'{gain[i]:+.3e}' for i in top_k[:5]]}")

    # ── 3. Null calibration: random pixels (exclude top-K from pool) ──
    rng = np.random.RandomState(seed * 100 + target_port)
    eligible = np.where(~excluded)[0]
    # exclude top-K so null is genuinely independent
    eligible_no_top = np.array([i for i in eligible if i not in top_k_set],
                               dtype=int)
    n_null_actual = min(null_draws, len(eligible_no_top))
    null_idx = [int(eligible_no_top[i]) for i in
                rng.choice(len(eligible_no_top), size=n_null_actual,
                           replace=False)]
    print(f"  null calibration: {n_null_actual} random pixels "
          f"(excluded top-{k} from pool)")

    # ── 4. Device-level measurement ───────────────────────────────────
    t0 = time.time()

    # Order: null first (so accept threshold can be null-calibrated),
    #        then top-K.
    all_candidates = ([(f"null_{i}", idx) for i, idx in enumerate(null_idx)]
                      + [(f"{tag_prefix}_{r}", idx) for r, idx in enumerate(top_k, start=1)])

    results = []
    with DeviceAuditSession(params, target_port, wl_nm,
                            tag=f"audit_{label}") as S:
        # baseline
        bl = S.read_baseline()
        J_bl = compute_j_wdm(bl["T_out"], bl["R_in"], target_port, lam_R, lam_X)
        print(f"  baseline J_WDM = {J_bl:.4e}  "
              f"(T_target={bl['T_out'][target_port - 1]:.4e}  "
              f"R_in={bl['R_in']:.4e}  "
              f"T_wrong_mean="
              f"{np.mean([bl['T_out'][i] for i in range(3) if i != target_port - 1]):.4e})")

        # null phase
        null_phase = True
        null_dJs = []

        n_total = len(all_candidates)
        for n, (tag, idx) in enumerate(all_candidates, start=1):
            t_c = time.time()
            m = S.measure([idx])
            J_m = compute_j_wdm(m["T_out"], m["R_in"], target_port, lam_R, lam_X)
            dJ = J_m - J_bl
            dT_target = (m["T_out"][target_port - 1]
                         - bl["T_out"][target_port - 1])
            dR_in = m["R_in"] - bl["R_in"]
            dT_wrong = (
                np.mean([m["T_out"][i] for i in range(3)
                         if i != target_port - 1])
                - np.mean([bl["T_out"][i] for i in range(3)
                          if i != target_port - 1]))
            dAbs2 = m["abs2"] - bl["abs2"]

            # defer accept/reject until after null phase
            if null_phase and not tag.startswith("null"):
                # transition: finalize null -> compute calibrated threshold
                if null_dJs:
                    null_mean = float(np.mean(null_dJs))
                    null_std = float(np.std(null_dJs, ddof=1)) if len(null_dJs) >= 2 else 0.0
                    null_2s = null_mean + 2.0 * null_std
                    if accept_threshold_user is not None:
                        accept_threshold = max(accept_threshold_user, null_2s)
                    else:
                        accept_threshold = max(1e-5, null_2s)
                    print(f"  -- null phase complete --")
                    print(f"  null dJ: mean={null_mean:+.3e}  std={null_std:.3e}  "
                          f"2sigma={null_2s:+.3e}")
                    print(f"  calibrated accept threshold: dJ > {accept_threshold:.3e}")
                else:
                    accept_threshold = accept_threshold_user or 1e-4
                null_phase = False

            if tag.startswith("null"):
                null_dJs.append(dJ)
                accepted = None  # null pixels aren't subject to accept/reject
            else:
                accepted = bool(dJ > accept_threshold)

            entry = {
                "rank": int(tag.split("_")[1]) if tag.startswith(tag_prefix) else None,
                "tag": tag,
                "idx": int(idx),
                "ix": idx // NY_2D,
                "iy": idx % NY_2D,
                "pred_gain": float(gain[idx]) if idx < len(gain) else None,
                "J_WDM": float(J_m),
                "dJ_WDM": float(dJ),
                "dT_target": float(dT_target),
                "dR_in": float(dR_in),
                "dT_wrong": float(dT_wrong),
                "dAbs2": float(dAbs2),
                "accepted": accepted,
                "wall_s": time.time() - t_c,
            }
            results.append(entry)

            if accepted is None:
                v_str = "null"
            elif accepted:
                v_str = "ACCEPT"
            else:
                v_str = "reject"
            print(f"  [{n}/{n_total}] {tag} idx={idx} "
                  f"(ix={idx // NY_2D},iy={idx % NY_2D}) "
                  f"dJ={dJ:+.3e} dT_target={dT_target:+.3e} "
                  f"dR_in={dR_in:+.3e} dAbs2={dAbs2:+.3e} "
                  f"-> {v_str} ({time.time() - t_c:.0f}s)")

    wall_s = time.time() - t0

    # ── 5. Summary ───────────────────────────────────────────────────
    top_results = [r for r in results if r["tag"].startswith(tag_prefix)]
    null_results = [r for r in results if r["tag"].startswith("null")]

    n_accepted = sum(1 for r in top_results if r["accepted"])
    top_dJ = [r["dJ_WDM"] for r in top_results]
    null_dJ_report = [r["dJ_WDM"] for r in null_results]
    top_dAbs2 = [r["dAbs2"] for r in top_results]

    null_mean = float(np.mean(null_dJ_report)) if null_dJ_report else 0.0
    null_std = (float(np.std(null_dJ_report, ddof=1))
                if len(null_dJ_report) >= 2 else float("nan"))

    # formal–device rank correlation within top-K
    if len(top_results) >= 5:
        rho, p_rho = _spearman_r(
            [r["pred_gain"] for r in top_results],
            [r["dJ_WDM"] for r in top_results],
        )
    else:
        rho, p_rho = float("nan"), float("nan")

    summary = {
        "label": label,
        "timestamp": _now_iso(),
        "seed": seed,
        "target_port": target_port,
        "wl_nm": wl_nm,
        "variant": variant,
        "jac_file": str(jac_npz.name),
        "mask_cols": mask_cols,
        "k": k,
        "direction": direction,
        f"n_{direction}_pool": n_candidates,
        "null_draws": null_draws,
        "null_actual": n_null_actual,
        "lam_R": lam_R,
        "lam_X": lam_X,
        "accept_threshold_user": accept_threshold_user,
        "accept_threshold_calibrated": accept_threshold,
        "baseline": {
            "abs2_jac": baseline_abs2_jac,
            "abs2_session": bl["abs2"],
            "T_out": [float(v) for v in bl["T_out"]],
            "R_in": float(bl["R_in"]),
            "J_WDM": float(J_bl),
        },
        "top_k": {
            "n_accepted": n_accepted,
            "n_total": len(top_results),
            "accept_rate": (float(n_accepted / len(top_results))
                            if top_results else 0),
            "dJ_mean": float(np.mean(top_dJ)) if top_dJ else float("nan"),
            "dJ_std": (float(np.std(top_dJ, ddof=1))
                       if len(top_dJ) >= 2 else float("nan")),
            "dJ_min": float(np.min(top_dJ)) if top_dJ else float("nan"),
            "dJ_max": float(np.max(top_dJ)) if top_dJ else float("nan"),
            "dAbs2_mean": float(np.mean(top_dAbs2)) if top_dAbs2 else float("nan"),
            "spearman_rho_pred_vs_dJ": float(rho),
            "spearman_p": float(p_rho),
        },
        "null": {
            "dJ_mean": null_mean,
            "dJ_std": null_std,
            "dJ_min": float(np.min(null_dJ_report)) if null_dJ_report else float("nan"),
            "dJ_max": float(np.max(null_dJ_report)) if null_dJ_report else float("nan"),
        },
        "verdict": _make_verdict(n_accepted, len(top_results),
                                 top_dJ, null_dJ_report, null_mean, null_std),
        "wall_s": wall_s,
        "entries": results,
    }

    # ── 6. Output ────────────────────────────────────────────────────
    out_name = _make_out_name(label, variant, k, mask_cols,
                              lam_R, lam_X, null_draws, bottom=bottom)
    out_json = out_dir / out_name
    _save_json(summary, out_json)

    print(f"\n{'=' * 60}")
    print("AUDIT SUMMARY")
    print(f"{'=' * 60}")
    print(f"  direction: {direction}  variant: {variant}  jac: {jac_npz.name}  mask: {mask_cols}")
    print(f"  null ({n_null_actual} draws): dJ mean={null_mean:+.3e}  "
          f"std={null_std:.3e}")
    print(f"  null-calibrated accept threshold: dJ > {accept_threshold:.3e}")
    print(f"  {direction}-{k}: {n_accepted}/{len(top_results)} accepted "
          f"(rate={summary['top_k']['accept_rate']:.0%}, "
          f"dJ mean={summary['top_k']['dJ_mean']:+.3e} "
          f"std={summary['top_k']['dJ_std']:.3e})")
    print(f"  formal pred_gain vs device dJ: rho = {rho:+.3f}"
          + (f" (p={p_rho:.3f})" if not np.isnan(p_rho) else ""))
    print(f"  verdict: {summary['verdict']}")
    print(f"  wall: {wall_s:.0f}s  ({wall_s / 60:.1f}min)")
    print(f"  -> {out_json}")

    return summary


def _make_verdict(n_acc, n_total, top_dJ, null_dJ,
                  null_mean, null_std):
    """Produce a one-line verdict string for the audit."""
    if n_total == 0:
        return ("NO CANDIDATES: zero positive-gain pixels in pool — "
                "the formal adjoint found no actionable direction at "
                "single-pixel granularity.")

    accept_rate = n_acc / n_total
    top_mean = float(np.mean(top_dJ)) if top_dJ else float("nan")
    top_max = float(np.max(top_dJ)) if top_dJ else float("nan")

    null_band_2s = (null_mean + 2.0 * null_std
                    if not np.isnan(null_std) else float("inf"))

    if accept_rate >= 0.25:
        return (f"POSITIVE TREND: {n_acc}/{n_total} accepted "
                f"({accept_rate:.0%}) — single-pixel proposals survive "
                f"device-level acceptance above expectation.")
    elif top_max > null_band_2s:
        return (f"MARGINAL: best dJ={top_max:+.3e} exceeds null 2sigma "
                f"({null_band_2s:+.3e}), but accept rate {accept_rate:.0%} "
                f"< 25% — isolated hits, not a reliable trend.")
    elif accept_rate >= 0.05:
        return (f"WEAK: {n_acc}/{n_total} accepted ({accept_rate:.0%}), "
                f"top dJ within null band ({null_band_2s:+.3e}) — "
                f"single-pixel action insufficient at device level.")
    else:
        return (f"REJECTED: {n_acc}/{n_total} accepted — "
                f"diagnostically selected single-pixel proposals fail at "
                f"device-level acceptance.  The same device-level null "
                f"calibration contained {null_std:.1e}-scale active pixels "
                f"(null sigma={null_std:.1e}), whereas formal top-ranked pixels "
                f"produced dJ in [{min(top_dJ):.1e}, {max(top_dJ):.1e}].  "
                f"Thus the failure occurred at the ranking/action-granularity "
                f"level rather than at the measurement threshold.")


# ═══════════════════════════════════════════════════════════════════════════
# Sequential replay: accumulate accepted flips, verify with fresh-sim
# ═══════════════════════════════════════════════════════════════════════════

def run_replay(audit_json, seed, target_port, wl_nm=WL_NM,
               lam_R=1.0, lam_X=0.5, accept_threshold=None,
               rank_order=True, out_dir=None):
    """Sequentially replay top-K accepted flips from an audit JSON.

    Loads the independent audit results, then opens a fresh incremental
    session and replays the top-ranked candidates in order, keeping flips
    that pass the device-level accept threshold.  At the end, a fresh
    FDTD session verifies the final device state.

    This measures epistasis: independently-accepted flips may interact
    when applied together.

    Parameters
    ----------
    audit_json : Path
        Path to a previous audit result JSON.
    seed, target_port, wl_nm : int
        Anchor spec (must match the audit).
    lam_R, lam_X : float
        J_WDM weights (must match the audit).
    accept_threshold : float | None
        If None, uses the null-calibrated threshold from the audit JSON.
    rank_order : bool
        If True, replay in original formal rank order. If False, replay
        in descending dJ order from the independent audit.
    out_dir : Path | None
        Output directory.
    """
    out_dir = Path(out_dir) if out_dir else AUDIT_DIR
    label = _label(seed, target_port, wl_nm)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load audit ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"SEQUENTIAL REPLAY: {label}")
    print(f"{'=' * 60}")

    with open(audit_json) as f:
        audit = json.load(f)

    top_entries = [e for e in audit["entries"] if e["tag"].startswith("top")]
    if not top_entries:
        print("  No top-ranked entries in audit -- nothing to replay.")
        return None

    # Sort: original rank order or independent dJ order
    if rank_order:
        top_entries.sort(key=lambda e: e["rank"])
        order_label = "original formal rank"
    else:
        top_entries.sort(key=lambda e: e.get("dJ_WDM", float("-inf")),
                         reverse=True)
        order_label = "independent dJ (descending)"

    if accept_threshold is None:
        accept_threshold = audit.get("accept_threshold_calibrated", 1e-4)

    n_independent_accepted = sum(1 for e in top_entries if e.get("accepted"))
    print(f"  audit: {audit_json.name}")
    print(f"  independent: {n_independent_accepted}/{len(top_entries)} accepted")
    print(f"  accept threshold: dJ > {accept_threshold:.3e}")
    print(f"  replay order: {order_label}")

    # ── 2. Load params (match the independent audit's jac baseline) ──
    jac_file = audit.get("jac_file")
    if jac_file:
        jac = load_jac_3d(seed, target_port, wl_nm, RESULTS_DIR,
                          label_suffix=jac_file.replace(f"jac_{label}", "").replace(".npz", ""))
        if jac is not None:
            params = np.asarray(jac["params"], dtype=float).reshape(-1)
            print(f"  params from jac: {jac_file}  |a|^2={jac['baseline']['abs2']:.6f}")
        else:
            params = np.asarray(load_warmstart_params(seed), dtype=float).reshape(-1)
            print(f"  params from warmstart (jac load failed)")
    else:
        params = np.asarray(load_warmstart_params(seed), dtype=float).reshape(-1)
        print(f"  params from warmstart (no jac_file in audit)")

    # ── 3. Sequential replay ─────────────────────────────────────────
    t0 = time.time()
    kept = []
    undone = []

    with DeviceAuditSession(params, target_port, wl_nm,
                            tag=f"replay_{label}") as S:
        bl = S.read_baseline()
        J_bl = compute_j_wdm(bl["T_out"], bl["R_in"], target_port, lam_R, lam_X)
        J_current = J_bl
        print(f"  baseline J_WDM = {J_bl:.4e}")

        n_total = len(top_entries)
        for n, entry in enumerate(top_entries, start=1):
            idx = entry["idx"]
            pred_gain = entry.get("pred_gain")
            indep_accepted = entry.get("accepted", False)
            indep_dJ = entry.get("dJ_WDM", float("nan"))

            t_c = time.time()
            m = S.measure([idx])
            J_m = compute_j_wdm(m["T_out"], m["R_in"], target_port, lam_R, lam_X)
            dJ = J_m - J_current
            accepted = bool(dJ > accept_threshold)

            record = {
                "rank": entry["rank"],
                "idx": int(idx),
                "ix": idx // NY_2D,
                "iy": idx % NY_2D,
                "pred_gain": pred_gain,
                "indep_dJ": indep_dJ,
                "indep_accepted": indep_accepted,
                "replay_dJ": float(dJ),
                "replay_accepted": accepted,
                "J_after": float(J_m) if accepted else float(J_current),
                "wall_s": time.time() - t_c,
            }

            if accepted:
                S._toggle(idx)  # re-apply: measure() reverts by default
                kept.append(record)
                J_current = J_m
                tag = "KEPT"
            else:
                undone.append(record)
                tag = "undone"

            was = "[+]" if indep_accepted else "[-]"
            print(f"  [{n}/{n_total}] rank={entry['rank']} idx={idx} "
                  f"(ix={idx // NY_2D},iy={idx % NY_2D}) "
                  f"indep={indep_dJ:+.3e}{was}  replay dJ={dJ:+.3e}  "
                  f"-> {tag}  ({time.time() - t_c:.0f}s)")
            sys.stdout.flush()

        # final state
        final_params = S.get_params()

    replay_wall_s = time.time() - t0

    # ── 4. Fresh-sim verification ────────────────────────────────────
    print(f"\n  -- fresh-sim verification --")
    t_fs = time.time()

    sim_v = _build_3d_sim(tag=f"replay_fresh_{label}")
    try:
        _apply_params_to_design(sim_v.fdtd,
                                np.asarray(final_params, dtype=float).reshape(-1))
        sim_v.fdtd.run()

        # Read device metrics (same pattern as DeviceAuditSession)
        T_out_v = []
        for port_i in [1, 2, 3]:
            coeff = _read_expansion_coefficient_3d(
                sim_v.fdtd, f"P{port_i}_prof_me")
            T_out_v.append(coeff.get("T_forward", float("nan")))
        in_coeff_v = _read_expansion_coefficient_3d(
            sim_v.fdtd, "P_in_prof_me")
        R_in_v = abs(in_coeff_v.get("T_backward", float("nan")))
        abs2_coeff_v = _read_expansion_coefficient_3d(
            sim_v.fdtd, f"P{target_port}_prof_me")
        abs2_v = abs2_coeff_v.get("abs2", float("nan"))
    finally:
        _close_sim(sim_v)
        _sweep_stray_engines()

    J_v = compute_j_wdm(T_out_v, R_in_v, target_port, lam_R, lam_X)
    fresh_wall_s = time.time() - t_fs

    # ── 5. Summary ───────────────────────────────────────────────────
    dJ_final = J_v - J_bl
    dT_final = T_out_v[target_port - 1] - bl["T_out"][target_port - 1]
    dR_final = R_in_v - bl["R_in"]
    dAbs2_final = abs2_v - bl["abs2"]

    n_retained = sum(1 for r in kept if r.get("indep_accepted"))
    n_newly_activated = sum(1 for r in kept if not r.get("indep_accepted"))

    replay_summary = {
        "label": label,
        "timestamp": _now_iso(),
        "audit_source": str(audit_json.name),
        "seed": seed,
        "target_port": target_port,
        "wl_nm": wl_nm,
        "lam_R": lam_R,
        "lam_X": lam_X,
        "accept_threshold": accept_threshold,
        "replay_order": order_label,
        "n_independent_accepted": n_independent_accepted,
        "n_replayed": len(top_entries),
        "n_kept": len(kept),
        "n_retained": n_retained,
        "n_newly_activated": n_newly_activated,
        "n_undone": len(undone),
        "retention_rate": float(n_retained / n_independent_accepted)
                          if n_independent_accepted > 0 else 0.0,
        "baseline": {
            "abs2": bl["abs2"],
            "T_out": [float(v) for v in bl["T_out"]],
            "R_in": float(bl["R_in"]),
            "J_WDM": float(J_bl),
        },
        "replay_incremental": {
            "final_J": float(J_current),
            "dJ_vs_baseline": float(J_current - J_bl),
        },
        "fresh_sim": {
            "abs2": float(abs2_v),
            "T_out": [float(v) for v in T_out_v],
            "R_in": float(R_in_v),
            "J_WDM": float(J_v),
            "dJ_vs_baseline": float(dJ_final),
            "dT_target": float(dT_final),
            "dR_in": float(dR_final),
            "dAbs2": float(dAbs2_final),
        },
        "kept": kept,
        "undone": undone,
        "wall_s": {
            "replay": replay_wall_s,
            "fresh_sim": fresh_wall_s,
            "total": time.time() - t0,
        },
        "verdict": _replay_verdict(n_retained, n_independent_accepted,
                                   dJ_final, J_bl, J_v),
    }

    # ── 6. Output ────────────────────────────────────────────────────
    # reuse audit source name as prefix
    audit_stem = Path(audit_json).stem
    out_json = out_dir / f"replay_{audit_stem}.json"
    _save_json(replay_summary, out_json)

    print(f"\n{'=' * 60}")
    print("REPLAY SUMMARY")
    print(f"{'=' * 60}")
    print(f"  kept: {len(kept)} total "
          f"(retained {n_retained}/{n_independent_accepted} "
          f"= {n_retained / max(n_independent_accepted, 1):.0%}; "
          f"newly-activated {n_newly_activated})")
    if kept:
        print(f"  kept flips: {[(r['rank'], r['idx']) for r in kept]}")
    print(f"  incremental final: J = {J_current:.4e} "
          f"(Delta = {J_current - J_bl:+.3e})")
    print(f"  fresh-sim final:   J = {J_v:.4e} "
          f"(Delta = {dJ_final:+.3e})")
    print(f"  fresh-sim: DeltaT_target={dT_final:+.3e}  "
          f"DeltaR_in={dR_final:+.3e}  Delta|a|^2={dAbs2_final:+.3e}")
    print(f"  verdict: {replay_summary['verdict']}")
    print(f"  wall: {replay_summary['wall_s']['total']:.0f}s")
    print(f"  -> {out_json}")

    return replay_summary


def _replay_verdict(n_retained, n_indep, dJ_final, J_bl, J_v):
    """One-line replay verdict based on retention of independently-accepted flips."""
    retention = n_retained / max(n_indep, 1)
    if n_retained == 0:
        return (f"ALL UNDONE: epistasis canceled all {n_indep} independently-"
                f"accepted flips — single-pixel additivity fails.")
    elif retention >= 0.67:
        if dJ_final > 0:
            return (f"STRONG: {n_retained}/{n_indep} retained ({retention:.0%}), "
                    f"DeltaJ={dJ_final:+.3e} — independent hits survive epistasis, "
                    f"modest device improvement achievable.")
        else:
            return (f"ANOMALOUS: {n_retained}/{n_indep} retained but fresh-sim "
                    f"DeltaJ={dJ_final:+.3e} < 0 — incremental vs fresh mismatch.")
    else:
        return (f"PARTIAL: {n_retained}/{n_indep} retained ({retention:.0%}), "
                f"DeltaJ={dJ_final:+.3e} — epistasis partially cancels "
                f"independent hits.  Patch/batch action indicated.")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Single-pixel actionability audit (WDM3 3D)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--port", type=int, default=1)
    ap.add_argument("--wl", type=float, default=WL_NM)
    ap.add_argument("--k", type=int, default=20,
                    help="Number of top-ranked pixels to audit")
    ap.add_argument("--variant", default="conj_a",
                    choices=list(GRAD_VARIANTS))
    ap.add_argument("--mask", type=str, default="72-79",
                    help="Column mask, e.g. '72-79' or '72-79,85-90' "
                         "or 'none'")
    ap.add_argument("--null", type=int, default=25,
                    help="Number of random pixels for null calibration "
                         "(≥25 recommended for publication)")
    ap.add_argument("--lam-R", type=float, default=1.0,
                    help="Reflection penalty weight")
    ap.add_argument("--lam-X", type=float, default=0.5,
                    help="Crosstalk penalty weight")
    ap.add_argument("--accept", type=float, default=None,
                    help="User-specified dJ_WDM threshold (if omitted, "
                         "null-calibrated)")
    ap.add_argument("--jac-suffix", type=str, default=None,
                    help="Explicit jac generation, e.g. '_g1'")
    ap.add_argument("--replay", type=str, default=None,
                    metavar="AUDIT_JSON",
                    help="Sequential replay: load audit JSON, replay "
                         "top-K in rank order, final fresh-sim verify")
    ap.add_argument("--replay-by-dj", action="store_true",
                    help="Replay in descending independent-dJ order "
                         "(default: original formal rank order)")
    ap.add_argument("--bottom", action="store_true",
                    help="Audit bottom-K (least promising) pixels instead of top-K")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Output directory for audit JSON files")
    args = ap.parse_args()

    # ── Replay mode ──────────────────────────────────────────────────
    if args.replay:
        audit_path = Path(args.replay)
        if not audit_path.exists():
            # try relative to default output dir
            audit_path = AUDIT_DIR / args.replay
        if not audit_path.exists():
            raise FileNotFoundError(f"Audit JSON not found: {args.replay}")
        run_replay(
            audit_json=audit_path,
            seed=args.seed,
            target_port=args.port,
            wl_nm=args.wl,
            lam_R=args.lam_R,
            lam_X=args.lam_X,
            accept_threshold=args.accept,
            rank_order=not args.replay_by_dj,
        )
        sys.exit(0)

    # ── Audit mode ───────────────────────────────────────────────────
    # Parse mask
    mask_cols = None
    if args.mask and args.mask.lower() != "none":
        mask_cols = []
        for part in args.mask.split(","):
            lo, hi = part.strip().split("-")
            mask_cols.append((int(lo), int(hi)))

    run_audit(
        seed=args.seed,
        target_port=args.port,
        wl_nm=args.wl,
        k=args.k,
        variant=args.variant,
        mask_cols=mask_cols,
        null_draws=args.null,
        lam_R=args.lam_R,
        lam_X=args.lam_X,
        accept_threshold_user=args.accept,
        jac_suffix=args.jac_suffix,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        bottom=args.bottom,
    )
