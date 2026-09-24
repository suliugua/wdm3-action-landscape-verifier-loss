#!/usr/bin/env python
"""3D single-port adjoint jacobian + short_push driver (post-T3D-4 line).

Route: manual double-run in ONE session — forward run, then switchtolayout,
disable forward source, enable a manually-built monochromatic backward mode
source at the target port, run again.  This bypasses lumopt's Optimization
framework entirely (opt.make_forward_sim + runjobs hangs in 3D even after the
broadband-source fix — see work record 2026-07-14).

FOM = |a|^2 at the target port (ModeMatch adjoint targets |a|^2; 2D sign
convention locked: abs2 ascent_sign = +1).

Gradient variants (all zero extra sims — global phase rotations of the same
complex per-pixel overlap O):
  raw       Re(O)                 — matched FD(|a|^2) sign in T3D-4 (scalar level)
  i         Re(1j*O)              — 2D lesson: ModeMatch scaling carries a 1j factor
  conj_a    Re(conj(a)*O)         — proper |a|^2 weighting up to source-norm phase
  i_conj_a  Re(1j*conj(a)*O)

Tasks:
  jac        compute per-pixel gradient (2 sims, one session), save .npz + .json
  fdcheck    discrete flip-response validation (2 sims): under thresholded
             extrusion eps is a FLIP BUDGET (pick via offline flip counts, both
             sides nonzero+comparable); judged on Δ+>0 and Δ+>Δ-, not d_fd
  push       discrete top-K flip push: flip the K best pixels by signed
             flip-gain under the chosen phase variant, K ladder 5/10/20
  randbase   matched random-flip baseline: does adjoint ranking beat flipping
             the same NUMBER of random pixels?

Usage:
  python -m wdm3_3d.wdm3_3d_jac --task jac      --seed 7 --port 1
  python -m wdm3_3d.wdm3_3d_jac --task fdcheck  --seed 7 --port 1 --eps 2e-2 --variant i_conj_a
  python -m wdm3_3d.wdm3_3d_jac --task push     --seed 7 --port 1 --ks 5,10,20
  python -m wdm3_3d.wdm3_3d_jac --task randbase --seed 7 --port 1 --k 5 --draws 3
"""

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Force 2026 R1 backend before any lumopt imports (3D requires 2026) ──
os.environ["LUMOPT_BACKEND"] = "2026"

import numpy as np

from .wdm3_3d_device import (
    DESIGN_X_NM,
    DESIGN_Y_NM,
    NX_2D,
    NY_2D,
    WL_NM,
    _build_3d_sim,
    _close_sim,
    _m,
    load_warmstart_params,
)
from .wdm3_3d_audit import read_formal_target_3d
from .wdm3_3d_smoke import _apply_params_to_design

RESULTS_DIR = Path(__file__).resolve().parent / "results_3d" / "jac3d"

GRAD_VARIANTS = ("raw", "i", "conj_a", "i_conj_a")


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  → saved {path}")


def _sweep_stray_engines():
    """Best-effort kill of stray FDTD engine workers.

    _close_sim's taskkill on the CAD process tree does not reliably reach the
    MPI solver workers — they accumulate (15+ observed on 2026-07-14).
    Safe here because runs are strictly sequential on this machine.
    """
    try:
        subprocess.run(["taskkill", "/F", "/T", "/IM", "fdtd-engine*"],
                       capture_output=True, timeout=15)
    except Exception:
        pass


def _label(seed, target_port, wl_nm):
    return f"seed{seed}_P{target_port}@{int(wl_nm)}nm"


# ═══════════════════════════════════════════════════════════════════════════
# Field readout + adjoint source
# ═══════════════════════════════════════════════════════════════════════════

def _read_opt_fields(fdtd):
    """Read E from the 3D opt_fields monitor.

    Returns
    -------
    E : np.ndarray complex, shape (nx, ny, nz, 3)
    x, y, z : np.ndarray, monitor axes in meters
    """
    data = fdtd.getresult("opt_fields", "E")
    E = np.asarray(data["E"])
    if E.ndim == 5:  # (nx, ny, nz, nf, 3) with nf=1
        E = E[:, :, :, 0, :]
    x = np.asarray(data["x"]).flatten()
    y = np.asarray(data["y"]).flatten()
    z = np.asarray(data["z"]).flatten()
    if E.shape[0] != len(x) or E.shape[1] != len(y):
        raise RuntimeError(
            f"opt_fields axis mismatch: E={E.shape} vs x={len(x)}, y={len(y)}"
        )
    return E, x, y, z


def _add_adjoint_source(fdtd, target_port, wl_nm, name="adj_src_manual"):
    """Manual monochromatic backward mode source at the target port monitor.

    Mirrors the 2026-07-14 broadband fix: monochromatic wl_nm, fundamental TE,
    Backward injection (toward -x, into the design region), geometry copied
    from P{target_port}_prof.
    """
    mon = f"P{target_port}_prof"
    fdtd.addmode()
    fdtd.set("name", name)
    fdtd.set("injection axis", "x-axis")
    fdtd.set("direction", "Backward")
    for prop in ["x", "y", "y span", "z", "z span"]:
        fdtd.setnamed(name, prop, fdtd.getnamed(mon, prop))
    fdtd.setnamed(name, "wavelength start", _m(wl_nm))
    fdtd.setnamed(name, "wavelength stop", _m(wl_nm))
    fdtd.setnamed(name, "mode selection", "fundamental TE mode")
    return name


def _overlap_to_design_grid(O_xy, x, y):
    """Interpolate the complex per-cell overlap O(x,y) onto 80x100 pixel centers."""
    from scipy.interpolate import RegularGridInterpolator

    dx = (DESIGN_X_NM[1] - DESIGN_X_NM[0]) / NX_2D
    dy = (DESIGN_Y_NM[1] - DESIGN_Y_NM[0]) / NY_2D
    xc = _m(DESIGN_X_NM[0] + (np.arange(NX_2D) + 0.5) * dx)
    yc = _m(DESIGN_Y_NM[0] + (np.arange(NY_2D) + 0.5) * dy)
    Xg, Yg = np.meshgrid(xc, yc, indexing="ij")

    interp = RegularGridInterpolator((x, y), O_xy, bounds_error=False,
                                     fill_value=0.0)
    O_pix = interp(np.column_stack([Xg.ravel(), Yg.ravel()]))
    return O_pix.reshape(NX_2D, NY_2D)


def _grad_variants(O_pix, a):
    """The four global-phase gradient variants, each flattened to (n_dims,)."""
    return {
        "raw": np.real(O_pix).flatten(),
        "i": np.real(1j * O_pix).flatten(),
        "conj_a": np.real(np.conj(a) * O_pix).flatten(),
        "i_conj_a": np.real(1j * np.conj(a) * O_pix).flatten(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# FOM evaluation (fresh sim per eval, T3D-3 pattern)
# ═══════════════════════════════════════════════════════════════════════════

def eval_fom_3d(params, target_port, wl_nm=WL_NM, tag="fom3d"):
    """Fresh-sim |a|^2 evaluation at the target port."""
    sim = _build_3d_sim(tag=tag)
    try:
        _apply_params_to_design(sim.fdtd, params)
        sim.fdtd.run()
        ft = read_formal_target_3d(sim.fdtd, target_port)
    finally:
        _close_sim(sim)
        _sweep_stray_engines()
    return ft


# ═══════════════════════════════════════════════════════════════════════════
# Task: jac — manual double-run adjoint gradient
# ═══════════════════════════════════════════════════════════════════════════

def compute_jac_3d(seed, target_port, wl_nm=WL_NM, out_dir=None, params=None,
                   label_suffix=""):
    """Per-pixel adjoint gradient via manual forward+adjoint double-run.

    One session, 2 sims.  Geometry is identical between runs (only sources
    toggle), so opt_fields grids match exactly — no cross-grid interpolation.

    *params*: explicit parameter vector (greedy rounds re-jac at the updated
    structure); defaults to the seed's warmstart.  *label_suffix* keeps the
    per-round .npz/.json files apart (e.g. "_g3" for greedy round 3).

    Returns dict; also saves <label>.npz (O_pix, grads, params, a) and .json.
    """
    out_dir = Path(out_dir) if out_dir else RESULTS_DIR
    label = _label(seed, target_port, wl_nm) + label_suffix

    print(f"\n{'='*60}")
    print(f"JAC3D: {label}")
    print(f"{'='*60}")

    if params is None:
        params = load_warmstart_params(seed)
    params = np.asarray(params, dtype=float).reshape(-1)
    t0 = time.time()

    sim = _build_3d_sim(tag=f"jac3d_{label}", wl_nm=wl_nm)
    try:
        _apply_params_to_design(sim.fdtd, params)

        # ── Forward run ──
        print("  [1/2] forward run ...", flush=True)
        t_f = time.time()
        sim.fdtd.run()
        ft = read_formal_target_3d(sim.fdtd, target_port)
        a = complex(ft["Re_a"], ft["Im_a"])
        E_fwd, x, y, z = _read_opt_fields(sim.fdtd)
        wall_fwd = time.time() - t_f
        print(f"    |a|^2={ft['abs2']:.6f}  Re(a)={ft['Re_a']:.6f}  "
              f"E_fwd max={np.max(np.abs(E_fwd)):.3f}  wall={wall_fwd:.0f}s")

        # ── Adjoint run (manual source switch) ──
        print("  [2/2] adjoint run ...", flush=True)
        t_a = time.time()
        sim.fdtd.switchtolayout()
        adj_name = _add_adjoint_source(sim.fdtd, target_port, wl_nm)
        sim.fdtd.setnamed("source", "enabled", False)
        sim.fdtd.setnamed(adj_name, "enabled", True)
        sim.fdtd.run()
        E_adj, x2, y2, z2 = _read_opt_fields(sim.fdtd)
        wall_adj = time.time() - t_a
        print(f"    E_adj max={np.max(np.abs(E_adj)):.3f}  wall={wall_adj:.0f}s")

        if E_adj.shape != E_fwd.shape:
            raise RuntimeError(
                f"fwd/adj grid mismatch: {E_fwd.shape} vs {E_adj.shape}"
            )
    finally:
        _close_sim(sim)
        _sweep_stray_engines()

    # ── Per-pixel complex overlap: sum components + z ──
    O3 = np.sum(E_fwd * E_adj, axis=-1)   # (nx, ny, nz)
    O_xy = np.sum(O3, axis=2)             # (nx, ny) complex
    O_pix = _overlap_to_design_grid(O_xy, x, y)  # (NX_2D, NY_2D) complex

    grads = _grad_variants(O_pix, a)

    # ── Diagnostics: norms + pairwise cosines ──
    norms = {k: float(np.linalg.norm(v)) for k, v in grads.items()}
    cosines = {}
    keys = list(GRAD_VARIANTS)
    for i1 in range(len(keys)):
        for i2 in range(i1 + 1, len(keys)):
            k1, k2 = keys[i1], keys[i2]
            d = norms[k1] * norms[k2]
            cosines[f"{k1}|{k2}"] = (
                float(np.dot(grads[k1], grads[k2]) / d) if d > 0 else float("nan")
            )

    wall_s = time.time() - t0

    result = {
        "label": label,
        "seed": seed,
        "target_port": target_port,
        "wl_nm": wl_nm,
        "timestamp": _now_iso(),
        "baseline": {"abs2": ft["abs2"], "Re_a": ft["Re_a"], "Im_a": ft["Im_a"],
                     "T_forward": ft["T_forward"]},
        "E_fwd_shape": list(E_fwd.shape),
        "E_fwd_max": float(np.max(np.abs(E_fwd))),
        "E_adj_max": float(np.max(np.abs(E_adj))),
        "overlap_total": float(np.real(np.sum(O_pix))),
        "overlap_abs_sum": float(np.sum(np.abs(O_pix))),
        "grad_norms": norms,
        "grad_cosines": cosines,
        "wall_fwd_s": wall_fwd,
        "wall_adj_s": wall_adj,
        "wall_s": wall_s,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"jac_{label}.npz"
    np.savez(npz_path, O_pix=O_pix, a=np.complex128(a), params=params,
             baseline_abs2=ft["abs2"],
             **{f"grad_{k}": v for k, v in grads.items()})
    print(f"  → saved {npz_path}")
    _save_json(result, out_dir / f"jac_{label}.json")

    print(f"\n  wall: {wall_s:.0f}s (fwd {wall_fwd:.0f}s + adj {wall_adj:.0f}s)")
    print(f"  grad norms: " + "  ".join(f"{k}={v:.3e}" for k, v in norms.items()))
    for k, v in cosines.items():
        print(f"  cos({k}) = {v:+.4f}")

    result["grads"] = grads
    result["params"] = params
    result["O_pix"] = O_pix
    result["a"] = a
    return result


def load_jac_3d(seed, target_port, wl_nm=WL_NM, out_dir=None, label_suffix=""):
    """Load a previously saved jac .npz, or None if absent."""
    out_dir = Path(out_dir) if out_dir else RESULTS_DIR
    npz_path = out_dir / f"jac_{_label(seed, target_port, wl_nm)}{label_suffix}.npz"
    if not npz_path.exists():
        return None
    d = np.load(npz_path)
    return {
        "grads": {k: d[f"grad_{k}"] for k in GRAD_VARIANTS},
        "O_pix": d["O_pix"],
        "params": d["params"],
        "a": complex(d["a"]),
        "baseline": {"abs2": float(d["baseline_abs2"])},
        "npz_path": str(npz_path),
    }


def _get_jac(seed, target_port, wl_nm, out_dir=None, reuse=True, params=None,
             label_suffix=""):
    if reuse:
        jac = load_jac_3d(seed, target_port, wl_nm, out_dir, label_suffix)
        if jac is not None:
            print(f"  [reuse] loaded saved jac: {jac['npz_path']}")
            return jac
    return compute_jac_3d(seed, target_port, wl_nm, out_dir, params=params,
                          label_suffix=label_suffix)


# ═══════════════════════════════════════════════════════════════════════════
# Task: fdcheck — directional FD along normalized grad (ascent gate)
# ═══════════════════════════════════════════════════════════════════════════

def fd_direction_check(seed, target_port, wl_nm=WL_NM, eps=1e-2,
                       variant="raw", out_dir=None, reuse=True):
    """FD of |a|^2 along the normalized *variant* gradient (2 sims).

    Gate: d_fd > 0 → the +grad direction is a true ascent direction for |a|^2
    → short_push is unlocked.  Also records every variant's predicted
    directional derivative along the tested direction for cross-diagnostics.
    """
    out_dir = Path(out_dir) if out_dir else RESULTS_DIR
    label = _label(seed, target_port, wl_nm)

    jac = _get_jac(seed, target_port, wl_nm, out_dir, reuse)
    grads = jac["grads"]
    params = np.asarray(jac["params"], dtype=float).reshape(-1)
    baseline = jac["baseline"]["abs2"]

    g = np.asarray(grads[variant], dtype=float).reshape(-1)
    gnorm = np.linalg.norm(g)
    if gnorm < 1e-30:
        raise RuntimeError(f"variant '{variant}' gradient is ~0 (norm={gnorm:.3e})")
    u = g / gnorm

    print(f"\n{'='*60}")
    print(f"FDCHECK: {label}  variant={variant}  eps={eps:.1e}")
    print(f"{'='*60}")
    print(f"  baseline |a|^2 = {baseline:.6f}")

    t0 = time.time()
    base_mask = params > 0.5

    p_plus = np.clip(params + eps * u, 0, 1)
    nf_p = int(np.sum((p_plus > 0.5) != base_mask))
    print(f"  f(+eps) (pixels flipped: {nf_p}) ...", flush=True)
    ft_p = eval_fom_3d(p_plus, target_port, wl_nm, tag=f"fdchk_{label}_p")
    f_plus = ft_p["abs2"]
    print(f"    |a|^2 = {f_plus:.6f}  (D={f_plus - baseline:+.4e})")

    p_minus = np.clip(params - eps * u, 0, 1)
    nf_m = int(np.sum((p_minus > 0.5) != base_mask))
    print(f"  f(-eps) (pixels flipped: {nf_m}) ...", flush=True)
    ft_m = eval_fom_3d(p_minus, target_port, wl_nm, tag=f"fdchk_{label}_m")
    f_minus = ft_m["abs2"]
    print(f"    |a|^2 = {f_minus:.6f}  (D={f_minus - baseline:+.4e})")

    d_fd = (f_plus - f_minus) / (2.0 * eps)
    delta_plus = f_plus - baseline
    delta_minus = f_minus - baseline

    # Every variant's prediction along the tested direction u
    predictions = {k: float(np.dot(np.asarray(v).reshape(-1), u))
                   for k, v in grads.items()}

    # ── Discrete flip-response criteria (2026-07-15 reframing) ──
    # Under thresholded extrusion eps is a FLIP BUDGET, not a physical step:
    # the signal is f(flip selected pixels) - f(x), not a directional
    # derivative.  d_fd is kept for reference only — its denominator is
    # ceremonial.  Judge on one-sided improvement + sign separation.
    criteria = {
        "one_sided_improvement": delta_plus > 0,        # Δ+ > 0
        "sign_separation": delta_plus > delta_minus,    # Δ+ > Δ-
        "flips_both_sides": nf_p > 0 and nf_m > 0,      # else phase verdict weak
    }
    passes = criteria["one_sided_improvement"] and criteria["sign_separation"]
    wall_s = time.time() - t0

    result = {
        "label": label,
        "seed": seed,
        "target_port": target_port,
        "wl_nm": wl_nm,
        "variant": variant,
        "eps": eps,
        "timestamp": _now_iso(),
        "passes": passes,
        "criteria": criteria,
        "baseline_abs2": baseline,
        "f_plus": f_plus,
        "f_minus": f_minus,
        "n_flipped_plus": nf_p,
        "n_flipped_minus": nf_m,
        "delta_plus": delta_plus,
        "delta_minus": delta_minus,
        "d_fd": d_fd,
        "variant_predictions_along_u": predictions,
        "wall_s": wall_s,
    }
    _save_json(result, out_dir / f"fdcheck_{label}_{variant}.json")

    print(f"\n  d_fd = {d_fd:+.6e}   {'PASS (ascent confirmed)' if passes else 'FAIL'}")
    for k, v in criteria.items():
        print(f"    {k}: {'OK' if v else 'NO'}")
    for k, v in predictions.items():
        agree = "agree" if (v * d_fd) > 0 else "DISAGREE"
        print(f"    pred[{k}] along u = {v:+.3e}  ({agree} with d_fd)")
    print(f"  wall: {wall_s:.0f}s")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Discrete flip machinery (2026-07-15 reframing: thresholded extrusion makes
# this a discrete pixel-flip problem, not a continuous line search)
# ═══════════════════════════════════════════════════════════════════════════

def _flip_gain(grad, params):
    """Predicted dF from flipping each pixel toward ascent.

    Flipping Si→SiO2 gives ΔF ≈ -g_i, SiO2→Si gives ΔF ≈ +g_i.
    """
    base_mask = np.asarray(params).reshape(-1) > 0.5
    return np.asarray(grad).reshape(-1) * (1.0 - 2.0 * base_mask)


def _apply_flips(params, flip_idx):
    """Return params with the given pixels flipped hard across the threshold."""
    p = np.asarray(params, dtype=float).reshape(-1).copy()
    base_mask = p > 0.5
    p[flip_idx] = np.where(base_mask[flip_idx], 0.0, 1.0)
    return p


def _isolation_report(params, flip_idx):
    """4-neighbor isolation check for flipped pixels (2D design grid).

    A flipped pixel is 'isolated' if none of its 4-neighbors shares its new
    material — flags floating Si dots / pinholes introduced by the push.
    """
    mask = (np.asarray(params).reshape(NX_2D, NY_2D) > 0.5)
    isolated = []
    for idx in np.asarray(flip_idx).reshape(-1):
        ix, iy = int(idx) // NY_2D, int(idx) % NY_2D
        val = mask[ix, iy]
        neigh = []
        if ix > 0:
            neigh.append(mask[ix - 1, iy])
        if ix < NX_2D - 1:
            neigh.append(mask[ix + 1, iy])
        if iy > 0:
            neigh.append(mask[ix, iy - 1])
        if iy < NY_2D - 1:
            neigh.append(mask[ix, iy + 1])
        if not any(n == val for n in neigh):
            isolated.append(int(idx))
    return isolated


# ═══════════════════════════════════════════════════════════════════════════
# Task: push — discrete top-K flip push (replaces continuous short_push)
# ═══════════════════════════════════════════════════════════════════════════

def discrete_push_3d(seed, target_port, wl_nm=WL_NM, ks=(5, 10, 20),
                     variant="i_conj_a", exclude_ix=None, select="top",
                     out_dir=None, reuse=True):
    """Flip the top-K pixels ranked by signed flip-gain under *variant*.

    For each K: flip the K highest positive-gain pixels hard across the
    threshold, evaluate |a|^2 (fresh sim), record ΔF + isolation diagnostics.

    *exclude_ix*: optional (ix_min, ix_max) column range to mask out of the
    ranking.  2026-07-16 correction: pollution is ONE-SIDED — the monitor/
    adjoint-injection near-field band (ix>=72) is where probes showed the
    per-pixel signal is coin-flip; the left near-source region measured
    clean (ix=5,6 strongest genuine single flips).  Use the rule-based mask
    from wdm3_3d_landscape.diagnose_anchor instead of hand-picked ranges.

    *select*: 'top' (default) flips the most-positive-gain pixels, expects
    improvement; 'bottom' flips the most-NEGATIVE-gain pixels, expects
    degradation — the reverse control proving direction matters (a PASS for
    'bottom' means every K degraded).

    PASS: ΔF > 0 ('top') / ΔF < 0 ('bottom') at every K.  Monotonicity in K
    is reported but not gated (linear prediction degrades as K grows).
    """
    out_dir = Path(out_dir) if out_dir else RESULTS_DIR
    label = _label(seed, target_port, wl_nm)

    jac = _get_jac(seed, target_port, wl_nm, out_dir, reuse)
    params = np.asarray(jac["params"], dtype=float).reshape(-1)
    baseline = jac["baseline"]["abs2"]
    gain = _flip_gain(jac["grads"][variant], params)
    excluded = np.zeros(len(params), dtype=bool)
    mask_tag = ""
    if exclude_ix is not None:
        ranges = [exclude_ix] if isinstance(exclude_ix[0], (int, float)) \
            else list(exclude_ix)
        ix_of = np.arange(len(params)) // NY_2D
        for lo, hi in ranges:
            excluded |= (ix_of >= lo) & (ix_of <= hi)
        mask_tag = "_mask" + "+".join(f"{lo}-{hi}" for lo, hi in ranges)
    if select == "bottom":
        sel_gain = np.where(excluded, np.inf, gain)
        order = np.argsort(sel_gain)               # most negative first
        n_pool = int(np.sum(sel_gain < 0))
        sel_tag, expect_improve = "_bottom", False
    else:
        sel_gain = np.where(excluded, -np.inf, gain)
        order = np.argsort(sel_gain)[::-1]         # most positive first
        n_pool = int(np.sum(sel_gain > 0))
        sel_tag, expect_improve = "", True
    n_positive = n_pool

    print(f"\n{'='*60}")
    print(f"DISCRETE PUSH 3D: {label}  variant={variant}  ks={list(ks)}"
          + (f"  exclude_ix={exclude_ix}" if exclude_ix else "")
          + (f"  select={select}" if select != "top" else ""))
    print(f"{'='*60}")
    print(f"  baseline |a|^2 = {baseline:.6f}  pool ({select}-gain pixels): {n_pool}")

    t0 = time.time()
    entries = []
    for k in ks:
        k_eff = min(int(k), n_pool)
        flip_idx = order[:k_eff]
        p_new = _apply_flips(params, flip_idx)
        isolated = _isolation_report(p_new, flip_idx)
        pred = float(np.sum(gain[flip_idx]))
        print(f"  K={k} (effective {k_eff}, predicted D={pred:+.3e}, "
              f"isolated after flip: {len(isolated)}) ...", flush=True)
        ft = eval_fom_3d(p_new, target_port, wl_nm,
                         tag=f"push3d_{label}{sel_tag}_k{k}")
        fom = ft["abs2"]
        delta = fom - baseline
        entries.append({
            "K": int(k), "K_effective": k_eff,
            "flip_idx": [int(i) for i in flip_idx],
            "predicted_delta": pred,
            "fom": float(fom), "delta_fom": float(delta),
            "improved": bool(delta > 0),
            "as_expected": bool((delta > 0) == expect_improve),
            "T_forward": ft["T_forward"],
            "isolated_pixels": isolated,
        })
        print(f"    |a|^2={fom:.6f}  D={delta:+.4e}  "
              f"{'improved' if delta > 0 else 'degraded'}  "
              f"T_fwd={ft['T_forward']:.6f}")

    n_improved = sum(1 for e in entries if e["improved"])
    n_expected = sum(1 for e in entries if e["as_expected"])
    deltas = [e["delta_fom"] for e in entries]
    monotonic = all(deltas[i] <= deltas[i + 1] for i in range(len(deltas) - 1))

    crit_name = "all_K_improve" if expect_improve else "all_K_degrade"
    criteria = {crit_name: n_expected == len(entries)}
    passes = all(criteria.values())
    if not monotonic:
        print("  [WARN] dF not monotonic in K — expected as linear prediction degrades")

    wall_s = time.time() - t0
    result = {
        "label": label, "seed": seed, "target_port": target_port,
        "wl_nm": wl_nm, "variant": variant, "objective": "abs2",
        "mode": "discrete_topK_flip",
        "select": select,
        "exclude_ix": list(exclude_ix) if exclude_ix else None,
        "timestamp": _now_iso(),
        "passes": passes, "criteria": criteria,
        "baseline_fom": baseline,
        "n_pool": n_pool,
        "entries": entries,
        "n_improved": n_improved,
        "n_as_expected": n_expected,
        "monotonic_in_K": monotonic,
        "wall_s": wall_s,
    }
    _save_json(result, out_dir / f"push_{label}_{variant}{mask_tag}{sel_tag}.json")

    print(f"\n  as-expected: {n_expected}/{len(entries)}  "
          f"push[{select}]: {'PASS' if passes else 'CHECK'}")
    print(f"  wall: {wall_s:.0f}s")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Task: linearity — small-contrast probe in the adjoint's own (linear) regime
# ═══════════════════════════════════════════════════════════════════════════

def linearity_probe(seed, target_port, wl_nm=WL_NM, variant="i_conj_a",
                    n_each=5, probe_index=2.0, out_dir=None, reuse=True):
    """Test the adjoint derivative with SMALL Δε probes (not full flips).

    Retrospective finding (2026-07-15): T3D-3/4 perturbed continuous params
    through the 0.5 threshold, so every prior 3D 'FD validation' was actually
    a full-contrast flip response (Δε≈10) — the adjoint was never tested in
    its own linear regime.  This probe places an n=*probe_index* rect
    (Δε = n²-1.44² ≈ 1.9 for n=2.0) at interior SiO2 pixels — the *n_each*
    most-positive-g and most-negative-g ones with all-SiO2 4-neighborhoods
    (avoiding boundary D-field corrections).

    H1 (chain healthy, flips just nonlinear): sign-match ≈ 1.0 here.
    H2 (real bug in z-weights / material sign / alignment): still random.
    """
    from .wdm3_3d_device import SI_THICK_NM

    out_dir = Path(out_dir) if out_dir else RESULTS_DIR
    label = _label(seed, target_port, wl_nm)

    jac = _get_jac(seed, target_port, wl_nm, out_dir, reuse)
    params = np.asarray(jac["params"], dtype=float).reshape(-1)
    g = np.asarray(jac["grads"][variant], dtype=float).reshape(-1)

    # Interior SiO2 pixels: pixel and all 4-neighbors below threshold
    mask2d = (params.reshape(NX_2D, NY_2D) > 0.5)
    interior_sio2 = np.zeros((NX_2D, NY_2D), dtype=bool)
    interior_sio2[1:-1, 1:-1] = (
        ~mask2d[1:-1, 1:-1] & ~mask2d[:-2, 1:-1] & ~mask2d[2:, 1:-1]
        & ~mask2d[1:-1, :-2] & ~mask2d[1:-1, 2:]
    )
    pool = np.where(interior_sio2.flatten())[0]
    if len(pool) < 2 * n_each:
        raise RuntimeError(f"only {len(pool)} interior SiO2 pixels available")

    pool_sorted = pool[np.argsort(g[pool])]
    neg_idx = pool_sorted[:n_each]           # most negative g → expect ΔF < 0
    pos_idx = pool_sorted[-n_each:][::-1]    # most positive g → expect ΔF > 0
    candidates = ([{"group": "pos", "idx": int(i)} for i in pos_idx]
                  + [{"group": "neg", "idx": int(i)} for i in neg_idx])

    eps_bg = 1.44 ** 2
    d_eps = probe_index ** 2 - eps_bg

    print(f"\n{'='*60}")
    print(f"LINEARITY PROBE: {label}  variant={variant}  "
          f"n={probe_index} (Δε={d_eps:.2f})  pixels: {n_each}+{n_each}")
    print(f"{'='*60}")

    dx = (DESIGN_X_NM[1] - DESIGN_X_NM[0]) / NX_2D
    dy = (DESIGN_Y_NM[1] - DESIGN_Y_NM[0]) / NY_2D
    probe_name = "lin_probe"

    t0 = time.time()
    sim = _build_3d_sim(tag=f"lin3d_{label}")
    entries = []
    try:
        _apply_params_to_design(sim.fdtd, params)
        print("  baseline run ...", flush=True)
        sim.fdtd.run()
        f0 = read_formal_target_3d(sim.fdtd, target_port)["abs2"]
        print(f"    baseline |a|^2 = {f0:.6f}", flush=True)

        for ci, cand in enumerate(candidates):
            idx = cand["idx"]
            ix, iy = divmod(idx, NY_2D)
            xc = DESIGN_X_NM[0] + (ix + 0.5) * dx
            yc = DESIGN_Y_NM[0] + (iy + 0.5) * dy
            sim.fdtd.switchtolayout()
            sim.fdtd.addrect()
            sim.fdtd.set("name", probe_name)
            sim.fdtd.set("x", _m(xc))
            sim.fdtd.set("x span", _m(dx))
            sim.fdtd.set("y", _m(yc))
            sim.fdtd.set("y span", _m(dy))
            sim.fdtd.set("z", _m(SI_THICK_NM / 2))
            sim.fdtd.set("z span", _m(SI_THICK_NM))
            sim.fdtd.set("material", "<Object defined dielectric>")
            sim.fdtd.set("index", float(probe_index))
            t_r = time.time()
            sim.fdtd.run()
            fom = read_formal_target_3d(sim.fdtd, target_port)["abs2"]
            dF = fom - f0
            g_i = float(g[idx])
            expect_pos = cand["group"] == "pos"
            match = (dF > 0) == expect_pos
            entries.append({
                **cand, "g": g_i, "p": float(params[idx]),
                "fom": float(fom), "delta_fom": float(dF),
                "sign_match": bool(match),
                "wall_s": time.time() - t_r,
            })
            print(f"  [{ci + 1}/{len(candidates)}] {cand['group']} idx={idx} "
                  f"g={g_i:+.4f} -> DF={dF:+.4e} "
                  f"{'MATCH' if match else 'MISS'} ({entries[-1]['wall_s']:.0f}s)",
                  flush=True)
            sim.fdtd.switchtolayout()
            sim.fdtd.select(probe_name)
            sim.fdtd.delete()
    finally:
        _close_sim(sim)
        _sweep_stray_engines()

    n_match = sum(1 for e in entries if e["sign_match"])
    # α scale estimate from matching probes: dF ≈ α · g · (Δε/Δε_full)?  Report
    # the raw ratio dF/g per probe instead — constant ratio = linear regime.
    ratios = [e["delta_fom"] / e["g"] for e in entries if abs(e["g"]) > 1e-12]

    wall_s = time.time() - t0
    result = {
        "label": label, "seed": seed, "target_port": target_port,
        "wl_nm": wl_nm, "variant": variant,
        "probe_index": probe_index, "d_eps": d_eps,
        "timestamp": _now_iso(),
        "baseline_abs2": f0,
        "entries": entries,
        "sign_match": f"{n_match}/{len(entries)}",
        "sign_match_rate": n_match / len(entries),
        "dF_over_g": {"mean": float(np.mean(ratios)),
                      "std": float(np.std(ratios)),
                      "values": [float(r) for r in ratios]},
        "wall_s": wall_s,
    }
    _save_json(result, out_dir / f"linearity_{label}_{variant}_n{probe_index}.json")

    print(f"\n  sign-match: {n_match}/{len(entries)}")
    print(f"  dF/g ratio: mean={result['dF_over_g']['mean']:+.3e} "
          f"std={result['dF_over_g']['std']:.3e} (constant => linear)")
    print(f"  wall: {wall_s:.0f}s")
    return result




# ═══════════════════════════════════════════════════════════════════════════
# Task: precision — single-pixel flip FD of ranked candidates (FAIL branch #1)
# ═══════════════════════════════════════════════════════════════════════════

def _flip_single_pixel(fdtd, idx, to_si):
    """Toggle pixel *idx* Si↔SiO2.

    Uses Python API addrect for to_si=True (always safe).
    Uses Lumerical eval for to_si=False (Python select+set is unreliable
    on eval-created rects).
    """
    from .wdm3_3d_device import SI_THICK_NM, MAT_SI, MAT_SIO2

    ix, iy = divmod(int(idx), NY_2D)
    name = f"ws_{ix}_{iy}"
    fdtd.switchtolayout()

    if to_si:
        dx = (DESIGN_X_NM[1] - DESIGN_X_NM[0]) / NX_2D
        dy = (DESIGN_Y_NM[1] - DESIGN_Y_NM[0]) / NY_2D
        xc = DESIGN_X_NM[0] + (ix + 0.5) * dx
        yc = DESIGN_Y_NM[0] + (iy + 0.5) * dy
        fdtd.addrect()
        fdtd.set("name", name)
        fdtd.set("x", _m(xc)); fdtd.set("x span", _m(dx))
        fdtd.set("y", _m(yc)); fdtd.set("y span", _m(dy))
        fdtd.set("z", _m(SI_THICK_NM / 2)); fdtd.set("z span", _m(SI_THICK_NM))
        fdtd.set("material", MAT_SI)
    else:
        try:
            fdtd.eval(f"select('{name}'); set('material','{MAT_SIO2}');")
        except Exception:
            pass  # rect doesn't exist = already SiO2, nothing to do


def precision_probe(seed, target_port, wl_nm=WL_NM, variant="i_conj_a",
                    n_top=10, mid_ranks=(400, 500, 600), n_rand=3,
                    rng_seed=99, pixels=None, out_dir=None, reuse=True):
    """Single-pixel flip ΔF for ranked candidates — precision@K of the ranking.

    Probes: top-*n_top* by signed flip-gain, a few mid-rank controls (default
    ranks 400/500/600 — the regime the fdcheck trio came from), and *n_rand*
    random pixels.  One session, incremental single-rect add/delete per probe
    (undone after each measurement), so each probe costs one solver run.

    If *pixels* is given (list of flat indices), probe exactly those pixels
    instead (group='custom') — used to full-flip-test the linearity-probe
    pixel set.

    Outputs precision@K (fraction of top-K with measured ΔF > 0), measured
    vs predicted gain per pixel, and Spearman rank correlation.
    """
    out_dir = Path(out_dir) if out_dir else RESULTS_DIR
    label = _label(seed, target_port, wl_nm)

    jac = _get_jac(seed, target_port, wl_nm, out_dir, reuse)
    params = np.asarray(jac["params"], dtype=float).reshape(-1)
    gain = _flip_gain(jac["grads"][variant], params)
    order = np.argsort(gain)[::-1]
    rank_of = np.empty(len(params), dtype=int)
    rank_of[order] = np.arange(len(params))

    if pixels is not None:
        candidates = [{"group": "custom", "rank": int(rank_of[int(i)]),
                       "idx": int(i)} for i in pixels]
        suffix = "_custom"
    else:
        rng = np.random.RandomState(rng_seed)
        candidates = [{"group": "top", "rank": r, "idx": int(order[r])}
                      for r in range(n_top)]
        candidates += [{"group": "mid", "rank": int(r), "idx": int(order[r])}
                       for r in mid_ranks]
        seen = {c["idx"] for c in candidates}
        rand_pool = [i for i in rng.choice(len(params), size=n_rand * 4,
                                           replace=False) if i not in seen]
        candidates += [{"group": "rand", "rank": int(rank_of[i]), "idx": int(i)}
                       for i in rand_pool[:n_rand]]
        suffix = ""

    print(f"\n{'='*60}")
    print(f"PRECISION PROBE: {label}  variant={variant}  "
          f"probes={len(candidates)} (top {n_top} / mid {len(mid_ranks)} / rand {n_rand})")
    print(f"{'='*60}")

    t0 = time.time()
    sim = _build_3d_sim(tag=f"prec3d_{label}")
    entries = []
    try:
        _apply_params_to_design(sim.fdtd, params)
        print("  baseline run ...", flush=True)
        sim.fdtd.run()
        f0 = read_formal_target_3d(sim.fdtd, target_port)["abs2"]
        print(f"    baseline |a|^2 = {f0:.6f}  "
              f"(jac session: {jac['baseline']['abs2']:.6f})")

        for ci, cand in enumerate(candidates):
            idx = cand["idx"]
            was_si = params[idx] > 0.5
            sim.fdtd.switchtolayout()
            _flip_single_pixel(sim.fdtd, idx, to_si=not was_si)
            t_r = time.time()
            sim.fdtd.run()
            fom = read_formal_target_3d(sim.fdtd, target_port)["abs2"]
            dF = fom - f0
            g_pred = float(gain[idx])
            entries.append({
                **cand,
                "dir": "Si->SiO2" if was_si else "SiO2->Si",
                "p": float(params[idx]),
                "predicted_gain": g_pred,
                "fom": float(fom),
                "delta_fom": float(dF),
                "sign_match": bool((dF > 0) == (g_pred > 0)),
                "wall_s": time.time() - t_r,
            })
            print(f"  [{ci + 1}/{len(candidates)}] {cand['group']}#rank{cand['rank']} "
                  f"idx={idx} pred={g_pred:+.4f} -> DF={dF:+.4e} "
                  f"{'MATCH' if entries[-1]['sign_match'] else 'MISS'} "
                  f"({entries[-1]['wall_s']:.0f}s)", flush=True)
            # undo
            sim.fdtd.switchtolayout()
            _flip_single_pixel(sim.fdtd, idx, to_si=was_si)
    finally:
        _close_sim(sim)
        _sweep_stray_engines()

    top_entries = [e for e in entries if e["group"] == "top"]
    precision_at = {}
    for K in (5, n_top):
        sub = top_entries[:K]
        precision_at[f"precision@{K}"] = (
            sum(1 for e in sub if e["delta_fom"] > 0) / len(sub) if sub else float("nan")
        )

    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr([e["predicted_gain"] for e in entries],
                              [e["delta_fom"] for e in entries])
        spearman = {"rho": float(rho), "p": float(pval)}
    except Exception:
        spearman = {"rho": float("nan"), "p": float("nan")}

    wall_s = time.time() - t0
    result = {
        "label": label, "seed": seed, "target_port": target_port,
        "wl_nm": wl_nm, "variant": variant,
        "timestamp": _now_iso(),
        "baseline_abs2_session": f0,
        "baseline_abs2_jac": jac["baseline"]["abs2"],
        "entries": entries,
        **precision_at,
        "sign_match_rate_all": sum(1 for e in entries if e["sign_match"]) / len(entries),
        "spearman": spearman,
        "wall_s": wall_s,
    }
    _save_json(result, out_dir / f"precision_{label}_{variant}{suffix}.json")

    print(f"\n  {precision_at}")
    print(f"  sign-match (all {len(entries)}): "
          f"{result['sign_match_rate_all']:.2f}   spearman rho={spearman['rho']:+.3f}")
    print(f"  wall: {wall_s:.0f}s")
    return result




def matched_random_baseline(seed, target_port, wl_nm=WL_NM, k=5, draws=3,
                            rng_seed=1234, exclude_ix=None, out_dir=None,
                            reuse=True):
    """Flip K *random* pixels (matched count) — does adjoint beat random?

    Without this, a push PASS only shows 'these flips helped', not
    'the adjoint picked them better than chance'.

    *exclude_ix*: optional column range(s) to exclude from the random pool —
    use the SAME mask as the push under comparison so random draws come from
    the same interior region.

    (Task: randbase — matched random-flip baseline)
    """
    out_dir = Path(out_dir) if out_dir else RESULTS_DIR
    label = _label(seed, target_port, wl_nm)

    jac = _get_jac(seed, target_port, wl_nm, out_dir, reuse)
    params = np.asarray(jac["params"], dtype=float).reshape(-1)
    baseline = jac["baseline"]["abs2"]
    rng = np.random.RandomState(rng_seed)

    pool = np.arange(len(params))
    mask_tag = ""
    if exclude_ix is not None:
        ranges = [exclude_ix] if isinstance(exclude_ix[0], (int, float)) \
            else list(exclude_ix)
        ix_of = pool // NY_2D
        excluded = np.zeros(len(params), dtype=bool)
        for lo, hi in ranges:
            excluded |= (ix_of >= lo) & (ix_of <= hi)
        pool = pool[~excluded]
        mask_tag = "_mask" + "+".join(f"{lo}-{hi}" for lo, hi in ranges)

    print(f"\n{'='*60}")
    print(f"MATCHED RANDOM BASELINE: {label}  K={k}  draws={draws}"
          + (f"  exclude_ix={exclude_ix} (pool {len(pool)})" if exclude_ix else ""))
    print(f"{'='*60}")
    print(f"  baseline |a|^2 = {baseline:.6f}")

    t0 = time.time()
    entries = []
    for d_i in range(draws):
        flip_idx = rng.choice(pool, size=int(k), replace=False)
        p_new = _apply_flips(params, flip_idx)
        print(f"  draw {d_i + 1}/{draws} ...", flush=True)
        ft = eval_fom_3d(p_new, target_port, wl_nm,
                         tag=f"rand3d_{label}_k{k}_d{d_i}")
        fom = ft["abs2"]
        delta = fom - baseline
        entries.append({
            "draw": d_i, "flip_idx": [int(i) for i in flip_idx],
            "fom": float(fom), "delta_fom": float(delta),
            "improved": bool(delta > 0),
        })
        print(f"    |a|^2={fom:.6f}  D={delta:+.4e}")

    deltas = [e["delta_fom"] for e in entries]
    wall_s = time.time() - t0
    result = {
        "label": label, "seed": seed, "target_port": target_port,
        "wl_nm": wl_nm, "K": int(k), "draws": draws, "rng_seed": rng_seed,
        "exclude_ix": list(exclude_ix) if exclude_ix else None,
        "timestamp": _now_iso(),
        "baseline_fom": baseline,
        "entries": entries,
        "delta_mean": float(np.mean(deltas)),
        "delta_max": float(np.max(deltas)),
        "n_improved": sum(1 for e in entries if e["improved"]),
        "wall_s": wall_s,
    }
    _save_json(result, out_dir / f"randbase_{label}_k{k}{mask_tag}.json")

    print(f"\n  random ΔF: mean={result['delta_mean']:+.4e}  "
          f"max={result['delta_max']:+.4e}  "
          f"improved {result['n_improved']}/{draws}")
    print(f"  wall: {wall_s:.0f}s")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse_exclude_ix(spec):
    """Parse 'lo,hi' or 'lo,hi;lo,hi' into a list of (lo, hi) tuples."""
    if not spec:
        return None
    return [tuple(int(v) for v in rng_s.split(",")) for rng_s in spec.split(";")]


def main():
    import argparse

    import lum_backend  # noqa: F401 — verifies backend path
    import _lumopt_compat  # noqa: F401

    parser = argparse.ArgumentParser(
        description="3D adjoint jac / fdcheck / short_push (manual double-run route)",
    )
    parser.add_argument("--task", required=True,
                        choices=["jac", "fdcheck", "push", "randbase",
                                 "precision", "linearity"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--port", type=int, default=1)
    parser.add_argument("--wl", type=float, default=1550.0)
    parser.add_argument("--eps", type=float, default=1e-2,
                        help="fdcheck flip-budget scale (pick via offline flip count)")
    parser.add_argument("--ks", type=str, default="5,10,20",
                        help="push top-K ladder, comma-separated")
    parser.add_argument("--k", type=int, default=5,
                        help="randbase matched flip count")
    parser.add_argument("--draws", type=int, default=3,
                        help="randbase random draws")
    parser.add_argument("--rng-seed", type=int, default=1234)
    parser.add_argument("--n-top", type=int, default=10,
                        help="precision: number of top-ranked pixels to probe")
    parser.add_argument("--probe-index", type=float, default=2.0,
                        help="linearity: refractive index of the small-contrast probe")
    parser.add_argument("--pixels", type=str, default=None,
                        help="precision: explicit comma-separated flat pixel indices")
    parser.add_argument("--exclude-ix", type=str, default=None,
                        help="push/randbase: mask column ranges 'lo,hi' or 'lo,hi;lo,hi'")
    parser.add_argument("--select", default="top", choices=["top", "bottom"],
                        help="push: 'bottom' flips most-negative-gain pixels (reverse control)")
    parser.add_argument("--variant", default="i_conj_a", choices=list(GRAD_VARIANTS))
    parser.add_argument("--no-reuse", action="store_true",
                        help="recompute jac even if a saved .npz exists")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    reuse = not args.no_reuse

    if args.task == "jac":
        r = compute_jac_3d(args.seed, args.port, args.wl, out_dir)
        ok = r["grad_norms"]["raw"] > 0
    elif args.task == "fdcheck":
        r = fd_direction_check(args.seed, args.port, args.wl, eps=args.eps,
                               variant=args.variant, out_dir=out_dir, reuse=reuse)
        ok = r["passes"]
    elif args.task == "push":
        ks = tuple(int(s.strip()) for s in args.ks.split(","))
        r = discrete_push_3d(args.seed, args.port, args.wl, ks=ks,
                             variant=args.variant,
                             exclude_ix=_parse_exclude_ix(args.exclude_ix),
                             select=args.select,
                             out_dir=out_dir, reuse=reuse)
        ok = r["passes"]
    elif args.task == "precision":
        pixels = None
        if args.pixels:
            pixels = [int(s.strip()) for s in args.pixels.split(",")]
        r = precision_probe(args.seed, args.port, args.wl,
                            variant=args.variant, n_top=args.n_top,
                            rng_seed=args.rng_seed, pixels=pixels,
                            out_dir=out_dir, reuse=reuse)
        ok = True  # diagnostic, not a gate
    elif args.task == "linearity":
        r = linearity_probe(args.seed, args.port, args.wl,
                            variant=args.variant, probe_index=args.probe_index,
                            out_dir=out_dir, reuse=reuse)
        ok = True  # diagnostic, not a gate
    else:
        r = matched_random_baseline(args.seed, args.port, args.wl, k=args.k,
                                    draws=args.draws, rng_seed=args.rng_seed,
                                    exclude_ix=_parse_exclude_ix(args.exclude_ix),
                                    out_dir=out_dir, reuse=reuse)
        ok = True  # baseline is informational, not a gate

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
