#!/usr/bin/env python
"""seed9 Route A/B counterfactual — Stage 1→2→3 (FROZEN 2026-07-24).

Stage 1: Compute Route A (-R adjoint) and Route B (Sigma grad T_i proxy)
         gradients via single-session 3-run FDTD.
Stage 2: Top-5 / bottom-5 batch flip + local random null, with mask from
         probe adjudication.
Stage 3: Sequential greedy accept/reject on the preferred route (optional).

Usage:
  python -m wdm3_3d.seed9_route_ab
"""
import os, sys, json, time
from pathlib import Path
import numpy as np

os.environ.setdefault("LUMOPT_BACKEND", "2026")
import lum_backend  # noqa

from wdm3_3d.wdm3_3d_device import (
    DESIGN_X_NM, DESIGN_Y_NM, NX_2D, NY_2D, WL_NM,
    _build_3d_sim, _close_sim, _m, load_warmstart_params,
)
from wdm3_3d.wdm3_3d_audit import _read_expansion_coefficient_3d, audit_3d_modes
from wdm3_3d.wdm3_3d_smoke import _apply_params_to_design
from wdm3_3d.wdm3_3d_jac import _now_iso, _save_json

# ── Config ────────────────────────────────────────────────────────────────
SEED = 9
TARGET_PORT = 1
WL = WL_NM
OUT_DIR = Path(__file__).resolve().parent / "results_3d" / "route_ab"
K_TOP = 5
K_NULL = 30
RNG_SEED = 42

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers (mirrored from stage1_negR.py for local use)
# ═══════════════════════════════════════════════════════════════════════════

def _read_opt_fields(fdtd):
    data = fdtd.getresult("opt_fields", "E")
    E = np.asarray(data["E"])
    if E.ndim == 5:
        E = E[:, :, :, 0, :]
    x = np.asarray(data["x"]).flatten()
    y = np.asarray(data["y"]).flatten()
    z = np.asarray(data["z"]).flatten()
    return E, x, y, z


def _overlap_to_design_grid(O_xy, x, y):
    from scipy.interpolate import RegularGridInterpolator
    dx = (DESIGN_X_NM[1] - DESIGN_X_NM[0]) / NX_2D
    dy = (DESIGN_Y_NM[1] - DESIGN_Y_NM[0]) / NY_2D
    xc = _m(DESIGN_X_NM[0] + (np.arange(NX_2D) + 0.5) * dx)
    yc = _m(DESIGN_Y_NM[0] + (np.arange(NY_2D) + 0.5) * dy)
    Xg, Yg = np.meshgrid(xc, yc, indexing="ij")
    interp = RegularGridInterpolator((x, y), O_xy, bounds_error=False, fill_value=0.0)
    O_pix = interp(np.column_stack([Xg.ravel(), Yg.ravel()]))
    return O_pix.reshape(NX_2D, NY_2D)


def _add_mode_source_at_port(fdtd, port_name, direction, wl_nm, name,
                              amplitude=1.0, phase_deg=0.0):
    fdtd.addmode()
    fdtd.set("name", name)
    fdtd.set("injection axis", "x-axis")
    fdtd.set("direction", direction)
    for prop in ["x", "y", "y span", "z", "z span"]:
        fdtd.setnamed(name, prop, fdtd.getnamed(port_name, prop))
    fdtd.setnamed(name, "wavelength start", _m(wl_nm))
    fdtd.setnamed(name, "wavelength stop", _m(wl_nm))
    fdtd.setnamed(name, "mode selection", "fundamental TE mode")
    fdtd.setnamed(name, "amplitude", amplitude)
    fdtd.setnamed(name, "phase", phase_deg)
    return name


def _phase_scan(C_pix, n_top=20, n_phi=36):
    phi_deg = np.linspace(0, 360, n_phi, endpoint=False)
    phi_rad = np.deg2rad(phi_deg)
    C_flat = C_pix.flatten()
    gains = np.zeros(n_phi)
    for i, phi in enumerate(phi_rad):
        g = np.real(np.exp(1j * phi) * C_flat)
        top_idx = np.argsort(np.abs(g))[-n_top:]
        gains[i] = np.sum(np.abs(g[top_idx]))
    best_idx = np.argmax(gains)
    return {
        "best_phi_deg": float(phi_deg[best_idx]),
        "max_gain_sum": float(gains[best_idx]),
        "phi_deg": phi_deg.tolist(),
        "gain_sum": gains.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Route A/B gradient computation
# ═══════════════════════════════════════════════════════════════════════════

def stage1_compute_gradients():
    params = load_warmstart_params(SEED)
    params = np.asarray(params, dtype=float).reshape(-1)
    label = f"seed{SEED}_P{TARGET_PORT}@{int(WL)}nm"

    print(f"\n{'='*60}")
    print(f"STAGE 1: Route A/B Gradient — {label}")
    print(f"{'='*60}")

    st1_path = OUT_DIR / f"stage1_summary_{label}.json"
    if st1_path.exists():
        print(f"  [reuse] Stage 1 already done: {st1_path}")
        return json.load(open(st1_path))

    sim = _build_3d_sim(tag=f"routeAB_{label}")
    try:
        _apply_params_to_design(sim.fdtd, params)

        # ── [1/3] Forward run ──
        print("[1/3] Forward run ...", flush=True)
        t0 = time.time()
        sim.fdtd.run()
        audit = audit_3d_modes(sim.fdtd, output_ports=3)
        R_base = audit["R_in_modal"]
        T_sum_base = audit["T_fwd_sum"]
        abs2_p1 = audit["per_port"].get(1, {}).get("abs2", float("nan"))
        print(f"  R_in={R_base:.6f}  T_sum={T_sum_base:.6f}  |a|^2(P1)={abs2_p1:.6f}  wall={time.time()-t0:.0f}s", flush=True)

        in_coeff = _read_expansion_coefficient_3d(sim.fdtd, "P_in_prof_me")
        b_in = complex(in_coeff["Re_b"], in_coeff["Im_b"])
        a_ports = {}
        for i in range(1, 4):
            coeff = _read_expansion_coefficient_3d(sim.fdtd, f"P{i}_prof_me")
            a_ports[i] = complex(coeff["Re_a"], coeff["Im_a"])
            print(f"  P{i}: a={a_ports[i]:.6e}  |a|^2={np.abs(a_ports[i])**2:.6f}", flush=True)

        E_fwd, x_opt, y_opt, _ = _read_opt_fields(sim.fdtd)

        # ── [2/3] Route A: -R adjoint (input port backward mode) ──
        print("[2/3] Route A adjoint (-R) ...", flush=True)
        t0 = time.time()
        sim.fdtd.switchtolayout()
        sim.fdtd.setnamed("source", "enabled", False)
        b_amp = float(np.abs(b_in))
        b_phase = float(-np.angle(b_in, deg=True)) if b_amp > 1e-20 else 0.0
        _add_mode_source_at_port(sim.fdtd, "P_in_prof", "Forward", WL,
                                 name="adj_routeA", amplitude=b_amp, phase_deg=b_phase)
        sim.fdtd.setnamed("adj_routeA", "enabled", True)
        sim.fdtd.run()
        E_adj_A, _, _, _ = _read_opt_fields(sim.fdtd)
        O_A = np.sum(E_fwd * E_adj_A, axis=-1)
        O_A_xy = np.sum(O_A, axis=2)
        C_A_pix = _overlap_to_design_grid(O_A_xy, x_opt, y_opt)
        grad_A = np.real(C_A_pix).flatten()
        print(f"  Route A |grad|={np.linalg.norm(grad_A):.4e}  wall={time.time()-t0:.0f}s", flush=True)

        # ── [3/3] Route B: Sigma grad(T_i) proxy ──
        print("[3/3] Route B adjoint (Sigma grad T_i) ...", flush=True)
        t0 = time.time()
        sim.fdtd.switchtolayout()
        sim.fdtd.setnamed("adj_routeA", "enabled", False)
        for i in range(1, 4):
            a_i = a_ports[i]
            amp = float(np.abs(a_i))
            phase = float(-np.angle(a_i, deg=True)) if amp > 1e-20 else 0.0
            _add_mode_source_at_port(sim.fdtd, f"P{i}_prof", "Backward", WL,
                                     name=f"adj_routeB_P{i}", amplitude=amp, phase_deg=phase)
            sim.fdtd.setnamed(f"adj_routeB_P{i}", "enabled", True)
        sim.fdtd.run()
        E_adj_B, _, _, _ = _read_opt_fields(sim.fdtd)
        O_B = np.sum(E_fwd * E_adj_B, axis=-1)
        O_B_xy = np.sum(O_B, axis=2)
        C_B_pix = _overlap_to_design_grid(O_B_xy, x_opt, y_opt)
        grad_B = np.real(C_B_pix).flatten()
        print(f"  Route B |grad|={np.linalg.norm(grad_B):.4e}  wall={time.time()-t0:.0f}s", flush=True)
    finally:
        _close_sim(sim)

    # ── Diagnostics ──
    dot_AB = float(np.dot(grad_A, grad_B))
    norm_A = float(np.linalg.norm(grad_A))
    norm_B = float(np.linalg.norm(grad_B))
    cos_AB = dot_AB / (norm_A * norm_B) if norm_A > 0 and norm_B > 0 else float("nan")

    ps_A = _phase_scan(C_A_pix)
    ps_B = _phase_scan(C_B_pix)

    top20_A = set(np.argsort(np.abs(grad_A))[-20:][::-1])
    top20_B = set(np.argsort(np.abs(grad_B))[-20:][::-1])
    overlap_top20 = len(top20_A & top20_B)

    summary = {
        "label": label, "seed": SEED, "target_port": TARGET_PORT,
        "timestamp": _now_iso(),
        "baseline": {
            "R_modal_in": float(R_base),
            "T_fwd_sum": float(T_sum_base),
            "abs2_P1": float(abs2_p1),
        },
        "route_comparison": {
            "cosine_similarity": cos_AB,
            "norm_A": norm_A, "norm_B": norm_B,
            "top20_overlap": overlap_top20,
            "phase_scan_A_best_deg": ps_A["best_phi_deg"],
            "phase_scan_B_best_deg": ps_B["best_phi_deg"],
        },
    }
    # Save gradients
    np.savez(OUT_DIR / f"route_A_{label}.npz", C_A_pix=C_A_pix, grad_A=grad_A, params=params)
    np.savez(OUT_DIR / f"route_B_{label}.npz", C_B_pix=C_B_pix, grad_B=grad_B, params=params)
    _save_json(summary, st1_path)
    print(f"  cos(A,B)={cos_AB:+.4f}  top20_overlap={overlap_top20}/20")
    print(f"  Phase A best: {ps_A['best_phi_deg']:.1f} deg  |  B best: {ps_B['best_phi_deg']:.1f} deg")
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: Top-5 / Bottom-5 FDTD acceptance + null baseline
# ═══════════════════════════════════════════════════════════════════════════

def stage2_acceptance(st1):
    """Batch flip top-5/bottom-5 for both routes, compare to random null."""
    from wdm3_3d.wdm3_3d_landscape import IncrementalSession
    label = st1["label"]
    params = load_warmstart_params(SEED)
    params = np.asarray(params, dtype=float).reshape(-1)

    st2_path = OUT_DIR / f"stage2_acceptance_{label}.json"
    if st2_path.exists():
        print(f"  [reuse] Stage 2 already done: {st2_path}")
        return json.load(open(st2_path))

    print(f"\n{'='*60}")
    print(f"STAGE 2: Top/Bottom Flip Acceptance — {label}")
    print(f"{'='*60}")

    # Load gradients
    rA = np.load(OUT_DIR / f"route_A_{label}.npz")
    rB = np.load(OUT_DIR / f"route_B_{label}.npz")
    grad_A = rA["grad_A"]
    grad_B = rB["grad_B"]

    # ── Local random null (k=1 x 30) ──
    rng = np.random.RandomState(RNG_SEED)
    null_idx = rng.choice(len(params), size=min(K_NULL, len(params)), replace=False)
    null_dFs = []
    with IncrementalSession(params.copy(), TARGET_PORT, WL, tag=f"null_{label}") as S:
        f0 = S.run_baseline()
        for idx in null_idx:
            f = S.measure([int(idx)])
            null_dFs.append(float(f - f0))
    sigma_single = float(np.std(null_dFs))
    print(f"  sigma_single = {sigma_single:.3e}  (n={len(null_dFs)} random flips)")

    results = {}
    for route_name, grad, clr in [("Route A", grad_A, "A"), ("Route B", grad_B, "B")]:
        # Top-5 and bottom-5 by |grad|
        ranked = np.argsort(np.abs(grad))[::-1]
        top5 = [int(ranked[i]) for i in range(K_TOP)]
        bot5 = [int(ranked[-i-1]) for i in range(K_TOP)]

        top_dFs, bot_dFs = [], []
        with IncrementalSession(params.copy(), TARGET_PORT, WL, tag=f"topbot_{route_name}_{label}") as S:
            f0 = S.run_baseline()
            # Batch flip top-5, measure total
            for idx in top5:
                S._toggle(int(idx))
            f_top = S._run_fom()
            dF_top = float(f_top - f0)
            # Revert
            for idx in top5:
                S._toggle(int(idx))
            top_dFs.append(dF_top)

            # Batch flip bottom-5
            for idx in bot5:
                S._toggle(int(idx))
            f_bot = S._run_fom()
            dF_bot = float(f_bot - f0)

        top_sigma = dF_top / sigma_single if sigma_single > 0 else 0
        bot_sigma = dF_bot / sigma_single if sigma_single > 0 else 0
        print(f"  {route_name}: top-5 dF={dF_top:+.4e} ({top_sigma:+.1f}σ)  "
              f"bottom-5 dF={dF_bot:+.4e} ({bot_sigma:+.1f}σ)")

        results[route_name] = {
            "top5_indices": top5, "bottom5_indices": bot5,
            "top5_dF": dF_top, "bottom5_dF": dF_bot,
            "top5_sigma": top_sigma, "bottom5_sigma": bot_sigma,
        }

    # ── Verdict ──
    # Preferred route = larger top-5 gain in sigma units
    verdict = {
        "sigma_single": sigma_single,
        "null_n": len(null_dFs),
        "preferred_route_by_top5": "Route A" if results["Route A"]["top5_sigma"] > results["Route B"]["top5_sigma"] else "Route B",
        "top5_sigma_A": results["Route A"]["top5_sigma"],
        "top5_sigma_B": results["Route B"]["top5_sigma"],
        "route_A_top5_positive": results["Route A"]["top5_dF"] > 0,
        "route_B_top5_positive": results["Route B"]["top5_dF"] > 0,
        "continuous_preferred": "Route A" if st1["route_comparison"]["cosine_similarity"] < 0 else "Route B",
        "continuous_discrete_agree": False,  # filled below
    }
    # Agreement: does the continuous-layer preferred route also win in discrete?
    cont_pref = verdict["continuous_preferred"]
    disc_pref = verdict["preferred_route_by_top5"]
    verdict["continuous_discrete_agree"] = (cont_pref == disc_pref)
    print(f"  Continuous preferred: {cont_pref}  |  Discrete preferred: {disc_pref}  |  Agree: {verdict['continuous_discrete_agree']}")

    _save_json({**verdict, "results": results, "label": label, "timestamp": _now_iso()}, st2_path)
    return verdict


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--port", type=int, default=1)
    ap.add_argument("--stage2-only", action="store_true")
    args = ap.parse_args()

    SEED = args.seed
    TARGET_PORT = args.port

    if not args.stage2_only:
        st1 = stage1_compute_gradients()
    else:
        label = f"seed{SEED}_P{TARGET_PORT}@{int(WL)}nm"
        st1_path = OUT_DIR / f"stage1_summary_{label}.json"
        st1 = json.load(open(st1_path))

    st2 = stage2_acceptance(st1)
    print(f"\n{'='*60}")
    print(f"DONE. Stage 2 verdict: {json.dumps({k: v for k, v in st2.items() if k != 'results'}, indent=2, default=str)}")
