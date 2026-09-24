#!/usr/bin/env python
"""3D SOI 1×2 Splitter Transfer-Lite (FROZEN 2026-07-24).

Reuses WDM3 simulation skeleton but reconfigures to 2-output splitter:
P1 (y=1400) + P3 (y=3720) as active outputs, P2 unused.
Dual-output adjoint: forward run → adjoint at P1+P3 combined.
Splitter objective: J = T1+T3 - R_in - |T1-T3|.

Usage:
  python -m wdm3_3d.splitter_transfer
"""
import os, sys, json, time
from pathlib import Path
import numpy as np

os.environ.setdefault("LUMOPT_BACKEND", "2026")
import lum_backend  # noqa

from wdm3_3d.wdm3_3d_device import (
    DESIGN_X_NM, DESIGN_Y_NM, NX_2D, NY_2D, WL_NM,
    _close_sim, _m, _build_3d_sim,
)
from wdm3_3d.wdm3_3d_smoke import _apply_params_to_design
from wdm3_3d.wdm3_3d_audit import (
    _read_expansion_coefficient_3d, audit_3d_modes, read_formal_target_3d,
)
from wdm3_3d.wdm3_3d_jac import _grad_variants, _flip_gain, _now_iso, _save_json
from wdm3_3d.wdm3_3d_landscape import IncrementalSession
from wdm3_3d.wdm3_3d_splitter import (
    make_splitter_initial, read_splitter_metrics_from_audit,
    SPLIT_OUTPUT_PORTS, SPLIT_OUTPUT_YS,
)

# ── Config ────────────────────────────────────────────────────────────────
WL = WL_NM
OUT_DIR = Path(__file__).resolve().parent / "results_3d" / "splitter_transfer"
N_PROBE = 10
K_TEST = 10
ACCEPT_SIGMA = 2.0
RNG_SEED = 42

OUT_DIR.mkdir(parents=True, exist_ok=True)
LABEL = "splitter_1x2@1550nm"


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
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


def _add_adjoint_source(fdtd, port_idx, wl_nm, name):
    mon = f"P{port_idx}_prof"
    fdtd.addmode()
    fdtd.set("name", name)
    fdtd.set("injection axis", "x-axis")
    fdtd.set("direction", "Backward")
    for prop in ["x", "y", "y span", "z", "z span"]:
        fdtd.setnamed(name, prop, fdtd.getnamed(mon, prop))
    fdtd.setnamed(name, "wavelength start", _m(wl_nm))
    fdtd.setnamed(name, "wavelength stop", _m(wl_nm))
    fdtd.setnamed(name, "mode selection", "fundamental TE mode")
    fdtd.setnamed(name, "amplitude", 1.0)
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
    return float(phi_deg[best_idx]), phi_deg, gains


def variant_phi_deg(variant, a):
    arg_a = np.degrees(np.angle(a))
    return {"raw": 0.0, "i": 90.0, "conj_a": (-arg_a) % 360,
            "i_conj_a": (90.0 - arg_a) % 360}[variant]


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Forward + dual-output adjoint
# ═══════════════════════════════════════════════════════════════════════════

def stage1_compute():
    params = make_splitter_initial()
    st1_path = OUT_DIR / f"stage1_{LABEL}.json"

    print(f"\n{'='*60}")
    print(f"STAGE 1: Forward + Dual-Output Adjoint — {LABEL}")
    print(f"  Active outputs: P{SPLIT_OUTPUT_PORTS[0]}(y={SPLIT_OUTPUT_YS[0]}) + "
          f"P{SPLIT_OUTPUT_PORTS[1]}(y={SPLIT_OUTPUT_YS[1]})")
    print(f"{'='*60}")

    sim = _build_3d_sim(tag="st1_splitter")
    try:
        _apply_params_to_design(sim.fdtd, params)

        # ── Forward run ──
        print("[1/2] Forward run ...", flush=True)
        t0 = time.time()
        sim.fdtd.run()
        E_fwd, x_opt, y_opt, _ = _read_opt_fields(sim.fdtd)
        aud = audit_3d_modes(sim.fdtd, output_ports=3)
        dev_base = read_splitter_metrics_from_audit(aud)
        # Read a at P1 for phase reference
        ft1 = read_formal_target_3d(sim.fdtd, 1)
        a_ref = complex(ft1["Re_a"], ft1["Im_a"])
        print(f"  T1={dev_base['T1']:.4f}  T3={dev_base['T3']:.4f}  "
              f"R_in={dev_base['R_in']:.4f}  J_dev={dev_base['J_dev']:.4f}  "
              f"wall={time.time()-t0:.0f}s", flush=True)

        # ── Dual-output adjoint: P1 + P3 sources enabled simultaneously ──
        print("[2/2] Dual adjoint (P1+P3) ...", flush=True)
        t0 = time.time()
        sim.fdtd.switchtolayout()
        sim.fdtd.setnamed("source", "enabled", False)
        for pi in SPLIT_OUTPUT_PORTS:
            aname = _add_adjoint_source(sim.fdtd, pi, WL, f"adj_P{pi}")
            sim.fdtd.setnamed(aname, "enabled", True)
        sim.fdtd.run()
        E_adj, _, _, _ = _read_opt_fields(sim.fdtd)
        print(f"  max|E_adj|={np.max(np.abs(E_adj)):.3f}  wall={time.time()-t0:.0f}s", flush=True)
    finally:
        _close_sim(sim)

    # ── O_pix + 4 variants ──
    O3 = np.sum(E_fwd * E_adj, axis=-1)
    O_xy = np.sum(O3, axis=2)
    O_pix = _overlap_to_design_grid(O_xy, x_opt, y_opt)
    grads = _grad_variants(O_pix, a_ref)

    # ── Phase scan ──
    peak_phi, phi_deg, gains = _phase_scan(O_pix)
    nearest_variant = min(["raw", "i", "conj_a", "i_conj_a"],
                          key=lambda v: abs(variant_phi_deg(v, a_ref) - peak_phi))

    summary = {
        "label": LABEL, "timestamp": _now_iso(),
        "active_ports": list(SPLIT_OUTPUT_PORTS),
        "baseline": {
            "T1": dev_base["T1"], "T3": dev_base["T3"],
            "T_sum": dev_base["T_sum"], "balance": dev_base["balance"],
            "R_in": dev_base["R_in"], "J_dev": dev_base["J_dev"],
            "a_ref": [float(a_ref.real), float(a_ref.imag)],
        },
        "phase_scan": {"peak_phi_deg": peak_phi, "nearest_variant": nearest_variant},
        "variants": {v: {"phi_deg": variant_phi_deg(v, a_ref),
                         "norm": float(np.linalg.norm(grads[v]))}
                      for v in ["raw", "i", "conj_a", "i_conj_a"]},
    }
    np.savez(OUT_DIR / f"grads_{LABEL}.npz", O_pix=O_pix,
             params=params, a_ref=np.complex128(a_ref),
             **{f"grad_{v}": grads[v] for v in grads})
    _save_json(summary, st1_path)
    print(f"  Phase peak: {peak_phi:.1f}° → {nearest_variant}")
    print(f"  Saved → {st1_path}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: Per-variant FDTD probe (uses |a|² at P1 as formal proxy)
# ═══════════════════════════════════════════════════════════════════════════

def stage2_probe(st1):
    data = np.load(OUT_DIR / f"grads_{LABEL}.npz")
    params = data["params"]
    variants = ["raw", "i", "conj_a", "i_conj_a"]
    a_ref = complex(data["a_ref"])

    st2_path = OUT_DIR / f"stage2_probe_{LABEL}.json"
    print(f"\n{'='*60}")
    print(f"STAGE 2: Per-variant FDTD probe — {LABEL}")
    print(f"{'='*60}")

    results = {}
    variant_probes = {}
    nets = {}
    # Use P1 as the formal readout port for IncrementalSession
    PROBE_PORT = 1  # read |a|² at P1 during probe

    for variant in variants:
        gain = np.real(data[f"grad_{variant}"]).flatten()
        pos_rank = np.argsort(gain)[::-1]
        pos_rank = pos_rank[gain[pos_rank] > 0]
        candidates = pos_rank[5:] if len(pos_rank) > N_PROBE + 5 else pos_rank
        n_take = min(N_PROBE, len(candidates))
        step = max(1, len(candidates) // n_take) if n_take else 1
        probe_idx = [int(candidates[i * step]) for i in range(n_take)]
        variant_probes[variant] = probe_idx

        probe_results = []
        with IncrementalSession(params.copy(), PROBE_PORT, WL,
                                tag=f"probe_{variant}") as S:
            f0 = S.run_baseline()
            for idx in probe_idx:
                f = S.measure([int(idx)])
                dF = float(f - f0)
                probe_results.append({
                    "idx": int(idx), "dF": dF,
                    "gain_pred": float(gain[idx]), "accept": dF > 0,
                })

        n_pos = sum(1 for r in probe_results if r["accept"])
        net = sum(r["dF"] for r in probe_results)
        nets[variant] = net
        results[variant] = probe_results
        print(f"  {variant:12s} (phi={variant_phi_deg(variant, a_ref):.1f}°): "
              f"{n_pos}/{n_take} positive, net={net:+.4e}")

    best_variant = max(nets, key=nets.get)
    adopted = best_variant if nets[best_variant] > 0 else None

    summary = {
        "label": LABEL, "timestamp": _now_iso(),
        "variant_probes": variant_probes, "results": results, "nets": nets,
        "adopted_variant": adopted,
        "phase_scan_nearest": st1["phase_scan"]["nearest_variant"],
        "agreement_with_scan": adopted == st1["phase_scan"]["nearest_variant"],
    }
    _save_json(summary, st2_path)
    print(f"  Adopted: {adopted}  |  Scan nearest: {st1['phase_scan']['nearest_variant']}  "
          f"|  Agree: {summary['agreement_with_scan']}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3: Held-out test (|a|² at P1)
# ═══════════════════════════════════════════════════════════════════════════

def stage3_test(st1, st2):
    data = np.load(OUT_DIR / f"grads_{LABEL}.npz")
    params = data["params"]
    adopted = st2["adopted_variant"]
    if adopted is None:
        print("No variant adopted — skipping Stage 3.")
        return None

    probe_set = set()
    for v_probes in st2["variant_probes"].values():
        probe_set.update(v_probes)

    st3_path = OUT_DIR / f"stage3_test_{LABEL}.json"
    print(f"\n{'='*60}")
    print(f"STAGE 3: Held-out test — {LABEL}")
    print(f"{'='*60}")

    # ── Sigma ──
    rng = np.random.RandomState(RNG_SEED)
    safe_pool = np.array([i for i in range(len(params)) if i not in probe_set])
    sigma_idx = rng.choice(safe_pool, size=min(15, len(safe_pool)), replace=False)
    single_dFs = []
    with IncrementalSession(params.copy(), 1, WL, tag="sigma") as S:
        f0 = S.run_baseline()
        for idx in sigma_idx:
            f = S.measure([int(idx)])
            single_dFs.append(float(f - f0))
    sigma = float(np.std(single_dFs))
    accept_thr = ACCEPT_SIGMA * sigma
    print(f"  sigma={sigma:.3e}  accept_thr={accept_thr:.3e}")

    # ── Test ──
    gain = np.real(data[f"grad_{adopted}"]).flatten()
    excl = np.zeros(len(params), dtype=bool)
    for px in probe_set:
        excl[int(px)] = True
    nom_idx = [int(i) for i in np.argsort(gain)[::-1] if not excl[int(i)] and gain[int(i)] > 0][:K_TEST]
    if not nom_idx:
        nom_idx = [int(i) for i in np.argsort(gain)[::-1] if not excl[int(i)]][:K_TEST]
    print(f"  Test candidates: {len(nom_idx)} (excluded {len(probe_set)} probes)")

    rounds = []
    total_dF = 0.0
    with IncrementalSession(params.copy(), 1, WL, tag="test") as S:
        f0 = S.run_baseline()
        for rank, idx in enumerate(nom_idx, start=1):
            dF, accept = S.try_flip(idx, accept_thr)
            rounds.append({"rank": rank, "idx": int(idx),
                           "dF": float(dF), "accepted": bool(accept),
                           "gain_pred": float(gain[idx])})
            if accept:
                total_dF += dF
            print(f"  [{rank}/{len(nom_idx)}] idx={idx} pred={gain[idx]:+.4f} "
                  f"dF={dF:+.4e} {'ACCEPT' if accept else 'reject'}")

    accepted = [r for r in rounds if r["accepted"]]
    all_dFs = [r["dF"] for r in rounds]
    max_i = int(np.argmax(all_dFs)) if all_dFs else 0
    primary = total_dF > 0 or len(accepted) > 0

    summary = {
        "label": LABEL, "timestamp": _now_iso(),
        "adopted_variant": adopted, "sigma": sigma, "accept_thr": accept_thr,
        "rounds": rounds, "n_tested": len(rounds), "n_accepted": len(accepted),
        "net_dF": float(total_dF), "max_dF_rank": max_i + 1 if all_dFs else 0,
        "primary_PASS": bool(primary),
        "accepted_indices": [r["idx"] for r in accepted],
    }
    _save_json(summary, st3_path)
    print(f"  VERDICT: {'PASS' if primary else 'FAIL'}  net={total_dF:+.4e}  "
          f"accept={len(accepted)}/{len(rounds)}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4: Fresh device T/R bridge (splitter objective)
# ═══════════════════════════════════════════════════════════════════════════

def stage4_bridge(st1, st3):
    if st3 is None or not st3.get("accepted_indices"):
        print("No accepted pixels — skipping Stage 4.")
        return None

    data = np.load(OUT_DIR / f"grads_{LABEL}.npz")
    params_base = data["params"].copy()
    params_acc = params_base.copy()
    for idx in st3["accepted_indices"]:
        params_acc[int(idx)] = 1.0 - params_acc[int(idx)]

    st4_path = OUT_DIR / f"stage4_bridge_{LABEL}.json"
    print(f"\n{'='*60}")
    print(f"STAGE 4: Fresh device T/R bridge — {LABEL}")
    print(f"{'='*60}")

    results = {}
    for tag, p in [("baseline", params_base), ("accepted", params_acc)]:
        sim = _build_3d_sim(tag=f"br_{tag}")
        try:
            _apply_params_to_design(sim.fdtd, p)
            sim.fdtd.run()
            aud = audit_3d_modes(sim.fdtd, output_ports=3)
            dev = read_splitter_metrics_from_audit(aud)
            results[tag] = dev
            print(f"  {tag}: T1={dev['T1']:.4f}  T3={dev['T3']:.4f}  "
                  f"T_sum={dev['T_sum']:.4f}  R_in={dev['R_in']:.4f}  "
                  f"J_dev={dev['J_dev']:.4f}", flush=True)
        finally:
            _close_sim(sim)

    b = results["baseline"]
    a = results["accepted"]
    delta = {k: a[k] - b[k] for k in ["T1", "T3", "T_sum", "R_in", "J_dev"]}
    delta["balance"] = a["balance"] - b["balance"]
    bridge_pass = bool(delta["J_dev"] > 0)

    summary = {
        "label": LABEL, "timestamp": _now_iso(),
        "baseline": b, "accepted": a, "delta": delta,
        "bridge_PASS": bridge_pass,
        "note": "bridge_PASS = J_dev increased (splitter objective improved)",
    }
    _save_json(summary, st4_path)
    print(f"  ΔT_sum={delta['T_sum']:+.4e}  ΔR_in={delta['R_in']:+.4e}  "
          f"ΔJ_dev={delta['J_dev']:+.4e}  Δbalance={delta['balance']:+.4e}")
    print(f"  Bridge: {'PASS' if bridge_pass else 'FAIL'}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Helpers for device-level simulation
# ═══════════════════════════════════════════════════════════════════════════

def _run_device_sim(params, tag="dev"):
    """Run a single device simulation and return splitter metrics dict."""
    sim = _build_3d_sim(tag=tag)
    try:
        _apply_params_to_design(sim.fdtd, np.asarray(params, dtype=float))
        sim.fdtd.run()
        aud = audit_3d_modes(sim.fdtd, output_ports=3)
        return read_splitter_metrics_from_audit(aud)
    finally:
        _close_sim(sim)


def _load_stage(path):
    """Load a stage JSON if it exists, else None."""
    if path.exists():
        return json.load(open(path))
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3b: Device-level null calibration + greedy test
# ═══════════════════════════════════════════════════════════════════════════

def stage3b_device_null_and_test(n_null=25, resume=False):
    """Null-calibrated device-level greedy test for the 1×2 splitter.

    Stage 3 (formal proxy, |a|² at P1) passed 8/10, but stage 4 (device
    bridge with all 8 flips at once) failed: J_dev dropped 27%.  Stage 3b
    answers: if the greedy test used the true device objective J_dev as
    the accept/reject criterion instead of the formal proxy, would it:

      (a) accept *different* pixels — proving the formal→device gap is a
          ranking failure (proxy ranks pixels device disagrees with), or
      (b) accept *similar* pixels — proving it's a threshold/nonlinearity
          failure (pixels individually good for both, but accumulate badly).

    Procedure:
      1. Flip n_null random single pixels through full device sim → σ_device
      2. accept_thr = 2 × σ_device
      3. Greedy test: top-10 candidates ranked by adopted variant's gradient,
         each tested via device sim, accept if ΔJ_dev > threshold.
    """
    st1_path = OUT_DIR / f"stage1_{LABEL}.json"
    st2_path = OUT_DIR / f"stage2_probe_{LABEL}.json"
    st3_path = OUT_DIR / f"stage3_test_{LABEL}.json"
    st3b_path = OUT_DIR / f"stage3b_device_{LABEL}.json"

    st1 = _load_stage(st1_path)
    st2 = _load_stage(st2_path)
    st3 = _load_stage(st3_path)
    if st1 is None or st2 is None or st3 is None:
        print("ERROR: Stages 1-3 must be run first.")
        print(f"  st1: {st1_path} — {'OK' if st1 else 'MISSING'}")
        print(f"  st2: {st2_path} — {'OK' if st2 else 'MISSING'}")
        print(f"  st3: {st3_path} — {'OK' if st3 else 'MISSING'}")
        return None

    data = np.load(OUT_DIR / f"grads_{LABEL}.npz")
    params_base = np.asarray(data["params"], dtype=float).reshape(-1).copy()
    adopted = st3["adopted_variant"]

    # Build exclusion set: all probe pixels + stage 3 test pixels
    excl_set = set()
    for v_probes in st2["variant_probes"].values():
        excl_set.update(v_probes)
    for r in st3["rounds"]:
        excl_set.add(r["idx"])

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Null calibration — random single-pixel device flips
    # ═══════════════════════════════════════════════════════════════
    rng = np.random.RandomState(RNG_SEED + 1000)
    safe_pool = np.array([i for i in range(len(params_base)) if i not in excl_set])
    null_indices = [int(i) for i in rng.choice(safe_pool,
                    size=min(n_null, len(safe_pool)), replace=False)]

    null_dFs = []
    null_details = []
    sigma_device = None
    accept_thr_device = None

    if resume and st3b_path.exists():
        prev = json.load(open(st3b_path))
        null_details = prev.get("null_calibration", {}).get("null_details", [])
        null_dFs = [d["dF"] for d in null_details]
        sigma_device = prev.get("null_calibration", {}).get("sigma_device")
        accept_thr_device = prev.get("null_calibration", {}).get("accept_thr_device")
        print(f"[resume] Loaded {len(null_details)} existing null flips "
              f"from {st3b_path}")

    if sigma_device is None:
        print(f"\n{'='*60}")
        print(f"STAGE 3b PHASE 1: Device null calibration — {LABEL}")
        print(f"  {n_null} random single-pixel flips via device simulation")
        print(f"  Excluded: {len(excl_set)} probe+test pixels")
        print(f"{'='*60}")

        # Baseline device sim
        t0 = time.time()
        print("[baseline] running device sim ...", end=" ", flush=True)
        dev_base = _run_device_sim(params_base, tag="st3b_base")
        print(f"J_dev={dev_base['J_dev']:.6f}  "
              f"T1={dev_base['T1']:.4f}  T3={dev_base['T3']:.4f}  "
              f"R_in={dev_base['R_in']:.4f}  wall={time.time()-t0:.0f}s",
              flush=True)

        n_done = len(null_details)
        for i in range(n_done, n_null):
            idx = null_indices[i]
            params_flipped = params_base.copy()
            params_flipped[int(idx)] = 1.0 - params_flipped[int(idx)]
            t0 = time.time()
            print(f"[null {i+1}/{n_null}] idx={idx} ...", end=" ", flush=True)
            dev = _run_device_sim(params_flipped, tag=f"st3b_null_{i}")
            dF = float(dev["J_dev"] - dev_base["J_dev"])
            null_dFs.append(dF)
            null_details.append({
                "idx": int(idx), "dF": dF,
                "T1": dev["T1"], "T3": dev["T3"],
                "T_sum": dev["T_sum"], "balance": dev["balance"],
                "R_in": dev["R_in"], "J_dev": dev["J_dev"],
            })
            print(f"dF={dF:+.6e}  wall={time.time()-t0:.0f}s", flush=True)
            # Save progress after each null flip
            _save_json({"null_calibration": {
                "n_null": n_null, "n_done": i + 1,
                "null_details": null_details, "null_dFs": null_dFs,
                "sigma_device": None, "accept_thr_device": None,
            }}, st3b_path)

        sigma_device = float(np.std(null_dFs))
        accept_thr_device = ACCEPT_SIGMA * sigma_device
        n_pos_null = sum(1 for d in null_dFs if d > 0)

        print(f"\n  σ_device = {sigma_device:.6e}")
        print(f"  accept_thr = {accept_thr_device:.6e}  "
              f"(= {ACCEPT_SIGMA}×σ)")
        print(f"  null dF range: [{min(null_dFs):+.6e}, "
              f"{max(null_dFs):+.6e}]")
        print(f"  null positive: {n_pos_null}/{n_null}")

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Device-level greedy test
    # ═══════════════════════════════════════════════════════════════
    dev_base = _run_device_sim(params_base, tag="st3b_base2")
    print(f"\n  Re-measured baseline: J_dev={dev_base['J_dev']:.6f}")

    print(f"\n{'='*60}")
    print(f"STAGE 3b PHASE 2: Device greedy test — {LABEL}")
    print(f"  Adopted variant: {adopted}")
    print(f"  accept_thr: {accept_thr_device:.6e}")
    print(f"{'='*60}")

    gain = np.real(data[f"grad_{adopted}"]).flatten()
    excl = np.zeros(len(params_base), dtype=bool)
    for px in excl_set:
        excl[int(px)] = True
    nom_idx = [int(i) for i in np.argsort(gain)[::-1]
               if not excl[int(i)] and gain[int(i)] > 0][:K_TEST]

    print(f"  Test candidates: {len(nom_idx)} "
          f"(excluded {len(excl_set)} probe+test pixels)")

    current_params = params_base.copy()
    current_dev = dev_base
    rounds = []
    total_dF = 0.0

    for rank, idx in enumerate(nom_idx, start=1):
        params_candidate = current_params.copy()
        params_candidate[int(idx)] = 1.0 - params_candidate[int(idx)]
        t0 = time.time()
        print(f"[{rank}/{len(nom_idx)}] idx={idx} "
              f"pred={gain[idx]:+.4f} ...", end=" ", flush=True)
        dev = _run_device_sim(params_candidate, tag=f"st3b_test_{rank}")
        dF = float(dev["J_dev"] - current_dev["J_dev"])
        accept = dF > accept_thr_device
        rounds.append({
            "rank": rank, "idx": int(idx),
            "dF": dF, "accepted": bool(accept),
            "gain_pred": float(gain[idx]),
            "J_dev_candidate": dev["J_dev"],
            "J_dev_baseline": current_dev["J_dev"],
            "T1_cand": dev["T1"], "T3_cand": dev["T3"],
            "R_in_cand": dev["R_in"],
        })
        if accept:
            total_dF += dF
            current_params = params_candidate
            current_dev = dev
            print(f"dF={dF:+.6e}  ACCEPT  wall={time.time()-t0:.0f}s",
                  flush=True)
        else:
            print(f"dF={dF:+.6e}  reject  wall={time.time()-t0:.0f}s",
                  flush=True)

    accepted = [r for r in rounds if r["accepted"]]
    device_primary = total_dF > 0 or len(accepted) > 0

    summary = {
        "label": LABEL, "timestamp": _now_iso(),
        "adopted_variant": adopted,
        "null_calibration": {
            "n_null": n_null,
            "sigma_device": sigma_device,
            "accept_thr_device": accept_thr_device,
            "null_dFs": null_dFs,
            "null_details": null_details,
            "null_n_positive": sum(1 for d in null_dFs if d > 0),
        },
        "test_rounds": rounds,
        "n_tested": len(rounds),
        "n_accepted": len(accepted),
        "net_dF_device": float(total_dF),
        "accepted_indices": [r["idx"] for r in accepted],
        "device_primary_PASS": bool(device_primary),
        "baseline_J_dev": dev_base["J_dev"],
        "final_J_dev": current_dev["J_dev"],
        "final_T1": current_dev["T1"],
        "final_T3": current_dev["T3"],
        "final_R_in": current_dev["R_in"],
        "final_T_sum": current_dev["T_sum"],
        "final_balance": current_dev["balance"],
    }
    _save_json(summary, st3b_path)
    print(f"\n  VERDICT: {'PASS' if device_primary else 'FAIL'}")
    print(f"  accept={len(accepted)}/{len(rounds)}  "
          f"net_dF={total_dF:+.6e}")
    print(f"  baseline J_dev={dev_base['J_dev']:.6f}  "
          f"final J_dev={current_dev['J_dev']:.6f}")
    print(f"  Saved → {st3b_path}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Splitter transfer-lite pipeline")
    ap.add_argument("--stage3b", action="store_true",
                    help="Run only stage 3b (device null calibration + greedy test)")
    ap.add_argument("--n-null", type=int, default=25,
                    help="Number of random null flips (default: 25)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume partial stage3b from saved file")
    args = ap.parse_args()

    if args.stage3b:
        stage3b_device_null_and_test(n_null=args.n_null, resume=args.resume)
    else:
        st1 = stage1_compute()
        st2 = stage2_probe(st1)
        st3 = stage3_test(st1, st2)
        stage4_bridge(st1, st3)
        print(f"\n{'='*60}")
        print("Splitter transfer-lite complete.")
        print(f"Results → {OUT_DIR}")
