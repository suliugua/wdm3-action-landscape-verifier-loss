#!/usr/bin/env python
"""Experiment C: Fresh T/R device readout for accepted structures.

For each 3D anchor's accepted pixel patterns (from greedy / ablation experiments),
run a fresh FDTD session and read:
  - R_modal_in (input reflection)
  - T_modal_i for each output port (i=1,2,3)
  - Sum(T) (total transmission)
  - |a|^2 (formal proxy — for comparison)

Purpose: bridge formal |a|^2 improvements to device port T/R readings.
Whether the bridge holds or not is itself informative.

Usage:
  python -m wdm3_3d.device_readout --seeds 0,7,9,10
"""

import json, os, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault("LUMOPT_BACKEND", "2026")
import lum_backend  # noqa

from .wdm3_3d_device import WL_NM, load_warmstart_params, _build_3d_sim, _close_sim
from .wdm3_3d_smoke import _apply_params_to_design
from .wdm3_3d_audit import read_formal_target_3d
from .wdm3_3d_jac import _label, _now_iso, _save_json, _sweep_stray_engines

READOUT_DIR = Path(__file__).resolve().parent / "results_3d" / "device_readout"


def read_device_metrics(seed, target_port=1, params_override=None,
                        wl_nm=WL_NM, tag="readout"):
    """Build fresh FDTD, apply params, run, read formal + device metrics.

    Parameters
    ----------
    params_override : np.ndarray or None
        If None, use warmstart params. Otherwise use these (accepted structure).

    Returns
    -------
    dict with keys: abs2, R_modal_in, T_modal_i (list), sum_T, wall_s
    """
    params = (np.asarray(params_override, dtype=float).reshape(-1)
              if params_override is not None
              else np.asarray(load_warmstart_params(seed), dtype=float).reshape(-1))

    t0 = time.time()
    sim = _build_3d_sim(tag=f"{tag}_{_label(seed, target_port, wl_nm)}")
    try:
        _apply_params_to_design(sim.fdtd, params)
        sim.fdtd.run()
        # Read formal |a|^2
        formal = read_formal_target_3d(sim.fdtd, target_port)
        # Read device: T_i for all ports
        T_modal = []
        for port_i in [1, 2, 3]:
            try:
                pt = read_formal_target_3d(sim.fdtd, port_i)
                T_modal.append(float(pt.get("T_forward", 0)))
            except Exception:
                T_modal.append(float("nan"))
        sum_T = float(np.nansum(T_modal))
        return {
            "seed": seed, "target_port": target_port,
            "abs2": float(formal["abs2"]),
            "R_modal_in": float(formal.get("R_modal_in", formal.get("Re_a", 0))),
            "T_modal": T_modal,
            "sum_T": sum_T,
            "Im_a": float(formal.get("Im_a", 0)),
            "Re_a": float(formal.get("Re_a", 0)),
            "wall_s": time.time() - t0,
        }
    finally:
        _close_sim(sim)
        _sweep_stray_engines()


def run_readout(seeds, target_port=1):
    """Run device readout for baseline (warmstart) + accepted structures."""
    READOUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    results = {}
    for seed in seeds:
        label = _label(seed, target_port, WL_NM)
        print(f"\n{'='*50}")
        print(f"DEVICE READOUT: {label}")
        print(f"{'='*50}")

        # 1. Baseline (warmstart)
        print("  [1/2] Baseline (warmstart)...")
        baseline = read_device_metrics(seed, target_port, tag="baseline")
        results[f"{label}_baseline"] = baseline
        print(f"  |a|^2={baseline['abs2']:.6f}  "
              f"R={baseline['R_modal_in']:.6f}  "
              f"Sum(T)={baseline['sum_T']:.6f}")

        # 2. Accepted structure (if available)
        # Search across all experiment directories for accepted flips
        accepted_params = None
        accepted_source = "none"

        from .wdm3_3d_landscape import LANDSCAPE_DIR

        def _extract_accepted(record, rounds_key="rounds"):
            """Return accepted pixel indices from a record dict."""
            items = record.get(rounds_key, [])
            return [r for r in items if r.get("accepted")]

        def _build_accepted_params(accepted_list):
            """Apply accepted flips to warmstart params."""
            p = np.asarray(load_warmstart_params(seed), dtype=float).reshape(-1)
            for r in accepted_list:
                p[int(r["idx"])] = 1.0 - p[int(r["idx"])]
            return p

        # Source 1: prospective_series / prospective arm files
        for exp_dir_name in ["prospective_series", "prospective"]:
            exp_dir = LANDSCAPE_DIR / exp_dir_name
            for arm_name in ["PHASE_MASKED_ACCEPT", "PHASE_DIAGNOSED",
                            "PHASE_MASKED", "ARM_DIAGNOSED"]:
                arm_file = exp_dir / f"arm_{arm_name}_{label}.json"
                if arm_file.exists():
                    accepted = _extract_accepted(json.load(open(arm_file)))
                    if accepted:
                        accepted_params = _build_accepted_params(accepted)
                        accepted_source = f"{exp_dir_name}/{arm_name}"
                        break
            if accepted_params is not None:
                break

        # Source 2: train_test Phase 2 results (seed0)
        if accepted_params is None:
            tt_file = LANDSCAPE_DIR / "train_test" / f"TEST_{label}.json"
            if tt_file.exists():
                accepted = _extract_accepted(json.load(open(tt_file)))
                if accepted:
                    accepted_params = _build_accepted_params(accepted)
                    accepted_source = f"train_test/TEST_{label}"

        # Source 3: main landscape greedy runs (seed7, seed10)
        if accepted_params is None:
            greedy_files = sorted(LANDSCAPE_DIR.glob(f"greedy_{label}_r*.json"))
            if greedy_files:
                # Use the last (most advanced) round
                last = json.load(open(greedy_files[-1]))
                accepted = _extract_accepted(last, rounds_key="nominations")
                if accepted:
                    accepted_params = _build_accepted_params(accepted)
                    accepted_source = f"greedy/{greedy_files[-1].stem}"

        if accepted_params is not None:
            print(f"  [2/2] Accepted structure (source: {accepted_source})...")
            accepted = read_device_metrics(seed, target_port,
                                          params_override=accepted_params,
                                          tag="accepted")
            results[f"{label}_accepted"] = accepted
            print(f"  |a|^2={accepted['abs2']:.6f}  "
                  f"R={accepted['R_modal_in']:.6f}  "
                  f"Sum(T)={accepted['sum_T']:.6f}")
            print(f"  Δ|a|^2 = {accepted['abs2'] - baseline['abs2']:+.4e}")
            print(f"  ΔR     = {accepted['R_modal_in'] - baseline['R_modal_in']:+.4e}")
            print(f"  ΔSumT  = {accepted['sum_T'] - baseline['sum_T']:+.4e}")
        else:
            print(f"  [2/2] No accepted structure found for {label}")

    # Summary
    summary = {"seeds": seeds, "wall_s": time.time() - t0,
               "results": {}}
    for seed in seeds:
        label = _label(seed, target_port, WL_NM)
        b = results.get(f"{label}_baseline", {})
        a = results.get(f"{label}_accepted", {})
        summary["results"][label] = {
            "baseline": {"abs2": b.get("abs2"), "R": b.get("R_modal_in"),
                        "SumT": b.get("sum_T")},
            "accepted": {"abs2": a.get("abs2"), "R": a.get("R_modal_in"),
                        "SumT": a.get("sum_T"),
                        "source": accepted_source if a else "none"},
            "delta": {
                "abs2": (a.get("abs2", 0) - b.get("abs2", 0)) if a else None,
                "R": (a.get("R_modal_in", 0) - b.get("R_modal_in", 0)) if a else None,
                "SumT": (a.get("sum_T", 0) - b.get("sum_T", 0)) if a else None,
            } if a else None,
        }

    _save_json(summary, READOUT_DIR / "device_readout_summary.json")
    _save_json(results, READOUT_DIR / "device_readout_details.json")
    print(f"\nSummary saved → {READOUT_DIR}")
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fresh T/R device readout")
    ap.add_argument("--seeds", type=str, default="0,7,9,10",
                    help="comma-separated seeds")
    ap.add_argument("--port", type=int, default=1)
    args = ap.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    run_readout(seeds, args.port)
