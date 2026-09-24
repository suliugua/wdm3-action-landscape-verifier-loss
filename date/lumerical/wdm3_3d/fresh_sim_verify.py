#!/usr/bin/env python
"""Fresh-sim verification of IncrementalSession.

Two experiments:
  1. Final structure: rebuild Stage 3 accepted state in fresh FDTD session,
     verify R_modal_in matches incremental final value (0.01935).
  2. #647 reject flip: rebuild pre-647 structure, measure baseline R,
     then flip #647 fresh, verify Δ(-R) is negative (should be ~-8.46e-3).

If both pass, we can claim in Methods:
  "The final state and the key rejected pixel were verified in independent
   fresh-session FDTD, confirming the incremental-measurement verdicts."

Usage:
  python -m wdm3_3d.fresh_sim_verify
"""

import json, os, sys, time
from pathlib import Path
import numpy as np

# Add project root to path
PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from wdm3_3d.wdm3_3d_device import _build_3d_sim, _close_sim, load_warmstart_params
from wdm3_3d.wdm3_3d_smoke import _apply_params_to_design
from wdm3_3d.wdm3_3d_audit import audit_3d_modes
from wdm3_3d.wdm3_3d_jac import _flip_single_pixel, _sweep_stray_engines

# ── Load Stage 3 decisions ────────────────────────────────────────────────
STAGE3_PATH = Path(__file__).resolve().parent / "results_3d" / ".." / ".." / \
              "fuwuqi" / "3D" / "results_3d" / "stage3_greedy" / "stage3_result.json"
STAGE3_PATH = Path(__file__).resolve().parent / ".." / "fuwuqi" / "3D" / \
              "results_3d" / "stage3_greedy" / "stage3_result.json"
STAGE3_PATH = STAGE3_PATH.resolve()

s3 = json.load(open(STAGE3_PATH, "r", encoding="utf-8"))
decisions = s3["decisions"]
R_INCR_FINAL = s3["R_final"]        # 0.01934831
T_INCR_FINAL = s3["T_sum_final"]    # 0.10793870
R_INCR_BASELINE = s3["R_baseline"]  # 0.03555870
TOP_THR = s3["TOP_THR"]             # 0.00297331

# Accepted and rejected pixel indices
accepted = [d["idx"] for d in decisions if d["accepted"]]
rejected = [d["idx"] for d in decisions if not d["accepted"]]

print("=" * 70)
print("FRESH-SIM VERIFICATION: Stage 3 negR sequential accept/reject")
print("=" * 70)
print(f"  Incremental baseline:  R={R_INCR_BASELINE:.6f}  T_sum={s3['T_sum_baseline']:.6f}")
print(f"  Incremental final:     R={R_INCR_FINAL:.6f}  T_sum={T_INCR_FINAL:.6f}")
print(f"  Accepted pixels ({len(accepted)}): {accepted}")
print(f"  Rejected pixels ({len(rejected)}): {rejected}")
print(f"  #647 delta_negR (incremental): {decisions[3]['delta_negR']:.6e}")
print(f"  TOP_THR: {TOP_THR:.6e}")
print()

# ── Load baseline (warmstart) params ──────────────────────────────────────
SEED, PORT, WL = 7, 1, 1550.0
baseline_params = load_warmstart_params(SEED)
baseline_binary = (np.asarray(baseline_params, dtype=float).reshape(-1) > 0.5).astype(float)
n_pixels = len(baseline_binary)
print(f"  Warmstart params loaded: seed={SEED}, port={PORT}, wl={WL}nm")
print(f"  Binary pixel count: {n_pixels}")

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Fresh-sim final structure verification
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 1: Fresh-sim final structure")
print("=" * 70)

# Build final structure: baseline + all accepted pixels flipped
final_params = baseline_binary.copy()
for idx in accepted:
    final_params[idx] = 1.0 - final_params[idx]  # toggle

# Count flips
n_flips = sum(1 for i in range(n_pixels) if final_params[i] != baseline_binary[i])
print(f"  Flips from baseline: {n_flips}")
print(f"  Accepted pixel locations (ix, iy):")
for idx in accepted:
    ix, iy = divmod(idx, 100)
    print(f"    #{idx} = ({ix}, {iy})")

sim = None
try:
    sim = _build_3d_sim(tag="fresh_verify_final")
    _apply_params_to_design(sim.fdtd, final_params)
    print("  Running fresh FDTD (final structure)...")
    t0 = time.time()
    sim.fdtd.run()
    print(f"  FDTD done in {time.time()-t0:.1f}s")

    audit = audit_3d_modes(sim.fdtd, output_ports=3)
    R_fresh_raw = audit["R_in_modal"]
    R_fresh = abs(R_fresh_raw)  # sign convention: fresh-sim may return negative T_backward
    T_fresh = audit["T_fwd_sum"]
    print(f"  Fresh-sim R_modal_in = {R_fresh_raw:.8f} (abs={R_fresh:.8f})")
    print(f"  Fresh-sim T_fwd_sum  = {T_fresh:.8f}")

    dR = abs(R_fresh - R_INCR_FINAL)
    dT = abs(T_fresh - T_INCR_FINAL)
    r_pass = dR < 1e-4
    t_pass = dT < 1e-4
    print(f"  |R_fresh - R_incr| = {dR:.2e}  {'PASS' if r_pass else 'FAIL'} (threshold: 1e-4)")
    print(f"  |T_fresh - T_incr| = {dT:.2e}  {'PASS' if t_pass else 'FAIL'} (threshold: 1e-4)")

    # Also check: is R_fresh significantly below baseline? (sanity check)
    r_improvement = R_INCR_BASELINE - R_fresh
    print(f"  R improvement vs baseline: {r_improvement:.6e}  ({r_improvement/R_INCR_BASELINE*100:.1f}%)")

    exp1_pass = r_pass and t_pass
finally:
    if sim:
        _close_sim(sim)
        _sweep_stray_engines()
        sim = None

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Fresh-sim #647 single-flip verification
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 2: Fresh-sim #647 reject-flip verification")
print("=" * 70)

# Pre-647 structure: baseline + accepted pixels BEFORE #647 in the sequence
# Sequence: #548 accept, #7631 accept, #449 accept, #647 REJECT, #7731 accept
pre_647_accepted = [548, 7631, 449]  # pixels accepted before #647
IDX_647 = 647

pre_647_params = baseline_binary.copy()
for idx in pre_647_accepted:
    pre_647_params[idx] = 1.0 - pre_647_params[idx]

ix_647, iy_647 = divmod(IDX_647, 100)
print(f"  Pre-647 accepted pixels: {pre_647_accepted}")
print(f"  #647 location: ({ix_647}, {iy_647})")
print(f"  #647 current state in pre-647 params: {'Si' if pre_647_params[IDX_647] > 0.5 else 'SiO2'}")
print(f"  Flipping #647 to: {'Si' if pre_647_params[IDX_647] < 0.5 else 'SiO2'}")

# Step A: measure baseline R on pre-647 structure
sim = None
try:
    sim = _build_3d_sim(tag="fresh_verify_pre647")
    _apply_params_to_design(sim.fdtd, pre_647_params)
    print("  Running fresh FDTD (pre-647 structure)...")
    t0 = time.time()
    sim.fdtd.run()
    print(f"  FDTD done in {time.time()-t0:.1f}s")

    audit_pre = audit_3d_modes(sim.fdtd, output_ports=3)
    R_pre = abs(audit_pre["R_in_modal"])
    T_pre = audit_pre["T_fwd_sum"]
    print(f"  Pre-647 R_modal_in = {audit_pre['R_in_modal']:.8f} (abs={R_pre:.8f})")
    print(f"  Pre-647 T_fwd_sum  = {T_pre:.8f}")

    # Step B: flip #647 and re-measure
    to_si = pre_647_params[IDX_647] < 0.5
    print(f"  Flipping #647 to {'Si' if to_si else 'SiO2'} (fresh-sim toggle)...")
    sim.fdtd.switchtolayout()  # must exit analysis mode before geometry edit
    _flip_single_pixel(sim.fdtd, IDX_647, to_si=to_si)
    print("  Re-running FDTD (post-647 flip)...")
    t0 = time.time()
    sim.fdtd.run()
    print(f"  FDTD done in {time.time()-t0:.1f}s")

    audit_post = audit_3d_modes(sim.fdtd, output_ports=3)
    R_post = abs(audit_post["R_in_modal"])
    T_post = audit_post["T_fwd_sum"]
    delta_negR = R_pre - R_post  # positive = improvement (R decreased)

    print(f"  Post-647 R_modal_in = {R_post:.8f}")
    print(f"  Post-647 T_fwd_sum  = {T_post:.8f}")
    print(f"  Δ(-R) = R_pre - R_post = {delta_negR:.6e}")
    print(f"  Incremental Δ(-R)    = {decisions[3]['delta_negR']:.6e}")
    print(f"  Incremental pred_gain = {decisions[3]['pred_gain']:.6e}")

    # Verdict criteria
    sign_ok = delta_negR < -1e-6  # must be negative (worsens R)
    magnitude_ok = delta_negR < -TOP_THR  # must be below BOT_THR (i.e., worse than null)
    reject_upheld = sign_ok  # the key criterion: reject verdict confirmed

    print(f"  Sign correct (negative): {sign_ok}  {'PASS' if sign_ok else 'FAIL'}")
    print(f"  Below -TOP_THR ({-TOP_THR:.6e}): {magnitude_ok}  {'PASS' if magnitude_ok else 'WARN'}")
    print(f"  Incremental reject verdict upheld: {'YES' if reject_upheld else 'NO — WOULD CHANGE PAPER'}")

    exp2_pass = reject_upheld
finally:
    if sim:
        _close_sim(sim)
        _sweep_stray_engines()
        sim = None

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)
print(f"  Experiment 1 (final structure): {'PASS' if exp1_pass else 'FAIL'}")
print(f"  Experiment 2 (#647 reject flip): {'PASS' if exp2_pass else 'FAIL'}")
print()

if exp1_pass and exp2_pass:
    print("Both experiments PASS. Methods can state:")
    print('  "The final device state and the key rejected pixel (#647) were')
    print('   verified in independent fresh-session FDTD runs. The incremental')
    print('   measurements agree with fresh-sim to |ΔR| < 1e-4 for the final')
    print('   state, and the reject verdict for #647 (Δ(-R) < 0) is confirmed.')
    print('   IncrementalSession is therefore a validated evaluation tool for')
    print('   this device class, not merely an engineering accelerator."')
else:
    print("WARNING: One or both experiments FAILED. Do not claim fresh-sim")
    print("verification in the manuscript until the discrepancy is understood.")
    print("Possible causes: session state bleed, warmstart loading differences,")
    print("or actual disagreement between incremental and fresh measurement chains.")
