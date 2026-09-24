"""3D mode expansion readout and modal metrics.

Uses profile monitor + mode expansion — the same pattern as 2D wdm3_audit.py,
adapted for 3D geometry (z span added to monitor properties).

Direct T_forward from Lumerical expansion avoids the N≈0 issue in 3D.
"""

import numpy as np


def _ensure_mode_expansion_3d(fdtd, monitor_name, me_name, mode_number=1):
    """Idempotent mode expansion creation on a 3D profile monitor.

    Mirrors wdm3_audit.py:_ensure_mode_expansion, adds z/z span propagation.
    """
    if fdtd.getnamednumber(me_name) == 0:
        fdtd.switchtolayout()
        fdtd.setnamed(monitor_name, "override global monitor settings", False)
        fdtd.addmodeexpansion()
        fdtd.set("name", me_name)
        fdtd.setexpansion(me_name, monitor_name)
        fdtd.setnamed(me_name, "auto update before analysis", True)
        fdtd.setnamed(me_name, "override global monitor settings", False)
        for prop_name in ["monitor type", "x", "y", "z", "y span", "z span"]:
            try:
                fdtd.setnamed(me_name, prop_name, fdtd.getnamed(monitor_name, prop_name))
            except Exception:
                pass
        fdtd.select(me_name)
        fdtd.setnamed(me_name, "mode selection", "user select")
        fdtd.updatemodes(mode_number)
        # Mode expansion added post-run — must re-run for results to populate
        return True  # caller must re-run
    return False  # expansion already existed, no re-run needed


def _read_modal_transmission_3d(fdtd, me_name):
    """Read modal transmission from a 3D mode expansion.

    Returns dict {wavelength_nm: T_forward} using the expansion's built-in
    T_forward field — no manual |a*sqrt(N)|^2 computation needed.
    """
    exp_result = f"expansion for {me_name}"
    if not fdtd.haveresult(me_name, exp_result):
        return {}

    data = fdtd.getresult(me_name, exp_result)

    # Direct T_forward from Lumerical (avoids N≈0 issue in 3D)
    if "T_forward" in data:
        wl_arr = np.asarray(data["lambda"]).flatten()
        tf_arr = np.asarray(data["T_forward"]).flatten()
        t_modal = {}
        for i, wl in enumerate(wl_arr):
            if i < len(tf_arr):
                t_modal[float(wl) * 1e9] = float(tf_arr[i])
        return t_modal

    # Fallback: compute from a and N
    wl_arr = np.asarray(data["lambda"]).flatten()
    a_arr = np.asarray(data["a"]).flatten()
    N_arr = np.asarray(data["N"]).real.flatten()
    t_modal = {}
    for i, wl in enumerate(wl_arr):
        if i < len(a_arr):
            a_c = a_arr[i] * np.sqrt(max(N_arr[i], 0.0))
            t_modal[float(wl) * 1e9] = float(np.abs(a_c) ** 2)
    return t_modal


def _read_expansion_coefficient_3d(fdtd, me_name):
    """Read the complex mode expansion coefficient 'a'.

    Returns dict with Re_a, Im_a, abs2, T_forward, T_backward, N.
    """
    exp_result = f"expansion for {me_name}"
    if not fdtd.haveresult(me_name, exp_result):
        return {"Re_a": float("nan"), "Im_a": float("nan"), "abs2": float("nan"),
                "T_forward": float("nan"), "T_backward": float("nan"),
                "N": float("nan")}

    data = fdtd.getresult(me_name, exp_result)
    a_arr = np.asarray(data["a"]).flatten()
    N_arr = np.asarray(data["N"]).real.flatten()

    a_c = complex(a_arr[0]) if len(a_arr) > 0 else complex(0, 0)
    N_v = float(N_arr[0]) if len(N_arr) > 0 else float("nan")

    # Read b coefficient (backward-going wave; only nonzero at input ports)
    b_c = complex(0, 0)
    if "b" in data:
        b_arr = np.asarray(data["b"]).flatten()
        b_c = complex(b_arr[0]) if len(b_arr) > 0 else complex(0, 0)

    tf = float("nan")
    if "T_forward" in data:
        tf_arr = np.asarray(data["T_forward"]).flatten()
        tf = float(tf_arr[0]) if len(tf_arr) > 0 else float("nan")

    tb = float("nan")
    if "T_backward" in data:
        tb_arr = np.asarray(data["T_backward"]).flatten()
        tb = float(tb_arr[0]) if len(tb_arr) > 0 else float("nan")

    return {
        "Re_a": float(np.real(a_c)),
        "Im_a": float(np.imag(a_c)),
        "abs2": float(np.abs(a_c) ** 2),
        "Re_b": float(np.real(b_c)),
        "Im_b": float(np.imag(b_c)),
        "abs2_b": float(np.abs(b_c) ** 2),
        "T_forward": tf,
        "T_backward": tb,
        "N": N_v,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Aggregate audit (T3D-1)
# ═══════════════════════════════════════════════════════════════════════════

def audit_3d_modes(fdtd, output_ports=3):
    """Full 3D mode expansion audit on all output and input profile monitors.

    Returns dict with per-port T_forward/abs2/Re_a, total T_fwd_sum,
    and input reflection (R_in_modal from backward mode).
    """
    per_port = {}
    per_port_raw = {}

    T_fwd_sum = 0.0
    needs_rerun = False
    for i in range(1, output_ports + 1):
        prof_name = f"P{i}_prof"
        me_name = f"P{i}_prof_me"
        if _ensure_mode_expansion_3d(fdtd, prof_name, me_name, mode_number=1):
            needs_rerun = True

    # Input reflection — backward mode coefficient (skip if monitor absent, e.g. probe base)
    try:
        if _ensure_mode_expansion_3d(fdtd, "P_in_prof", "P_in_prof_me", mode_number=1):
            needs_rerun = True
    except Exception:
        pass

    if needs_rerun:
        fdtd.run()

    for i in range(1, output_ports + 1):
        coeff = _read_expansion_coefficient_3d(fdtd, f"P{i}_prof_me")
        per_port_raw[i] = coeff
        per_port[i] = {
            "abs2": coeff["abs2"],
            "T_forward": coeff["T_forward"],
            "T_backward": coeff["T_backward"],
            "Re_a": coeff["Re_a"],
            "N": coeff["N"],
        }
        if not np.isnan(coeff["T_forward"]):
            T_fwd_sum += coeff["T_forward"]

    # Read input reflection
    try:
        in_coeff = _read_expansion_coefficient_3d(fdtd, "P_in_prof_me")
        R_in_modal = in_coeff["T_backward"] if not np.isnan(in_coeff["T_backward"]) else float("nan")
    except Exception:
        in_coeff = {"T_backward": float("nan")}
        R_in_modal = float("nan")

    return {
        "per_port": per_port,
        "per_port_raw": per_port_raw,
        "T_fwd_sum": T_fwd_sum,
        "R_in_modal": R_in_modal,
        "in_coeff": in_coeff,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Formal target readout (T3D-2)
# ═══════════════════════════════════════════════════════════════════════════

def read_formal_target_3d(fdtd, target_port):
    """Read formal target quantities for a single output port via mode expansion.

    Returns dict with: Re_a, Im_a, abs2, T_forward, T_backward, N, target_port.
    """
    prof_name = f"P{target_port}_prof"
    me_name = f"P{target_port}_prof_me"
    needs_rerun = _ensure_mode_expansion_3d(fdtd, prof_name, me_name, mode_number=1)
    if needs_rerun:
        fdtd.run()
    coeff = _read_expansion_coefficient_3d(fdtd, me_name)
    coeff["target_port"] = target_port
    return coeff
