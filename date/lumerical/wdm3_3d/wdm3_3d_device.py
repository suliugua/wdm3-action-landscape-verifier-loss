"""3D WDM3 device geometry and simulation builder.

SOI platform: 220nm Si on SiO2, air cladding.
Fixed-structure verification — no TopologyOptimization (that comes in 3b).

Reuses patterns from:
  - wdm3_device.py:_build_base() — geometry + monitor setup
  - lumerical_wdm3_3d.py:_build_3d_base() — 3D port + mode source
"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# Geometry constants
# ═══════════════════════════════════════════════════════════════════════════

DOMAIN_X_NM   = 6000
DOMAIN_Y_NM   = 6000
SI_THICK_NM   = 220
Z_SPAN_NM     = 2000

# Design region must match 2D pixel grid (80×100 @ 40nm) for warmstart compatibility.
# 3D ports use different Y positions than 2D — the extruded structure may not align
# optimally, but T3D-1 C4 only tests port readout, not device performance.
DESIGN_X_NM   = (1000, 4200)   # span = 3200nm → 80px × 40nm  (matches 2D)
DESIGN_Y_NM   = (560, 4560)    # span = 4000nm → 100px × 40nm (matches 2D)

WG_W_NM       = 400
INPUT_WG_Y_NM = 2560            # matches 2D layout for warmstart alignment
OUTPUT_WG_YS  = (1400, 2000, 3720)  # matches 2D: P1, P2, P3

MAT_SI   = "Si (Silicon) - Palik"
MAT_SIO2 = "SiO2 (Glass) - Palik"
WL_NM    = 1550.0  # single-wavelength for smoke

# Lumopt-compatible permittivity and filter constants (mirrors wdm3_device.py)
EPS_MIN, EPS_MAX = 1.44**2, 3.48**2
FILTER_R_NM = 80.0


def _m(v):
    """nm → m."""
    return v * 1e-9


def _close_sim(sim):
    """Kill the FDTD process tree, then close the Python handle."""
    try:
        pid = int(sim.fdtd.getprocessid())
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, timeout=30)
    except Exception:
        pass
    try:
        sim.fdtd.close()
    except Exception:
        pass


def _add_mode_expansion(fdtd, monitor_name, mode_number=1):
    """Add a mode expansion to a profile monitor (must be called BEFORE run())."""
    me_name = f"{monitor_name}_me"
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


# ═══════════════════════════════════════════════════════════════════════════
# Material fill helper
# ═══════════════════════════════════════════════════════════════════════════

def _build_3d_base(fdtd):
    """Build the complete 3D SOI WDM3 simulation.

    Monitors:
      - opt_fields: 3D field profile (for adjoint E_fwd/E_adj extraction)
      - P{i}_prof: 2D X-normal profile at each output waveguide (mode expansion)
      - P_in_prof: 2D X-normal profile at input (backward mode / reflection)
      - mode source at input waveguide
    """
    # ── FDTD region ──
    fdtd.addfdtd()
    fdtd.set("dimension", "3D")
    fdtd.set("x", _m(DOMAIN_X_NM / 2))
    fdtd.set("x span", _m(DOMAIN_X_NM))
    fdtd.set("y", _m(DOMAIN_Y_NM / 2))
    fdtd.set("y span", _m(DOMAIN_Y_NM))
    fdtd.set("z", _m(SI_THICK_NM / 2))
    fdtd.set("z span", _m(Z_SPAN_NM))
    fdtd.set("background material", MAT_SIO2)

    # ── Global monitor settings (single wavelength) ──
    fdtd.setglobalmonitor("use linear wavelength spacing", 1)
    fdtd.setglobalmonitor("wavelength center", _m(WL_NM))
    fdtd.setglobalmonitor("wavelength span", _m(0))
    fdtd.setglobalmonitor("frequency points", 1)

    # ── Mesh override ──
    fdtd.addmesh()
    for k, v in [("name", "wg_mesh"),
                 ("dx", _m(40)), ("dy", _m(40)), ("dz", _m(30)),
                 ("override x mesh", 1), ("override y mesh", 1), ("override z mesh", 1)]:
        fdtd.set(k, v)

    # ── Helper: add a waveguide rectangle ──
    def _wg_rect(name, xc, yc, xs, ys):
        fdtd.addrect()
        for k, v in [("name", name),
                     ("x", _m(xc)), ("x span", _m(xs)),
                     ("y", _m(yc)), ("y span", _m(ys)),
                     ("z", _m(SI_THICK_NM / 2)), ("z span", _m(SI_THICK_NM)),
                     ("material", MAT_SI)]:
            fdtd.set(k, v)

    # ── Helper: add a 2D X-normal profile monitor ──
    def _prof_mon(name, x_nm, y_nm):
        fdtd.addprofile()
        for k, v in [("name", name), ("monitor type", "2D X-normal"),
                     ("x", _m(x_nm)), ("y", _m(y_nm)),
                     ("y span", _m(WG_W_NM * 3)),
                     ("z", _m(SI_THICK_NM / 2)), ("z span", _m(SI_THICK_NM * 2)),
                     ("override global monitor settings", 1),
                     ("use linear wavelength spacing", 1),
                     ("wavelength center", _m(WL_NM)),
                     ("wavelength span", _m(0)),
                     ("frequency points", 1)]:
            fdtd.set(k, v)

    # ── Input waveguide ──
    _wg_rect("wg_in", DESIGN_X_NM[0] / 2, INPUT_WG_Y_NM,
             DESIGN_X_NM[0], WG_W_NM)

    # ── Output waveguides ──
    out_len = DOMAIN_X_NM - DESIGN_X_NM[1]
    out_x   = (DESIGN_X_NM[1] + DOMAIN_X_NM) / 2
    for i, wy in enumerate(OUTPUT_WG_YS):
        _wg_rect(f"wg_out_{i+1}", out_x, wy, out_len, WG_W_NM)

    # ── Output profile monitors (for mode expansion / formal readout) ──
    prof_x = DESIGN_X_NM[1] + 100
    for i, wy in enumerate(OUTPUT_WG_YS):
        _prof_mon(f"P{i+1}_prof", prof_x, wy)

    # ── Input profile monitor (for backward mode / reflection readout) ──
    _prof_mon("P_in_prof", DESIGN_X_NM[0] - 200, INPUT_WG_Y_NM)

    # ── Pre-create mode expansions on all profile monitors ──
    for prof_name in ["P1_prof", "P2_prof", "P3_prof", "P_in_prof"]:
        _add_mode_expansion(fdtd, prof_name)

    # ── Mode source ──
    fdtd.addmode()
    for k, v in [("name", "source"),
                 ("injection axis", "x-axis"),
                 ("direction", "Forward"),
                 ("x", _m(DESIGN_X_NM[0] - 100)),
                 ("y", _m(INPUT_WG_Y_NM)),
                 ("y span", _m(WG_W_NM * 3)),
                 ("z", _m(SI_THICK_NM / 2)),
                 ("z span", _m(SI_THICK_NM * 2)),
                 ("wavelength start", _m(WL_NM)),
                 ("wavelength stop", _m(WL_NM)),
                 ("mode selection", "fundamental TE mode")]:
        fdtd.set(k, v)

    # ── 3D field monitor (for adjoint E_fwd/E_adj extraction) ──
    fdtd.addprofile()
    fdtd.set("name", "opt_fields")
    fdtd.set("monitor type", "3D")
    fdtd.set("x", _m(DOMAIN_X_NM / 2))
    fdtd.set("x span", _m(DOMAIN_X_NM))
    fdtd.set("y", _m(DOMAIN_Y_NM / 2))
    fdtd.set("y span", _m(DOMAIN_Y_NM))
    fdtd.set("z", _m(SI_THICK_NM / 2))
    fdtd.set("z span", _m(SI_THICK_NM))
    fdtd.set("override global monitor settings", 1)
    fdtd.set("use linear wavelength spacing", 1)
    fdtd.set("wavelength center", _m(WL_NM))
    fdtd.set("wavelength span", _m(0))
    fdtd.set("frequency points", 1)

    # ── Design region placeholder (SiO2 fill, replaced per test case) ──
    fdtd.addrect()
    fdtd.set("name", "design_region")
    fdtd.set("x", _m((DESIGN_X_NM[0] + DESIGN_X_NM[1]) / 2))
    fdtd.set("x span", _m(DESIGN_X_NM[1] - DESIGN_X_NM[0]))
    fdtd.set("y", _m((DESIGN_Y_NM[0] + DESIGN_Y_NM[1]) / 2))
    fdtd.set("y span", _m(DESIGN_Y_NM[1] - DESIGN_Y_NM[0]))
    fdtd.set("z", _m(SI_THICK_NM / 2))
    fdtd.set("z span", _m(SI_THICK_NM))
    fdtd.set("material", MAT_SIO2)


# ═══════════════════════════════════════════════════════════════════════════
# Probe base — minimal 2-port straight waveguide for T3D-1 C1
# ═══════════════════════════════════════════════════════════════════════════

def _build_3d_probe_base(fdtd):
    """Minimal straight-waveguide 3D setup for fast port result enumeration."""
    domain_x_nm = 3200
    domain_y_nm = 2000
    probe_wg_x_nm = 2200
    probe_y_nm = domain_y_nm / 2

    fdtd.addfdtd()
    fdtd.set("dimension", "3D")
    fdtd.set("x", _m(domain_x_nm / 2))
    fdtd.set("x span", _m(domain_x_nm))
    fdtd.set("y", _m(domain_y_nm / 2))
    fdtd.set("y span", _m(domain_y_nm))
    fdtd.set("z", _m(SI_THICK_NM / 2))
    fdtd.set("z span", _m(1200))
    fdtd.set("background material", MAT_SIO2)

    fdtd.setglobalmonitor("use linear wavelength spacing", 1)
    fdtd.setglobalmonitor("wavelength center", _m(WL_NM))
    fdtd.setglobalmonitor("wavelength span", _m(0))
    fdtd.setglobalmonitor("frequency points", 1)

    fdtd.addmesh()
    for k, v in [("name", "probe_mesh"),
                 ("dx", _m(60)), ("dy", _m(40)), ("dz", _m(30)),
                 ("override x mesh", 1), ("override y mesh", 1), ("override z mesh", 1)]:
        fdtd.set(k, v)

    fdtd.addrect()
    for k, v in [("name", "probe_wg"),
                 ("x", _m(domain_x_nm / 2)), ("x span", _m(probe_wg_x_nm)),
                 ("y", _m(probe_y_nm)), ("y span", _m(WG_W_NM)),
                 ("z", _m(SI_THICK_NM / 2)), ("z span", _m(SI_THICK_NM)),
                 ("material", MAT_SI)]:
        fdtd.set(k, v)

    # Output profile monitor
    fdtd.addprofile()
    for k, v in [("name", "P1_prof"), ("monitor type", "2D X-normal"),
                 ("x", _m(domain_x_nm - 200)), ("y", _m(probe_y_nm)),
                 ("y span", _m(WG_W_NM * 3)),
                 ("z", _m(SI_THICK_NM / 2)), ("z span", _m(SI_THICK_NM * 2)),
                 ("override global monitor settings", 1),
                 ("use linear wavelength spacing", 1),
                 ("wavelength center", _m(WL_NM)),
                 ("wavelength span", _m(0)),
                 ("frequency points", 1)]:
        fdtd.set(k, v)

    # Pre-create mode expansion on output profile
    _add_mode_expansion(fdtd, "P1_prof")

    fdtd.addmode()
    for k, v in [("name", "source"),
                 ("injection axis", "x-axis"), ("direction", "Forward"),
                 ("x", _m(600)), ("y", _m(probe_y_nm)),
                 ("y span", _m(WG_W_NM * 3)),
                 ("z", _m(SI_THICK_NM / 2)), ("z span", _m(SI_THICK_NM * 2)),
                 ("wavelength start", _m(WL_NM)), ("wavelength stop", _m(WL_NM)),
                 ("mode selection", "fundamental TE mode")]:
        fdtd.set(k, v)

    # 3D field monitor for probe
    fdtd.addprofile()
    fdtd.set("name", "opt_fields")
    fdtd.set("monitor type", "3D")
    fdtd.set("x", _m(domain_x_nm / 2))
    fdtd.set("x span", _m(domain_x_nm))
    fdtd.set("y", _m(domain_y_nm / 2))
    fdtd.set("y span", _m(domain_y_nm))
    fdtd.set("z", _m(SI_THICK_NM / 2))
    fdtd.set("z span", _m(SI_THICK_NM))
    fdtd.set("override global monitor settings", 1)
    fdtd.set("use linear wavelength spacing", 1)
    fdtd.set("wavelength center", _m(WL_NM))
    fdtd.set("wavelength span", _m(0))
    fdtd.set("frequency points", 1)


# ═══════════════════════════════════════════════════════════════════════════
# Simulation builder
# ═══════════════════════════════════════════════════════════════════════════

def _retarget_wavelength(fdtd, wl_nm):
    """Re-point the single-wavelength source + monitors from WL_NM to *wl_nm*.

    _build_3d_base hardcodes WL_NM (1550 nm) in the forward source, opt_fields
    monitor, profile monitors, and global monitor settings.  This helper is
    called after the base sim is built so the whole FDTD pipeline (forward run,
    adjoint run, audit) can execute at a non-default wavelength.  The forward
    mode profile is recomputed at *wl_nm*; mode expansions auto-update before
    analysis (already enabled in _add_mode_expansion).
    """
    fdtd.setnamed("source", "wavelength start", _m(wl_nm))
    fdtd.setnamed("source", "wavelength stop", _m(wl_nm))
    fdtd.setglobalmonitor("wavelength center", _m(wl_nm))
    fdtd.setnamed("opt_fields", "wavelength center", _m(wl_nm))
    # Profile monitors (P1..P3_prof, P_in_prof) inherit the global monitor
    # wavelength (their "override global monitor settings" is False after
    # _add_mode_expansion), so their "wavelength center" is inactive and must
    # NOT be set directly.


def _build_3d_sim(tag="", probe=False, wl_nm=None):
    """Build a 3D Simulation object.

    Parameters
    ----------
    tag : str
        Disambiguation tag for the temp working directory.
    probe : bool
        If True, use the minimal probe geometry (T3D-1 C1).
    wl_nm : float | None
        If given, retarget the source + monitors to this wavelength (nm).
        Default None keeps the hardcoded WL_NM (1550 nm) behavior.

    Returns
    -------
    sim : lumopt.utilities.simulation.Simulation
    """
    from lumopt.utilities.base_script import BaseScript
    from lumopt.utilities.simulation import Simulation

    wd = str(Path(tempfile.gettempdir()) / f"_lum3d_{tag}_{os.getpid()}")
    os.makedirs(wd, exist_ok=True)
    sim = Simulation(wd, use_var_fdtd=False, hide_fdtd_cad=True)
    sim.fdtd.save(str(Path(wd) / "_init.fsp"))

    base_fn = _build_3d_probe_base if probe else _build_3d_base
    BaseScript(base_fn).eval(sim.fdtd)
    if wl_nm is not None:
        _retarget_wavelength(sim.fdtd, wl_nm)
    return sim


# ═══════════════════════════════════════════════════════════════════════════
# Warmstart extrusion helper
# ═══════════════════════════════════════════════════════════════════════════

# 2D pixel grid (must match wdm3_device.py for warmstart compatibility)
NX_2D, NY_2D = 80, 100
PIXEL_NM_2D = 40.0  # (DESIGN_X_NM[1]-DESIGN_X_NM[0])/NX for the 2D device

WARMSTART_DIR = Path(__file__).resolve().parent.parent / \
    "results_lumerical_wdm3_phase1_port_power_continuation"


def load_warmstart_params(seed):
    """Read the 2D warmstart continuation params for *seed* (no geometry side effects).

    Returns
    -------
    params_2d : np.ndarray  shape (NX_2D * NY_2D,)
    """
    import json

    cont_file = WARMSTART_DIR / f"port_power_continuation_seed{seed}.json"
    if not cont_file.exists():
        raise FileNotFoundError(
            f"Warmstart file not found: {cont_file}\n"
            f"Available seeds: {sorted([int(p.stem.replace('port_power_continuation_seed', '')) for p in WARMSTART_DIR.glob('*.json')])}"
        )

    with open(cont_file) as f:
        data = json.load(f)

    # The continuation result stores params under 'final_params' or 'params'.
    # Some files use 'last_params' or nested 'summary.final_params'.
    if "final_params" in data:
        params_2d = np.asarray(data["final_params"], dtype=float).flatten()
    elif "params" in data:
        params_2d = np.asarray(data["params"], dtype=float).flatten()
    elif "last_params" in data:
        params_2d = np.asarray(data["last_params"], dtype=float).flatten()
    elif "summary" in data and "final_params" in data["summary"]:
        params_2d = np.asarray(data["summary"]["final_params"], dtype=float).flatten()
    else:
        available = sorted(data.keys())
        raise KeyError(
            f"Cannot find params in warmstart file {cont_file.name}. "
            f"Available top-level keys: {available}"
        )

    if len(params_2d) != NX_2D * NY_2D:
        raise ValueError(
            f"Expected {NX_2D * NY_2D} params, got {len(params_2d)}. "
            f"2D pixel grid mismatch?"
        )

    return params_2d


def load_warmstart_extrusion(seed, fdtd):
    """Load 2D warmstart params and extrude uniformly along z into the 3D design region.

    Reads the 2D continuation result for *seed*, reshapes to (NX, NY),
    then places the same pattern at every z-slice in the 3D design region.

    Parameters
    ----------
    seed : int
        Warmstart seed index (0-14).
    fdtd : lumapi.FDTD handle
        An open FDTD session with _build_3d_base() already applied.

    Returns
    -------
    params_2d : np.ndarray  shape (NX_2D * NY_2D,)
        The loaded 2D parameters.
    """
    params_2d = load_warmstart_params(seed)

    # Reshape to 2D and extrude uniformly in z
    p2d = params_2d.reshape(NX_2D, NY_2D)

    # Place Si where p2d > 0.5, SiO2 otherwise — simple threshold extrusion
    # (no projection/filter needed for verification)
    dx = (DESIGN_X_NM[1] - DESIGN_X_NM[0]) / NX_2D
    dy = (DESIGN_Y_NM[1] - DESIGN_Y_NM[0]) / NY_2D

    for ix in range(NX_2D):
        for iy in range(NY_2D):
            if p2d[ix, iy] <= 0.5:
                continue  # SiO2 — already the background
            xc = DESIGN_X_NM[0] + (ix + 0.5) * dx
            yc = DESIGN_Y_NM[0] + (iy + 0.5) * dy
            fdtd.addrect()
            fdtd.set("name", f"ws_{ix}_{iy}")
            fdtd.set("x", _m(xc))
            fdtd.set("x span", _m(dx))
            fdtd.set("y", _m(yc))
            fdtd.set("y span", _m(dy))
            fdtd.set("z", _m(SI_THICK_NM / 2))
            fdtd.set("z span", _m(SI_THICK_NM))
            fdtd.set("material", MAT_SI)

    return params_2d


def fill_design_region(fdtd, material):
    """Fill the design region with *material* (for T3D-1 C2/C3)."""
    fdtd.select("design_region")
    fdtd.set("material", material)
