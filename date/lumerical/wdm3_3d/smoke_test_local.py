"""Minimal 3D port smoke test — local 2024 R1 environment.

Tests: can 3D FDTD ports return mode-expansion a/b/N?
This is the single most critical question for the entire 3D line.
"""

import os, sys, subprocess, tempfile, time, json
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import lum_backend  # noqa: F401 — 2024 R1 local
import _lumopt_compat  # noqa: F401
import lumapi
import numpy as np

WL_NM = 1550.0
SI_THICK_NM = 220


def _m(v):
    return v * 1e-9


# ── Build minimal 3D straight waveguide with 2 ports ──
def build_probe():
    """Build 3D probe via BaseScript — same pattern as 2D _build_sim()."""
    from lumopt.utilities.base_script import BaseScript
    from lumopt.utilities.simulation import Simulation

    wd = tempfile.mkdtemp(prefix="_lum3d_smoke_")
    sim = Simulation(wd, use_var_fdtd=False, hide_fdtd_cad=True)
    sim.fdtd.save(str(Path(wd) / "probe.fsp"))

    def _base(fdtd):
        domain_x, domain_y = 3000, 2000
        wg_len = 2000
        wg_w = 400
        wg_y = domain_y / 2

        fdtd.addfdtd()
        fdtd.set("dimension", "3D")
        fdtd.set("x", _m(domain_x / 2))
        fdtd.set("x span", _m(domain_x))
        fdtd.set("y", _m(domain_y / 2))
        fdtd.set("y span", _m(domain_y))
        fdtd.set("z", _m(SI_THICK_NM / 2))
        fdtd.set("z span", _m(1200))
        fdtd.set("background material", "SiO2 (Glass) - Palik")

        fdtd.setglobalmonitor("use linear wavelength spacing", 1)
        fdtd.setglobalmonitor("wavelength center", _m(WL_NM))
        fdtd.setglobalmonitor("wavelength span", _m(0))
        fdtd.setglobalmonitor("frequency points", 1)

        fdtd.addmesh()
        fdtd.set("name", "mesh")
        fdtd.set("dx", _m(40)); fdtd.set("dy", _m(40)); fdtd.set("dz", _m(30))
        fdtd.set("override x mesh", 1)
        fdtd.set("override y mesh", 1)
        fdtd.set("override z mesh", 1)

        fdtd.addrect()
        fdtd.set("name", "wg")
        fdtd.set("x", _m(domain_x / 2)); fdtd.set("x span", _m(wg_len))
        fdtd.set("y", _m(wg_y)); fdtd.set("y span", _m(wg_w))
        fdtd.set("z", _m(SI_THICK_NM / 2)); fdtd.set("z span", _m(SI_THICK_NM))
        fdtd.set("material", "Si (Silicon) - Palik")

        fdtd.addport()
        fdtd.set("name", "port_in"); fdtd.set("injection axis", "x-axis")
        fdtd.set("direction", "Forward")
        fdtd.set("x", _m(600)); fdtd.set("y", _m(wg_y))
        fdtd.set("y span", _m(wg_w * 3))
        fdtd.set("z", _m(SI_THICK_NM / 2)); fdtd.set("z span", _m(SI_THICK_NM * 2))

        fdtd.addport()
        fdtd.set("name", "port_out"); fdtd.set("injection axis", "x-axis")
        fdtd.set("direction", "Forward")
        fdtd.set("x", _m(domain_x - 600)); fdtd.set("y", _m(wg_y))
        fdtd.set("y span", _m(wg_w * 3))
        fdtd.set("z", _m(SI_THICK_NM / 2)); fdtd.set("z span", _m(SI_THICK_NM * 2))

        fdtd.addmode()
        fdtd.set("name", "source"); fdtd.set("injection axis", "x-axis")
        fdtd.set("direction", "Forward")
        fdtd.set("x", _m(500)); fdtd.set("y", _m(wg_y))
        fdtd.set("y span", _m(wg_w * 3))
        fdtd.set("z", _m(SI_THICK_NM / 2)); fdtd.set("z span", _m(SI_THICK_NM * 2))
        fdtd.set("wavelength start", _m(WL_NM))
        fdtd.set("wavelength stop", _m(WL_NM))
        fdtd.set("mode selection", "fundamental TE mode")

    BaseScript(_base).eval(sim.fdtd)
    return sim


def update_port_modes(fdtd, port_names):
    errors = []
    for pn in port_names:
        try:
            fdtd.eval(
                f"groupscope('FDTD::ports'); "
                f"select('{pn}'); "
                "updateportmodes(1); "
                "groupscope('::model');"
            )
        except Exception as exc:
            errors.append({"port": pn, "error": str(exc)})
    return errors


def probe_port(fdtd, port_name):
    """Try to read every result type from a port."""
    result_names = [
        "neff", "T", "S", "expansion for port monitor",
        "mode profiles", "T_in", "T_out", "a", "b", "N", "P",
    ]
    path = f"FDTD::ports::{port_name}"
    findings = {}
    for rn in result_names:
        try:
            raw = fdtd.getresult(path, rn)
            if raw is None:
                findings[rn] = {"ok": False, "reason": "None"}
            elif isinstance(raw, dict):
                keys = sorted(k for k in raw if k != "Lumerical_dataset")
                findings[rn] = {"ok": True, "keys": keys}
                # Extract scalar samples
                for k in keys[:3]:
                    arr = np.asarray(raw[k]).flatten()
                    if len(arr) > 0:
                        findings[rn][f"sample_{k}"] = (
                            float(np.real(arr[0])),
                            float(np.imag(arr[0])) if np.iscomplexobj(arr) else float(arr[0]),
                        )
            else:
                arr = np.asarray(raw).flatten()
                findings[rn] = {"ok": True, "shape": str(arr.shape), "sample": float(arr[0]) if len(arr) > 0 else None}
        except Exception as exc:
            findings[rn] = {"ok": False, "reason": str(exc)[:120]}
    return findings


# ── Main ──
print("=" * 60)
print("3D Port Smoke Test — straight waveguide, 2 ports")
print(f"Lumerical backend: {lum_backend.LUMOPT_BACKEND}")
print("=" * 60)

t0 = time.time()
sim = None

try:
    sim = build_probe()
    fdtd = sim.fdtd
    print("\n[1] Running 3D FDTD simulation ...")
    fdtd.run()
    print(f"    done ({time.time() - t0:.0f}s)")

    print("\n[2] Updating port modes ...")
    port_names = ["port_in", "port_out"]
    errors = update_port_modes(fdtd, port_names)
    if errors:
        for e in errors:
            print(f"    ERROR [{e['port']}]: {e['error']}")
    else:
        print("    OK — no errors")

    print("\n[3] Probing port results ...")
    results = {}
    for pn in port_names:
        results[pn] = probe_port(fdtd, pn)

    for pn in port_names:
        r = results[pn]
        print(f"\n  --- {pn} ---")
        for rn, info in sorted(r.items()):
            if info["ok"]:
                keys_str = ", ".join(info.get("keys", [])[:5])
                extras = []
                for k in info:
                    if k.startswith("sample_"):
                        extras.append(f"{k}={info[k]}")
                extra_str = " | ".join(extras[:2]) if extras else ""
                print(f"    [OK]  {rn:35s}  keys=[{keys_str}]  {extra_str}")
            else:
                print(f"    [--]  {rn:35s}  {info.get('reason', '?')}")

    # ── Quick verdict ──
    print("\n" + "=" * 60)
    expansion_ok = {}
    for pn in port_names:
        exp = results[pn].get("expansion for port monitor", {})
        expansion_ok[pn] = exp.get("ok", False)

    a_readable = results["port_out"].get("a", {}).get("ok", False)
    b_readable = results["port_in"].get("b", {}).get("ok", False)

    print(f"Port expansion readable: {expansion_ok}")
    print(f"  port_out 'a' readable: {a_readable}")
    print(f"  port_in  'b' readable: {b_readable}")

    if all(expansion_ok.values()):
        print("\n>>> VERDICT: 3D port mode-expansion WORKS on 2024 R1 local <<<")
        print("    T3D-1 can proceed on server with 2026 R1.")
    elif any(expansion_ok.values()):
        print("\n>>> VERDICT: Partial success — some ports readable <<<")
    else:
        print("\n>>> VERDICT: 3D port mode-expansion FAILED on 2024 R1 local <<<")
        print("    This matches 2D behavior — ports may need 2026 R1 or server environment.")

    # Save results
    out = Path("wdm3_3d/results_3d") / "smoke_test_local.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "backend": lum_backend.LUMOPT_BACKEND,
            "wall_s": time.time() - t0,
            "expansion_ok": expansion_ok,
            "a_readable": a_readable,
            "b_readable": b_readable,
        }, f, indent=2)
    print(f"\nResults saved to {out}")

finally:
    if sim is not None:
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

print(f"\nTotal wall time: {time.time() - t0:.0f}s")
