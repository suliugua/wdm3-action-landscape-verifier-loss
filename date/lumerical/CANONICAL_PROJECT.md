# Canonical project

## There is no persistent .fsp source of truth

The device is **built from source**, not opened from a stored Lumerical project
file. The scripts under `lumerical/wdm3_3d/` construct the simulation
programmatically (geometry, monitors, ports, mesh) and write the results as
JSON. A `.fsp` file may be produced as a by-product of a run, but it is never
the artefact of record, and nothing in the analysis chain reads one back.

Consequences worth stating, because they are what makes the package
reproducible at all:

* the geometry is fixed by code, so a run is reproducible from the package
  without shipping a binary project file;
* the figures and tables are computed from the JSON the runs emit, so the
  package's `probe_records/` and `fresh_records/` are the primary evidence;
* the device is not edited by hand between runs.

## Simulator versions

| Axis | Version | Notes |
|---|---|---|
| 3D FDTD (the results in the paper) | Ansys Lumerical **2026 R1** | selected by `LUMOPT_BACKEND`; the run log prints the resolved API path |
| 2D FDTD | Ansys Lumerical **2024 R1** | used by earlier Paper-1 work, not by the results here |

The 3D adjoint is evaluated by a **manual two-run single-frequency protocol**
rather than by a built-in sweep, because the adjoint solve is run by hand and
the two frequencies are not exposed as one call.

## Device

Three-channel silicon-on-insulator WDM multiplexer, one input port and three
output monitors (P1--P3), an 80x100 binary pixel region, operated at 1550 nm on
port P1. Anchors are different random initializations of this one topology and
therefore different landscape states, not different devices.
