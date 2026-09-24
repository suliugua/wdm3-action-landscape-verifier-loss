"""3D WDM3 verification line — independent of the 2D mainline.

This package verifies that 3D SOI FDTD resolves the 2D port mode-expansion
hard limit, then unlocks PortTransmission and negR true-jac.

Phases:
  T3D-1: Power vs Modal Audit
  T3D-2: Single-Target Formal Readout
  T3D-3: Directional FD Validation
  T3D-4: Adjoint Validation
  3b:    PortTransmission integration (after T3D-4)
  3c:    negR true-jac (after 3b)
"""
