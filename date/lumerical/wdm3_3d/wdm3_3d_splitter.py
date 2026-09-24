"""3D SOI 1×2 splitter — reusing WDM3 simulation skeleton.

Uses WDM3's 3-port geometry but routes light to exactly 2 outputs
(P1 at y=1400, P3 at y=3720), leaving P2 unused.  Same platform:
3D FDTD, SOI 220nm, 80×100@40nm, modal T/R, IncrementalSession.

Splitter objective: J = T1 + T3 - lambda*R_in - mu*|T1-T3|
"""
import numpy as np
from wdm3_3d.wdm3_3d_device import (
    DESIGN_X_NM, DESIGN_Y_NM, NX_2D, NY_2D, INPUT_WG_Y_NM, WG_W_NM, WL_NM,
)

# ── Splitter port config ─────────────────────────────────────────────────
# Use WDM3 P1 (y=1400) and P3 (y=3720) as the two splitter outputs.
# P2 (y=2000) is unused.
SPLIT_OUTPUT_PORTS = (1, 3)   # WDM3 port indices
SPLIT_OUTPUT_YS = (1400, 3720)  # corresponding y positions
SPLIT_Y_MID = (1400 + 3720) / 2  # 2560 = input y — symmetric!

# Device acceptor weights (pre-registered, frozen)
LAMBDA_R = 1.0
MU_BAL = 1.0


def make_splitter_initial():
    """1×2 splitter: input (y=2560) → P1 (y=1400) + P3 (y=3720).

    Uses cosine S-bends from center to each output.  Wide waveguides
    (600nm half-width) compensate for 40nm grid scattering.
    Returns binary np.array of shape (8000,).
    """
    dy = (DESIGN_Y_NM[1] - DESIGN_Y_NM[0]) / NY_2D
    yc = DESIGN_Y_NM[0] + dy * (np.arange(NY_2D) + 0.5)
    p = np.zeros((NX_2D, NY_2D))
    wg_hw = 300.0  # 600nm half-width = 15px

    for ix in range(NX_2D):
        frac = ix / NX_2D
        # Cosine split: start at center, end at each output
        y1 = INPUT_WG_Y_NM + (SPLIT_OUTPUT_YS[0] - INPUT_WG_Y_NM) * (1 - np.cos(np.pi * frac)) / 2
        y2 = INPUT_WG_Y_NM + (SPLIT_OUTPUT_YS[1] - INPUT_WG_Y_NM) * (1 - np.cos(np.pi * frac)) / 2
        mask1 = np.abs(yc - y1) < wg_hw
        mask2 = np.abs(yc - y2) < wg_hw
        p[ix, :] = (mask1 | mask2).astype(float)

    return p.flatten()


def compute_splitter_objective(T1, T3, R_in):
    """J_dev = T1 + T3 - lambda*R_in - mu*|T1-T3|."""
    bal = abs(T1 - T3)
    return T1 + T3 - LAMBDA_R * R_in - MU_BAL * bal, bal


def read_splitter_metrics_from_audit(audit_result):
    """Extract splitter-specific metrics from audit_3d_modes output."""
    pp = audit_result["per_port"]
    T1 = pp.get(1, {}).get("T_forward", 0.0) or 0.0
    T3 = pp.get(3, {}).get("T_forward", 0.0) or 0.0
    R_in = audit_result.get("R_in_modal", float("nan"))
    if np.isnan(R_in):
        R_in = 0.0
    J_dev, bal = compute_splitter_objective(T1, T3, R_in)
    return {
        "T1": float(T1), "T3": float(T3),
        "T_sum": float(T1 + T3), "balance": float(bal),
        "R_in": float(R_in), "J_dev": float(J_dev),
    }
