"""
Global configuration for Paper 2: landscape-gated performance optimization
"""

import os

# -- paths ----------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)  # parent of the project root
WDM3_DIR = os.path.join(PARENT_ROOT, "wdm3_3d")
RESULTS_BASE = os.path.join(PROJECT_ROOT, "results")

# -- run modes ------------------------------------------------------------
RUN_MODE = "protocol"  # "protocol" | "performance" | "selector"

def get_results_dir(mode: str = None) -> str:
    """Mode-specific results directory. Creates if needed."""
    m = mode or RUN_MODE
    d = os.path.join(RESULTS_BASE, m)
    os.makedirs(d, exist_ok=True)
    return d

# Backward-compat: default module-level RESULTS_DIR
RESULTS_DIR = get_results_dir()

# -- per-mode defaults ----------------------------------------------------
PROTOCOL_DEFAULTS = {
    "z_threshold": 2.0,
    "patch_sizes": [2, 3, 5],
    "patch_types": ["sparse", "dense"],
    "progressive_relaxation": True,
    "fresh_sim": True,
    # NOTE(2026-08-01): refresh_policy/refresh_every_N are DOCUMENTATION-ONLY.
    # No strategy function reads them. Actual behavior:
    #   single+patch (no +trajectory) → jac computed once, NEVER refreshed
    #   +trajectory → jac refreshed every round via _refresh_jac_inplace (ignores these fields)
    "refresh_policy": "every_accept",  # DOCUMENTATION-ONLY (see PROTOCOL_DEFAULTS note)
    "refresh_every_N": 1,
}

PERFORMANCE_DEFAULTS = {
    "z_threshold": 2.0,
    "patch_sizes": [2, 3, 5],
    "patch_types": ["sparse", "dense"],
    "progressive_relaxation": True,
    "fresh_sim": True,
    "refresh_policy": "every_accept",  # DOCUMENTATION-ONLY (see PROTOCOL_DEFAULTS note)
    "refresh_every_N": 1,
}

SELECTOR_DEFAULTS = {
    "z_threshold": 2.0,
    "patch_sizes": [2, 3, 5],
    "patch_types": ["sparse", "dense"],
    "progressive_relaxation": False,
    "fresh_sim": True,
    "refresh_policy": "none",
    "refresh_every_N": 0,
    "selector_top_m": 100,
}

# -- device ---------------------------------------------------------------
DEVICE = "wdm3"
PIXEL_GRID = (80, 100)  # ix (columns) × iy (rows)
PIXEL_SIZE_NM = 40.0
SI_INDEX = 3.476  # 1550nm
SIO2_INDEX = 1.444
WL_NM = 1550.0

# -- anchors --------------------------------------------------------------
ANCHORS = {
    "seed7": {
        "port": 1,
        "warmstart": "seed7_P1_warmstart",
        "mask_paper1": [(72, 79)],
        "variant_paper1": "conj_a",
        "baseline_abs2": 0.041841,
    },
    "seed10": {
        "port": 2,
        "warmstart": "seed10_P2_warmstart",
        "mask_paper1": [(0, 2)],
        "variant_paper1": "conj_a",
        "baseline_abs2": 0.038233,
    },
    "seed9": {
        "port": 1,
        "warmstart": "seed9_P1_warmstart",
        "mask_paper1": None,       # to be determined by Phase-0
        "variant_paper1": None,    # to be determined by Phase-0
        "baseline_abs2": 0.094415,  # from jac
    },
    "seed11": {
        "port": 1,
        "warmstart": "seed11_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": 0.036125,
    },
    "seed1": {
        "port": 1,
        "warmstart": "seed1_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    "seed13": {
        "port": 1,
        "warmstart": "seed13_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    "seed2": {
        "port": 1,
        "warmstart": "seed2_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    "seed3": {
        "port": 1,
        "warmstart": "seed3_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": 0.017420,
    },
    "seed4": {
        "port": 1,
        "warmstart": "seed4_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    "seed5": {
        "port": 1,
        "warmstart": "seed5_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    "seed0": {
        "port": 1,
        "warmstart": "seed0_P1_warmstart",
        "mask_paper1": None,       # to be determined by Phase-0
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    "seed6": {
        "port": 1,
        "warmstart": "seed6_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    "seed8": {
        "port": 1,
        "warmstart": "seed8_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    "seed12": {
        "port": 1,
        "warmstart": "seed12_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    "seed14": {
        "port": 1,
        "warmstart": "seed14_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    "seed15": {
        "port": 1,
        "warmstart": "seed15_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    "seed16": {
        "port": 1,
        "warmstart": "seed16_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    "seed17": {
        "port": 1,
        "warmstart": "seed17_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    "seed18": {
        "port": 1,
        "warmstart": "seed18_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    "seed19": {
        "port": 1,
        "warmstart": "seed19_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    "seed20": {
        "port": 1,
        "warmstart": "seed20_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    "seed21": {
        "port": 1,
        "warmstart": "seed21_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    "seed22": {
        "port": 1,
        "warmstart": "seed22_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    "seed23": {
        "port": 1,
        "warmstart": "seed23_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    "seed24": {
        "port": 1,
        "warmstart": "seed24_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    "seed25": {
        "port": 1,
        "warmstart": "seed25_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    "seed26": {
        "port": 1,
        "warmstart": "seed26_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    "seed27": {
        "port": 1,
        "warmstart": "seed27_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    "seed28": {
        "port": 1,
        "warmstart": "seed28_P1_warmstart",
        "mask_paper1": None,
        "variant_paper1": None,
        "baseline_abs2": None,
    },
    # seed29-36: virgin pool for the sigma-leg cohort (B2).
    # Registered 2026-09-17 per preregistration_sigma_leg_seed20plus_2026-09-17.md AMENDMENT #1.
    # jac + warmstart verified present; warmstart config identical to seed28 apart from the seed field.
    "seed29": {"port": 1, "warmstart": "seed29_P1_warmstart", "mask_paper1": None, "variant_paper1": None, "baseline_abs2": None},
    "seed30": {"port": 1, "warmstart": "seed30_P1_warmstart", "mask_paper1": None, "variant_paper1": None, "baseline_abs2": None},
    "seed31": {"port": 1, "warmstart": "seed31_P1_warmstart", "mask_paper1": None, "variant_paper1": None, "baseline_abs2": None},
    "seed32": {"port": 1, "warmstart": "seed32_P1_warmstart", "mask_paper1": None, "variant_paper1": None, "baseline_abs2": None},
    "seed33": {"port": 1, "warmstart": "seed33_P1_warmstart", "mask_paper1": None, "variant_paper1": None, "baseline_abs2": None},
    "seed34": {"port": 1, "warmstart": "seed34_P1_warmstart", "mask_paper1": None, "variant_paper1": None, "baseline_abs2": None},
    "seed35": {"port": 1, "warmstart": "seed35_P1_warmstart", "mask_paper1": None, "variant_paper1": None, "baseline_abs2": None},
    "seed36": {"port": 1, "warmstart": "seed36_P1_warmstart", "mask_paper1": None, "variant_paper1": None, "baseline_abs2": None},
}

# -- budget ---------------------------------------------------------------
BUDGET_LEVELS = [100, 500]
FDTD_EVAL_UNIT = "endpoint_evaluation"

# Fixed diagnostic overhead (default; can be ablated)
DIAG_COST_DEFAULT = {
    "null_k1_draws": 30,
    "null_k5_draws": 25,
    "probe_per_zone": 1,       # one probe per suspect column
    "phase_probe_count": 5,    # phase variant flip-response
}

DIAG_COST_LITE = {             # ablated variant
    "null_k1_draws": 15,
    "null_k5_draws": 12,
    "probe_per_zone": 1,
    "phase_probe_count": 3,
}

# Gradient refresh cost (outside the budget; reported separately)
GRAD_REFRESH_COST = {
    "wall_clock_s": 360,       # ~6 min forward+adjoint+CAD
    "equivalent_eval": 24,     # 360/15 ≈ 24
}

# -- strategies -----------------------------------------------------------
# Names reflect the stages actually executed: "+" means composed in order
# e.g. "naive_single+patch" = naive_single then naive_patch, matching the
STRATEGIES = [
    "naive_single",
    "diag_single",
    "naive_single+patch",
    "diag_single+patch",
    "naive_single+patch+trajectory",
    "diag_single+patch+trajectory",
]

# which strategies run at which budgets
STRATEGY_BUDGET_MAP = {
    100: ["naive_single", "diag_single"],
    500: ["naive_single", "diag_single",
          "naive_single+patch", "diag_single+patch",
          "naive_single+patch+trajectory", "diag_single+patch+trajectory"],
}

# ── Atlas mode ─────────────────────────────────────────────────────────
ATLAS_DEFAULTS = {
    "z_threshold": 2.0,
    "patch_sizes": [2, 3, 5],
    "patch_types": ["sparse", "dense"],
    "selector_top_m": 50,
}
ATLAS_DRIFT_CHECK_INTERVAL = 25  # baseline re-check every N evals

# -- greedy parameters ----------------------------------------------------
DEFAULT_K = 5                # nominations per greedy round
MAX_CONSECUTIVE_MISS = 2     # stop after this many consecutive zero-accept rounds
L4_MAX_EVAL_CAP = 50         # max candidate tests per L4 rescue round
PATCH_SIZES = [2, 3, 5]     # patch window sizes
PATCH_TYPES = ["sparse", "dense"]
POSITIVE_FRACTION_MIN = 0.6  # minimum positive fraction for a sparse patch

# -- statistics -----------------------------------------------------------
N_REPEATS = 3               # repeats per (strategy, budget, anchor)
SIGNIFICANCE_LEVEL = 2.0     # acceptance threshold: mu + 2 sigma
NULL_DRAWS_K1 = 30           # single-flip null draws
NULL_DRAWS_K5 = 25           # batch null draws
MATCHED_NULL_DRAWS = 30      # patch matched-null draws
RNG_BASE_SEED = 2678         # Paper-1 seed, offset by +repeat_id*100

# -- verifier -------------------------------------------------------------
J_WDM_PARAMS = {
    "lambda_R": 1.0,         # weight on R_in
    "lambda_X": 0.5,         # weight on T_wrong
}

# -- output ---------------------------------------------------------------
os.makedirs(RESULTS_DIR, exist_ok=True)
