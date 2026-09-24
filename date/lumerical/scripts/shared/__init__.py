"""
Shared infrastructure: IncrementalSession wrapper, verifier, null calibration, jac tools, clustering
"""

import json
import numpy as np
from typing import Optional

from .index_utils import (
    N_COLS, N_ROWS, N_PIXELS,
    flat_to_col, col_row_to_flat, is_masked, filter_masked,
    apply_mask_to_field, get_top_k_indices, get_bottom_k_indices,
    effective_pool_size, patch_window, patch_indices,
)


# -- IncrementalSession wrapper ------------------------------------------

class SessionWrapper:
    """Thin wrapper around IncrementalSession."""

    def __init__(self, warmstart_params_path: str, anchor_name: str):
        self.warmstart_path = warmstart_params_path
        self.anchor = anchor_name
        self._session = None
        self._baseline_abs2: Optional[float] = None
        self._baseline_J_wdm: Optional[float] = None

    def start(self):
        raise NotImplementedError("a Lumerical session is required")

    def flip_single(self, idx: int) -> dict:
        raise NotImplementedError

    def flip_patch(self, indices: list[int]) -> dict:
        raise NotImplementedError

    def accept(self):
        pass

    def reject(self):
        pass

    def measure_J_wdm(self) -> float:
        raise NotImplementedError

    def close(self):
        pass


# -- verifier -------------------------------------------------------------

def compute_J_wdm(T_target: float, R_in: float, T_wrong_ports: list[float],
                  lambda_R: float = 1.0, lambda_X: float = 0.5) -> float:
    T_wrong_mean = np.mean(T_wrong_ports) if T_wrong_ports else 0.0
    return T_target - lambda_R * R_in - lambda_X * T_wrong_mean


# -- null calibration -----------------------------------------------------

def sample_null_single(session: SessionWrapper, n_draws: int,
                       rng: np.random.Generator,
                       mask: Optional[list] = None) -> dict:
    """Single-flip null calibration (k=1).

    Draws at random from the valid pool using Paper-1 flat indices.
    """
    pool = effective_pool_size(mask)
    dFs = []
    attempts = 0
    while len(dFs) < n_draws and attempts < n_draws * 10:
        idx = int(rng.integers(0, N_PIXELS))
        if mask and is_masked(idx, mask):
            attempts += 1
            continue
        dF = session.flip_single(idx)["dJ"]
        dFs.append(dF)
        session.reject()
        attempts += 1

    mu = float(np.mean(dFs))
    sigma = float(np.std(dFs, ddof=1))
    return {
        "mu": mu, "sigma": sigma,
        "threshold": mu + 2 * sigma,
        "dFs": [float(x) for x in dFs],
        "n_draws": len(dFs),
    }


def sample_null_batch(session: SessionWrapper, K: int, n_draws: int,
                      rng: np.random.Generator,
                      mask: Optional[list] = None) -> dict:
    """Batch null calibration (k=K).

    Each draw samples K non-mask pixels from the valid pool without replacement,
    """
    # precompute the valid pixel pool
    all_indices = np.arange(N_PIXELS)
    if mask:
        valid_mask_arr = np.array([not is_masked(i, mask) for i in range(N_PIXELS)])
        valid_pool = all_indices[valid_mask_arr]
    else:
        valid_pool = all_indices

    dFs = []
    for _ in range(n_draws):
        indices = rng.choice(valid_pool, size=min(K, len(valid_pool)), replace=False).tolist()
        dF = session.flip_patch(indices)["dJ"]
        dFs.append(dF)
        session.reject()

    mu = float(np.mean(dFs))
    sigma = float(np.std(dFs, ddof=1))
    return {
        "mu": mu, "sigma": sigma,
        "threshold": mu + 2 * sigma,
        "dFs": [float(x) for x in dFs],
        "n_draws": len(dFs), "K": K,
    }


def sample_matched_null(session: SessionWrapper, size: int, ptype: str,
                        n_draws: int, rng: np.random.Generator,
                        n_px: int = None) -> dict:
    """Build a matched random null for a given (size, type).

    ptype controls which pixels are actually flipped:
      - "sparse": flip n_px pixels drawn at random in the window (matching the
      - "dense":  flip every pixel in the window
    n_px: required for "sparse" (matching the candidate patch's positive-pixel
    """
    if ptype == "sparse" and n_px is None:
        raise ValueError(
            "sample_matched_null: ptype='sparse' requires n_px (number of "
            "positive-gain pixels to match). Got n_px=None."
        )

    dFs = []
    window_px = size * size
    if ptype == "sparse":
        n_flip = min(n_px, window_px)
    else:
        n_flip = window_px

    for _ in range(n_draws):
        cx = int(rng.integers(size // 2, N_COLS - size // 2))
        cy = int(rng.integers(size // 2, N_ROWS - size // 2))
        all_idx = patch_indices(cx, cy, size)
        if n_flip < len(all_idx):
            chosen = rng.choice(all_idx, size=n_flip, replace=False).tolist()
        else:
            chosen = all_idx
        dF = session.flip_patch(chosen)["dJ"]
        dFs.append(dF)
        session.reject()

    mu = float(np.mean(dFs))
    sigma = float(np.std(dFs, ddof=1))
    return {
        "mu": mu, "sigma": sigma,
        "threshold": mu + 2 * sigma,
        "n_draws": n_draws, "size": size, "type": ptype,
        "n_px_matched": n_flip,
    }


# -- jac tools ------------------------------------------------------------

def load_jac_npz(path: str) -> dict:
    """Load a Paper-1 adjoint jacobian and convert it to the unified 2D layout.

    Paper-1 npz layout:
      - grad_raw / grad_i / grad_conj_a / grad_i_conj_a: (8000,) 1D flat
      - O_pix: (80, 100) complex
      - params: (8000,) float
      - baseline_abs2: scalar

    Converted to the unified layout:
      - gain_raw / gain_i / gain_conj_a / gain_i_conj_a: (80, 100) 2D
      - params / baseline_abs2 passed through
    """
    import numpy as np
    data = np.load(path, allow_pickle=True)
    result = {}

    # Rename grad_* → gain_*, reshape 1D→2D
    for grad_key in ["grad_raw", "grad_i", "grad_conj_a", "grad_i_conj_a"]:
        if grad_key in data:
            gain_key = grad_key.replace("grad_", "gain_")
            flat = data[grad_key]
            if flat.ndim == 1 and flat.shape[0] == N_COLS * N_ROWS:
                result[gain_key] = flat.reshape(N_COLS, N_ROWS, order='C')
            else:
                result[gain_key] = flat

    # pass through the scalars and the parameters
    for k in ["params", "baseline_abs2"]:
        if k in data:
            result[k] = data[k]

    return result


# -- clustering -----------------------------------------------------------

def spatial_cluster(indices: list[int],
                    distance_threshold: int = 3) -> list[tuple[int, int]]:
    """Cluster accepted pixel indices spatially; return cluster centres (ix, iy)."""
    if not indices:
        return []

    ixs = [flat_to_col(idx) for idx in indices]
    iys = [idx % N_ROWS for idx in indices]
    points = np.column_stack([ixs, iys])

    visited = set()
    clusters = []
    for i, pt in enumerate(points):
        if i in visited:
            continue
        cluster = [pt]
        visited.add(i)
        for j, pt2 in enumerate(points):
            if j in visited:
                continue
            if np.linalg.norm(pt - pt2) <= distance_threshold:
                cluster.append(pt2)
                visited.add(j)
        center = np.mean(cluster, axis=0)
        clusters.append((int(round(center[0])), int(round(center[1]))))

    return clusters


# -- trajectory recording -------------------------------------------------

class Trajectory:
    """Record the full trajectory of one optimization line."""

    def __init__(self, strategy: str, anchor: str, budget: int, repeat: int):
        self.strategy = strategy
        self.anchor = anchor
        self.budget = budget
        self.repeat = repeat
        self.steps: list[dict] = []
        self.eval_count = 0
        self.cumulative_dJ = 0.0

    def record_step(self, step: dict):
        step["eval_count"] = self.eval_count
        step["cumulative_dJ"] = self.cumulative_dJ
        self.steps.append(step)

    def extend_steps(self, child_steps: list[dict]):
        """Append child trajectory steps WITHOUT overwriting their fields.

        Unlike record_step(), this preserves the child's original eval_count
        and cumulative_dJ values.  Use this when merging sub-strategy
        trajectories into a composite parent.
        """
        self.steps.extend(child_steps)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "anchor": self.anchor,
            "budget": self.budget,
            "repeat": self.repeat,
            "steps": self.steps,
            "eval_count": self.eval_count,
            "cumulative_dJ": self.cumulative_dJ,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        obj = cls(d["strategy"], d["anchor"], d["budget"], d["repeat"])
        obj.steps = d.get("steps", [])
        obj.eval_count = d.get("eval_count", 0)
        obj.cumulative_dJ = d.get("cumulative_dJ", 0.0)
        return obj

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "strategy": self.strategy,
                "anchor": self.anchor,
                "budget": self.budget,
                "repeat": self.repeat,
                "steps": self.steps,
                "final_eval_count": self.eval_count,
                "final_cumulative_dJ": self.cumulative_dJ,
            }, f, indent=2)
