"""
Index-coordinate convention

Paper 1 and IncrementalSession use a C-order flat index:
  grid shape: (n_cols=80, n_rows=100)
  gain_field[ix, iy]  ← numpy shape (80, 100), C-order
  flat_idx = ix * N_ROWS + iy = ix * 100 + iy
  ix = flat_idx // N_ROWS = flat_idx // 100
  iy = flat_idx % N_ROWS = flat_idx % 100

Every function here follows that convention; do not reimplement the ix/iy to
flat_idx conversion in another file.
"""

# -- grid constants -------------------------------------------------------
N_COLS = 80   # ix in [0, 80)
N_ROWS = 100  # iy in [0, 100)
N_PIXELS = N_COLS * N_ROWS  # 8000

# -- conversion functions --------------------------------------------------

def flat_to_col_row(flat_idx: int) -> tuple[int, int]:
    """flat_idx → (ix, iy)"""
    return flat_idx // N_ROWS, flat_idx % N_ROWS


def col_row_to_flat(ix: int, iy: int) -> int:
    """(ix, iy) → flat_idx"""
    return ix * N_ROWS + iy


def flat_to_col(flat_idx: int) -> int:
    """flat_idx → ix (column)"""
    return flat_idx // N_ROWS


def flat_to_row(flat_idx: int) -> int:
    """flat_idx → iy (row)"""
    return flat_idx % N_ROWS


# -- mask checks ----------------------------------------------------------

def is_masked(flat_idx: int, mask: list[tuple[int, int]]) -> bool:
    """True if flat_idx falls inside the mask. The mask is a column (ix) range."""
    ix = flat_to_col(flat_idx)
    return any(lo <= ix <= hi for lo, hi in mask)


def filter_masked(indices: list[int], mask: list[tuple[int, int]]) -> list[int]:
    """Drop the indices that fall inside the mask."""
    return [idx for idx in indices if not is_masked(idx, mask)]


def apply_mask_to_field(gain_field: "np.ndarray", mask: list[tuple[int, int]]) -> "np.ndarray":
    """Set the gain to -inf across the mask region (a column range)."""
    import numpy as np
    masked = gain_field.copy()
    for lo, hi in mask:
        masked[lo:hi + 1, :] = -np.inf
    return masked


# -- ranking extraction ---------------------------------------------------

def get_top_k_indices(gain_field: "np.ndarray", K: int) -> list[int]:
    """Return the Paper-1 flat indices of the K highest-gain pixels.

    gain_field shape = (N_COLS, N_ROWS) = (80, 100), C-order.
    The C-order flat index from np.flatnonzero is exactly ix*100+iy, matching Paper 1.
    """
    import numpy as np
    valid = gain_field != -np.inf
    flat_indices = np.flatnonzero(valid)
    sorted_local = np.argsort(gain_field[valid])[::-1][:K]
    return flat_indices[sorted_local].tolist()


def get_bottom_k_indices(gain_field: "np.ndarray", K: int) -> list[int]:
    """Return the Paper-1 flat indices of the K lowest-gain pixels."""
    import numpy as np
    valid = gain_field != -np.inf
    flat_indices = np.flatnonzero(valid)
    sorted_local = np.argsort(gain_field[valid])[:K]
    return flat_indices[sorted_local].tolist()


# -- pool size ------------------------------------------------------------

def effective_pool_size(mask: list[tuple[int, int]] = None) -> int:
    """Size of the valid pixel pool after masking."""
    if not mask:
        return N_PIXELS
    masked_cols = sum(hi - lo + 1 for lo, hi in mask)
    return N_PIXELS - masked_cols * N_ROWS


# -- patch window tools ---------------------------------------------------

def patch_window(center_ix: int, center_iy: int, size: int) -> tuple[int, int, int, int]:
    """Return the patch window bounds (x0, x1, y0, y1), all half-open.

    For even size the window starts at center - size//2.
    For odd size it starts at center - size//2 and has length size.
    Either way the window width is size.
    """
    half = size // 2
    x0 = center_ix - half
    x1 = x0 + size  # half-open
    y0 = center_iy - half
    y1 = y0 + size
    return x0, x1, y0, y1


def patch_indices(center_ix: int, center_iy: int, size: int,
                  sparse_mask: "np.ndarray" = None) -> list[int]:
    """Return the Paper-1 flat indices of every pixel in the patch window.

    sparse_mask: bool array of shape (size, size); if given, only the True
    """
    x0, x1, y0, y1 = patch_window(center_ix, center_iy, size)
    indices = []
    for dx in range(size):
        ix = x0 + dx
        if ix < 0 or ix >= N_COLS:
            continue
        for dy in range(size):
            iy = y0 + dy
            if iy < 0 or iy >= N_ROWS:
                continue
            if sparse_mask is not None and not sparse_mask[dx, dy]:
                continue
            indices.append(col_row_to_flat(ix, iy))
    return indices
