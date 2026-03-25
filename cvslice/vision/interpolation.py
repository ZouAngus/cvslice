"""Interpolation for missing (NaN) 3D joint data.

Strategy (per-joint, per-axis):
1. Gaps bounded on both sides:
   - gap <= 30 frames: cubic spline (smooth)
   - gap > 30 frames: linear interpolation (safe)
2. Unbounded gaps (at start/end): nearest-valid hold (forward/backward fill)
"""
import numpy as np

try:
    from scipy.interpolate import CubicSpline
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Threshold: gaps longer than this use linear instead of cubic
CUBIC_MAX_GAP = 30


def interpolate_joints(pts3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate NaN values in a (T, J, 3) array.

    Returns:
        pts3d_filled: (T, J, 3) array with NaN replaced by interpolated values.
        was_nan: (T, J) boolean mask — True where the original data was NaN.
    """
    T, J, D = pts3d.shape
    result = pts3d.copy()
    was_nan = np.isnan(pts3d[:, :, 0])  # (T, J) — NaN on x implies NaN on y,z

    for j in range(J):
        nan_mask = was_nan[:, j]
        if not nan_mask.any():
            continue
        if nan_mask.all():
            # Entire joint is NaN — leave as zero
            result[:, j, :] = 0.0
            continue

        valid_idx = np.where(~nan_mask)[0]

        for d in range(D):  # x, y, z
            series = result[:, j, d].copy()
            valid_vals = series[valid_idx]

            # Find NaN runs
            runs = _find_nan_runs(nan_mask)

            for start, end in runs:
                gap_len = end - start
                has_left = start > 0 and not nan_mask[start - 1]
                has_right = end < T and not nan_mask[end]

                if has_left and has_right:
                    # Bounded gap — interpolate
                    if gap_len <= CUBIC_MAX_GAP and HAS_SCIPY and len(valid_idx) >= 4:
                        # Use cubic spline with a local window for context
                        series[start:end] = _cubic_fill(
                            valid_idx, valid_vals, start, end)
                    else:
                        # Linear interpolation
                        left_val = series[start - 1]
                        right_val = series[end]
                        series[start:end] = np.linspace(
                            left_val, right_val, gap_len + 2)[1:-1]
                elif has_left and not has_right:
                    # Unbounded right — forward fill
                    series[start:end] = series[start - 1]
                elif has_right and not has_left:
                    # Unbounded left — backward fill
                    series[start:end] = series[end]
                # else: both unbounded — leave as 0 (set above)

            result[:, j, d] = series

    return result, was_nan


def _find_nan_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous runs of True in a boolean array.

    Returns list of (start, end) where end is exclusive.
    """
    runs = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            start = i
            while i < n and mask[i]:
                i += 1
            runs.append((start, i))
        else:
            i += 1
    return runs


def _cubic_fill(valid_idx: np.ndarray, valid_vals: np.ndarray,
                gap_start: int, gap_end: int) -> np.ndarray:
    """Fill a gap using cubic spline, using nearby valid points as knots."""
    # Use a local window of valid points around the gap for better conditioning
    margin = max(CUBIC_MAX_GAP, gap_end - gap_start) * 2
    local_mask = ((valid_idx >= gap_start - margin) &
                  (valid_idx <= gap_end + margin))
    local_idx = valid_idx[local_mask]
    local_vals = valid_vals[local_mask]

    if len(local_idx) < 4:
        # Not enough points for cubic — fall back to linear
        left_val = valid_vals[valid_idx < gap_start][-1]
        right_val = valid_vals[valid_idx >= gap_end][0]
        gap_len = gap_end - gap_start
        return np.linspace(left_val, right_val, gap_len + 2)[1:-1]

    cs = CubicSpline(local_idx, local_vals, bc_type='natural')
    fill_idx = np.arange(gap_start, gap_end)
    return cs(fill_idx)
