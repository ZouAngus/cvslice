"""Data discovery: find CSVs, video folders, and cameras for a scene."""
import os
import re
import numpy as np
import pandas as pd
from ..core.constants import CAMERA_NAMES


def _normalize_scene_key(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())


def find_data_subfolder(data_root: str, sheet_name: str) -> str | None:
    """Find the data subfolder matching a scene name."""
    if not data_root or not os.path.isdir(data_root):
        return None
    key = _normalize_scene_key(sheet_name)
    # Exact match
    for entry in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, entry)
        if os.path.isdir(full) and _normalize_scene_key(entry) == key:
            return full
    # Fuzzy match
    for entry in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, entry)
        if os.path.isdir(full):
            ek = _normalize_scene_key(entry)
            if key in ek or ek in key:
                return full
    return None


def find_csv_in_folder(folder: str) -> str | None:
    """Find the first 'extracted*.csv' in a folder."""
    if not folder or not os.path.isdir(folder):
        return None
    for fn in sorted(os.listdir(folder)):
        if fn.lower().startswith("extracted") and fn.lower().endswith(".csv"):
            return os.path.join(folder, fn)
    return None


def find_csv_for_scene(data_root: str, sheet_name: str) -> tuple[str | None, str | None]:
    """Find CSV + video folder for a scene.

    Returns (csv_path | None, video_folder | None).
    """
    subfolder = find_data_subfolder(data_root, sheet_name)
    # 1. CSV inside subfolder
    if subfolder:
        csv_path = find_csv_in_folder(subfolder)
        if csv_path:
            return csv_path, subfolder
    # 2. CSV in data root matching scene name
    csv_path = None
    if data_root and os.path.isdir(data_root):
        key = _normalize_scene_key(sheet_name)
        for fn in sorted(os.listdir(data_root)):
            if not fn.lower().endswith(".csv"):
                continue
            if not fn.lower().startswith("extracted"):
                continue
            fk = _normalize_scene_key(
                os.path.splitext(fn)[0].replace("extracted", "").strip("_"))
            if key in fk or fk in key:
                csv_path = os.path.join(data_root, fn)
                break
    return csv_path, subfolder


def find_cameras_in_folder(folder: str) -> list[str]:
    """Detect available camera names by scanning for matching .mp4 files."""
    if not folder or not os.path.isdir(folder):
        return []
    cams = []
    for cn in CAMERA_NAMES:
        for fn in os.listdir(folder):
            if cn in fn.lower() and fn.lower().endswith(".mp4"):
                cams.append(cn)
                break
    return cams


def load_csv_as_pts3d(csv_path: str) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Load extracted CSV -> (T, J, 3) array with NaN interpolation.

    Returns (pts3d_array, valid_mask, was_nan_mask):
        - pts3d_array: (T, J, 3) with NaN replaced by interpolated values
        - valid_mask: (T,) bool — True if frame has at least one originally valid joint
        - was_nan_mask: (T, J) bool — True where original data was NaN (now interpolated)
    """
    from ..vision.interpolation import interpolate_joints

    df = pd.read_csv(csv_path)
    df = df.apply(pd.to_numeric, errors="coerce")
    nc = df.shape[1]
    if nc % 3 != 0:
        return None, None, None
    pts_raw = df.values.reshape(-1, nc // 3, 3)

    # Interpolate NaN gaps
    pts_filled, was_nan = interpolate_joints(pts_raw)

    # Valid mask: frame has at least one originally non-NaN joint
    valid = ~np.all(was_nan, axis=1)
    return pts_filled, valid, was_nan
