"""Data discovery: find CSVs, video folders, and cameras for a scene."""
import os
import re
import numpy as np
import pandas as pd
from ..core.constants import CAMERA_NAMES


_SCENE_ALIASES = {
    "sword": {"sword", "elsdon"},
    "elsdon": {"sword", "elsdon"},
}


def _normalize_scene_key(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())


def scene_keys(name: str | None) -> set[str]:
    """Return normalized scene keys including known aliases."""
    key = _normalize_scene_key(name or "")
    if not key:
        return set()
    return set(_SCENE_ALIASES.get(key, {key}))


def scene_name_matches(candidate: str, scene_name: str | None) -> bool:
    """True if *candidate* matches *scene_name* or one of its aliases."""
    cand = _normalize_scene_key(candidate)
    if not cand:
        return False
    keys = scene_keys(scene_name)
    if not keys:
        return True
    return any(k in cand or cand in k for k in keys)


def find_data_subfolder(data_root: str, sheet_name: str) -> str | None:
    """Find the data subfolder matching a scene name."""
    if not data_root or not os.path.isdir(data_root):
        return None
    keys = scene_keys(sheet_name)
    # Exact/alias match
    for entry in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, entry)
        if os.path.isdir(full) and _normalize_scene_key(entry) in keys:
            return full
    # Fuzzy match with aliases
    for entry in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, entry)
        if os.path.isdir(full) and scene_name_matches(entry, sheet_name):
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
        for fn in sorted(os.listdir(data_root)):
            if not fn.lower().endswith(".csv"):
                continue
            if not fn.lower().startswith("extracted"):
                continue
            fk = os.path.splitext(fn)[0].replace("extracted", "").strip("_")
            if scene_name_matches(fk, sheet_name):
                csv_path = os.path.join(data_root, fn)
                break
    return csv_path, subfolder


def find_cameras_in_folder(folder: str, scene_hint: str | None = None) -> list[str]:
    """Detect available camera names by scanning for matching .mp4 files.

    If *scene_hint* is given, only consider files whose name contains the
    normalised scene key (e.g. 'boss' inside 'boss_15_topleft.mp4').
    """
    if not folder or not os.path.isdir(folder):
        return []
    scene_hint_present = bool(scene_hint)
    cams = []
    for cn in CAMERA_NAMES:
        for fn in os.listdir(folder):
            fl = fn.lower()
            if not fl.endswith(".mp4"):
                continue
            if cn not in fl:
                continue
            if scene_hint_present and not scene_name_matches(fl, scene_hint):
                continue
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
