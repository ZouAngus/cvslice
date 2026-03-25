"""3D joint manual adjustment via 2D drag with depth-preserving unprojection.

When a user drags a joint in the 2D view, we:
1. Keep the joint's depth in camera coordinates (z_cam) fixed
2. Compute the new 2D position from the drag
3. Unproject (u', v', z_cam) back to 3D world coordinates
4. Update the 3D point in-place

This is geometrically exact for the current camera view — the joint moves
along the camera's image plane at its current depth.
"""
import cv2
import numpy as np


def unproject_2d_to_3d(u: float, v: float, z_cam: float,
                       K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Unproject a 2D pixel (u, v) to 3D world coordinates at a given camera depth.

    Args:
        u, v: Pixel coordinates.
        z_cam: Depth in camera coordinate frame.
        K: (3, 3) camera intrinsic matrix.
        R: (3, 3) rotation matrix (world -> camera).
        t: (3,) translation vector (world -> camera).

    Returns:
        (3,) 3D point in world coordinates.
    """
    K_inv = np.linalg.inv(K)
    p_cam = z_cam * K_inv @ np.array([u, v, 1.0])
    p_world = R.T @ (p_cam - t)
    return p_world


def get_camera_depth(pt3d: np.ndarray, R: np.ndarray, t: np.ndarray) -> float:
    """Get the depth of a 3D world point in camera coordinates."""
    p_cam = R @ pt3d + t
    return float(p_cam[2])


def extract_R_t(extr: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract R, t from an extrinsic dict."""
    ext = None
    for k in ("best_extrinsic", "extrinsic", "extrinsics"):
        if k not in extr:
            continue
        v = extr[k]
        if k == "extrinsics" and isinstance(v, list) and v:
            v = v[0]
        ext = np.array(v, dtype=np.float64)
        break
    if ext is None:
        return None
    if ext.shape == (4, 4):
        ext = ext[:3, :]
    if ext.shape != (3, 4):
        return None
    R = ext[:, :3]
    t = ext[:, 3]
    return R, t


# Joint selection radius in pixels
PICK_RADIUS = 15


def find_nearest_joint(click_x: int, click_y: int,
                       proj: np.ndarray) -> int | None:
    """Find the joint index nearest to (click_x, click_y) within PICK_RADIUS.

    Returns joint index or None.
    """
    if proj is None or len(proj) == 0:
        return None
    dists = np.sqrt((proj[:, 0] - click_x) ** 2 + (proj[:, 1] - click_y) ** 2)
    min_idx = int(np.argmin(dists))
    if dists[min_idx] <= PICK_RADIUS:
        return min_idx
    return None
