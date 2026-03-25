"""Vision package: projection, skeleton drawing, interpolation, manual adjustment."""
from .projection import project_pts, draw_skel, draw_skel_with_confidence, clear_projection_cache
from .interpolation import interpolate_joints
from .adjustment import (
    unproject_2d_to_3d, get_camera_depth, extract_R_t,
    find_nearest_joint, PICK_RADIUS,
)
