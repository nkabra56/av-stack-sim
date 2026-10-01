"""Arc-length bookkeeping for a curved lane centerline: converts Euclidean position into
distance travelled along the road. See DESIGN.md section 12's closed-loop drive entry."""

import numpy as np


def build_arc_length_table(centerline: np.ndarray) -> np.ndarray:
    """Geometric arc length (m) at each waypoint, from actual consecutive (x,y) distances.
    Anchored at centerline[0, 0], not 0, so it's comparable to other absolute-frame positions."""
    xy = centerline[:, :2]
    seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    return centerline[0, 0] + np.concatenate([[0.0], np.cumsum(seg)])


def project_to_arc_length(x: float, y: float, centerline: np.ndarray, arc_length: np.ndarray) -> float:
    """Nearest-waypoint projection of a point onto the centerline, as cumulative arc
    length: same nearest-index approach StanleyController.control() uses."""
    idx = int(np.argmin(np.hypot(centerline[:, 0] - x, centerline[:, 1] - y)))
    return float(arc_length[idx])


def pose_at_arc_length(s: float, centerline: np.ndarray, arc_length: np.ndarray) -> tuple[float, float, float]:
    """Inverse of project_to_arc_length: for scenario setup only (placing a vehicle's
    initial x/y/theta at a desired distance along the road)."""
    idx = int(np.argmin(np.abs(arc_length - s)))
    x, y, theta = centerline[idx]
    return float(x), float(y), float(theta)


def pose_at_arc_length_interpolated(s, centerline: np.ndarray, arc_length: np.ndarray):
    """Continuous counterpart to pose_at_arc_length: linearly interpolates (np.interp) between
    the two bracketing waypoints instead of snapping to the nearest one, so it carries none of
    project_to_arc_length's 2 m quantization error. Used only for after-the-fact geometric truth
    (tests, the viewer), never for the live control loop. s may be a scalar or an array.
    np.interp clamps (not extrapolates) past the centerline's own arc-length range: callers
    must keep s within it, or the returned pose silently freezes at the nearer endpoint."""
    x = np.interp(s, arc_length, centerline[:, 0])
    y = np.interp(s, arc_length, centerline[:, 1])
    theta = np.interp(s, arc_length, centerline[:, 2])
    return x, y, theta


def geometric_following_gap(ego_x, ego_y, ego_theta, lead_arc_length, lead_length: float,
                             centerline: np.ndarray, arc_length_table: np.ndarray):
    """True bumper-to-bumper following gap, free of project_to_arc_length's quantization: the
    lead's rear bumper comes from continuous interpolation along the centerline (its pose, then
    a Euclidean offset back by half its length), projected onto the ego's real (x, y, theta)
    along its real heading rather than its quantized arc length. See DESIGN.md section 9.
    All of ego_x/ego_y/ego_theta/lead_arc_length may be scalars or equal-length arrays."""
    lead_x, lead_y, lead_theta = pose_at_arc_length_interpolated(
        lead_arc_length - lead_length / 2, centerline, arc_length_table
    )
    rear_x = lead_x - lead_length / 2 * np.cos(lead_theta)
    rear_y = lead_y - lead_length / 2 * np.sin(lead_theta)
    return (rear_x - ego_x) * np.cos(ego_theta) + (rear_y - ego_y) * np.sin(ego_theta)
