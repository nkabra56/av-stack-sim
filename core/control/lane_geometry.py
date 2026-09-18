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
    length -- same nearest-index approach StanleyController.control() uses."""
    idx = int(np.argmin(np.hypot(centerline[:, 0] - x, centerline[:, 1] - y)))
    return float(arc_length[idx])


def pose_at_arc_length(s: float, centerline: np.ndarray, arc_length: np.ndarray) -> tuple[float, float, float]:
    """Inverse of project_to_arc_length -- for scenario setup only (placing a vehicle's
    initial x/y/theta at a desired distance along the road)."""
    idx = int(np.argmin(np.abs(arc_length - s)))
    x, y, theta = centerline[idx]
    return float(x), float(y), float(theta)
