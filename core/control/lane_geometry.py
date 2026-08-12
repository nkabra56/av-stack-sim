"""Arc-length bookkeeping for a curved lane centerline (full closed-loop highway
drive). H1's gap/position math and H4's stop-line placement were built assuming a
straight road (a single scalar "position"); once the ego is genuinely following a
curved centerline (H3), longitudinal progress has to mean distance travelled *along
the road*, not raw Euclidean x -- this module is the one place that conversion lives,
reused by HighwayVehicleNode (ground truth) and, once H4 is layered on top,
IntersectionControllerNode (estimated pose). See DESIGN.md section 12's closed-loop
drive entry.

On the real committed `lane_centerline.csv` specifically, this is close to a no-op:
the real curvature there is gentle enough (~0.16 degrees max heading deviation) that
`position_m` is already arc length to within centimeters. Implemented generally
anyway rather than assuming a straight road, since a future lane with real curvature
should just work without anyone having to revisit this.
"""

import numpy as np


def build_arc_length_table(centerline: np.ndarray) -> np.ndarray:
    """Geometric arc length (m) at each waypoint of an (N,3) x/y/theta centerline,
    computed from actual consecutive (x,y) distances -- not read off a "position"
    column, which may not be true arc length in general.

    Anchored at `centerline[0, 0]` (the first waypoint's x), NOT at 0: arc length is
    meant to be directly comparable to any other "distance along this road" value in
    the same absolute frame -- e.g. a lead vehicle's recorded position -- and a
    0-based convention would silently discard that frame, making every arc-length
    value off by `centerline[0, 0]` relative to anything computed independently (a
    real bug caught exactly this way: an ego start position landing ~15m off from
    the intended gap behind a real NGSIM lead vehicle, before this anchor fix)."""
    xy = centerline[:, :2]
    seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    return centerline[0, 0] + np.concatenate([[0.0], np.cumsum(seg)])


def project_to_arc_length(x: float, y: float, centerline: np.ndarray, arc_length: np.ndarray) -> float:
    """Nearest-waypoint projection of a point onto the centerline, returned as that
    waypoint's cumulative arc length -- the same nearest-index approach
    StanleyController.control() already uses for cross-track error, consistent with
    this centerline's own waypoint spacing rather than a new interpolation scheme."""
    idx = int(np.argmin(np.hypot(centerline[:, 0] - x, centerline[:, 1] - y)))
    return float(arc_length[idx])


def pose_at_arc_length(s: float, centerline: np.ndarray, arc_length: np.ndarray) -> tuple[float, float, float]:
    """Inverse of project_to_arc_length -- for scenario setup only (placing a vehicle's
    initial x/y/theta at a desired distance along the road)."""
    idx = int(np.argmin(np.abs(arc_length - s)))
    x, y, theta = centerline[idx]
    return float(x), float(y), float(theta)
