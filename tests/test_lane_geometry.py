import numpy as np

from core.control.lane_geometry import (
    build_arc_length_table,
    geometric_following_gap,
    pose_at_arc_length,
    pose_at_arc_length_interpolated,
)

# A straight centerline, waypoints 2 m apart along x: arc length equals x exactly, so the
# continuous and nearest-waypoint projections can both be hand-computed and compared directly.
STRAIGHT = np.array([[float(2 * i), 0.0, 0.0] for i in range(11)])  # x = 0, 2, 4, ..., 20
STRAIGHT_TABLE = build_arc_length_table(STRAIGHT)


def test_interpolated_pose_is_exact_between_waypoints_on_a_straight_line():
    """Quantization case DESIGN.md section 9 describes: s=5 falls between the waypoints at
    x=4 and x=6. The nearest-waypoint version snaps to one of them; np.interp does not."""
    x, y, theta = pose_at_arc_length_interpolated(5.0, STRAIGHT, STRAIGHT_TABLE)
    assert x == 5.0
    assert y == 0.0
    assert theta == 0.0


def test_nearest_waypoint_pose_quantizes_the_same_query():
    x, _, _ = pose_at_arc_length(5.0, STRAIGHT, STRAIGHT_TABLE)
    assert x == 4.0  # up to 1 m (half the 2 m spacing) away from the true x=5.0


def test_interpolated_pose_on_a_gently_curved_centerline():
    """x, y and theta each interpolate independently; values chosen so the midpoint
    (s=3, halfway between the s=2 and s=4 waypoints) is hand-computable."""
    centerline = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 1.0, 0.1],
        [4.0, 2.0, 0.2],
        [6.0, 2.5, 0.3],
    ])
    arc_length = np.array([0.0, 2.0, 4.0, 6.0])
    x, y, theta = pose_at_arc_length_interpolated(3.0, centerline, arc_length)
    assert x == 3.0
    assert y == 1.5
    assert abs(theta - 0.15) < 1e-12


def test_geometric_following_gap_matches_hand_computation_on_a_straight_line():
    """Ego heading along +x: the gap is just the along-track distance to the lead's rear
    bumper, independent of any lateral offset (sin(theta)=0 kills the y term)."""
    gap = geometric_following_gap(
        ego_x=1.0, ego_y=0.5, ego_theta=0.0,
        lead_arc_length=10.0, lead_length=4.0,
        centerline=STRAIGHT, arc_length_table=STRAIGHT_TABLE,
    )
    # Lead's rear bumper is at arc length 10 - 4 = 6, i.e. x=6 on this centerline.
    assert gap == 5.0


def test_geometric_following_gap_projects_onto_a_rotated_ego_heading():
    """Ego facing +y (theta=pi/2): cos(theta)=0 zeroes the x term, so only the y
    separation contributes. Exercises the sin(theta) term the straight-ahead case above does not."""
    gap = geometric_following_gap(
        ego_x=100.0, ego_y=-3.0, ego_theta=np.pi / 2,
        lead_arc_length=10.0, lead_length=4.0,
        centerline=STRAIGHT, arc_length_table=STRAIGHT_TABLE,
    )
    # Lead's rear bumper is at (x=6, y=0); gap = (0 - (-3)) * sin(pi/2) = 3.
    assert abs(gap - 3.0) < 1e-12


def test_geometric_following_gap_is_vectorized():
    """The harness and the viewer both call this once over a whole time series, so array
    inputs must match the elementwise scalar results."""
    ego_x = np.array([1.0, 100.0])
    ego_y = np.array([0.5, -3.0])
    ego_theta = np.array([0.0, np.pi / 2])
    lead_arc_length = np.array([10.0, 10.0])
    gaps = geometric_following_gap(
        ego_x, ego_y, ego_theta, lead_arc_length, lead_length=4.0,
        centerline=STRAIGHT, arc_length_table=STRAIGHT_TABLE,
    )
    assert np.allclose(gaps, [5.0, 3.0])
