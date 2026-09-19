"""KNOWN_BUGS.md entry 4: H4's IntersectionNavigator models right-of-way at a single
conflict point; these tests run several through a real 2D intersection and check real (x,y) proximity."""

import numpy as np
import pytest

from core.control.intersection import IntersectionNavigator
from core.control.intersection_geometry import EAST, NORTH, SOUTH, WEST, build_turn_path, is_to_the_right
from core.intersection2d_harness import VehicleSpec, run_multi_approach_scenario
from core.vehicle import wrap_angle


def _proceed_time(result, name: str) -> float | None:
    vehicle = next(v for v in result.vehicles if v.name == name)
    for t, s in zip(result.times, vehicle.states, strict=True):
        if s.name == "PROCEEDING":
            return t
    return None


# --- Geometry itself ------------------------------------------------------------------


def test_is_to_the_right_matches_all_four_cardinal_pairs():
    assert is_to_the_right(NORTH.heading, WEST.heading)
    assert is_to_the_right(EAST.heading, NORTH.heading)
    assert is_to_the_right(SOUTH.heading, EAST.heading)
    assert is_to_the_right(WEST.heading, SOUTH.heading)


def test_is_to_the_right_is_not_true_for_opposite_or_left():
    assert not is_to_the_right(NORTH.heading, SOUTH.heading)  # opposite, not a real pair here
    assert not is_to_the_right(NORTH.heading, EAST.heading)  # EAST is on NORTH's left, not right


# --- Two approaches: reproduces H4's own validated branches, now over real geometry ---


def test_parallel_approaches_never_conflict():
    """North and South share a road but not a lane or crossing path: the single-
    conflict-point model can't express "these don't interact"; real geometry can."""
    result = run_multi_approach_scenario([VehicleSpec(NORTH), VehicleSpec(SOUTH)])
    assert not result.collided
    assert result.ran_stop_sign == []


def test_first_to_arrive_proceeds_first_and_never_collides():
    result = run_multi_approach_scenario([VehicleSpec(NORTH, start_distance=80.0), VehicleSpec(EAST, start_distance=100.0)])
    assert not result.collided
    assert result.ran_stop_sign == []
    assert _proceed_time(result, "N") < _proceed_time(result, "E")


def test_simultaneous_arrival_yields_to_the_right_and_never_collides():
    result = run_multi_approach_scenario([VehicleSpec(NORTH, start_distance=100.0), VehicleSpec(EAST, start_distance=100.0)])
    assert not result.collided
    n_stop, e_stop = (next(v for v in result.vehicles if v.name == n).stop_time for n in ("N", "E"))
    assert abs(n_stop - e_stop) < 0.5  # genuinely simultaneous
    assert _proceed_time(result, "E") > _proceed_time(result, "N")  # E yields, N is to E's right


# --- More than two approaches: the actual gap KNOWN_BUGS entry 4 called out -----------


def test_three_way_staggered_arrival_never_collides():
    result = run_multi_approach_scenario(
        [VehicleSpec(NORTH, start_distance=70.0), VehicleSpec(EAST, start_distance=90.0), VehicleSpec(WEST, start_distance=110.0)],
        max_steps=5000,
    )
    assert not result.collided
    assert result.ran_stop_sign == []
    assert _proceed_time(result, "N") < _proceed_time(result, "E") < _proceed_time(result, "W")


def test_four_way_staggered_arrival_never_collides():
    result = run_multi_approach_scenario(
        [
            VehicleSpec(NORTH, start_distance=70.0),
            VehicleSpec(EAST, start_distance=85.0),
            VehicleSpec(SOUTH, start_distance=100.0),
            VehicleSpec(WEST, start_distance=115.0),
        ],
        max_steps=5000,
    )
    assert not result.collided
    assert result.ran_stop_sign == []
    order = [_proceed_time(result, n) for n in ("N", "E", "S", "W")]
    assert order == sorted(order)


# --- The safety net itself has to be real, not vacuous --------------------------------


class _RecklessNavigator(IntersectionNavigator):
    """Never yields: always claims right of way, regardless of who else is there."""

    def has_right_of_way(self, others):
        return True


def test_collision_check_actually_catches_a_non_compliant_vehicle():
    """If every scenario above passed only because nothing was at risk, they'd be worthless
    as regression tests. Give one vehicle a reckless navigator and confirm collision is detected."""
    specs = [VehicleSpec(NORTH, start_distance=100.0), VehicleSpec(EAST, start_distance=100.0)]
    navigators = [
        IntersectionNavigator(stop_line_position=specs[0].start_distance - 9.5 - 1.5, v_cruise=15.0),
        _RecklessNavigator(stop_line_position=specs[1].start_distance - 9.5 - 1.5, v_cruise=15.0),
    ]
    result = run_multi_approach_scenario(specs, navigators=navigators)
    assert result.collided
    assert ("N", "E") in result.collision_pairs


# --- Turning movements: real curved connectors ----------------------------------------


def test_turn_geometry_meets_curvature_limit_and_is_short_for_every_approach_and_direction():
    """Every approach x {left, right} must produce a clean, direct Dubins connector, not
    the pathological ~5x-too-long path an under-sized turn_lead produced (KNOWN_BUGS.md entry 4)."""
    turning_radius = 6.0
    for approach in [NORTH, EAST, SOUTH, WEST]:
        for turn in ["left", "right"]:
            _exit_approach, path = build_turn_path(approach, turn, lane_offset=3.0, turning_radius=turning_radius)
            seg_lens = np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))
            total = float(seg_lens.sum())
            straight_dist = float(np.hypot(path[-1, 0] - path[0, 0], path[-1, 1] - path[0, 1]))
            dtheta = np.abs(wrap_angle(np.diff(path[:, 2])))
            mask = seg_lens > 1e-9
            max_curv = float((dtheta[mask] / seg_lens[mask]).max())
            assert max_curv <= 1.0 / turning_radius + 1e-3  # discretized path; small numerical slack
            assert total / straight_dist < 1.5  # a direct connector, not a loop


def test_left_turn_exits_heading_matches_the_straight_through_traffic_it_merges_with():
    """A left turn from NORTH ends up heading the same direction WEST's straight-through
    traffic travels: the real-driving convention turn_exit_heading documents."""
    exit_n, _ = build_turn_path(NORTH, "left", lane_offset=3.0, turning_radius=4.0)
    assert abs(wrap_angle(exit_n.heading - WEST.heading)) < 1e-6
    exit_s, _ = build_turn_path(SOUTH, "left", lane_offset=3.0, turning_radius=4.0)
    assert abs(wrap_angle(exit_s.heading - EAST.heading)) < 1e-6


def test_right_turn_exits_heading_is_opposite_the_left_turn_from_the_same_approach():
    exit_left, _ = build_turn_path(NORTH, "left", lane_offset=3.0, turning_radius=4.0)
    exit_right, _ = build_turn_path(NORTH, "right", lane_offset=3.0, turning_radius=4.0)
    assert abs(wrap_angle(exit_right.heading - exit_left.heading - np.pi)) < 1e-6


def test_left_turn_yields_to_oncoming_straight_traffic_despite_arriving_first():
    """A left-turner that reaches its stop line first still waits for oncoming straight
    traffic: regression for the circular-deadlock bug this rule introduced (KNOWN_BUGS.md entry 4)."""
    specs = [VehicleSpec(NORTH, start_distance=70.0, turn="left"), VehicleSpec(SOUTH, start_distance=130.0, turn="straight")]
    result = run_multi_approach_scenario(specs, max_steps=6000)
    assert not result.collided
    n = next(v for v in result.vehicles if v.name == "N")
    s = next(v for v in result.vehicles if v.name == "S")
    assert n.stop_time < s.stop_time  # N really did arrive first
    assert _proceed_time(result, "N") > _proceed_time(result, "S")  # but N still waits for S


def test_left_turn_proceeds_once_oncoming_traffic_has_already_cleared():
    """Control for the test above: if the oncoming vehicle clears well before the
    left-turner is ready, the left-turner isn't stuck waiting on it forever."""
    specs = [VehicleSpec(NORTH, start_distance=70.0, turn="left"), VehicleSpec(SOUTH, start_distance=40.0, turn="straight")]
    result = run_multi_approach_scenario(specs, max_steps=6000)
    assert not result.collided
    assert _proceed_time(result, "N") is not None
    assert _proceed_time(result, "N") > _proceed_time(result, "S")


def test_right_turn_does_not_defer_to_oncoming_traffic_via_the_left_turn_rule():
    """Only left turns get the special oncoming-traffic rule; a right-turner competes
    on ordinary arrival-order/yield-to-right terms like any straight-through vehicle."""
    specs = [VehicleSpec(NORTH, start_distance=100.0, turn="right"), VehicleSpec(SOUTH, start_distance=100.0, turn="straight")]
    result = run_multi_approach_scenario(specs, max_steps=5000)
    assert not result.collided
    n_proceed = _proceed_time(result, "N")
    s_proceed = _proceed_time(result, "S")
    assert n_proceed is not None and s_proceed is not None
    assert abs(n_proceed - s_proceed) < 1.0  # neither made to wait out the other


def test_mixed_turn_random_sweep_never_collides():
    """Broad regression: random start distances and turns across all four approaches must
    never produce a real collision. This is what actually caught the two bugs above."""
    rng = np.random.default_rng(0)
    for _ in range(60):
        turns = rng.choice(["straight", "left", "right"], size=4)
        dists = rng.uniform(60, 140, size=4)
        specs = [
            VehicleSpec(NORTH, start_distance=float(dists[0]), turn=str(turns[0])),
            VehicleSpec(EAST, start_distance=float(dists[1]), turn=str(turns[1])),
            VehicleSpec(SOUTH, start_distance=float(dists[2]), turn=str(turns[2])),
            VehicleSpec(WEST, start_distance=float(dists[3]), turn=str(turns[3])),
        ]
        result = run_multi_approach_scenario(specs, max_steps=8000)
        assert not result.collided, [(s.approach.name, s.turn, s.start_distance) for s in specs]


def test_left_turn_yield_can_gridlock_but_never_collides():
    """Known, accepted liveness limitation (KNOWN_BUGS.md entry 4): the unconditional
    left-yields-to-oncoming rule can form a 3-vehicle wait cycle, which is never unsafe, just not always live."""
    specs = [
        VehicleSpec(NORTH, start_distance=128.7, turn="straight"),
        VehicleSpec(EAST, start_distance=115.8, turn="right"),
        VehicleSpec(SOUTH, start_distance=67.5, turn="left"),
        VehicleSpec(WEST, start_distance=138.0, turn="left"),
    ]
    result = run_multi_approach_scenario(specs, max_steps=8000)
    assert not result.collided
    assert all(v.states[-1].name == "STOPPED" for v in result.vehicles)  # the cycle, not a slow resolution


# --- Guards against regressing the geometric-inconsistency bug ------------------------


def test_guard_rejects_turn_lead_too_large_for_start_distance():
    """Code-review finding: the original guard only related turn_lead to the conflict box,
    never to this vehicle's own start_distance, which could reproduce the same mid-curve bug."""
    with pytest.raises(ValueError, match="turn_lead"):
        run_multi_approach_scenario(
            [VehicleSpec(NORTH, start_distance=8.0, turn="left", turning_radius=4.4), VehicleSpec(SOUTH)]
        )


def test_clear_distance_for_a_turning_vehicle_is_past_the_conflict_box_not_just_the_curve():
    """Code-review finding: _clear_distance's turning branch checked only the curve's endpoint,
    not conflict_half_width, so a turning vehicle could be marked "cleared" while still inside the box."""
    from core.intersection2d_harness import _build_route, _clear_distance, _pose_at

    conflict_half_width, clear_margin = 9.5, 3.0
    spec = VehicleSpec(NORTH, start_distance=100.0, turn="left", turning_radius=1.0)  # tiny turn_lead=2.5
    route = _build_route(spec)
    clear_distance = _clear_distance(spec, route, conflict_half_width, clear_margin)

    x, y, _ = _pose_at(spec, route, clear_distance)
    distance_from_center = float(np.hypot(x, y))
    assert distance_from_center >= conflict_half_width + clear_margin - 1e-6
