"""KNOWN_BUGS.md entry 4: H4's `IntersectionNavigator` models right-of-way as mutual
exclusion + arrival-order priority at a single point two approaches share -- correct
reasoning, but no way to represent real crossing paths, more than two approaches, or
verify geometrically that the reasoning actually avoided a collision. These tests run
several real vehicles (still each just an unmodified `IntersectionNavigator`, same reuse
principle H4 itself already used for `IDMController`) through one shared, real 2D
intersection built from core/control/intersection_geometry.py, and check real (x, y)
proximity, not just arrival-order bookkeeping.

Turning movements are still out of scope -- see intersection_geometry.py's module
docstring -- so every vehicle here goes straight through.
"""

import pytest

from core.control.intersection import IntersectionNavigator
from core.control.intersection_geometry import EAST, NORTH, SOUTH, WEST, is_to_the_right
from core.intersection2d_harness import VehicleSpec, run_multi_approach_scenario


def _proceed_time(result, name: str) -> float | None:
    vehicle = next(v for v in result.vehicles if v.name == name)
    for t, s in zip(result.times, vehicle.states):
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
    """North and South share a road but not a lane or a crossing path -- the single-
    conflict-point model can't express "these two don't actually interact at all";
    real geometry can."""
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
    """Never yields -- always claims right of way, regardless of who else is there."""

    def has_right_of_way(self, others):
        return True


def test_collision_check_actually_catches_a_non_compliant_vehicle():
    """If every scenario above passed only because nothing was ever really at risk, that
    would make them worthless as regression tests. Prove the checker is live: give one
    vehicle a navigator that ignores right-of-way entirely on a simultaneous-arrival
    pair that the compliant version above (test_simultaneous_arrival_...) already
    proved is otherwise safe, and confirm a real geometric collision is detected."""
    specs = [VehicleSpec(NORTH, start_distance=100.0), VehicleSpec(EAST, start_distance=100.0)]
    navigators = [
        IntersectionNavigator(stop_line_position=specs[0].start_distance - 7.0 - 1.5, v_cruise=15.0),
        _RecklessNavigator(stop_line_position=specs[1].start_distance - 7.0 - 1.5, v_cruise=15.0),
    ]
    result = run_multi_approach_scenario(specs, navigators=navigators)
    assert result.collided
    assert ("N", "E") in result.collision_pairs
