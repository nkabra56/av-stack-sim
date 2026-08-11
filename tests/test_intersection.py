import pytest

from core.control.intersection import IntersectionState
from core.intersection_harness import (
    no_other_vehicle,
    other_vehicle_present_from,
    run_intersection_scenario,
)

# Hand-authored scenarios (same pattern as the parking side's obstacle scenarios --
# not derived from a real dataset, since the point is exercising specific right-of-way
# logic branches, not real-world realism).


def test_never_runs_the_stop_sign_alone():
    result = run_intersection_scenario(no_other_vehicle)
    assert not result.ran_stop_sign
    assert result.ego_stop_time is not None


def test_never_runs_the_stop_sign_with_another_vehicle_present():
    other = other_vehicle_present_from(arrival_time=5.0, clear_time=15.0)
    result = run_intersection_scenario(other)
    assert not result.ran_stop_sign


def test_proceeds_promptly_with_no_conflict():
    result = run_intersection_scenario(no_other_vehicle)
    assert result.proceed_time is not None
    assert result.proceed_time - result.ego_stop_time < 1.0


def test_yields_to_a_vehicle_that_arrived_first():
    """Other vehicle stops well before ego does and hasn't cleared yet -- ego must
    wait for it, not proceed as soon as ego itself stops."""
    other = other_vehicle_present_from(arrival_time=2.0, clear_time=20.0)
    result = run_intersection_scenario(other)
    assert result.ego_stop_time is not None
    assert result.proceed_time is not None
    assert result.proceed_time >= 20.0


def test_proceeds_first_when_ego_arrived_first():
    """Other vehicle doesn't stop until well after ego already has -- ego shouldn't
    wait for a vehicle that wasn't even there yet when it committed to stopping."""
    other = other_vehicle_present_from(arrival_time=13.0, clear_time=25.0)
    result = run_intersection_scenario(other)
    assert result.proceed_time is not None
    assert result.proceed_time < 13.0


def test_yields_to_the_right_on_simultaneous_arrival():
    other = other_vehicle_present_from(arrival_time=10.4, clear_time=20.0, is_to_the_right=True)
    result = run_intersection_scenario(other)
    assert result.proceed_time is not None
    assert result.proceed_time >= 20.0


def test_does_not_yield_to_a_simultaneous_vehicle_on_the_left():
    """Same timing as the yield-to-the-right case, but the other vehicle is on the
    ego's left -- ego should have priority and not wait."""
    other = other_vehicle_present_from(arrival_time=10.4, clear_time=20.0, is_to_the_right=False)
    result = run_intersection_scenario(other)
    assert result.proceed_time is not None
    assert result.proceed_time < 20.0


def test_state_sequence_is_well_formed():
    result = run_intersection_scenario(no_other_vehicle)
    seen = []
    for s in result.states:
        if not seen or seen[-1] != s:
            seen.append(s)
    assert seen == [IntersectionState.APPROACHING, IntersectionState.STOPPED, IntersectionState.PROCEEDING]
