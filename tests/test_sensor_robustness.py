"""End-to-end robustness of the closed loop under sensor dropout/latency (DESIGN.md
section 10's future-extensions list) -- SensorNode's dropout_prob/latency_ticks are
unit-tested directly in test_sensor_node.py; these confirm the *closed loop* (EKF +
planner + controller together) stays collision-free under them across real scenarios,
not just that individual messages are dropped/delayed correctly in isolation.

Thresholds below (dropout_prob<=0.2, latency_ticks<=10) aren't arbitrary -- a real
sweep (dropout 0.1-0.4, latency 5-50, both controllers, all 5 scenarios, 5 seeds each)
found this is where safety actually holds; beyond it, real collisions start (dropout
0.3 reopens KNOWN_BUGS.md entry 2's already-razor-thin parallel_between_cars/
pure_pursuit margin; latency 20+ lets EKF corrections lag long enough that dead-
reckoning drift alone -- confirmed directly: true/estimated position error reached
2.7-4.0m in one such run -- lets a reactive controller steer the *true* vehicle into
an obstacle the *estimated* vehicle would have cleared, a materially different failure
mode than anything a stopping-buffer margin can fix). See KNOWN_BUGS.md for the full
account of both residuals.
"""

import pytest

from core.control.mpc import MPCController
from core.control.pure_pursuit import PurePursuitAdaptive
from core.harness import ParkingHarness
from core.planning.hybrid_astar import HybridAStarPlanner
from core.scenario_loader import list_scenarios, load_scenario

SEEDS = [1, 2, 3]
CONTROLLERS = {
    "pure_pursuit": lambda v: PurePursuitAdaptive(wheelbase=v.wheelbase, v_max=1.5, max_steer=v.max_steer),
    "mpc": lambda v: MPCController(wheelbase=v.wheelbase, delta_max=v.max_steer, v_max=1.5),
}


def _run(scenario_name: str, controller_name: str, seed: int, **sensor_kwargs):
    scenario = load_scenario(scenario_name)
    planner = HybridAStarPlanner()
    controller = CONTROLLERS[controller_name](scenario.vehicle)
    harness = ParkingHarness(scenario.vehicle, scenario.environment, planner, controller, seed=seed, **sensor_kwargs)
    return harness.run(max_steps=1000)


@pytest.mark.parametrize("scenario_name", list_scenarios())
@pytest.mark.parametrize("controller_name", list(CONTROLLERS))
def test_never_collides_under_sensor_dropout(scenario_name, controller_name):
    for seed in SEEDS:
        result = _run(scenario_name, controller_name, seed, sensor_dropout_prob=0.2)
        assert not result.collision


@pytest.mark.parametrize("scenario_name", list_scenarios())
@pytest.mark.parametrize("controller_name", list(CONTROLLERS))
def test_never_collides_under_sensor_latency(scenario_name, controller_name):
    for seed in SEEDS:
        result = _run(scenario_name, controller_name, seed, sensor_latency_ticks=10)
        assert not result.collision


def test_latency_margin_is_what_actually_closes_the_gap():
    """Regression for the fix itself: on this exact case, a controller that doesn't
    know about the delay (latency_margin forced to 0, simulating the pre-fix governor)
    collides; the real fix (latency_margin computed from sensor_latency_ticks) does
    not -- confirms the governor's extra margin is load-bearing, not redundant with
    something else that would have caught it anyway. `parallel_between_cars` is
    already KNOWN_BUGS.md entry 2's tightest-margin scenario, so it's also the most
    sensitive one to a missing margin -- collides reliably at latency=5 without the
    fix (verified across all 5 seeds, both controllers) despite that latency level
    being otherwise fully safe (see test_never_collides_under_sensor_latency above)."""
    scenario = load_scenario("parallel_between_cars")
    planner = HybridAStarPlanner()
    controller = MPCController(wheelbase=scenario.vehicle.wheelbase, delta_max=scenario.vehicle.max_steer, v_max=1.5)
    harness = ParkingHarness(
        scenario.vehicle, scenario.environment, planner, controller, seed=1, sensor_latency_ticks=5
    )
    harness.controller_node.latency_margin = 0.0  # simulate the pre-fix governor directly
    result = harness.run(max_steps=1000)
    assert result.collision
