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

import numpy as np
import pytest

from core.control.mpc import MPCController
from core.control.pure_pursuit import PurePursuitAdaptive
from core.environment import VEHICLE_RADIUS
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


def _min_clearance(true_history: np.ndarray, obstacles) -> float:
    """Minimum signed vehicle-to-obstacle clearance across a run: negative means the
    vehicle's collision circle actually overlapped an obstacle's (by that many meters),
    positive means it stayed clear by that much. Same circle-circle geometry as
    `ParkingHarness._collided`, just continuous instead of thresholded at exactly 0."""
    xy = true_history[:, :2]
    return min(
        float(np.min(np.hypot(xy[:, 0] - o.x, xy[:, 1] - o.y) - (o.radius + VEHICLE_RADIUS)))
        for o in obstacles
    )


def test_latency_margin_is_what_actually_closes_the_gap():
    """Regression for the fix itself: on this exact case, a controller that doesn't
    know about the delay (latency_margin forced to 0, simulating the pre-fix governor)
    penetrates a real obstacle by several centimeters; the real fix (latency_margin
    computed from sensor_latency_ticks) keeps a comfortable multi-decimeter clearance
    instead -- confirms the governor's extra margin is load-bearing, not redundant with
    something else that would have caught it anyway. `parallel_between_cars` is
    already KNOWN_BUGS.md entry 2's tightest-margin scenario, so it's also the most
    sensitive one to a missing margin.

    Asserts on continuous minimum clearance, not the boolean `result.collision`, with
    real numerical headroom on both sides of zero -- KNOWN_BUGS.md entry 8: at this
    test's original `latency_ticks=5`, the no-fix case is itself only a ~2-7mm
    penetration (measured across seeds 1-5), smaller than the floating-point
    differences a different BLAS-backend/CPU platform's SLSQP solve (`MPCController`)
    produces over this trajectory's ~300+ ticks -- observed to flip the boolean outcome
    between a Windows host and a Linux Docker container despite byte-identical
    numpy/scipy versions on both. `latency_ticks=10` -- still inside this scenario's
    own documented verified-safe upper bound (entry 7) -- produces a consistent ~6cm
    penetration without the fix and >15cm clearance with it, confirmed matching between
    host and container to within ~3mm, well clear of that noise floor."""

    def _run(latency_margin_override: float | None) -> float:
        # Reloaded per call, not shared: VehicleNode.update() mutates the Vehicle object
        # it's given in place, so reusing one `scenario` across both calls would have the
        # second run silently start from wherever the first run's vehicle ended up.
        scenario = load_scenario("parallel_between_cars")
        planner = HybridAStarPlanner()
        controller = MPCController(
            wheelbase=scenario.vehicle.wheelbase, delta_max=scenario.vehicle.max_steer, v_max=1.5
        )
        harness = ParkingHarness(
            scenario.vehicle, scenario.environment, planner, controller, seed=1, sensor_latency_ticks=10
        )
        if latency_margin_override is not None:
            harness.controller_node.latency_margin = latency_margin_override  # simulate the pre-fix governor
        result = harness.run(max_steps=1000)
        return _min_clearance(result.true_history, scenario.environment.obstacles)

    assert _run(0.0) < -0.03  # decisive penetration without the fix, not a hairline zero-crossing
    assert _run(None) > 0.05  # and a comfortable, non-hairline gap with the real fix
