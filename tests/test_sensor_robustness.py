"""End-to-end robustness of the closed loop under sensor dropout/latency: SensorNode's
own params are unit-tested in test_sensor_node.py. Thresholds below come from a real sweep (KNOWN_BUGS.md)."""

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
    vehicle's collision circle overlapped an obstacle's. Same geometry as ParkingHarness._collided."""
    xy = true_history[:, :2]
    return min(
        float(np.min(np.hypot(xy[:, 0] - o.x, xy[:, 1] - o.y) - (o.radius + VEHICLE_RADIUS)))
        for o in obstacles
    )


def test_latency_margin_is_what_actually_closes_the_gap():
    """Regression for the fix itself: forcing latency_margin to 0 (the pre-fix governor)
    penetrates a real obstacle; the real fix keeps a comfortable clearance instead (KNOWN_BUGS.md entry 8)."""

    def _run(latency_margin_override: float | None) -> float:
        # Reloaded per call, not shared: VehicleNode.update() mutates the Vehicle in place,
        # so reusing one scenario would have the second run start where the first ended.
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
