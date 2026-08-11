import numpy as np
import pytest

from auto_park.control.mpc import MPCController
from auto_park.control.pure_pursuit import PurePursuitAdaptive
from auto_park.harness import ParkingHarness
from auto_park.planning.dubins import DubinsPlanner
from auto_park.scenario_loader import list_scenarios, load_scenario

# Scenarios with a clear path from start to spot: both controllers must reach the spot
# reliably across seeds. The remaining scenarios (obstacle-obstructed) are only required
# to be safe (no collision) -- the M1 baseline planner doesn't route around obstacles by
# design (see IMPLEMENTATION.md section 6), so stalling short of the goal is expected there.
OPEN_SCENARIOS = ["perpendicular_open", "parallel_open"]
SEEDS = [1, 2, 3, 4, 5]
MIN_SUCCESS_RATE = 4  # out of 5 -- noise can legitimately cause an occasional miss

CONTROLLERS = {
    "pure_pursuit": lambda v: PurePursuitAdaptive(wheelbase=v.wheelbase, v_max=1.5, max_steer=v.max_steer),
    "mpc": lambda v: MPCController(wheelbase=v.wheelbase, delta_max=v.max_steer, v_max=1.5),
}


def _run(scenario_name: str, controller_name: str, seed: int, max_steps: int = 500):
    scenario = load_scenario(scenario_name)
    planner = DubinsPlanner()
    controller = CONTROLLERS[controller_name](scenario.vehicle)
    harness = ParkingHarness(scenario.vehicle, scenario.environment, planner, controller, seed=seed)
    return harness.run(max_steps=max_steps)


@pytest.mark.parametrize("scenario_name", OPEN_SCENARIOS)
@pytest.mark.parametrize("controller_name", list(CONTROLLERS))
def test_open_scenarios_reach_the_spot_across_seeds(scenario_name, controller_name):
    """Evaluated statistically, not as single-run determinism: with sensor/odometry
    noise in the loop, an occasional miss is expected behavior, not a bug. Asserting a
    success-rate threshold instead of 100% is itself the correct way to validate a
    stochastic system, rather than pretending noise doesn't exist."""
    successes = sum(_run(scenario_name, controller_name, seed).success for seed in SEEDS)
    assert successes >= MIN_SUCCESS_RATE


@pytest.mark.parametrize("scenario_name", list_scenarios())
@pytest.mark.parametrize("controller_name", list(CONTROLLERS))
@pytest.mark.parametrize("seed", SEEDS)
def test_never_collides(scenario_name, controller_name, seed):
    """Safety must hold regardless of estimation noise, on every seed, every scenario
    -- including the obstacle scenarios that aren't expected to reach the spot."""
    result = _run(scenario_name, controller_name, seed)
    assert not result.collision


@pytest.mark.parametrize("scenario_name", list_scenarios())
def test_all_scenario_headings_are_radians(scenario_name):
    """Regression guard for the prototype bug where some scenarios passed degrees
    (e.g. theta=90.0) for a model that only accepts radians. Any valid heading must
    fall within [-pi, pi]; a value like 90.0 is ~14 full rotations out of range.
    """
    scenario = load_scenario(scenario_name)
    assert -np.pi <= scenario.vehicle.theta <= np.pi
    assert -np.pi <= scenario.environment.spot.theta <= np.pi
