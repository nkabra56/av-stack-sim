import numpy as np
import pytest

from core.control.mpc import MPCController
from core.control.pure_pursuit import PurePursuitAdaptive
from core.harness import ParkingHarness
from core.planning.hybrid_astar import HybridAStarPlanner
from core.scenario_loader import list_scenarios, load_scenario

SEEDS = [1, 2, 3, 4, 5]
MIN_SUCCESS_RATE = 4  # out of 5: noise can legitimately cause an occasional miss
MAX_STEPS = 1000  # Hybrid A*'s avoidance routes are longer than M1's direct Dubins paths:
# parallel_between_cars measured up to ~880 of 1000 steps to converge under MPC.

CONTROLLERS = {
    "pure_pursuit": lambda v: PurePursuitAdaptive(wheelbase=v.wheelbase, v_max=1.5, max_steer=v.max_steer),
    "mpc": lambda v: MPCController(wheelbase=v.wheelbase, delta_max=v.max_steer, v_max=1.5),
}

# Pure Pursuit's documented curvature-saturation weakness (DESIGN.md section 7): the governor
# stops it safely here (KNOWN_BUGS.md entry 2) but it can't complete the maneuver; MPC does (5/5).
NEVER_SUCCEEDS = {("parallel_between_cars", "pure_pursuit")}


def _run(scenario_name: str, controller_name: str, seed: int, max_steps: int = MAX_STEPS):
    scenario = load_scenario(scenario_name)
    planner = HybridAStarPlanner()
    controller = CONTROLLERS[controller_name](scenario.vehicle)
    harness = ParkingHarness(scenario.vehicle, scenario.environment, planner, controller, seed=seed)
    return harness.run(max_steps=max_steps)


def _combinations(exclude: set[tuple[str, str]]) -> list[tuple[str, str]]:
    return [
        (scenario_name, controller_name)
        for scenario_name in list_scenarios()
        for controller_name in CONTROLLERS
        if (scenario_name, controller_name) not in exclude
    ]


@pytest.mark.parametrize("scenario_name,controller_name", _combinations(exclude=NEVER_SUCCEEDS))
def test_reaches_the_spot_across_seeds(scenario_name, controller_name):
    """Evaluated statistically, not single-run determinism: with sensor/odometry noise,
    an occasional miss is expected. Hybrid A* is expected to solve every pair below (5/5)."""
    successes = sum(_run(scenario_name, controller_name, seed).success for seed in SEEDS)
    assert successes >= MIN_SUCCESS_RATE


@pytest.mark.parametrize("scenario_name,controller_name", _combinations(exclude=set()))
@pytest.mark.parametrize("seed", SEEDS)
def test_never_collides(scenario_name, controller_name, seed):
    """Safety must hold on every seed for every combination, including
    parallel_between_cars/pure_pursuit, which now fails safe instead of colliding (KNOWN_BUGS.md entry 2)."""
    result = _run(scenario_name, controller_name, seed)
    assert not result.collision


@pytest.mark.parametrize("seed", SEEDS)
def test_parallel_between_cars_pure_pursuit_fails_safe_not_success(seed):
    """Pins down NEVER_SUCCEEDS as an actual regression guard: if a future change makes
    this combination succeed, this test should start failing (a good thing)."""
    result = _run("parallel_between_cars", "pure_pursuit", seed)
    assert not result.collision
    assert not result.success


@pytest.mark.parametrize("scenario_name", list_scenarios())
def test_all_scenario_headings_are_radians(scenario_name):
    """Regression guard for the prototype bug where some scenarios passed degrees instead
    of radians. Any valid heading must fall within [-pi, pi]."""
    scenario = load_scenario(scenario_name)
    assert -np.pi <= scenario.vehicle.theta <= np.pi
    assert -np.pi <= scenario.environment.spot.theta <= np.pi
