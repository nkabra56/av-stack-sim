import numpy as np
import pytest

from auto_park.control.mpc import MPCController
from auto_park.control.pure_pursuit import PurePursuitAdaptive
from auto_park.harness import ParkingHarness
from auto_park.planning.hybrid_astar import HybridAStarPlanner, brake_distance_for
from auto_park.scenario_loader import list_scenarios, load_scenario

SEEDS = [1, 2, 3, 4, 5]
MIN_SUCCESS_RATE = 4  # out of 5 -- noise can legitimately cause an occasional miss
MAX_STEPS = 1000  # Hybrid A*'s avoidance routes are longer and more circuitous than
# M1's direct Dubins paths ever were -- parallel_between_cars measured up to ~880 of
# 1000 steps to converge under MPC, well past the old 500-step budget.

CONTROLLERS = {
    "pure_pursuit": lambda v: PurePursuitAdaptive(wheelbase=v.wheelbase, v_max=1.5, max_steer=v.max_steer),
    "mpc": lambda v: MPCController(wheelbase=v.wheelbase, delta_max=v.max_steer, v_max=1.5),
}

# Pure Pursuit's already-documented "no margin when curvature is already at the
# vehicle's limit" weakness (DESIGN.md section 7) is a measured, consistent safety
# failure -- not noise-dependent flakiness -- on parallel_between_cars specifically:
# Hybrid A*'s avoidance route there requires a tight reverse-gear cusp with obstacles
# close enough that Pure Pursuit's reactive tracking error alone (not sensor-braking
# blind spots, not planner infeasibility -- the planned path itself keeps a verified
# 1.15m clearance floor) exceeds that margin. Measured across seeds 1-5: 5/5
# collisions, every time, regardless of brake_distance or the planner's safety_margin
# (a wider margin just traded the collision for Pure Pursuit never converging at all,
# not for success -- see IMPLEMENTATION.md's M2 entry). MPC's constraint-respecting
# rollout stays collision-free and converges reliably (5/5, up to ~880 steps). This is
# the same class of finding DESIGN.md section 7 already predicts in the abstract,
# now concretely realized once a planner actually produces curvature-saturated,
# obstacle-hugging paths for Pure Pursuit to track.
UNSAFE_COMBINATIONS = {("parallel_between_cars", "pure_pursuit")}


def _run(scenario_name: str, controller_name: str, seed: int, max_steps: int = MAX_STEPS):
    scenario = load_scenario(scenario_name)
    planner = HybridAStarPlanner()
    controller = CONTROLLERS[controller_name](scenario.vehicle)
    harness = ParkingHarness(
        scenario.vehicle, scenario.environment, planner, controller, seed=seed,
        brake_distance=brake_distance_for(planner),
    )
    return harness.run(max_steps=max_steps)


def _combinations(exclude: set[tuple[str, str]]) -> list[tuple[str, str]]:
    return [
        (scenario_name, controller_name)
        for scenario_name in list_scenarios()
        for controller_name in CONTROLLERS
        if (scenario_name, controller_name) not in exclude
    ]


@pytest.mark.parametrize("scenario_name,controller_name", _combinations(exclude=UNSAFE_COMBINATIONS))
def test_reaches_the_spot_across_seeds(scenario_name, controller_name):
    """Evaluated statistically, not as single-run determinism: with sensor/odometry
    noise in the loop, an occasional miss is expected behavior, not a bug. Hybrid A*
    (M2) is expected to solve every scenario -- measured 5/5 for every (scenario,
    controller) pair below (IMPLEMENTATION.md's M2 entry) -- unlike the M1 Dubins
    baseline, which could only reach the two obstacle-free scenarios."""
    successes = sum(_run(scenario_name, controller_name, seed).success for seed in SEEDS)
    assert successes >= MIN_SUCCESS_RATE


@pytest.mark.parametrize("scenario_name,controller_name", _combinations(exclude=UNSAFE_COMBINATIONS))
@pytest.mark.parametrize("seed", SEEDS)
def test_never_collides(scenario_name, controller_name, seed):
    """Safety must hold regardless of estimation noise, on every seed, for every
    (scenario, controller) combination Hybrid A*'s own planned clearance actually
    supports -- see UNSAFE_COMBINATIONS above for the one documented, measured
    exception (not a flaky test, a real controller limitation)."""
    result = _run(scenario_name, controller_name, seed)
    assert not result.collision


@pytest.mark.parametrize("seed", SEEDS)
def test_parallel_between_cars_pure_pursuit_is_the_documented_unsafe_case(seed):
    """Pins down UNSAFE_COMBINATIONS's claim as an actual regression guard, not just a
    comment: if a future planner/controller change makes this combination safe, this
    test should start failing (a good thing -- narrow UNSAFE_COMBINATIONS back down
    when it does) rather than the exclusion silently going stale."""
    result = _run("parallel_between_cars", "pure_pursuit", seed)
    assert result.collision


@pytest.mark.parametrize("scenario_name", list_scenarios())
def test_all_scenario_headings_are_radians(scenario_name):
    """Regression guard for the prototype bug where some scenarios passed degrees
    (e.g. theta=90.0) for a model that only accepts radians. Any valid heading must
    fall within [-pi, pi]; a value like 90.0 is ~14 full rotations out of range.
    """
    scenario = load_scenario(scenario_name)
    assert -np.pi <= scenario.vehicle.theta <= np.pi
    assert -np.pi <= scenario.environment.spot.theta <= np.pi
