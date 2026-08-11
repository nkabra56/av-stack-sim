import numpy as np
import pytest

from auto_park.environment import VEHICLE_RADIUS
from auto_park.planning.hybrid_astar import HybridAStarPlanner
from auto_park.planning.reeds_shepp import ReedsSheppPlanner
from auto_park.scenario_loader import list_scenarios, load_scenario
from auto_park.vehicle import wrap_angle

PLANNERS = {
    "reeds_shepp": ReedsSheppPlanner(),
    "hybrid_astar": HybridAStarPlanner(),
}
OBSTACLE_SCENARIOS = ["perpendicular_flanked", "perpendicular_obstructed_lane", "parallel_between_cars"]


def _scenario_pose(name: str):
    scenario = load_scenario(name)
    turning_radius = scenario.vehicle.turning_radius
    start = (scenario.vehicle.x, scenario.vehicle.y, scenario.vehicle.theta)
    goal = (scenario.environment.spot.x, scenario.environment.spot.y, scenario.environment.spot.theta)
    return scenario, start, goal, turning_radius


def _curvature(path: np.ndarray) -> np.ndarray:
    dx, dy = np.diff(path[:, 0]), np.diff(path[:, 1])
    ds = np.hypot(dx, dy)
    dtheta = np.abs(wrap_angle(np.diff(path[:, 2])))
    mask = ds > 1e-6  # guard div-by-zero at duplicate/cusp points
    return dtheta[mask] / ds[mask]


@pytest.mark.parametrize("planner_name", list(PLANNERS))
@pytest.mark.parametrize("scenario_name", list_scenarios())
def test_path_starts_and_ends_at_the_given_poses(planner_name, scenario_name):
    """Tight tolerance, not the controller's noisy tol=0.4: both planners are
    guaranteed to land exactly on the given start/goal poses by construction
    (closed-form endpoint for Reeds-Shepp, a verified analytic-expansion connection
    for Hybrid A* -- see hybrid_astar.py's module docstring)."""
    scenario, start, goal, turning_radius = _scenario_pose(scenario_name)
    path = PLANNERS[planner_name].plan(start, goal, scenario.environment.obstacles, turning_radius)
    assert path[0] == pytest.approx(start, abs=1e-2)
    assert path[-1] == pytest.approx(goal, abs=1e-2)


@pytest.mark.parametrize("planner_name", list(PLANNERS))
@pytest.mark.parametrize("scenario_name", list_scenarios())
def test_path_never_exceeds_the_vehicle_turning_radius(planner_name, scenario_name):
    """Same style of discrete curvature check that caught the M1 Bezier-curvature bug
    (IMPLEMENTATION.md section 6): curvature must never exceed 1/turning_radius
    anywhere along the path. +1e-3 epsilon absorbs the chord-vs-arc-length
    discretization slop (chord distance is always slightly shorter than true arc
    length, so dtheta/chord slightly *over*-estimates true curvature -- the safe
    direction), not a real tolerance widening."""
    scenario, start, goal, turning_radius = _scenario_pose(scenario_name)
    path = PLANNERS[planner_name].plan(start, goal, scenario.environment.obstacles, turning_radius)
    curvature = _curvature(path)
    assert np.all(curvature <= 1.0 / turning_radius + 1e-3)


@pytest.mark.parametrize("scenario_name", OBSTACLE_SCENARIOS)
def test_hybrid_astar_never_comes_within_the_vehicle_radius_of_an_obstacle(scenario_name):
    scenario, start, goal, turning_radius = _scenario_pose(scenario_name)
    path = HybridAStarPlanner().plan(start, goal, scenario.environment.obstacles, turning_radius)
    for obstacle in scenario.environment.obstacles:
        dist = np.hypot(path[:, 0] - obstacle.x, path[:, 1] - obstacle.y)
        assert dist.min() >= obstacle.radius + VEHICLE_RADIUS - 1e-6
