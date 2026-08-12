import numpy as np
import pytest

from core.environment import VEHICLE_RADIUS
from core.planning.dubins import _solve_csc
from core.planning.hybrid_astar import HybridAStarPlanner
from core.planning.reeds_shepp import ReedsSheppPlanner, _solve_ccc, reeds_shepp_length, reeds_shepp_path
from core.scenario_loader import list_scenarios, load_scenario
from core.vehicle import wrap_angle

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


# --- CCC (LRL/RLR, KNOWN_BUGS.md entry 5): reeds_shepp.py's own close-pose family ----
#
# Built via direct geometric construction (tangent-circle centers), not a from-memory
# trig formula -- verified below by reconstructing every candidate's (t, p, q) through
# the same arc-stepping machinery the rest of this module already trusts
# (dubins.py's _arc_points, via reeds_shepp.py's _ccc_points) and checking it actually
# lands on the goal pose, rather than just trusting the derivation.


def test_ccc_candidates_reconstruct_to_the_goal_pose():
    from core.planning.reeds_shepp import _ccc_points

    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(300):
        start = (0.0, 0.0, float(rng.uniform(-np.pi, np.pi)))
        d = rng.uniform(0.01, 3.9)
        ang = rng.uniform(-np.pi, np.pi)
        goal = (float(d * np.cos(ang)), float(d * np.sin(ang)), float(rng.uniform(-np.pi, np.pi)))
        turning_radius = float(rng.uniform(0.5, 2.0))

        result = _solve_ccc(start, goal, turning_radius)
        if result is None:
            continue  # start/goal turning circles farther apart than 4x turning_radius -- CCC not needed here
        _length, first, mid, last, t, p, q = result
        path = _ccc_points(start, first, mid, last, t, p, q, turning_radius, step=0.5)
        assert path[-1] == pytest.approx(goal, abs=1e-3)
        checked += 1

    assert checked > 100  # sanity: the close-pose sampling actually exercised CCC, not just skipped every time


def test_ccc_is_infeasible_only_when_turning_circles_are_far_apart():
    """The 4x-turning_radius regime KNOWN_BUGS.md entry 5 describes: CCC needs the
    start/goal turning circles within 4*turning_radius of each other (two tangent
    circles of radius 2r fitting between them) -- confirm both sides of that boundary
    behave as expected, not just that *some* CCC cases work."""
    close = _solve_ccc((0.0, 0.0, 0.0), (0.5, 0.0, np.pi), 1.0)  # d=0.5, well under 4x
    assert close is not None

    far = _solve_ccc((0.0, 0.0, 0.0), (20.0, 0.0, np.pi), 1.0)  # d=20, way over 4x
    assert far is None


def test_reeds_shepp_planner_never_raises_in_the_close_pose_regime():
    """Direct regression guard for KNOWN_BUGS.md entry 5's actual finding: CSC alone
    (all 4 families) never made ReedsSheppPlanner.plan() raise in this regime to begin
    with (verified separately, 20,000 random trials) -- this pins that down as a
    permanent test, not just a one-off investigation."""
    planner = ReedsSheppPlanner()
    rng = np.random.default_rng(11)
    for _ in range(300):
        start = (0.0, 0.0, float(rng.uniform(-np.pi, np.pi)))
        d = rng.uniform(0.01, 3.9)
        ang = rng.uniform(-np.pi, np.pi)
        goal = (float(d * np.cos(ang)), float(d * np.sin(ang)), float(rng.uniform(-np.pi, np.pi)))
        planner.plan(start, goal, [], turning_radius=1.0)  # must not raise


def test_reeds_shepp_length_matches_the_actual_generated_path_length():
    """Regression guard for the Euclidean-fallback bug found while validating CCC
    (see reeds_shepp_length's docstring): previously this almost always returned the
    straight-line distance regardless of what path was actually produced, since it was
    unconditionally included in the min(). `include_ccc=True` (the default, used by the
    standalone planner) must report the real selected path's length."""
    rng = np.random.default_rng(13)
    for _ in range(200):
        start = (0.0, 0.0, float(rng.uniform(-np.pi, np.pi)))
        d = rng.uniform(0.5, 8.0)
        ang = rng.uniform(-np.pi, np.pi)
        goal = (float(d * np.cos(ang)), float(d * np.sin(ang)), float(rng.uniform(-np.pi, np.pi)))
        r = float(rng.uniform(0.5, 2.0))

        path = reeds_shepp_path(start, goal, r)
        actual_length = float(np.sum(np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))))
        reported_length = reeds_shepp_length(start, goal, r)
        assert reported_length == pytest.approx(actual_length, rel=0.05)


def test_ccc_produces_meaningfully_shorter_paths_in_the_close_pose_regime():
    """Quantifies the actual value CCC adds (KNOWN_BUGS.md entry 5's real finding: not
    a crash fix, a path-quality one) -- if this ever drops near zero, CCC isn't doing
    anything and the added complexity wouldn't be worth it."""
    rng = np.random.default_rng(17)
    shorter_count = 0
    trials = 300
    for _ in range(trials):
        start = (0.0, 0.0, float(rng.uniform(-np.pi, np.pi)))
        d = rng.uniform(0.01, 3.9)
        ang = rng.uniform(-np.pi, np.pi)
        goal = (float(d * np.cos(ang)), float(d * np.sin(ang)), float(rng.uniform(-np.pi, np.pi)))
        r = 1.0

        best_csc = min(
            (res[0] for res in [_solve_csc(start, goal, r), _solve_csc(goal, start, r)] if res is not None),
            default=None,
        )
        best_ccc = min(
            (res[0] for res in [_solve_ccc(start, goal, r), _solve_ccc(goal, start, r)] if res is not None),
            default=None,
        )
        if best_csc is not None and best_ccc is not None and best_ccc < best_csc - 1e-9:
            shorter_count += 1

    assert shorter_count / trials > 0.15  # measured ~30% on a larger sample; 15% is a safe floor


def test_hybrid_astar_does_not_use_ccc():
    """KNOWN_BUGS.md entry 5 / reeds_shepp.py's module docstring: CCC is deliberately
    NOT available to HybridAStarPlanner (all 4 of its reeds_shepp_length/reeds_shepp_path
    call sites pass include_ccc=False), because it's more curvature-aggressive than CSC
    and reopened Pure Pursuit's curvature-saturation collision risk (KNOWN_BUGS.md bug 1)
    on scenarios that were previously safe when tried unconditionally. This doesn't
    re-derive that finding (tests/test_simulation.py's collision tests already do,
    every run) -- it just pins the source-level guard so a future edit can't silently
    drop `include_ccc=False` from one of those call sites without a test noticing."""
    import inspect

    source = inspect.getsource(HybridAStarPlanner)
    # reeds_shepp_path is called twice (_reconstruct's local connector, plan's analytic
    # expansion), reeds_shepp_length twice more (root heuristic, per-node heuristic).
    assert source.count("include_ccc=False") == 4
