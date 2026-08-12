"""KNOWN_BUGS.md bug 3 / IMPLEMENTATION.md's M4 entry: PlannerNode used to plan once and
never again, so a sensed obstacle the original plan didn't account for just made the
vehicle brake and stay stopped forever. Every scenario this project ships is static and
fully known to the planner up front, so that never actually happened there -- these
tests construct the one case that does: a new obstacle dropped onto an already-planned,
originally obstacle-free route partway through the run.

Two levels of test, deliberately: `PlannerNode`/`ControllerNode`'s re-plan wiring is
tested directly and deterministically (no dependence on a specific nonlinear closed-loop
trajectory actually converging), plus one closed-loop integration test that shows the
whole thing wired together. A pure closed-loop test alone turned out to be a poor primary
tool here -- see `test_replanning_produces_a_materially_different_obstacle_avoiding_path`
for why the closed-loop case stops short of *completing* the detour and why that's a
separate, narrower issue from the bug being fixed.
"""

import numpy as np

from core.control.pure_pursuit import PurePursuitAdaptive
from core.environment import Environment, Obstacle, Spot
from core.harness import ParkingHarness
from core.interfaces import HasPose
from core.messaging.bus import Bus
from core.messaging.messages import ObstacleRangeMsg, PathMsg, PoseEstimateMsg, ReplanRequestMsg
from core.nodes.controller_node import STALL_TICKS, ControllerNode
from core.nodes.planner_node import PlannerNode
from core.planning.hybrid_astar import HybridAStarPlanner
from core.vehicle import Vehicle

TURNING_RADIUS = 3.946579057110876  # Vehicle(wheelbase=2.7, max_steer=0.6).turning_radius


def _pose(x: float, y: float, theta: float) -> PoseEstimateMsg:
    return PoseEstimateMsg(x, y, theta, covariance=np.zeros((3, 3)))


# --- PlannerNode: does it actually re-plan against the *live* obstacle list? ---------


def test_replan_request_produces_a_new_obstacle_avoiding_path():
    bus = Bus()
    planner = HybridAStarPlanner()
    environment = Environment(Spot(0.0, 0.0, 0.0), obstacles=[])
    node = PlannerNode(bus, planner, environment, TURNING_RADIUS, max_replans=2)

    published: list[np.ndarray] = []
    bus.subscribe("path", lambda msg: published.append(msg.path))

    bus.publish("pose_estimate", _pose(-10.0, 0.0, 0.0))
    assert len(published) == 1
    original_path = published[0]
    # The original, obstacle-free plan is a straight line -- confirms it actually runs
    # through where the obstacle below will be dropped, not around it.
    assert np.allclose(original_path[:, 1], 0.0, atol=1e-6)

    obstacle = Obstacle(x=-5.0, y=0.0, radius=1.3)
    environment.obstacles.append(obstacle)
    bus.publish("replan_request", ReplanRequestMsg())

    assert node._replans == 1
    assert len(published) == 2
    new_path = published[1]
    assert new_path.shape != original_path.shape or not np.allclose(new_path, original_path)
    clearance = np.hypot(new_path[:, 0] - obstacle.x, new_path[:, 1] - obstacle.y).min()
    assert clearance >= obstacle.radius + 1.0  # VEHICLE_RADIUS


def test_replan_requests_are_capped_at_max_replans():
    bus = Bus()
    planner = HybridAStarPlanner()
    environment = Environment(Spot(0.0, 0.0, 0.0), obstacles=[])
    node = PlannerNode(bus, planner, environment, TURNING_RADIUS, max_replans=2)

    published = []
    bus.subscribe("path", lambda msg: published.append(msg.path))
    bus.publish("pose_estimate", _pose(-10.0, 0.0, 0.0))

    environment.obstacles.append(Obstacle(x=-5.0, y=0.0, radius=1.3))
    for _ in range(5):  # well past max_replans
        bus.publish("replan_request", ReplanRequestMsg())

    assert node._replans == 2  # capped, not 5
    assert len(published) == 1 + 2  # initial plan + exactly 2 replans


def test_a_planner_that_cannot_find_a_route_leaves_the_old_path_in_place():
    """Every shipped Planner fails loud with some RuntimeError subclass (PlanningFailure
    for Hybrid A*, plain RuntimeError for Dubins/ReedsShepp) rather than returning a
    partial path -- PlannerNode must not crash the simulation when a re-plan attempt hits
    that, and must not publish a nonexistent path."""

    class AlwaysFailsPlanner:
        def plan(self, start, goal, obstacles, turning_radius):
            raise RuntimeError("no route")

    bus = Bus()
    environment = Environment(Spot(0.0, 0.0, 0.0), obstacles=[])
    node = PlannerNode(bus, AlwaysFailsPlanner(), environment, TURNING_RADIUS, max_replans=2)

    published = []
    bus.subscribe("path", lambda msg: published.append(msg.path))
    bus.publish("pose_estimate", _pose(-10.0, 0.0, 0.0))
    assert published == []  # initial plan also failed -- nothing published, no crash

    bus.publish("replan_request", ReplanRequestMsg())
    assert node._replans == 1
    assert published == []  # still nothing -- the failure was swallowed, not crashed on


# --- ControllerNode: does a sustained governed stall actually ask to re-plan? --------


class _AlwaysWantsToMove:
    """Stub Controller: always wants to drive forward, regardless of the path -- isolates
    ControllerNode's stall-detection from any real path-tracking law."""

    def control(self, pose: HasPose, path: np.ndarray) -> tuple[float, float]:
        return 2.0, 0.0


def test_sustained_governed_stall_requests_a_replan():
    bus = Bus()
    node = ControllerNode(bus, _AlwaysWantsToMove(), a_max=0.8, stopping_buffer=0.5)
    requests = []
    bus.subscribe("replan_request", lambda msg: requests.append(msg))

    bus.publish("pose_estimate", _pose(0.0, 0.0, 0.0))
    bus.publish("path", PathMsg(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])))
    # closest_range=1.0 -> gap = 1.0 - VEHICLE_RADIUS(1.0) - stopping_buffer(0.5) < 0 -> v_safe=0
    bus.publish("obstacle_ranges", ObstacleRangeMsg({0.0: 1.0}))

    for tick in range(1, STALL_TICKS + 1):
        node.step()
        if tick < STALL_TICKS:
            assert len(requests) == 0, f"fired early, at tick {tick}"
    assert len(requests) == 1


def test_a_stall_that_never_recovers_keeps_asking_periodically():
    """Not just once: KNOWN_BUGS.md bug 3's residual note -- a re-plan can itself land on
    a route the governor still won't move along -- means a stall has to keep retrying,
    not give up after a single attempt."""
    bus = Bus()
    node = ControllerNode(bus, _AlwaysWantsToMove(), a_max=0.8, stopping_buffer=0.5)
    requests = []
    bus.subscribe("replan_request", lambda msg: requests.append(msg))

    bus.publish("pose_estimate", _pose(0.0, 0.0, 0.0))
    bus.publish("path", PathMsg(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])))
    bus.publish("obstacle_ranges", ObstacleRangeMsg({0.0: 1.0}))

    for _ in range(3 * STALL_TICKS):
        node.step()
    assert len(requests) == 3


def test_recovering_then_re_stalling_asks_again():
    bus = Bus()
    node = ControllerNode(bus, _AlwaysWantsToMove(), a_max=0.8, stopping_buffer=0.5)
    requests = []
    bus.subscribe("replan_request", lambda msg: requests.append(msg))

    bus.publish("pose_estimate", _pose(0.0, 0.0, 0.0))
    bus.publish("path", PathMsg(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])))

    bus.publish("obstacle_ranges", ObstacleRangeMsg({0.0: 1.0}))  # close: stall
    for _ in range(STALL_TICKS):
        node.step()
    assert len(requests) == 1

    bus.publish("obstacle_ranges", ObstacleRangeMsg({0.0: 100.0}))  # far: recovers
    node.step()
    assert node._stall_ticks == 0

    bus.publish("obstacle_ranges", ObstacleRangeMsg({0.0: 1.0}))  # close again
    for _ in range(STALL_TICKS):
        node.step()
    assert len(requests) == 2


# --- Closed-loop: the whole thing wired together through ParkingHarness -------------

# A straight, same-heading start/goal so the *first* (obstacle-free) plan is a clean
# line -- see test_replan_request_produces_a_new_obstacle_avoiding_path, which confirms
# this geometrically. Placed off-center (not dead ahead) so the vehicle's approach isn't
# perfectly symmetric, which -- verified separately while building this test -- is what
# makes both a real stall *and* a real (if incomplete) re-plan attempt reliably
# reproducible here, unlike a dead-ahead obstacle where Hybrid A*'s primitive search
# can't find any detour at all from such a tight, centered near-stop.
DYNAMIC_OBSTACLE = Obstacle(x=-4.0, y=1.8, radius=1.0)
SPAWN_TICK = 3


def _run(max_replans: int, max_steps: int = 500, seed: int = 1):
    vehicle = Vehicle(x=-10.0, y=0.0, theta=0.0, wheelbase=2.7)
    environment = Environment(Spot(0.0, 0.0, 0.0), obstacles=[])
    planner = HybridAStarPlanner()
    controller = PurePursuitAdaptive(wheelbase=vehicle.wheelbase, v_max=1.5, max_steer=vehicle.max_steer)
    harness = ParkingHarness(vehicle, environment, planner, controller, seed=seed, max_replans=max_replans)

    spawned = False

    def on_tick(tick: int) -> None:
        nonlocal spawned
        if tick == SPAWN_TICK and not spawned:
            environment.obstacles.append(DYNAMIC_OBSTACLE)
            spawned = True

    result = harness.run(max_steps=max_steps, on_tick=on_tick)
    return result, harness


def test_never_collides_with_a_dynamically_appearing_obstacle():
    """Safety holds regardless of when the obstacle showed up -- ControllerNode's speed
    governor (KNOWN_BUGS.md bug 1's fix) is what guarantees this on its own, with or
    without re-planning; the tests below are about whether it can also make *progress*,
    not just survive."""
    for max_replans in (0, 3):
        result, _ = _run(max_replans)
        assert not result.collision


def test_replanning_produces_a_materially_different_obstacle_avoiding_path():
    """The direct fix for the bug: without re-planning, the stale (now-invalid) original
    path is never updated, so the vehicle stops short and stays there. With it enabled,
    PlannerNode picks up the live obstacle and computes a real detour.

    This does *not* assert the vehicle goes on to reach the goal in this run: verified
    while building this test that the detour Hybrid A* finds still passes closer to the
    obstacle than ControllerNode's `stopping_buffer` (tuned generously for KNOWN_BUGS.md
    bug 1's slower-approach scenario) allows at speed, so the governor pins it near-zero
    again right at the start of the new route too. That's a real, separate, narrower
    interaction between the governor's fixed buffer and the planner's own tighter
    clearance margin -- worth a future look, but distinct from bug 3 ("never re-plans at
    all"), which is what this test pins down: the path genuinely changes and genuinely
    avoids the obstacle, which it provably did not before this fix existed.
    """
    without_replanning, _ = _run(max_replans=0)
    with_replanning, harness = _run(max_replans=3)

    assert without_replanning.path is not None
    assert harness.planner_node._replans >= 1
    assert with_replanning.path.shape != without_replanning.path.shape or not np.allclose(
        with_replanning.path, without_replanning.path
    )
    clearance = np.hypot(
        with_replanning.path[:, 0] - DYNAMIC_OBSTACLE.x, with_replanning.path[:, 1] - DYNAMIC_OBSTACLE.y
    ).min()
    assert clearance >= DYNAMIC_OBSTACLE.radius + 1.0  # VEHICLE_RADIUS -- a genuinely valid detour


def test_replanning_gives_up_after_max_replans_instead_of_looping_forever():
    """A capped-at-zero re-plan budget must still terminate -- fail-safe (stopped, no
    collision), not an infinite retry loop -- within the step budget."""
    result, _ = _run(max_replans=0, max_steps=300)
    assert not result.collision
    assert not result.success
    assert len(result.true_history) == 300  # ran out the clock rather than hanging
