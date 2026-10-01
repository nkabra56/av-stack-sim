"""Closes DESIGN.md's future-extensions gap: re-planning previously only ever handled a
*static* obstacle appearing once (KNOWN_BUGS.md entry 3). This covers a genuinely *moving*
one, with the EKF-landmark isolation the task that built this called out as the one thing
to get right (Environment.moving_obstacles must never be enumerated as an EKF landmark)."""

import numpy as np
import pytest

from core.control.pure_pursuit import PurePursuitAdaptive
from core.environment import VEHICLE_RADIUS, Environment, Obstacle, Spot
from core.harness import ParkingHarness
from core.messaging.bus import Bus
from core.messaging.messages import PoseEstimateMsg, ReplanRequestMsg, TrueStateMsg
from core.moving_obstacle import MovingObstacleSpec, obstacle_at
from core.nodes.planner_node import PlannerNode
from core.nodes.sensor_node import SensorNode
from core.planning.hybrid_astar import HybridAStarPlanner
from core.scenarios.pedestrian_crossing import PEDESTRIAN, build_scenario
from core.sensors import UltrasonicArray
from core.vehicle import Vehicle

TURNING_RADIUS = 3.946579057110876  # Vehicle(wheelbase=2.7, max_steer=0.6).turning_radius


def _pose(x: float, y: float, theta: float) -> PoseEstimateMsg:
    return PoseEstimateMsg(x, y, theta, covariance=np.zeros((3, 3)))


# --- core/moving_obstacle.py: the pure position function on its own ------------------


def test_obstacle_at_is_absent_before_start_time():
    spec = MovingObstacleSpec(start_x=0.0, start_y=0.0, vx=1.0, vy=0.0, radius=0.5, start_time=2.0)
    assert obstacle_at(spec, 0.0) is None
    assert obstacle_at(spec, 1.999) is None


def test_obstacle_at_moves_at_constant_velocity_after_start_time():
    spec = MovingObstacleSpec(start_x=1.0, start_y=-2.0, vx=0.5, vy=1.0, radius=0.3, start_time=1.0)
    o = obstacle_at(spec, 3.0)  # 2s elapsed
    assert o is not None
    assert o.x == pytest.approx(1.0 + 0.5 * 2.0)
    assert o.y == pytest.approx(-2.0 + 1.0 * 2.0)
    assert o.radius == 0.3


def test_obstacle_at_freezes_after_duration_elapses():
    spec = MovingObstacleSpec(start_x=0.0, start_y=0.0, vx=1.0, vy=0.0, radius=0.3, start_time=0.0, duration=2.0)
    frozen_at_end = obstacle_at(spec, 2.0)
    frozen_later = obstacle_at(spec, 100.0)
    assert frozen_at_end is not None and frozen_later is not None
    assert (frozen_later.x, frozen_later.y) == (frozen_at_end.x, frozen_at_end.y) == (2.0, 0.0)


# --- ParkingHarness: moving_obstacles wired in without a hand-rolled on_tick --------


def test_harness_updates_moving_obstacles_from_specs_without_on_tick():
    """Point 2's requirement: no test should need to hand-roll an on_tick mutation for a
    moving obstacle (that pattern is still used for a plain static appearance, see
    tests/test_replanning.py); ParkingHarness must do it internally, every tick."""
    vehicle, environment, moving_obstacles = build_scenario()
    planner = HybridAStarPlanner()
    controller = PurePursuitAdaptive(wheelbase=vehicle.wheelbase, v_max=1.5, max_steer=vehicle.max_steer)
    harness = ParkingHarness(vehicle, environment, planner, controller, seed=1, moving_obstacles=moving_obstacles)

    assert environment.moving_obstacles == []  # not yet appeared (start_time=0.8s)
    harness.run(max_steps=20)  # 2.0s: well past the pedestrian's start_time
    assert len(environment.moving_obstacles) == 1
    expected = obstacle_at(PEDESTRIAN, 19 * harness.dt)  # last tick's mutation, before the loop exits
    assert environment.moving_obstacles[0] == expected


# --- Safety: never collides with the moving obstacle, across seeds, incl. a close call ---


# Tuned by simulation (see core/scenarios/pedestrian_crossing.py's docstring): seed 8 is the
# tightest timing (ego and pedestrian reach the same point almost simultaneously).
SEEDS = list(range(1, 11))
CLOSE_CALL_SEED = 8


def _run_pedestrian_crossing(seed: int, max_replans: int = 3, max_steps: int = 400):
    vehicle, environment, moving_obstacles = build_scenario()
    planner = HybridAStarPlanner()
    controller = PurePursuitAdaptive(wheelbase=vehicle.wheelbase, v_max=1.5, max_steer=vehicle.max_steer)
    harness = ParkingHarness(
        vehicle, environment, planner, controller, seed=seed, max_replans=max_replans,
        moving_obstacles=moving_obstacles,
    )
    result = harness.run(max_steps=max_steps)
    return result, harness


def _min_clearance_to_pedestrian(result, dt: float) -> float:
    """Clearance (can go negative on contact) between the ego's true position and the
    pedestrian's live position at that same tick, minimized over the whole run."""
    clearances = []
    for i, (x, y, _theta) in enumerate(result.true_history):
        o = obstacle_at(PEDESTRIAN, i * dt)
        if o is None:
            continue
        clearances.append(np.hypot(x - o.x, y - o.y) - o.radius - VEHICLE_RADIUS)
    return min(clearances)


def test_never_collides_with_the_crossing_pedestrian_across_seeds():
    for seed in SEEDS:
        result, _ = _run_pedestrian_crossing(seed)
        assert not result.collision, f"seed {seed} collided with the pedestrian"


def test_close_call_seed_clears_by_a_small_but_positive_margin():
    """Proves the safety margin above isn't trivially loose: this seed's ego and pedestrian
    nearly occupy the same point at the same time, and it still clears, by only a few cm."""
    result, harness = _run_pedestrian_crossing(CLOSE_CALL_SEED)
    clearance = _min_clearance_to_pedestrian(result, harness.dt)
    assert not result.collision
    assert 0.0 < clearance < 0.15  # genuinely tight, not a comfortable miss


def test_crossing_pedestrian_forces_at_least_one_real_replan():
    for seed in SEEDS:
        _, harness = _run_pedestrian_crossing(seed)
        assert harness.planner_node._replans >= 1, f"seed {seed} never re-planned"


def test_most_seeds_still_reach_the_goal_despite_the_crossing():
    successes = sum(_run_pedestrian_crossing(seed)[0].success for seed in SEEDS)
    assert successes == len(SEEDS)


def test_deterministic_for_a_fixed_seed():
    result_a, _ = _run_pedestrian_crossing(seed=3)
    result_b, _ = _run_pedestrian_crossing(seed=3)
    assert np.array_equal(result_a.true_history, result_b.true_history)
    assert np.array_equal(result_a.controls, result_b.controls)
    assert result_a.collision == result_b.collision
    assert result_a.success == result_b.success


# --- The re-plan actually tracks the obstacle's *current* position, not its start ----


def test_replans_target_the_moving_obstacles_live_position_not_its_start():
    """Direct PlannerNode-level proof (same pattern as test_replanning.py's static-obstacle
    coverage): two replan_requests, with the moving obstacle at two different live positions
    in between, produce two different paths, each clearing the position it was replanned
    against, not the spec's start position and not the other replan's position."""
    bus = Bus()
    planner = HybridAStarPlanner()
    environment = Environment(Spot(0.0, 0.0, 0.0), obstacles=[])
    PlannerNode(bus, planner, environment, TURNING_RADIUS, max_replans=2)  # subscribes itself to the bus

    # Spy on exactly what obstacle list each plan() call received: direct, non-flaky proof
    # each replan used the *current* moving_obstacles entry, not a stale snapshot.
    seen_obstacles: list[list[Obstacle]] = []
    real_plan = planner.plan

    def spy_plan(start, goal, obstacles, turning_radius):
        seen_obstacles.append(obstacles)
        return real_plan(start, goal, obstacles, turning_radius)

    planner.plan = spy_plan

    published: list[np.ndarray] = []
    bus.subscribe("path", lambda msg: published.append(msg.path))
    bus.publish("pose_estimate", _pose(-10.0, 0.0, 0.0))
    assert len(published) == 1
    assert seen_obstacles[0] == []  # initial plan: no moving obstacle present yet

    position_at_first_replan = Obstacle(x=-5.0, y=0.0, radius=0.5)
    environment.moving_obstacles = [position_at_first_replan]
    bus.publish("replan_request", ReplanRequestMsg())
    assert len(published) == 2
    assert seen_obstacles[1] == [position_at_first_replan]
    first_replan_path = published[1]
    clearance_1 = np.hypot(
        first_replan_path[:, 0] - position_at_first_replan.x, first_replan_path[:, 1] - position_at_first_replan.y
    ).min()
    assert clearance_1 >= position_at_first_replan.radius + VEHICLE_RADIUS

    # The pedestrian kept walking: a different live position by the second replan.
    position_at_second_replan = Obstacle(x=-5.0, y=1.5, radius=0.5)
    environment.moving_obstacles = [position_at_second_replan]
    bus.publish("replan_request", ReplanRequestMsg())
    assert len(published) == 3
    assert seen_obstacles[2] == [position_at_second_replan]  # not the first replan's position
    second_replan_path = published[2]

    assert second_replan_path.shape != first_replan_path.shape or not np.allclose(
        second_replan_path, first_replan_path
    )
    clearance_2 = np.hypot(
        second_replan_path[:, 0] - position_at_second_replan.x, second_replan_path[:, 1] - position_at_second_replan.y
    ).min()
    assert clearance_2 >= position_at_second_replan.radius + VEHICLE_RADIUS


# --- EKF-landmark isolation: the one thing this feature must not break --------------


def test_sensor_node_never_enumerates_a_moving_obstacle_as_a_landmark():
    """The exact boundary DESIGN.md section 1 depends on (localization, not SLAM): a moving
    obstacle sitting well within landmark_range must produce byte-identical landmark_bearings
    whether or not it's present, because SensorNode's landmark loop must stay scoped to
    `environment.obstacles`, never `all_obstacles()`."""
    bus = Bus()
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    ultrasonic = UltrasonicArray(angles=[0.0], max_range=8.0)

    static_landmark = Obstacle(x=3.0, y=0.0, radius=1.0)
    env_without_moving = Environment(Spot(0.0, 0.0, 0.0), obstacles=[static_landmark])
    env_with_moving = Environment(
        Spot(0.0, 0.0, 0.0), obstacles=[static_landmark], moving_obstacles=[Obstacle(x=1.0, y=0.5, radius=0.4)]
    )

    node_a = SensorNode(bus, ultrasonic, env_without_moving, rng_a, position_fix_period=1)
    readings_a: list = []
    bus.subscribe("landmark_bearings", lambda msg: readings_a.append(msg))
    bus.publish("true_state", TrueStateMsg(x=0.0, y=0.0, theta=0.0, v=0.0, delta=0.0))
    node_a.step()

    bus_b = Bus()
    node_b = SensorNode(bus_b, ultrasonic, env_with_moving, rng_b, position_fix_period=1)
    readings_b: list = []
    bus_b.subscribe("landmark_bearings", lambda msg: readings_b.append(msg))
    bus_b.publish("true_state", TrueStateMsg(x=0.0, y=0.0, theta=0.0, v=0.0, delta=0.0))
    node_b.step()

    assert len(readings_a) == 1 and len(readings_b) == 1
    msg_a, msg_b = readings_a[0], readings_b[0]
    assert len(msg_a.readings) == len(msg_b.readings) == 1  # only the static landmark, both times
    assert msg_a.readings[0].landmark_id == msg_b.readings[0].landmark_id == 0
    assert msg_a.readings[0].range == pytest.approx(msg_b.readings[0].range)
    assert msg_a.readings[0].bearing == pytest.approx(msg_b.readings[0].bearing)


def test_ekf_pose_estimate_unaffected_by_a_moving_obstacle_outside_its_path():
    """End-to-end version of the same isolation check: a moving obstacle placed where it
    never enters the ultrasonic beams, collision radius, or planned path (so the true
    trajectory is identical either way) must leave the EKF's pose-estimate history, and
    hence its error, bit-for-bit unchanged, same seed, with vs. without."""
    far_pedestrian = MovingObstacleSpec(start_x=-5.0, start_y=50.0, vx=0.0, vy=1.0, radius=0.4, start_time=0.0)

    def run(moving_obstacles):
        vehicle = Vehicle(x=-10.0, y=0.0, theta=0.0, wheelbase=2.7)
        environment = Environment(Spot(0.0, 0.0, 0.0), obstacles=[])
        planner = HybridAStarPlanner()
        controller = PurePursuitAdaptive(wheelbase=vehicle.wheelbase, v_max=1.5, max_steer=vehicle.max_steer)
        harness = ParkingHarness(
            vehicle, environment, planner, controller, seed=5, moving_obstacles=moving_obstacles
        )
        return harness.run(max_steps=250)

    without_moving = run(None)
    with_moving = run([far_pedestrian])

    assert np.array_equal(without_moving.true_history, with_moving.true_history)
    assert np.array_equal(without_moving.estimated_history, with_moving.estimated_history)
    assert np.allclose(without_moving.covariance_history, with_moving.covariance_history)


# --- A genuine, scoped limitation found while wiring this in: see the final report ---


def test_known_limitation_a_fast_closing_pedestrian_can_still_make_contact():
    """NOT a desired outcome: pins a real, scoped, documented gap rather than hiding it
    (see this feature's final report for the full root-cause writeup). ControllerNode's
    governor brakes from the sensed range using only the ego's own kinematics (KNOWN_BUGS.md
    entry 2/3); it has no notion of the obstacle's own closing velocity. A moving obstacle
    walking into an already-planned detour can keep ego speed just above STALL_SPEED,
    never tripping the stall-based re-plan trigger, while it closes the rest of the gap
    itself. If this test starts failing (no collision), the gap has closed: update/remove it."""
    vehicle = Vehicle(x=-10.0, y=0.0, theta=0.0, wheelbase=2.7)
    environment = Environment(Spot(0.0, 0.0, 0.0), obstacles=[])
    planner = HybridAStarPlanner()
    controller = PurePursuitAdaptive(wheelbase=vehicle.wheelbase, v_max=1.5, max_steer=vehicle.max_steer)
    fast_closing = MovingObstacleSpec(start_x=-5.0, start_y=-2.0, vx=0.0, vy=0.8, radius=0.5, start_time=1.5)
    harness = ParkingHarness(
        vehicle, environment, planner, controller, seed=1, max_replans=3, moving_obstacles=[fast_closing]
    )
    result = harness.run(max_steps=200)
    assert result.collision
    assert harness.planner_node._replans == 0  # the stall-based trigger never fired
