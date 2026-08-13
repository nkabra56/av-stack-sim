"""Fast, deterministic unit coverage for ParkingEnv's Gym contract and reward shape --
no training involved (that's core/rl/train.py / test_rl_training.py's job). Requires
the `rl` optional dependency group (gymnasium) -- skipped entirely if unavailable, the
same pattern any optional-dependency test suite uses.
"""

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")

from core.rl.parking_env import GOAL_TOL, ParkingEnv  # noqa: E402


def test_gymnasium_check_env_passes():
    """Standard Gym API-contract checker: correct reset()/step() signatures, spaces
    match what's actually returned, dtypes are right, etc."""
    from gymnasium.utils.env_checker import check_env

    check_env(ParkingEnv().unwrapped, skip_render_check=True)


def test_reset_returns_an_observation_in_bounds():
    env = ParkingEnv()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_driving_straight_toward_the_goal_gives_positive_reward():
    env = ParkingEnv(scenario_name="perpendicular_open")
    env.reset(seed=0)
    # perpendicular_open's vehicle starts facing roughly toward the spot -- driving
    # forward should reduce distance-to-goal, which the dense reward term rewards.
    _, reward, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert reward > 0
    assert not info["collided"]


def test_driving_straight_into_a_flanking_obstacle_collides_and_penalizes_heavily():
    """Places the vehicle directly facing an obstacle at close range, rather than
    relying on some chosen action sequence happening to steer into one -- direct and
    unambiguous, not dependent on this scenario's specific curvature/geometry."""
    env = ParkingEnv(scenario_name="perpendicular_flanked")
    env.reset(seed=0)
    obstacle = env._environment.obstacles[0]
    env._vehicle.x, env._vehicle.y = obstacle.x - obstacle.radius - 1.0, obstacle.y
    env._vehicle.theta = 0.0  # facing directly at the obstacle

    _, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert info["collided"]
    assert terminated
    assert reward < -10.0  # the collision penalty dominates this step's reward


def test_action_is_clipped_to_the_declared_bounds():
    """An out-of-range action shouldn't silently drive the vehicle faster/harder than
    the declared action_space allows -- clip, don't pass through."""
    env = ParkingEnv()
    env.reset(seed=0)
    x0 = env._vehicle.x
    env.step(np.array([1000.0, 0.0], dtype=np.float32))
    # at v_cmd clipped to V_MAX=1.5 and dt=0.1, the vehicle can move at most 0.15m
    assert abs(env._vehicle.x - x0) <= 0.15 + 1e-6


def test_episode_truncates_at_max_steps_if_never_terminated():
    env = ParkingEnv(scenario_name="perpendicular_open", max_steps=5)
    env.reset(seed=0)
    for i in range(5):
        _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))  # sit still
    assert truncated
    assert not terminated


def test_success_uses_the_same_position_only_tolerance_as_the_baseline_harness():
    """Regression for a specific design choice: success is position-only (matches
    ParkingHarness.run()'s own criterion exactly), not also gated on heading -- so the
    RL policy isn't held to a stricter bar than the planner+controller baseline is
    when they're compared later."""
    env = ParkingEnv(scenario_name="perpendicular_open")
    env.reset(seed=0)
    spot = env._environment.spot
    env._vehicle.x, env._vehicle.y, env._vehicle.theta = spot.x, spot.y, spot.theta + 2.5  # way off heading
    _, reward, terminated, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert info["success"]
    assert terminated
    assert reward > 50.0  # the +100 success bonus, minus a tiny progress/time term
