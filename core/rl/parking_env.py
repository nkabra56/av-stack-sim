"""Gymnasium environment wrapping the parking simulation for a learned end-to-end policy
(raw sensing -> v/delta). Uses ground-truth state directly, no sensor noise -- see DESIGN.md
section 10; core/validation/rl_comparison.py evaluates it against the real noisy baseline."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.environment import VEHICLE_RADIUS, Environment
from core.harness import DEFAULT_SENSOR_ANGLES
from core.scenario_loader import load_scenario
from core.sensors import UltrasonicArray
from core.vehicle import Vehicle, wrap_angle

V_MAX = 1.5  # matches every controller's own v_max elsewhere in this project
DELTA_MAX = 0.6  # matches Vehicle's own default max_steer
GOAL_TOL = 0.4  # matches ParkingHarness's own default tol -- same "close enough to
# call it parked" bar the baseline is judged against, for a fair comparison
SENSOR_MAX_RANGE = 8.0


class ParkingEnv(gym.Env):
    metadata = {"render_modes": []}  # noqa: RUF012 -- standard Gymnasium API convention

    def __init__(self, scenario_name: str = "perpendicular_open", dt: float = 0.1, max_steps: int = 500):
        super().__init__()
        self.scenario_name = scenario_name
        self.dt = dt
        self.max_steps = max_steps
        self._ultrasonic = UltrasonicArray(angles=DEFAULT_SENSOR_ANGLES, max_range=SENSOR_MAX_RANGE)

        # Normalized to [-1, 1]^2, not raw m/s/radians -- aids SB3 training stability.
        # step() rescales to (V_MAX, DELTA_MAX) internally.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        n_beams = len(DEFAULT_SENSOR_ANGLES)
        # [dx, dy (goal offset, body frame), sin(dtheta), cos(dtheta)] + normalized sensor
        # ranges. Body-frame so the policy generalizes across scenarios, not one geometry.
        obs_dim = 4 + n_beams
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._vehicle: Vehicle | None = None
        self._environment: Environment | None = None
        self._steps = 0
        self._prev_dist = 0.0

    def _observation(self) -> np.ndarray:
        v, spot = self._vehicle, self._environment.spot
        dx_world, dy_world = spot.x - v.x, spot.y - v.y
        cos_t, sin_t = np.cos(-v.theta), np.sin(-v.theta)
        dx_body = dx_world * cos_t - dy_world * sin_t
        dy_body = dx_world * sin_t + dy_world * cos_t
        dtheta = wrap_angle(spot.theta - v.theta)

        readings = self._ultrasonic.sense(v, self._environment.obstacles)
        ranges = np.array([readings[a] for a in DEFAULT_SENSOR_ANGLES]) / SENSOR_MAX_RANGE

        return np.concatenate([[dx_body, dy_body, np.sin(dtheta), np.cos(dtheta)], ranges]).astype(np.float32)

    def _collided(self) -> bool:
        v = self._vehicle
        return any(
            np.hypot(v.x - o.x, v.y - o.y) < o.radius + VEHICLE_RADIUS for o in self._environment.obstacles
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        scenario = load_scenario(self.scenario_name)
        self._vehicle = scenario.vehicle
        self._environment = scenario.environment
        self._steps = 0
        self._prev_dist = float(np.hypot(self._environment.spot.x - self._vehicle.x, self._environment.spot.y - self._vehicle.y))
        return self._observation(), {}

    def step(self, action: np.ndarray):
        v_cmd = float(np.clip(action[0], -1.0, 1.0)) * V_MAX
        delta_cmd = float(np.clip(action[1], -1.0, 1.0)) * DELTA_MAX
        self._vehicle.update(v_cmd, delta_cmd, self.dt)
        self._steps += 1

        spot = self._environment.spot
        dist = float(np.hypot(spot.x - self._vehicle.x, spot.y - self._vehicle.y))
        collided = self._collided()
        success = dist < GOAL_TOL and not collided  # position-only, matching
        # ParkingHarness.run()'s own success criterion (no heading term).

        # Shaped reward: dense progress term (dominant every step) plus sparse terminal
        # bonuses/penalties -- dense alone would let the policy loiter without committing.
        reward = (self._prev_dist - dist) - 0.01
        if collided:
            reward -= 50.0
        if success:
            reward += 100.0
        self._prev_dist = dist

        terminated = collided or success
        truncated = self._steps >= self.max_steps
        return self._observation(), reward, terminated, truncated, {"collided": collided, "success": success}
