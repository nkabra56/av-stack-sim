"""Perception node: the only node besides the harness allowed to see true_state.
Publishes obstacle ranges (for braking), an always-on noisy compass, a low-rate noisy
position fix, and opportunistic noisy landmark range-bearing readings against the
environment's known obstacles. See DESIGN.md's EKF design section for why each of
these three measurement types exists.
"""

import numpy as np

from core.environment import Environment
from core.messaging.bus import Bus
from core.messaging.messages import (
    CompassMsg,
    LandmarkBearingMsg,
    LandmarkReading,
    ObstacleRangeMsg,
    PositionFixMsg,
    TrueStateMsg,
)
from core.sensors import UltrasonicArray
from core.vehicle import wrap_angle


class SensorNode:
    def __init__(
        self,
        bus: Bus,
        ultrasonic: UltrasonicArray,
        environment: Environment,
        rng: np.random.Generator,
        landmark_range: float = 8.0,
        compass_std: float = 0.02,
        position_std: float = 0.3,
        position_fix_period: int = 10,
        landmark_range_std: float = 0.2,
        landmark_bearing_std: float = 0.03,
    ):
        self.bus = bus
        self.ultrasonic = ultrasonic
        self.environment = environment
        self.rng = rng
        self.landmark_range = landmark_range
        self.compass_std = compass_std
        self.position_std = position_std
        self.position_fix_period = position_fix_period
        self.landmark_range_std = landmark_range_std
        self.landmark_bearing_std = landmark_bearing_std

        self._true_state: TrueStateMsg | None = None
        self._tick = 0
        bus.subscribe("true_state", self._on_true_state)

    def _on_true_state(self, msg: TrueStateMsg) -> None:
        self._true_state = msg

    def step(self) -> None:
        ts = self._true_state
        if ts is None:
            return

        obstacle_readings = self.ultrasonic.sense(ts, self.environment.obstacles)
        self.bus.publish("obstacle_ranges", ObstacleRangeMsg(obstacle_readings))

        compass_meas = wrap_angle(ts.theta + self.rng.normal(0.0, self.compass_std))
        self.bus.publish("compass", CompassMsg(compass_meas))

        self._tick += 1
        if self._tick % self.position_fix_period == 0:
            x_meas = ts.x + self.rng.normal(0.0, self.position_std)
            y_meas = ts.y + self.rng.normal(0.0, self.position_std)
            self.bus.publish("position_fix", PositionFixMsg(x_meas, y_meas))

        readings = []
        for landmark_id, obstacle in enumerate(self.environment.obstacles):
            dx, dy = obstacle.x - ts.x, obstacle.y - ts.y
            true_range = np.hypot(dx, dy)
            if true_range <= self.landmark_range:
                true_bearing = wrap_angle(np.arctan2(dy, dx) - ts.theta)
                range_meas = true_range + self.rng.normal(0.0, self.landmark_range_std)
                bearing_meas = wrap_angle(true_bearing + self.rng.normal(0.0, self.landmark_bearing_std))
                readings.append(LandmarkReading(landmark_id, range_meas, bearing_meas))
        if readings:
            self.bus.publish("landmark_bearings", LandmarkBearingMsg(readings))
