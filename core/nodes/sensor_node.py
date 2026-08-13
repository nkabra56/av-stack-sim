"""Perception node: the only node besides the harness allowed to see true_state.
Publishes obstacle ranges (for braking), an always-on noisy compass, a low-rate noisy
position fix, and opportunistic noisy landmark range-bearing readings against the
environment's known obstacles. See DESIGN.md's EKF design section for why each of
these three measurement types exists.

**Dropout/latency** (DESIGN.md section 10's future-extensions list): real sensors
don't just add noise to every reading, they sometimes miss a cycle entirely or
deliver late. `dropout_prob` independently drops each of this tick's four messages
before publishing (never arrives at all -- the EKF simply doesn't correct that tick,
and ControllerNode/PlannerNode keep acting on whatever they last received, exactly the
same "stale reading persists" behavior a real subscriber sees from a real dropped
message). `latency_ticks` instead queues a survived message and releases it
`latency_ticks` ticks later, in the order it was computed -- modeling a late arrival
rather than a lost one. Both default to 0/off, in which case `_publish_or_defer`
degrades to the original unconditional `bus.publish` (`dropout_prob > 0.0` short-
circuits the RNG draw entirely, so every existing caller's noise-sample sequence is
byte-for-byte unchanged, not just statistically similar)."""

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
        dropout_prob: float = 0.0,
        latency_ticks: int = 0,
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
        self.dropout_prob = dropout_prob
        self.latency_ticks = latency_ticks

        self._true_state: TrueStateMsg | None = None
        self._tick = 0
        self._pending: list[tuple[int, str, object]] = []
        bus.subscribe("true_state", self._on_true_state)

    def _on_true_state(self, msg: TrueStateMsg) -> None:
        self._true_state = msg

    def _publish_or_defer(self, topic: str, msg: object) -> None:
        if self.dropout_prob > 0.0 and self.rng.random() < self.dropout_prob:
            return  # never arrives -- subscribers keep acting on their last-received value
        if self.latency_ticks <= 0:
            self.bus.publish(topic, msg)
        else:
            self._pending.append((self._tick + self.latency_ticks, topic, msg))

    def _release_due_messages(self) -> None:
        if not self._pending:
            return
        due = [entry for entry in self._pending if entry[0] <= self._tick]
        self._pending = [entry for entry in self._pending if entry[0] > self._tick]
        for _, topic, msg in due:
            self.bus.publish(topic, msg)

    def step(self) -> None:
        ts = self._true_state
        if ts is None:
            return

        self._tick += 1
        self._release_due_messages()

        obstacle_readings = self.ultrasonic.sense(ts, self.environment.obstacles)
        self._publish_or_defer("obstacle_ranges", ObstacleRangeMsg(obstacle_readings))

        compass_meas = wrap_angle(ts.theta + self.rng.normal(0.0, self.compass_std))
        self._publish_or_defer("compass", CompassMsg(compass_meas))

        if self._tick % self.position_fix_period == 0:
            x_meas = ts.x + self.rng.normal(0.0, self.position_std)
            y_meas = ts.y + self.rng.normal(0.0, self.position_std)
            self._publish_or_defer("position_fix", PositionFixMsg(x_meas, y_meas))

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
            self._publish_or_defer("landmark_bearings", LandmarkBearingMsg(readings))
