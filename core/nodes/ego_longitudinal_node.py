"""Owns the ego vehicle's longitudinal state as a simple 1D point-mass -- H1 (ACC) is
straight-line only; the full 2D Vehicle comes back in H3. See DESIGN.md's ACC section."""

import numpy as np

from core.messaging.bus import Bus
from core.messaging.messages import (
    AccelOdometryMsg,
    EgoLongitudinalStateMsg,
    LongitudinalCmdMsg,
    SpeedometerMsg,
)


class EgoLongitudinalNode:
    def __init__(
        self,
        bus: Bus,
        position: float,
        speed: float,
        dt: float,
        rng: np.random.Generator,
        a_min: float = -9.0,
        a_max: float = 3.0,
        accel_odom_std: float = 0.15,
        speedometer_std: float = 0.2,
    ):
        self.bus = bus
        self.position = position
        self.speed = speed
        self.dt = dt
        self.rng = rng
        self.a_min = a_min
        self.a_max = a_max
        self.accel_odom_std = accel_odom_std
        self.speedometer_std = speedometer_std
        self._last_cmd = LongitudinalCmdMsg(0.0)
        bus.subscribe("longitudinal_cmd", self._on_cmd)

    def _on_cmd(self, msg: LongitudinalCmdMsg) -> None:
        self._last_cmd = msg

    def step(self) -> None:
        accel = float(np.clip(self._last_cmd.accel, self.a_min, self.a_max))
        self.speed = max(0.0, self.speed + accel * self.dt)  # can't reverse under ACC
        self.position += self.speed * self.dt
        self.bus.publish("ego_state", EgoLongitudinalStateMsg(self.position, self.speed, accel))

        self.bus.publish("accel_odometry", AccelOdometryMsg(accel + self.rng.normal(0.0, self.accel_odom_std)))
        self.bus.publish("speedometer", SpeedometerMsg(self.speed + self.rng.normal(0.0, self.speedometer_std)))
