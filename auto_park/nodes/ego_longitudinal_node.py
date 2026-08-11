"""Owns the ego vehicle's longitudinal state (position, speed) as a simple 1D
point-mass -- H1 (ACC) is straight-line following only, so the full 2D kinematic
bicycle model (`Vehicle`) isn't needed yet; it comes back in H3 once lateral control
(lane centering) is in the picture. See DESIGN.md's ACC section.

Clamps the commanded acceleration to physical actuator limits, same principle as
VehicleNode clamping steering/accel for the parking mode: a controller can command
anything, this node is what enforces what's actually achievable.
"""

import numpy as np

from auto_park.messaging.bus import Bus
from auto_park.messaging.messages import EgoLongitudinalStateMsg, LongitudinalCmdMsg


class EgoLongitudinalNode:
    def __init__(
        self,
        bus: Bus,
        position: float,
        speed: float,
        dt: float,
        a_min: float = -9.0,
        a_max: float = 3.0,
    ):
        self.bus = bus
        self.position = position
        self.speed = speed
        self.dt = dt
        self.a_min = a_min
        self.a_max = a_max
        self._last_cmd = LongitudinalCmdMsg(0.0)
        bus.subscribe("longitudinal_cmd", self._on_cmd)

    def _on_cmd(self, msg: LongitudinalCmdMsg) -> None:
        self._last_cmd = msg

    def step(self) -> None:
        accel = float(np.clip(self._last_cmd.accel, self.a_min, self.a_max))
        self.speed = max(0.0, self.speed + accel * self.dt)  # can't reverse under ACC
        self.position += self.speed * self.dt
        self.bus.publish("ego_state", EgoLongitudinalStateMsg(self.position, self.speed, accel))
