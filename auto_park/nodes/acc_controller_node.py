"""Wraps an ACC controller (IDM or MPC, control/acc.py). Consumes radar (range,
range_rate) plus the ego vehicle's own directly-known speed (a real speedometer, not a
noisy externally-sensed quantity like the lead vehicle's state) and publishes exactly
one acceleration command per tick via an explicit step(), same "store latest, act once
per tick" pattern as ControllerNode uses for parking. Lead speed is derived from radar
(ego_speed - range_rate), not read directly from lead-vehicle ground truth -- the
controller only ever sees what the radar actually measured.

H1 scope note: ego speed here is the *true* ego speed, not a noisy/estimated one. H2
(extending the EKF with a fused speed state) is what later tests these controllers
under ego-speed uncertainty too -- kept separate so H1 stays focused on the
longitudinal control problem itself. See DESIGN.md's ACC section.
"""

from typing import Protocol

from auto_park.messaging.bus import Bus
from auto_park.messaging.messages import EgoLongitudinalStateMsg, LongitudinalCmdMsg, RadarMsg


class AccController(Protocol):
    def control(self, ego_speed: float, gap: float, lead_speed: float) -> float: ...


class AccControllerNode:
    def __init__(self, bus: Bus, controller: AccController):
        self.bus = bus
        self.controller = controller
        self._ego: EgoLongitudinalStateMsg | None = None
        self._radar: RadarMsg | None = None
        bus.subscribe("ego_state", self._on_ego_state)
        bus.subscribe("radar", self._on_radar)

    def _on_ego_state(self, msg: EgoLongitudinalStateMsg) -> None:
        self._ego = msg

    def _on_radar(self, msg: RadarMsg) -> None:
        self._radar = msg

    def step(self) -> None:
        if self._ego is None or self._radar is None:
            self.bus.publish("longitudinal_cmd", LongitudinalCmdMsg(0.0))
            return
        lead_speed = self._ego.speed - self._radar.range_rate
        accel = self.controller.control(self._ego.speed, self._radar.range, lead_speed)
        self.bus.publish("longitudinal_cmd", LongitudinalCmdMsg(accel))
