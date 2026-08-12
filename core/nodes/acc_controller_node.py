"""Wraps an ACC controller (IDM or MPC, control/acc.py). Consumes radar (range,
range_rate) plus the ego vehicle's *fused speed estimate* (EgoSpeedEstimateMsg, H2 --
not ground truth; SpeedEstimatorNode's EKF output) and publishes exactly one
acceleration command per tick via an explicit step(), same "store latest, act once per
tick" pattern as ControllerNode uses for parking. Lead speed is derived from radar
(ego_speed_estimate - range_rate), not read directly from lead-vehicle ground truth --
the controller only ever sees estimates/measurements, never ground truth, same
principle as everywhere else in this project.
"""

from typing import Protocol

from core.messaging.bus import Bus
from core.messaging.messages import EgoSpeedEstimateMsg, LongitudinalCmdMsg, RadarMsg


class AccController(Protocol):
    def control(self, ego_speed: float, gap: float, lead_speed: float) -> float: ...


class AccControllerNode:
    def __init__(self, bus: Bus, controller: AccController, output_topic: str = "longitudinal_cmd"):
        self.bus = bus
        self.controller = controller
        self.output_topic = output_topic  # H5 Phase B: LongitudinalArbiterNode composes this
        # node's accel candidate with IntersectionControllerNode's via a separate topic per
        # candidate, defaulting to "longitudinal_cmd" so H1/H2-standalone (AccHarness) and H5
        # Phase A (no intersection) are completely unaffected.
        self._ego_speed_estimate: EgoSpeedEstimateMsg | None = None
        self._radar: RadarMsg | None = None
        bus.subscribe("ego_speed_estimate", self._on_ego_speed_estimate)
        bus.subscribe("radar", self._on_radar)

    def _on_ego_speed_estimate(self, msg: EgoSpeedEstimateMsg) -> None:
        self._ego_speed_estimate = msg

    def _on_radar(self, msg: RadarMsg) -> None:
        self._radar = msg

    def step(self) -> None:
        if self._ego_speed_estimate is None or self._radar is None:
            self.bus.publish(self.output_topic, LongitudinalCmdMsg(0.0))
            return
        ego_speed = self._ego_speed_estimate.speed
        lead_speed = ego_speed - self._radar.range_rate
        accel = self.controller.control(ego_speed, self._radar.range, lead_speed)
        self.bus.publish(self.output_topic, LongitudinalCmdMsg(accel))
