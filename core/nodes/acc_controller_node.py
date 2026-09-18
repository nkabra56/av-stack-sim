"""Wraps an ACC controller (IDM or MPC). Consumes radar plus the ego's fused speed
estimate (H2, not ground truth) and publishes one acceleration command per tick."""

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
        # node's candidate with IntersectionControllerNode's; defaults keep H1/H2/Phase A unaffected.
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
