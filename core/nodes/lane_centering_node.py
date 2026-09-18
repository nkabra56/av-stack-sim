"""Wraps StanleyController for the full closed-loop highway drive: consumes the fused
pose+speed estimate (H2, not ground truth) and publishes one steering command per tick."""

import numpy as np

from core.control.lane_centering import StanleyController
from core.messaging.bus import Bus
from core.messaging.messages import EgoSpeedEstimateMsg, LateralCmdMsg


class LaneCenteringControllerNode:
    def __init__(self, bus: Bus, controller: StanleyController, centerline: np.ndarray):
        self.bus = bus
        self.controller = controller
        self.centerline = centerline
        self._estimate: EgoSpeedEstimateMsg | None = None
        bus.subscribe("ego_speed_estimate", self._on_estimate)

    def _on_estimate(self, msg: EgoSpeedEstimateMsg) -> None:
        self._estimate = msg

    def step(self) -> None:
        if self._estimate is None:
            self.bus.publish("lateral_cmd", LateralCmdMsg(0.0))
            return
        delta = self.controller.control(self._estimate, self.centerline, self._estimate.speed)
        self.bus.publish("lateral_cmd", LateralCmdMsg(delta))
