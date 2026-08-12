"""Wraps StanleyController (control/lane_centering.py) for the full closed-loop
highway drive: consumes the fused pose+speed estimate (EgoSpeedEstimateMsg, H2 -- not
ground truth, same "controllers only see estimates" rule as everywhere else in this
project) and publishes exactly one steering command per tick via an explicit step(),
same "store latest, act once per tick" pattern as ControllerNode/AccControllerNode.

EgoSpeedEstimateMsg satisfies interfaces.HasPose directly (x/y/theta), so no adapter
is needed between the estimator and Stanley -- the same principle that already lets
Stanley track either a real Vehicle or a parking PoseEstimateMsg unchanged.
"""

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
