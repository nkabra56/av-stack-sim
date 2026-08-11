"""Simulated forward radar: noisy bumper-to-bumper range and closing range-rate to the
lead vehicle. See DESIGN.md's ACC section.
"""

import numpy as np

from auto_park.messaging.bus import Bus
from auto_park.messaging.messages import EgoLongitudinalStateMsg, LeadVehicleStateMsg, RadarMsg


class RadarNode:
    def __init__(
        self,
        bus: Bus,
        rng: np.random.Generator,
        lead_length: float,
        range_std: float = 0.5,
        range_rate_std: float = 0.3,
    ):
        self.bus = bus
        self.rng = rng
        self.lead_length = lead_length
        self.range_std = range_std
        self.range_rate_std = range_rate_std

        self._ego: EgoLongitudinalStateMsg | None = None
        self._lead: LeadVehicleStateMsg | None = None
        bus.subscribe("ego_state", self._on_ego_state)
        bus.subscribe("lead_state", self._on_lead_state)

    def _on_ego_state(self, msg: EgoLongitudinalStateMsg) -> None:
        self._ego = msg

    def _on_lead_state(self, msg: LeadVehicleStateMsg) -> None:
        self._lead = msg

    def step(self) -> None:
        if self._ego is None or self._lead is None:
            return
        true_range = max(0.0, (self._lead.position - self.lead_length) - self._ego.position)
        true_range_rate = self._ego.speed - self._lead.speed

        range_meas = max(0.0, true_range + self.rng.normal(0.0, self.range_std))
        range_rate_meas = true_range_rate + self.rng.normal(0.0, self.range_rate_std)
        self.bus.publish("radar", RadarMsg(range_meas, range_rate_meas))
