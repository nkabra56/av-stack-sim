"""Simulated forward radar: noisy bumper-to-bumper range and closing range-rate to the
lead vehicle. See DESIGN.md's ACC section."""

import numpy as np

from core.messaging.bus import Bus
from core.messaging.messages import EgoLongitudinalStateMsg, LeadVehicleStateMsg, RadarMsg

IN_LANE_TOLERANCE = 0.5  # meters; a candidate this close to lane_offset=0.0 counts as "in lane"


class RadarNode:
    def __init__(
        self,
        bus: Bus,
        rng: np.random.Generator,
        lead_length: float,
        range_std: float = 0.5,
        range_rate_std: float = 0.3,
        merging_length: float | None = None,
    ):
        self.bus = bus
        self.rng = rng
        self.lead_length = lead_length
        self.range_std = range_std
        self.range_rate_std = range_rate_std
        self.merging_length = merging_length if merging_length is not None else lead_length

        self._ego: EgoLongitudinalStateMsg | None = None
        self._lead: LeadVehicleStateMsg | None = None
        self._merging: LeadVehicleStateMsg | None = None
        bus.subscribe("ego_state", self._on_ego_state)
        bus.subscribe("lead_state", self._on_lead_state)
        bus.subscribe("merging_vehicle_state", self._on_merging_state)

    def _on_ego_state(self, msg: EgoLongitudinalStateMsg) -> None:
        self._ego = msg

    def _on_lead_state(self, msg: LeadVehicleStateMsg) -> None:
        self._lead = msg

    def _on_merging_state(self, msg: LeadVehicleStateMsg) -> None:
        self._merging = msg

    def _in_lane_candidates(self) -> list[tuple[LeadVehicleStateMsg, float]]:
        """(candidate, length) pairs that are in-lane and ahead of the ego, closest first."""
        candidates = []
        for candidate, length in ((self._lead, self.lead_length), (self._merging, self.merging_length)):
            if candidate is None or abs(candidate.lane_offset) >= IN_LANE_TOLERANCE:
                continue
            if candidate.position - length <= self._ego.position:
                continue
            candidates.append((candidate, length))
        candidates.sort(key=lambda pair: pair[0].position - pair[1])
        return candidates

    def step(self) -> None:
        if self._ego is None or self._lead is None:
            return
        candidates = self._in_lane_candidates()
        if candidates:
            nearest, nearest_length = candidates[0]
        else:
            nearest, nearest_length = self._lead, self.lead_length

        true_range = max(0.0, (nearest.position - nearest_length) - self._ego.position)
        true_range_rate = self._ego.speed - nearest.speed

        range_meas = max(0.0, true_range + self.rng.normal(0.0, self.range_std))
        range_rate_meas = true_range_rate + self.rng.normal(0.0, self.range_rate_std)
        self.bus.publish("radar", RadarMsg(range_meas, range_rate_meas))
