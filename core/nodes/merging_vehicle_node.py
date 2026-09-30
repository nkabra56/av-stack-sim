"""Scripted-but-reactive merging vehicle: cruises in the adjacent lane, then decides to
merge into the ego's lane once its gap ahead of the ego crosses an acceptance threshold."""

from core.messaging.bus import Bus
from core.messaging.messages import EgoLongitudinalStateMsg, LeadVehicleStateMsg


class MergingVehicleNode:
    def __init__(
        self,
        bus: Bus,
        position: float,
        speed: float,
        dt: float,
        gap_threshold: float = 20.0,  # meters; gap ahead of the ego it waits for before cutting in
        initial_lane_offset: float = 3.7,  # meters, real US lane width
        merge_duration: float = 3.0,  # seconds to complete the lane change once accepted
    ):
        self.bus = bus
        self.position = position
        self.speed = speed
        self.dt = dt
        self.gap_threshold = gap_threshold
        self.initial_lane_offset = initial_lane_offset
        self.merge_duration = merge_duration

        self.lane_offset = initial_lane_offset
        self.merging = False
        self._merge_elapsed = 0.0
        self._ego: EgoLongitudinalStateMsg | None = None
        bus.subscribe("ego_state", self._on_ego_state)

    def _on_ego_state(self, msg: EgoLongitudinalStateMsg) -> None:
        self._ego = msg

    def step(self) -> None:
        self.position += self.speed * self.dt

        if not self.merging and self._ego is not None:
            gap_ahead_of_ego = self.position - self._ego.position
            if gap_ahead_of_ego >= self.gap_threshold:
                self.merging = True

        if self.merging and self._merge_elapsed < self.merge_duration:
            self._merge_elapsed += self.dt
            frac = min(1.0, self._merge_elapsed / self.merge_duration)
            self.lane_offset = (1.0 - frac) * self.initial_lane_offset

        self.bus.publish("merging_vehicle_state", LeadVehicleStateMsg(self.position, self.speed, self.lane_offset))
