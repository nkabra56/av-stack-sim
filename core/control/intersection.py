"""Rule-based right-of-way navigator for a stop-sign intersection (H4). Reuses IDMController
for longitudinal control: the stop line is a stationary virtual lead vehicle. See DESIGN.md
section 12."""

from dataclasses import dataclass
from enum import Enum, auto

from core.control.acc import IDMController


class IntersectionState(Enum):
    APPROACHING = auto()
    STOPPED = auto()
    PROCEEDING = auto()


@dataclass
class OtherVehicleStatus:
    """What the ego can observe about one other vehicle: known facts about the world,
    not the other vehicle's own internal state."""

    stopped: bool
    stop_time: float | None
    cleared: bool
    is_to_the_right: bool


class IntersectionNavigator:
    def __init__(
        self,
        stop_line_position: float,
        wheelbase: float = 2.7,
        v_cruise: float = 15.0,
        stop_gap: float = 1.0,
        stop_speed_threshold: float = 0.3,
    ):
        self.stop_line_position = stop_line_position
        self.v_cruise = v_cruise
        self.stop_speed_threshold = stop_speed_threshold
        self.state = IntersectionState.APPROACHING
        self.stop_time: float | None = None
        # s0=stop_gap: the stop line is a virtual lead vehicle, so IDM naturally settles
        # stop_gap meters short of it.
        self._approach_idm = IDMController(v0=v_cruise, s0=stop_gap, time_headway=0.5, a_max=2.0, b_comfortable=2.5)
        self._cruise_idm = IDMController(v0=v_cruise, a_max=2.0)

    def has_right_of_way(self, others: list[OtherVehicleStatus]) -> bool:
        if self.stop_time is None:
            return False
        for other in others:
            if other.cleared or not other.stopped or other.stop_time is None:
                continue
            if other.stop_time < self.stop_time - 1e-6:
                return False  # other arrived first
            if abs(other.stop_time - self.stop_time) < 1e-6 and other.is_to_the_right:
                return False  # simultaneous arrival, yield to the right
        return True

    def control(self, ego_position: float, ego_speed: float, t: float, others: list[OtherVehicleStatus]) -> float:
        distance_to_stop = self.stop_line_position - ego_position

        if self.state == IntersectionState.APPROACHING:
            if ego_speed < self.stop_speed_threshold and distance_to_stop < 2.0 * (1 + self._approach_idm.s0):
                self.state = IntersectionState.STOPPED
                self.stop_time = t
                return -2.0 if ego_speed > 0 else 0.0
            gap = max(distance_to_stop, 1e-3)
            return self._approach_idm.control(ego_speed, gap, lead_speed=0.0)

        if self.state == IntersectionState.STOPPED:
            if self.has_right_of_way(others):
                self.state = IntersectionState.PROCEEDING
            else:
                return -1.0 if ego_speed > 0 else 0.0

        return self._cruise_idm.control(ego_speed, gap=1000.0, lead_speed=self.v_cruise)
