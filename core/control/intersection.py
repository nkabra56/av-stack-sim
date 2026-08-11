"""Rule-based right-of-way navigator for a stop-sign-controlled intersection (H4).
See DESIGN.md section 12's H4 entry.

Deliberately reuses IDMController rather than inventing new longitudinal control: the
stop line is modeled as a stationary virtual lead vehicle (lead_speed=0) at a fixed
position, so "decelerate smoothly and stop behind it" is exactly the car-following
behavior IDM already does and H1/H2 already validated (including the standstill case
specifically, from real NGSIM stop-and-go traffic) -- H4 is mostly the *reasoning*
about when it's this vehicle's turn, wired on top of control that already exists.

Scope: models the intersection as a single conflict point two independent approaches
share, not full 2D multi-direction intersection geometry -- this captures the actual
substance of right-of-way reasoning (mutual exclusion + arrival-order priority) without
needing to simulate a full 4-way intersection's road geometry, consistent with how H1
stayed longitudinal-only and H3 wasn't yet combined with H1.
"""

from dataclasses import dataclass
from enum import Enum, auto

from core.control.acc import IDMController


class IntersectionState(Enum):
    APPROACHING = auto()
    STOPPED = auto()
    PROCEEDING = auto()


@dataclass
class OtherVehicleStatus:
    """What the ego can observe about one other vehicle at the intersection --
    analogous to a landmark reading: known facts about the world, not the other
    vehicle's own internal state."""

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
        # s0 (min standstill gap) set to stop_gap: the "stop line" virtual lead vehicle
        # sits at stop_line_position, so IDM naturally settles with the ego stop_gap
        # meters short of it, same equilibrium behavior already validated in H1/H2.
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
