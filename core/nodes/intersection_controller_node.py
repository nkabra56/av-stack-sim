"""Wraps IntersectionNavigator (H4, control/intersection.py) for the full closed-loop
highway drive (H5 Phase B): publishes an accel candidate for LongitudinalArbiterNode
to compose with ACC's, same "store latest, act once per tick" pattern as every other
controller node here. See DESIGN.md section 12's H5 entry.

Feeds the navigator the FUSED pose+speed estimate (via lane_geometry.project_to_arc_length),
not ground truth -- extending the "controllers only see estimates" rule to H4 for the
first time. H4's own standalone harness (intersection_harness.py) uses true state
directly, justified there by "no sensor noise happening in that mode" -- that
justification stops applying once it's wired into a loop that has a running EKF, so
this is a deliberate, visible choice, not an oversight.
"""

from core.control.intersection import IntersectionNavigator, OtherVehicleStatus
from core.control.lane_geometry import project_to_arc_length
from core.messaging.bus import Bus
from core.messaging.messages import EgoSpeedEstimateMsg, LongitudinalCmdMsg


class IntersectionControllerNode:
    def __init__(
        self,
        bus: Bus,
        navigator: IntersectionNavigator,
        centerline,
        arc_length_table,
        dt: float,
        output_topic: str = "intersection_cmd_candidate",
    ):
        self.bus = bus
        self.navigator = navigator
        self.centerline = centerline
        self.arc_length_table = arc_length_table
        self.dt = dt
        self.output_topic = output_topic
        self._estimate: EgoSpeedEstimateMsg | None = None
        self._others: list[OtherVehicleStatus] = []
        self._t = 0.0
        bus.subscribe("ego_speed_estimate", self._on_estimate)
        bus.subscribe("other_vehicle_status", self._on_other_vehicle_status)

    def _on_estimate(self, msg: EgoSpeedEstimateMsg) -> None:
        self._estimate = msg

    def _on_other_vehicle_status(self, others: list[OtherVehicleStatus]) -> None:
        self._others = others

    def step(self) -> None:
        if self._estimate is None:
            self.bus.publish(self.output_topic, LongitudinalCmdMsg(0.0))
            self._t += self.dt
            return
        arc_length = project_to_arc_length(self._estimate.x, self._estimate.y, self.centerline, self.arc_length_table)
        accel = self.navigator.control(arc_length, self._estimate.speed, self._t, self._others)
        self.bus.publish(self.output_topic, LongitudinalCmdMsg(accel))
        self._t += self.dt
