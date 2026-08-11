"""Wraps the existing Controller (Pure Pursuit or MPC) unchanged -- both already only
touch .x/.y/.theta on whatever "vehicle" they're given, so a PoseEstimateMsg satisfies
them without modification. Stores the latest pose_estimate/path/obstacle_ranges via
subscriptions (cheap), but only computes and publishes a control_cmd when the harness
explicitly calls step() once per tick -- this keeps exactly one control decision per
tick even though pose_estimate can be republished several times a tick (once per
sensor update), instead of reacting to every intermediate estimate.

Braking on close-range obstacle detection lives here (moved from the old
ParkingSimulation loop): it's a control-layer safety decision, made from the sensor's
obstacle_ranges, independent of which path-tracking law is in use underneath.
"""

from auto_park.interfaces import Controller
from auto_park.messaging.bus import Bus
from auto_park.messaging.messages import ControlCmdMsg, ObstacleRangeMsg, PathMsg, PoseEstimateMsg


class ControllerNode:
    def __init__(self, bus: Bus, controller: Controller, brake_distance: float = 2.0):
        self.bus = bus
        self.controller = controller
        self.brake_distance = brake_distance

        self._pose_estimate: PoseEstimateMsg | None = None
        self._path: PathMsg | None = None
        self._obstacle_ranges: ObstacleRangeMsg | None = None

        bus.subscribe("pose_estimate", self._on_pose_estimate)
        bus.subscribe("path", self._on_path)
        bus.subscribe("obstacle_ranges", self._on_obstacle_ranges)

    def _on_pose_estimate(self, msg: PoseEstimateMsg) -> None:
        self._pose_estimate = msg

    def _on_path(self, msg: PathMsg) -> None:
        self._path = msg

    def _on_obstacle_ranges(self, msg: ObstacleRangeMsg) -> None:
        self._obstacle_ranges = msg

    def step(self) -> None:
        if self._pose_estimate is None or self._path is None:
            self.bus.publish("control_cmd", ControlCmdMsg(0.0, 0.0))
            return

        closest_range = min(self._obstacle_ranges.readings.values()) if self._obstacle_ranges else float("inf")
        if closest_range < self.brake_distance:
            v, delta = 0.0, 0.0
        else:
            v, delta = self.controller.control(self._pose_estimate, self._path.path)

        self.bus.publish("control_cmd", ControlCmdMsg(v, delta))
