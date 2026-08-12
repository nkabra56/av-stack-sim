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

**Speed governor, not a binary brake** (see KNOWN_BUGS.md's bug 1 for the finding that
motivated this): a fixed "brake within X meters" cutoff has no notion of how fast the
vehicle is actually going, so a distance picked to sit below a planner's own intentional
obstacle clearance (`hybrid_astar.brake_distance_for`'s old approach) leaves it with no
real stopping margin at speed -- verified directly: with the old formula's default
`buffer` equal to `HybridAStarPlanner.safety_margin`, `brake_distance` collapsed to
*exactly* `VEHICLE_RADIUS`, i.e. the literal collision boundary, so it fired at the
moment of contact rather than before it. `_safe_speed` instead computes the speed the
vehicle can be going right now and still stop (at `a_max`) before its own body reaches
the obstacle, re-derived every tick from the live sensor range -- correct for any
planner without needing planner-specific tuning, since it naturally has no effect when
obstacles are far and only binds as they get close.
"""

import math

from core.environment import VEHICLE_RADIUS
from core.interfaces import Controller
from core.messaging.bus import Bus
from core.messaging.messages import ControlCmdMsg, ObstacleRangeMsg, PathMsg, PoseEstimateMsg


class ControllerNode:
    def __init__(self, bus: Bus, controller: Controller, a_max: float = 0.8, stopping_buffer: float = 0.5):
        self.bus = bus
        self.controller = controller
        self.a_max = a_max
        # Extra cushion beyond the exact kinematic stopping distance, to absorb the one-tick
        # sense-decide-act latency (obstacle_ranges is read at tick t, but the resulting slower
        # command isn't actually applied by VehicleNode until tick t+1) and the fact that
        # VehicleNode's accel limiting is a proportional ramp toward v_desired, not an instant
        # switch to -a_max -- both of which make the naive sqrt(2*a*d) figure optimistic. Picked
        # empirically: 0.5 m was the smallest value in a sweep (0.2/0.3/0.5/0.8/1.2) that reached
        # zero collisions across all 5 parallel_between_cars/pure_pursuit seeds -- see
        # KNOWN_BUGS.md bug 1.
        self.stopping_buffer = stopping_buffer

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

    def _safe_speed(self) -> float:
        if not self._obstacle_ranges or not self._obstacle_ranges.readings:
            return float("inf")
        closest_range = min(self._obstacle_ranges.readings.values())
        # closest_range is measured from the vehicle's own center to the obstacle's surface (see
        # UltrasonicArray), so the body-to-body gap still available to brake within is
        # closest_range - VEHICLE_RADIUS, not closest_range itself.
        gap = closest_range - VEHICLE_RADIUS - self.stopping_buffer
        return math.sqrt(2 * self.a_max * gap) if gap > 0 else 0.0

    def step(self) -> None:
        if self._pose_estimate is None or self._path is None:
            self.bus.publish("control_cmd", ControlCmdMsg(0.0, 0.0))
            return

        v, delta = self.controller.control(self._pose_estimate, self._path.path)
        v_safe = self._safe_speed()
        v = max(-v_safe, min(v_safe, v))

        self.bus.publish("control_cmd", ControlCmdMsg(v, delta))
