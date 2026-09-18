"""Wraps the existing Controller (Pure Pursuit or MPC) unchanged, one control decision per
tick. Also owns the speed governor (braking on close obstacles) -- see KNOWN_BUGS.md entries 2/3."""

import math

import numpy as np

from core.environment import VEHICLE_RADIUS
from core.interfaces import Controller
from core.messaging.bus import Bus
from core.messaging.messages import ControlCmdMsg, ObstacleRangeMsg, PathMsg, PoseEstimateMsg, ReplanRequestMsg
from core.vehicle import wrap_angle


def _distance_to_polyline(x: float, y: float, path_xy: np.ndarray) -> float:
    """True perpendicular distance from (x, y) to the piecewise-linear path, not just to
    its nearest waypoint -- avoids misclassifying on-plan driving as "not tracking"."""
    if len(path_xy) < 2:
        return float(np.hypot(path_xy[:, 0] - x, path_xy[:, 1] - y).min())
    a, b = path_xy[:-1], path_xy[1:]
    ab = b - a
    ab_len_sq = np.sum(ab * ab, axis=1)
    t = np.where(ab_len_sq > 1e-12, np.sum(((x, y) - a) * ab, axis=1) / np.where(ab_len_sq > 1e-12, ab_len_sq, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[:, None] * ab
    return float(np.hypot(closest[:, 0] - x, closest[:, 1] - y).min())

STALL_SPEED = 0.1  # m/s -- below this, the governor is treated as "essentially stopping" the vehicle
STALL_TICKS = 15  # ~1.5s at dt=0.1 -- long enough to be a real stall, not ordinary governor tightening


class ControllerNode:
    def __init__(
        self,
        bus: Bus,
        controller: Controller,
        a_max: float = 0.8,
        stopping_buffer: float = 0.5,
        tracked_stopping_buffer: float | None = None,
        tracking_threshold: float = 0.03,
        latency_margin: float = 0.0,
    ):
        self.bus = bus
        self.controller = controller
        self.a_max = a_max
        self._stall_ticks = 0
        # Worst-case extra gap erosion while an obstacle_ranges reading was in flight under
        # sensor latency; 0.0 (default) is the original, unaffected behavior.
        self.latency_margin = latency_margin
        # Cushion beyond the exact kinematic stopping distance, absorbing one-tick latency
        # and VehicleNode's ramped (not instant) braking. Picked empirically (KNOWN_BUGS.md entry 2).
        self.stopping_buffer = stopping_buffer
        # Smaller buffer used only while accurately tracking the path (see _effective_buffer).
        # None (default) disables the feature, always using stopping_buffer.
        self.tracked_stopping_buffer = tracked_stopping_buffer
        # Cross-track distance below which the vehicle counts as "accurately tracking".
        # Deliberately tight (3cm) -- see KNOWN_BUGS.md entry 3 for the sweep behind this value.
        self.tracking_threshold = tracking_threshold

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

    def _effective_buffer(self) -> float:
        """`tracked_stopping_buffer` while accurately tracking the path, else the fully
        conservative `stopping_buffer` (see KNOWN_BUGS.md entry 3)."""
        if self.tracked_stopping_buffer is None or self._path is None or self._pose_estimate is None:
            return self.stopping_buffer
        path_xy = self._path.path[:, :2]
        cross_track = _distance_to_polyline(self._pose_estimate.x, self._pose_estimate.y, path_xy)
        return self.tracked_stopping_buffer if cross_track < self.tracking_threshold else self.stopping_buffer

    def _safe_speed(self, forward: bool) -> float:
        """Safe speed toward `forward` (front beams) or not (rear beams), ignoring the
        other side entirely -- classified by angle, not a hardcoded front/rear list."""
        if not self._obstacle_ranges or not self._obstacle_ranges.readings:
            return float("inf")
        relevant = [
            r for angle, r in self._obstacle_ranges.readings.items() if (abs(wrap_angle(angle)) < math.pi / 2) == forward
        ]
        if not relevant:
            return float("inf")
        closest_range = min(relevant)
        # closest_range is measured from the vehicle's center, so the body-to-body gap still
        # available to brake within subtracts VEHICLE_RADIUS too.
        gap = closest_range - VEHICLE_RADIUS - self._effective_buffer() - self.latency_margin
        return math.sqrt(2 * self.a_max * gap) if gap > 0 else 0.0

    def _note_stall(self, stalled: bool) -> None:
        if stalled:
            self._stall_ticks += 1
            if self._stall_ticks % STALL_TICKS == 0:
                self.bus.publish("replan_request", ReplanRequestMsg())
        else:
            self._stall_ticks = 0

    def step(self) -> None:
        if self._pose_estimate is None:
            self.bus.publish("control_cmd", ControlCmdMsg(0.0, 0.0))
            return

        if self._path is None:
            # No plan to track yet -- PlannerNode hasn't run yet, or its last attempt raised.
            # Counts as a stall too, so a boxed-in start pose still triggers a re-plan retry.
            self._note_stall(stalled=True)
            self.bus.publish("control_cmd", ControlCmdMsg(0.0, 0.0))
            return

        v, delta = self.controller.control(self._pose_estimate, self._path.path)
        v_safe = self._safe_speed(forward=v >= 0)
        v = max(-v_safe, min(v_safe, v))
        self._note_stall(stalled=v_safe < STALL_SPEED)

        self.bus.publish("control_cmd", ControlCmdMsg(v, delta))
