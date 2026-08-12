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

**Stall detection -> re-plan request** (see KNOWN_BUGS.md entry 3 / IMPLEMENTATION.md's M4
entry): the speed governor below is a safety backstop, not a routing decision -- if it
ends up binding for a sustained stretch (an obstacle close enough, for long enough, that
the vehicle can't make real progress), that's the same signal a real AV stack would use
to trigger re-planning, not just sit there. Every `STALL_TICKS`-th consecutive governed-
near-zero tick publishes a `replan_request` -- not just the first: even with the
tracking-aware buffer below, a re-plan can still land somewhere the vehicle can't
immediately move from (e.g. capped-out `max_replans`), so a stall that never recovers
still deserves periodic retries, not exactly one attempt ever. The counter resets the
moment the vehicle moves again, so recovering and re-stalling later also asks again
after its own `STALL_TICKS`.

**Speed governor, not a binary brake** (see KNOWN_BUGS.md's entry 2 for the finding that
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

**Direction-aware, not just distance-aware** (found in code review after entry 2's fix
shipped): `harness.py`'s ultrasonic array now carries both a forward cone and a mirrored
rear cone (`DEFAULT_SENSOR_ANGLES`). `_safe_speed(forward=...)` only looks at the beams
on the side the vehicle is actually trying to move toward -- the old version took the
closest reading across *all* beams regardless of travel direction, which had it exactly
backwards for the case that actually needs a rear sensor: a reverse-gear maneuver (e.g.
the parallel_between_cars cusp) got zero protection from an obstacle behind the vehicle
(outside the old forward-only cone, so `_safe_speed` saw nothing and never throttled),
while simultaneously being needlessly throttled while reversing *away* from something in
front. Splitting by direction fixes both: reversing is now governed by the rear beams
only, forward driving by the front beams only.

**No path yet is also a stall** (found in a second, broader code review pass): `step()`
used to early-return whenever `self._path is None`, before ever reaching the stall
counter below -- which meant that if `PlannerNode`'s *initial* plan attempt raised (start
pose genuinely boxed in, or any other up-front infeasibility), no `path` was ever
published, this node would sit at `control_cmd(0, 0)` forever, and the stall-detection
logic that exists specifically to ask for a retry would never even run, since it lived
entirely inside the branch that requires a path to already exist. The re-planning
mechanism (KNOWN_BUGS.md entry 3) was therefore unreachable for the one failure mode it
should most obviously cover. Fixed by treating "have a pose estimate but no path" as its
own stall condition, counted and retried the same way a governed-to-zero stall is.

**Tracking-aware buffer, closing KNOWN_BUGS.md entry 3**: `stopping_buffer` alone can't
be shrunk to fit inside a planner's own tight (but real, verified) clearance -- doing so
unconditionally would reopen entry 2's collision (Pure Pursuit drifting off a curvature-
saturated path is exactly a *large* cross-track error, and the whole point of
`stopping_buffer` was to survive that). But a small buffer *is* safe specifically when
the vehicle is accurately tracking a path a planner already checked for
`safety_margin` clearance -- entry 3's actual finding was that the governor had no way
to tell "accurately tracking a tight, planner-verified route" apart from "genuinely
drifting toward unplanned proximity," and used the same conservative buffer for both.
`_effective_buffer` now checks live cross-track distance to the current path: below
`tracking_threshold`, it's safe to use the smaller `tracked_stopping_buffer` (derived by
`ParkingHarness` from the active planner's own `safety_margin`, when it exposes one --
`getattr(planner, "safety_margin", None)`, so planners with no obstacle-clearance
guarantee at all, like Dubins/ReedsShepp, automatically keep the fully conservative
buffer instead of silently gaining a smaller one they never earned). Above the
threshold, it falls back to the same `stopping_buffer` as before -- unchanged behavior
for entry 2's exact failure mode, verified directly in tests/test_replanning.py.
"""

import math

import numpy as np

from core.environment import VEHICLE_RADIUS
from core.interfaces import Controller
from core.messaging.bus import Bus
from core.messaging.messages import ControlCmdMsg, ObstacleRangeMsg, PathMsg, PoseEstimateMsg, ReplanRequestMsg
from core.vehicle import wrap_angle


def _distance_to_polyline(x: float, y: float, path_xy: np.ndarray) -> float:
    """True perpendicular distance from (x, y) to the piecewise-linear path, not just
    to its nearest waypoint -- a vehicle sitting exactly on a straight segment between
    two waypoints has zero real cross-track error, but nearest-vertex distance alone
    would report up to half the segment spacing, misclassifying genuinely on-plan
    driving as "not tracking" whenever waypoint spacing approaches
    `ControllerNode.tracking_threshold` (Hybrid A*'s default 0.1m `output_step` puts
    that within a factor of ~3, not a large margin). Clamps each segment's projection
    parameter to [0, 1] so it degrades to vertex distance at the path's own ends,
    where there's no segment beyond the last/before the first waypoint to project onto."""
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
STALL_TICKS = 15  # ~1.5s at the harness's dt=0.1 -- long enough that this is a real stall, not the
# governor's ordinary momentary tightening as the vehicle brushes past an intentionally-close
# obstacle on a valid path.


class ControllerNode:
    def __init__(
        self,
        bus: Bus,
        controller: Controller,
        a_max: float = 0.8,
        stopping_buffer: float = 0.5,
        tracked_stopping_buffer: float | None = None,
        tracking_threshold: float = 0.03,
    ):
        self.bus = bus
        self.controller = controller
        self.a_max = a_max
        self._stall_ticks = 0
        # Extra cushion beyond the exact kinematic stopping distance, to absorb the one-tick
        # sense-decide-act latency (obstacle_ranges is read at tick t, but the resulting slower
        # command isn't actually applied by VehicleNode until tick t+1) and the fact that
        # VehicleNode's accel limiting is a proportional ramp toward v_desired, not an instant
        # switch to -a_max -- both of which make the naive sqrt(2*a*d) figure optimistic. Picked
        # empirically: 0.5 m was the smallest value in a sweep (0.2/0.3/0.5/0.8/1.2) that reached
        # zero collisions across all 5 parallel_between_cars/pure_pursuit seeds -- see
        # KNOWN_BUGS.md entry 2.
        self.stopping_buffer = stopping_buffer
        # Smaller buffer used only while accurately tracking the current path (see
        # `_effective_buffer` / module docstring's "Tracking-aware buffer" entry). None
        # (the default) disables the feature entirely -- always uses `stopping_buffer`,
        # identical to this node's behavior before entry 3 was closed.
        self.tracked_stopping_buffer = tracked_stopping_buffer
        # Cross-track distance below which the vehicle counts as "accurately tracking".
        # Deliberately tight (3cm): a real parameter sweep (KNOWN_BUGS.md entry 3 /
        # tests/test_replanning.py) found this needs to be small enough that the
        # classification is essentially never wrong, not just "usually right" -- a
        # looser threshold (tried 0.05-0.15m) let real cross-track noise cross it while
        # genuinely diverging (entry 2's curvature-saturation failure mode reaches
        # ~0.66m, but the danger starts well before that), reopening entry 2's
        # collision at several (threshold, tracked_buffer_extra) combinations. 0.03m
        # combined with harness.py's default `tracked_buffer_extra=0.3` was the
        # smallest/safest combination found: 0/5 collisions on both the entry 2
        # regression scenario and the entry 3 scenario, 5/5 full recoveries on the
        # latter, across 5 seeds each.
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
        """`tracked_stopping_buffer` while accurately tracking the current path, else
        the fully conservative `stopping_buffer` -- see module docstring's
        "Tracking-aware buffer" entry. Falls back to `stopping_buffer` whenever the
        feature is disabled (`tracked_stopping_buffer is None`) or there's no path/pose
        to measure tracking accuracy against yet."""
        if self.tracked_stopping_buffer is None or self._path is None or self._pose_estimate is None:
            return self.stopping_buffer
        path_xy = self._path.path[:, :2]
        cross_track = _distance_to_polyline(self._pose_estimate.x, self._pose_estimate.y, path_xy)
        return self.tracked_stopping_buffer if cross_track < self.tracking_threshold else self.stopping_buffer

    def _safe_speed(self, forward: bool) -> float:
        """Safe speed toward `forward` (True = the front beams' side, False = the rear
        beams'), ignoring beams on the *other* side entirely -- an obstacle behind the
        vehicle has no bearing on how fast it's safe to drive forward, and vice versa.
        Beams are classified by angle alone (front = within +/-90 degrees of heading),
        not by a hardcoded front/rear list, so this works for any beam layout
        `harness.py` hands the sensor, not just the current one."""
        if not self._obstacle_ranges or not self._obstacle_ranges.readings:
            return float("inf")
        relevant = [
            r for angle, r in self._obstacle_ranges.readings.items() if (abs(wrap_angle(angle)) < math.pi / 2) == forward
        ]
        if not relevant:
            return float("inf")
        closest_range = min(relevant)
        # closest_range is measured from the vehicle's own center to the obstacle's surface (see
        # UltrasonicArray), so the body-to-body gap still available to brake within is
        # closest_range - VEHICLE_RADIUS, not closest_range itself.
        gap = closest_range - VEHICLE_RADIUS - self._effective_buffer()
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
            # No plan to track yet -- either PlannerNode hasn't run this tick, or its
            # last attempt (initial or a re-plan) raised. Either way this is a stall
            # (see module docstring's "no path yet is also a stall"), not a reason to
            # skip stall-tracking entirely.
            self._note_stall(stalled=True)
            self.bus.publish("control_cmd", ControlCmdMsg(0.0, 0.0))
            return

        v, delta = self.controller.control(self._pose_estimate, self._path.path)
        v_safe = self._safe_speed(forward=v >= 0)
        v = max(-v_safe, min(v_safe, v))
        self._note_stall(stalled=v_safe < STALL_SPEED)

        self.bus.publish("control_cmd", ControlCmdMsg(v, delta))
