"""ControllerNode's speed governor (see its module docstring, KNOWN_BUGS.md entry 2)
must be direction-aware, not just distance-aware -- found in code review after entry
2's fix shipped: the original version took the closest ultrasonic reading across *all*
beams regardless of which way the vehicle was actually trying to move, which meant a
reverse-gear maneuver got zero protection from an obstacle behind the vehicle (outside
the forward-only sensor cone at the time) while simultaneously being needlessly
throttled while reversing *away* from something in front. These tests pin the fix
directly and deterministically, independent of any specific scenario's closed-loop
dynamics.
"""

import numpy as np

from core.messaging.bus import Bus
from core.messaging.messages import ObstacleRangeMsg, PathMsg, PoseEstimateMsg
from core.nodes.controller_node import ControllerNode

FRONT_ANGLE = 0.0  # dead ahead
REAR_ANGLE = np.pi  # dead behind


class _FixedController:
    """Stub Controller: always returns the same (v, delta), regardless of path --
    isolates the governor's clamping from any real path-tracking law."""

    def __init__(self, v: float):
        self._v = v

    def control(self, pose, path):
        return self._v, 0.0


def _step_with_reading(controller_v: float, angle: float, range_reading: float) -> float:
    """Builds a fresh ControllerNode, feeds it one obstacle_ranges reading at `angle`,
    and returns the resulting commanded speed for a controller that always wants `controller_v`."""
    bus = Bus()
    node = ControllerNode(bus, _FixedController(controller_v), a_max=0.8, stopping_buffer=0.5)
    commands = []
    bus.subscribe("control_cmd", lambda msg: commands.append(msg))

    bus.publish("pose_estimate", PoseEstimateMsg(0.0, 0.0, 0.0, covariance=np.zeros((3, 3))))
    bus.publish("path", PathMsg(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])))
    bus.publish("obstacle_ranges", ObstacleRangeMsg({angle: range_reading}))
    node.step()
    return commands[-1].v


# closest_range=1.1 -> gap = 1.1 - VEHICLE_RADIUS(1.0) - stopping_buffer(0.5) < 0 -> v_safe=0
CLOSE_READING = 1.1
FAR_READING = 100.0


def test_rear_obstacle_does_not_throttle_forward_driving():
    v = _step_with_reading(controller_v=2.0, angle=REAR_ANGLE, range_reading=CLOSE_READING)
    assert v == 2.0  # unthrottled: the close reading is behind, forward driving is unaffected


def test_rear_obstacle_throttles_reversing():
    """The actual bug: previously a rear-only obstacle was invisible to the governor
    entirely (no rear beams existed), so reversing into it was never caught."""
    v = _step_with_reading(controller_v=-2.0, angle=REAR_ANGLE, range_reading=CLOSE_READING)
    assert v == 0.0  # governed to a stop, not allowed to back into it


def test_front_obstacle_does_not_throttle_reversing_away():
    v = _step_with_reading(controller_v=-2.0, angle=FRONT_ANGLE, range_reading=CLOSE_READING)
    assert v == -2.0  # unthrottled: reversing away from a front obstacle only increases the gap


def test_front_obstacle_throttles_forward_driving():
    v = _step_with_reading(controller_v=2.0, angle=FRONT_ANGLE, range_reading=CLOSE_READING)
    assert v == 0.0


def test_far_obstacles_never_throttle_either_direction():
    for angle in (FRONT_ANGLE, REAR_ANGLE):
        for controller_v in (2.0, -2.0):
            assert _step_with_reading(controller_v, angle, FAR_READING) == controller_v


def test_harness_ultrasonic_array_has_a_rear_cone():
    """Regression guard for harness.py's DEFAULT_SENSOR_ANGLES: at least one beam angle
    must point behind the vehicle (|angle| > pi/2), or the governor has nothing to be
    direction-aware *about*."""
    from core.harness import DEFAULT_SENSOR_ANGLES
    from core.vehicle import wrap_angle

    assert any(abs(wrap_angle(a)) > np.pi / 2 for a in DEFAULT_SENSOR_ANGLES)
    assert any(abs(wrap_angle(a)) < np.pi / 2 for a in DEFAULT_SENSOR_ANGLES)


# --- Tracking-aware buffer (KNOWN_BUGS.md entry 3): direct, deterministic coverage --
# of `_effective_buffer` in isolation, independent of any closed-loop scenario's
# dynamics -- the closed-loop proof that this doesn't reopen entry 2's collision (or
# that it actually lets entry 3's scenario complete) lives in test_replanning.py, which
# needed a real parameter sweep since it's a genuinely dynamical question; whether
# `_effective_buffer` itself picks the right buffer for a given cross-track distance is
# not, and deserves the same direct-and-fast coverage as the rest of this file.


def _node_with_pose_and_path(bus: Bus, path_y: float, pose_y: float, **kwargs) -> ControllerNode:
    node = ControllerNode(bus, _FixedController(0.0), a_max=0.8, **kwargs)
    bus.publish("pose_estimate", PoseEstimateMsg(0.0, pose_y, 0.0, covariance=np.zeros((3, 3))))
    bus.publish("path", PathMsg(np.array([[0.0, path_y, 0.0], [10.0, path_y, 0.0]])))
    return node


def test_effective_buffer_defaults_to_stopping_buffer_when_tracking_disabled():
    """tracked_stopping_buffer=None (the ControllerNode default) must disable the
    feature entirely, regardless of how well the vehicle happens to be tracking --
    this is what keeps every planner without an exposed `safety_margin`
    (Dubins/ReedsShepp) on the fully conservative buffer."""
    bus = Bus()
    node = _node_with_pose_and_path(bus, path_y=0.0, pose_y=0.0, stopping_buffer=0.5)
    assert node._effective_buffer() == 0.5


def test_effective_buffer_uses_tracked_buffer_when_accurately_tracking():
    bus = Bus()
    node = _node_with_pose_and_path(
        bus, path_y=0.0, pose_y=0.0, stopping_buffer=0.5, tracked_stopping_buffer=0.2, tracking_threshold=0.03
    )
    assert node._effective_buffer() == 0.2  # cross-track distance is 0 -- well within threshold


def test_effective_buffer_falls_back_when_drifted_off_path():
    bus = Bus()
    node = _node_with_pose_and_path(
        bus, path_y=0.0, pose_y=0.5, stopping_buffer=0.5, tracked_stopping_buffer=0.2, tracking_threshold=0.03
    )
    assert node._effective_buffer() == 0.5  # 0.5m cross-track error is well past the 0.03m threshold


def test_effective_buffer_transitions_exactly_at_the_threshold():
    bus = Bus()
    just_inside = _node_with_pose_and_path(
        bus, path_y=0.0, pose_y=0.02, stopping_buffer=0.5, tracked_stopping_buffer=0.2, tracking_threshold=0.03
    )
    assert just_inside._effective_buffer() == 0.2

    bus2 = Bus()
    just_outside = _node_with_pose_and_path(
        bus2, path_y=0.0, pose_y=0.04, stopping_buffer=0.5, tracked_stopping_buffer=0.2, tracking_threshold=0.03
    )
    assert just_outside._effective_buffer() == 0.5
