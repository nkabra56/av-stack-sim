"""Reference adapter mapping `messaging/bus.py`'s in-process pub/sub onto real ROS2
topics -- DESIGN.md section 10's future-extensions item, made concrete: "the node/
topic boundaries... were drawn deliberately close to how a real ROS2 graph would be
structured, specifically so that swapping the in-process Bus for real ROS2 topics
later wouldn't require redesigning the nodes themselves -- only how they publish/
subscribe." This module is that "how they publish/subscribe" layer.

**Not verified against a real ROS2 install -- said plainly, not buried.** ROS2/rclpy
could not be installed in this development environment: there's no existing ROS2
install, WSL2 (the standard way to run ROS2 on Windows) isn't available here (the
Windows Subsystem for Linux service itself reports a "class not registered" error,
meaning WSL isn't functional on this machine, not just that a distro isn't
installed), and `rclpy` isn't pip-installable standalone -- it ships bundled with a
real ROS2 distribution's binary install, not as an independent PyPI package. So
nothing here has run against a real `rclpy.node.Node`, a real DDS discovery, or a
real ROS2 message type.

**What *is* actually tested** (`tests/test_ros2_bridge.py`): the message-conversion
functions (`*_to_ros_kwargs` below) against known inputs/expected outputs, and
`Ros2Bridge`'s subscribe -> convert -> publish wiring, using minimal stand-ins for
`rclpy`'s `Node`/`Publisher` (see `RosNode`/`RosPublisher` below) rather than the real
thing. `Ros2Bridge` is written against those two narrow `Protocol`s specifically so it
never imports `rclpy` at all -- this module stays importable, and its actual logic
stays testable, on a machine with no ROS2 install, which is exactly the situation
this was written in. What's unverified is whether real `rclpy`'s actual API matches
these stand-ins closely enough (constructor signatures, QoS object shape, executor
spinning behavior) -- that needs a real ROS2 environment to check, and this module
(plus real `geometry_msgs`/`sensor_msgs`/`nav_msgs` types passed in from the caller)
is what to point one at once it's available.

**The Bus/ROS2 semantic mismatch this doesn't try to hide**: `bus.py`'s dispatch is
synchronous and immediate, a deliberate tradeoff for deterministic tests (see its own
docstring) -- a real ROS2 graph is asynchronous with nondeterministic timing.
`Ros2Bridge` doesn't change Bus's own dispatch model at all; it mirrors each Bus
message onto a real ROS2 `publish()` call the instant it's produced, and (for
inbound topics) mirrors each incoming ROS2 message onto a Bus `publish()` call from
within `rclpy`'s own executor callback thread. Cross-network timing becomes real
(async); the in-process Bus graph this project's nodes actually talk over stays
exactly as deterministic as it always was.
"""

from typing import Any, Callable, Protocol

import numpy as np

from core.messaging.bus import Bus
from core.messaging.messages import ControlCmdMsg, ObstacleRangeMsg, PathMsg, PoseEstimateMsg


class RosPublisher(Protocol):
    def publish(self, msg: Any) -> None: ...


class RosNode(Protocol):
    """The narrow subset of `rclpy.node.Node`'s real interface this bridge actually
    needs -- kept intentionally small so a minimal stand-in can satisfy it for testing
    without importing `rclpy` at all. A real `rclpy.node.Node` satisfies this already;
    nothing here is a redefinition of it, just the slice this module depends on."""

    def create_publisher(self, msg_type: Any, topic: str, qos_profile: int) -> RosPublisher: ...


def pose_estimate_to_ros_kwargs(msg: PoseEstimateMsg) -> dict:
    """-> geometry_msgs/PoseWithCovarianceStamped-shaped constructor kwargs. Heading
    becomes a yaw-only quaternion (this project's pose is always planar -- roll/pitch
    are always 0); the 3x3 [x, y, theta] covariance this project tracks is embedded
    into ROS2's flattened row-major 6x6 [x,y,z,roll,pitch,yaw] layout at the (x, y,
    yaw) sub-indices (0, 1, 5), zero elsewhere -- this project never estimates
    z/roll/pitch uncertainty, so declaring it exactly 0 is correct, not a placeholder."""
    half_yaw = msg.theta / 2.0
    cov_3x3 = msg.covariance
    cov_6x6 = np.zeros((6, 6))
    for a, ra in enumerate((0, 1, 5)):
        for b, rb in enumerate((0, 1, 5)):
            cov_6x6[ra, rb] = cov_3x3[a, b]
    return {
        "position": {"x": float(msg.x), "y": float(msg.y), "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": float(np.sin(half_yaw)), "w": float(np.cos(half_yaw))},
        "covariance": cov_6x6.flatten().tolist(),
    }


def control_cmd_to_ros_kwargs(msg: ControlCmdMsg) -> dict:
    """-> geometry_msgs/Twist-shaped kwargs. `angular.z` carries steering *angle*
    (delta, rad), not yaw *rate* -- Twist has no dedicated steering-angle field, and
    this project's controllers command an Ackermann angle, not a rate. Documented
    here rather than silently reusing the field for something a real Twist consumer
    (e.g. a diff-drive stack) would misinterpret -- an `ackermann_msgs/AckermannDrive`
    (speed, steering_angle) is the more correct real-world target for this project's
    actual control convention, if that message package is available."""
    return {"linear": {"x": float(msg.v), "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": float(msg.delta)}}


def obstacle_range_to_ros_kwargs_list(msg: ObstacleRangeMsg, max_range: float, field_of_view: float = 0.05) -> list[dict]:
    """-> one sensor_msgs/Range-shaped kwargs dict per beam (ROS2 has no native
    "named beam array" message; a real bridge would either publish `len(readings)`
    separate Range topics, one per beam frame_id, or a custom message -- this
    returns the per-beam kwargs list either approach would consume, not just one)."""
    return [
        {"radiation_type": 0, "field_of_view": field_of_view, "min_range": 0.0, "max_range": max_range, "range": float(r)}
        for r in msg.readings.values()
    ]


def path_to_ros_kwargs(msg: PathMsg) -> dict:
    """-> nav_msgs/Path-shaped kwargs: a list of PoseStamped-shaped waypoints, each
    with a yaw-only quaternion orientation like pose_estimate_to_ros_kwargs above."""
    poses = []
    for x, y, theta in msg.path:
        half_yaw = theta / 2.0
        poses.append(
            {
                "pose": {
                    "position": {"x": float(x), "y": float(y), "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": float(np.sin(half_yaw)), "w": float(np.cos(half_yaw))},
                }
            }
        )
    return {"poses": poses}


class Ros2Bridge:
    """Mirrors a set of Bus topics onto real ROS2 topics, one direction (this
    simulation -> a real ROS2 network) -- e.g. so a real RViz instance or `ros2 topic
    echo` could observe a run of this project's simulation live. Inbound bridging
    (a real ROS2 sensor driver feeding *into* this simulation in place of
    SensorNode's own synthetic UltrasonicArray) follows the identical registration
    pattern in reverse (subscribe on the ROS2 side, `bus.publish` on receipt) and
    isn't included here -- this project has no real sensor hardware to receive from,
    so there's nothing concrete to write it against yet.

    Registration is intentionally a single generic method, not one method per message
    type -- adding another Bus topic to bridge is `register(...)` plus a conversion
    function shaped like the four above, not a new class."""

    def __init__(self, bus: Bus, node: RosNode):
        self._bus = bus
        self._node = node

    def register(self, bus_topic: str, ros_topic: str, ros_msg_type: Any, convert: Callable[[Any], dict], qos_profile: int = 10) -> None:
        publisher = self._node.create_publisher(ros_msg_type, ros_topic, qos_profile)

        def on_bus_message(msg: Any) -> None:
            publisher.publish(ros_msg_type(**convert(msg)))

        self._bus.subscribe(bus_topic, on_bus_message)
