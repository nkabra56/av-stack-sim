"""Reference adapter mapping bus.py's in-process pub/sub onto real ROS2 topics (DESIGN.md
section 10). Not verified against a real ROS2 install -- see tests/test_ros2_bridge.py."""

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from core.messaging.bus import Bus
from core.messaging.messages import ControlCmdMsg, ObstacleRangeMsg, PathMsg, PoseEstimateMsg


class RosPublisher(Protocol):
    def publish(self, msg: Any) -> None: ...


class RosNode(Protocol):
    """The narrow subset of `rclpy.node.Node`'s interface this bridge needs, kept small so a
    minimal stand-in can satisfy it for testing without importing `rclpy` at all."""

    def create_publisher(self, msg_type: Any, topic: str, qos_profile: int) -> RosPublisher: ...


def pose_estimate_to_ros_kwargs(msg: PoseEstimateMsg) -> dict:
    """-> geometry_msgs/PoseWithCovarianceStamped-shaped kwargs. Heading becomes a yaw-only
    quaternion; the 3x3 [x,y,theta] covariance embeds into ROS2's 6x6 layout at (x,y,yaw)."""
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
    """-> geometry_msgs/Twist-shaped kwargs. `angular.z` carries steering *angle* (delta),
    not yaw rate -- Twist has no steering-angle field; `ackermann_msgs/AckermannDrive` would be more correct."""
    return {"linear": {"x": float(msg.v), "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": float(msg.delta)}}


def obstacle_range_to_ros_kwargs_list(msg: ObstacleRangeMsg, max_range: float, field_of_view: float = 0.05) -> list[dict]:
    """-> one sensor_msgs/Range-shaped kwargs dict per beam (ROS2 has no native "named
    beam array" message; a real bridge would publish one Range topic per beam or a custom message)."""
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
    """Mirrors Bus topics onto real ROS2 topics, one direction (sim -> ROS2), e.g. so RViz
    could observe a run live. Registration is a single generic method, not one per message type."""

    def __init__(self, bus: Bus, node: RosNode):
        self._bus = bus
        self._node = node

    def register(self, bus_topic: str, ros_topic: str, ros_msg_type: Any, convert: Callable[[Any], dict], qos_profile: int = 10) -> None:
        publisher = self._node.create_publisher(ros_msg_type, ros_topic, qos_profile)

        def on_bus_message(msg: Any) -> None:
            publisher.publish(ros_msg_type(**convert(msg)))

        self._bus.subscribe(bus_topic, on_bus_message)
