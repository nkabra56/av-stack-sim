"""Tests what's actually testable in ros2_bridge.py without a real ROS2 install (see
that module's docstring for what is and isn't verified): the message-conversion
functions against known inputs, and Ros2Bridge's subscribe -> convert -> publish
wiring, using minimal fakes for rclpy's Node/Publisher rather than the real thing.
"""

import numpy as np
import pytest

from core.messaging.bus import Bus
from core.messaging.messages import ControlCmdMsg, ObstacleRangeMsg, PathMsg, PoseEstimateMsg
from core.messaging.ros2_bridge import (
    Ros2Bridge,
    control_cmd_to_ros_kwargs,
    obstacle_range_to_ros_kwargs_list,
    path_to_ros_kwargs,
    pose_estimate_to_ros_kwargs,
)


# --- Conversion functions ---------------------------------------------------------


def test_pose_estimate_conversion_maps_position_and_yaw_only_quaternion():
    msg = PoseEstimateMsg(x=1.0, y=2.0, theta=np.pi / 2, covariance=np.eye(3) * 0.5)
    kwargs = pose_estimate_to_ros_kwargs(msg)
    assert kwargs["position"] == {"x": 1.0, "y": 2.0, "z": 0.0}
    # theta=pi/2 -> quaternion (0, 0, sin(pi/4), cos(pi/4))
    assert kwargs["orientation"]["z"] == pytest.approx(np.sin(np.pi / 4))
    assert kwargs["orientation"]["w"] == pytest.approx(np.cos(np.pi / 4))
    assert kwargs["orientation"]["x"] == 0.0 and kwargs["orientation"]["y"] == 0.0


def test_pose_estimate_conversion_embeds_covariance_at_the_right_flattened_indices():
    """The 3x3 [x, y, theta] covariance must land at ROS2's (x, y, yaw) sub-indices
    (0, 1, 5) of the flattened row-major 6x6 layout, not just anywhere plausible-
    looking -- a wrong index here would silently corrupt every downstream consumer's
    uncertainty estimate without ever raising an error."""
    cov = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    msg = PoseEstimateMsg(x=0.0, y=0.0, theta=0.0, covariance=cov)
    flat = pose_estimate_to_ros_kwargs(msg)["covariance"]
    cov_6x6 = np.array(flat).reshape(6, 6)
    for a, ra in enumerate((0, 1, 5)):
        for b, rb in enumerate((0, 1, 5)):
            assert cov_6x6[ra, rb] == pytest.approx(cov[a, b])
    # everything outside those 9 cells must be exactly 0
    mask = np.ones((6, 6), dtype=bool)
    for ra in (0, 1, 5):
        for rb in (0, 1, 5):
            mask[ra, rb] = False
    assert np.all(cov_6x6[mask] == 0.0)


def test_control_cmd_conversion_maps_v_and_delta():
    msg = ControlCmdMsg(v=1.2, delta=-0.3)
    kwargs = control_cmd_to_ros_kwargs(msg)
    assert kwargs["linear"]["x"] == pytest.approx(1.2)
    assert kwargs["angular"]["z"] == pytest.approx(-0.3)


def test_obstacle_range_conversion_produces_one_entry_per_beam():
    msg = ObstacleRangeMsg(readings={0.0: 3.0, 0.5: 5.0, -0.5: 8.0})
    kwargs_list = obstacle_range_to_ros_kwargs_list(msg, max_range=8.0)
    assert len(kwargs_list) == 3
    assert {k["range"] for k in kwargs_list} == {3.0, 5.0, 8.0}
    assert all(k["max_range"] == 8.0 for k in kwargs_list)


def test_path_conversion_produces_one_pose_per_waypoint():
    path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, np.pi / 2]])
    kwargs = path_to_ros_kwargs(PathMsg(path))
    assert len(kwargs["poses"]) == 2
    assert kwargs["poses"][0]["pose"]["position"] == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert kwargs["poses"][1]["pose"]["orientation"]["z"] == pytest.approx(np.sin(np.pi / 4))


# --- Ros2Bridge wiring, via fake rclpy stand-ins -----------------------------------


class _FakeRosPublisher:
    def __init__(self):
        self.published: list = []

    def publish(self, msg) -> None:
        self.published.append(msg)


class _FakeRosNode:
    def __init__(self):
        self.publishers: dict[str, _FakeRosPublisher] = {}
        self.created: list[tuple[str, str, int]] = []  # (msg_type_name, topic, qos)

    def create_publisher(self, msg_type, topic, qos_profile):
        self.created.append((msg_type.__name__, topic, qos_profile))
        publisher = _FakeRosPublisher()
        self.publishers[topic] = publisher
        return publisher


class _FakeRosMsg:
    """Stand-in for a real ROS2 message class -- records whatever kwargs it was
    constructed with, so a test can check the bridge passed the converted fields
    through unchanged."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_register_creates_a_publisher_on_the_declared_ros_topic():
    bus = Bus()
    node = _FakeRosNode()
    bridge = Ros2Bridge(bus, node)

    bridge.register("control_cmd", "/cmd_vel", _FakeRosMsg, control_cmd_to_ros_kwargs)

    assert node.created == [("_FakeRosMsg", "/cmd_vel", 10)]


def test_bus_message_is_converted_and_published_on_the_ros_topic():
    bus = Bus()
    node = _FakeRosNode()
    bridge = Ros2Bridge(bus, node)
    bridge.register("control_cmd", "/cmd_vel", _FakeRosMsg, control_cmd_to_ros_kwargs)

    bus.publish("control_cmd", ControlCmdMsg(v=0.7, delta=0.1))

    published = node.publishers["/cmd_vel"].published
    assert len(published) == 1
    assert published[0].kwargs["linear"]["x"] == pytest.approx(0.7)
    assert published[0].kwargs["angular"]["z"] == pytest.approx(0.1)


def test_multiple_bus_messages_each_produce_one_ros_publish():
    bus = Bus()
    node = _FakeRosNode()
    bridge = Ros2Bridge(bus, node)
    bridge.register("control_cmd", "/cmd_vel", _FakeRosMsg, control_cmd_to_ros_kwargs)

    for v in (0.1, 0.2, 0.3):
        bus.publish("control_cmd", ControlCmdMsg(v=v, delta=0.0))

    assert len(node.publishers["/cmd_vel"].published) == 3


def test_bridge_does_not_interfere_with_other_bus_subscribers():
    """The bridge subscribes like any other node -- it shouldn't prevent or reorder
    delivery to subscribers that were already listening on the same topic."""
    bus = Bus()
    received = []
    bus.subscribe("control_cmd", lambda msg: received.append(msg))

    node = _FakeRosNode()
    bridge = Ros2Bridge(bus, node)
    bridge.register("control_cmd", "/cmd_vel", _FakeRosMsg, control_cmd_to_ros_kwargs)

    bus.publish("control_cmd", ControlCmdMsg(v=1.0, delta=0.0))

    assert len(received) == 1
    assert len(node.publishers["/cmd_vel"].published) == 1
