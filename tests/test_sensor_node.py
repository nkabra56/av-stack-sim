"""SensorNode's dropout/latency modeling (DESIGN.md section 10's future-extensions
list): a message that's dropped never arrives at all; a message under latency arrives
`latency_ticks` ticks after it was computed, not immediately. Direct, deterministic
unit coverage on the node itself, independent of any full harness run."""

import numpy as np
import pytest

from core.environment import Environment, Spot
from core.messaging.bus import Bus
from core.messaging.messages import TrueStateMsg
from core.nodes.sensor_node import SensorNode
from core.sensors import UltrasonicArray


def _sensor_node(bus, rng, **kwargs) -> SensorNode:
    ultrasonic = UltrasonicArray(angles=[0.0], max_range=8.0)
    environment = Environment(Spot(100.0, 100.0, 0.0), obstacles=[])  # far away -- no landmarks in range
    node = SensorNode(bus, ultrasonic, environment, rng, position_fix_period=1, **kwargs)
    bus.publish("true_state", TrueStateMsg(x=0.0, y=0.0, theta=0.0, v=0.0, delta=0.0))
    return node


def test_zero_dropout_and_latency_publishes_every_tick_immediately():
    bus = Bus()
    rng = np.random.default_rng(0)
    node = _sensor_node(bus, rng)
    compass_msgs = []
    bus.subscribe("compass", lambda msg: compass_msgs.append(msg))

    for _ in range(5):
        node.step()
    assert len(compass_msgs) == 5


def test_dropout_prob_one_never_delivers_anything():
    bus = Bus()
    rng = np.random.default_rng(0)
    node = _sensor_node(bus, rng, dropout_prob=1.0)
    received = []
    for topic in ("obstacle_ranges", "compass", "position_fix", "landmark_bearings"):
        bus.subscribe(topic, lambda msg, t=topic: received.append(t))

    for _ in range(10):
        node.step()
    assert received == []


def test_dropout_is_probabilistic_not_all_or_nothing():
    """A middling dropout_prob should deliver *some* but not *all* compass readings
    over enough ticks -- confirms the RNG draw actually gates delivery per-message,
    not just per-run."""
    bus = Bus()
    rng = np.random.default_rng(0)
    node = _sensor_node(bus, rng, dropout_prob=0.5)
    compass_msgs = []
    bus.subscribe("compass", lambda msg: compass_msgs.append(msg))

    for _ in range(200):
        node.step()
    assert 40 < len(compass_msgs) < 160  # binomial(200, 0.5): overwhelmingly in this band


def test_latency_delays_delivery_by_exactly_latency_ticks():
    bus = Bus()
    rng = np.random.default_rng(0)
    node = _sensor_node(bus, rng, latency_ticks=3)
    arrivals = []
    bus.subscribe("compass", lambda msg: arrivals.append(msg))

    for tick in range(1, 6):
        node.step()
        if tick < 4:
            assert arrivals == []  # nothing due yet
    assert len(arrivals) == 2  # ticks 1 and 2's readings, released on ticks 4 and 5


def test_latency_preserves_the_original_measurement_not_a_stale_recompute():
    """The delayed message should carry the value computed *when the reading was
    taken*, not a value re-derived from wherever the vehicle is once it's released --
    modeling a real late-arriving packet, not a teleporting one."""
    bus = Bus()
    rng = np.random.default_rng(0)
    ultrasonic = UltrasonicArray(angles=[0.0], max_range=8.0)
    environment = Environment(Spot(100.0, 100.0, 0.0), obstacles=[])
    node = SensorNode(bus, ultrasonic, environment, rng, latency_ticks=2, compass_std=0.0)

    bus.publish("true_state", TrueStateMsg(x=0.0, y=0.0, theta=0.3, v=0.0, delta=0.0))
    compass_msgs = []
    bus.subscribe("compass", lambda msg: compass_msgs.append(msg))
    node.step()  # theta=0.3 reading queued, not yet delivered

    bus.publish("true_state", TrueStateMsg(x=0.0, y=0.0, theta=1.5, v=0.0, delta=0.0))
    node.step()
    node.step()  # tick 3: the tick-1 (theta=0.3) reading is now due

    assert compass_msgs[0].theta == pytest.approx(0.3)
