from core.messaging.bus import Bus


def test_publish_delivers_to_all_subscribers():
    bus = Bus()
    received_a, received_b = [], []
    bus.subscribe("topic", received_a.append)
    bus.subscribe("topic", received_b.append)

    bus.publish("topic", "hello")

    assert received_a == ["hello"]
    assert received_b == ["hello"]


def test_unsubscribed_topic_does_not_error():
    bus = Bus()
    bus.publish("nobody_listens", 42)  # must not raise


def test_multiple_publishes_delivered_in_order():
    bus = Bus()
    received = []
    bus.subscribe("topic", received.append)

    bus.publish("topic", 1)
    bus.publish("topic", 2)
    bus.publish("topic", 3)

    assert received == [1, 2, 3]


def test_subscribers_only_receive_their_own_topic():
    bus = Bus()
    received = []
    bus.subscribe("topic_a", received.append)

    bus.publish("topic_b", "should not arrive")

    assert received == []
