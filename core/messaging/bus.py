"""Minimal synchronous pub/sub bus. See DESIGN.md section 2 (pub/sub architecture).

Nodes talk to each other only through named topics and typed messages, never through
direct references to one another -- that decoupling is the actual property that
matters for the "ROS2-style" signal this architecture is going for, not the dispatch
mechanism. Dispatch here is synchronous and immediate (publish() calls subscriber
callbacks directly, in registration order) rather than a real async/threaded executor:
a real ROS2 graph runs concurrently with nondeterministic timing, but a simulation used
for tests and reproducible demos needs determinism more than it needs realistic async
timing. That tradeoff is deliberate, not a missing feature.
"""

from typing import Any, Callable

Callback = Callable[[Any], None]


class Bus:
    def __init__(self):
        self._subscribers: dict[str, list[Callback]] = {}

    def subscribe(self, topic: str, callback: Callback) -> None:
        self._subscribers.setdefault(topic, []).append(callback)

    def publish(self, topic: str, message: Any) -> None:
        for callback in self._subscribers.get(topic, []):
            callback(message)
