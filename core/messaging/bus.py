"""Minimal synchronous pub/sub bus -- deterministic dispatch, not a real async executor,
since tests and demos need reproducibility more than realistic ROS2 timing. See DESIGN.md section 2."""

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
