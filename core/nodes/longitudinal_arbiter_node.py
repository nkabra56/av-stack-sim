"""Composes two or more longitudinal-accel sources (H5 Phase B: ACC's and
IntersectionNavigator's) via min() -- the more conservative demand wins each tick. Sound
since both controllers are memoryless functions of the current (position, speed). See DESIGN.md section 12's H5 entry."""

from core.messaging.bus import Bus
from core.messaging.messages import LongitudinalCmdMsg


class LongitudinalArbiterNode:
    def __init__(self, bus: Bus, sources: list[str]):
        self.bus = bus
        self.sources = sources
        self._candidates = dict.fromkeys(sources, 0.0)
        for source in sources:
            bus.subscribe(source, self._make_handler(source))

    def _make_handler(self, source: str):
        def handler(msg: LongitudinalCmdMsg) -> None:
            self._candidates[source] = msg.accel

        return handler

    def step(self) -> None:
        self.bus.publish("longitudinal_cmd", LongitudinalCmdMsg(min(self._candidates.values())))
