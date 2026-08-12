"""Composes two or more independent longitudinal-accel sources (H5 Phase B: ACC's
real-lead-vehicle accel and IntersectionNavigator's stop-line/right-of-way accel)
into a single command via min() -- the more conservative (harder-braking) demand
wins each tick. See DESIGN.md section 12's H5 entry.

Sound because both IDMController.control() and IntersectionNavigator.control() are
memoryless functions of the CURRENT tick's actual (position, speed) -- neither
persists state that could desync from whichever candidate actually got applied last
tick -- so recomputing both fresh every tick off the one real resulting state and
taking min() is architecturally safe: the composed accel at any instant is always at
least as conservative as either candidate alone. It does NOT guarantee the resulting
*trajectory* matches either controller's own standalone-validated trajectory (which
candidate wins can and does alternate tick to tick) -- see DESIGN.md section 12's H5
Phase B entry for the specific edge case this implies and how it's tested.
"""

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
