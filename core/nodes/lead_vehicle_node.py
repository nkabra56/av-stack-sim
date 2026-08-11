"""Replays a real recorded lead-vehicle trajectory (e.g. from NGSIM) tick by tick. This
is a scripted external agent, not something our stack controls -- it doesn't get a
controller, it publishes ground truth being replayed. See DESIGN.md's ACC section.
"""

import numpy as np

from core.messaging.bus import Bus
from core.messaging.messages import LeadVehicleStateMsg


class LeadVehicleNode:
    def __init__(self, bus: Bus, position: np.ndarray, speed: np.ndarray):
        self.bus = bus
        self.position = position
        self.speed = speed

    def __len__(self) -> int:
        return len(self.position)

    def step(self, tick: int) -> bool:
        """Publish this tick's recorded state. Returns False once the recording runs out."""
        if tick >= len(self.position):
            return False
        self.bus.publish("lead_state", LeadVehicleStateMsg(self.position[tick], self.speed[tick]))
        return True
