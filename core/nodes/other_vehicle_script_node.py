"""Publishes a scripted other-vehicle's status over the Bus each tick (H5 Phase B),
reusing H4's existing OtherVehicleScript/OtherVehicleStatus/other_vehicle_present_from
(core/intersection_harness.py) completely unchanged -- H4's own standalone harness
calls the script directly since it has no Bus at all; this is the thin adapter that
lets the same hand-authored scripts drive IntersectionControllerNode instead.
"""

from core.intersection_harness import OtherVehicleScript
from core.messaging.bus import Bus


class OtherVehicleScriptNode:
    def __init__(self, bus: Bus, script: OtherVehicleScript, dt: float):
        self.bus = bus
        self.script = script
        self.dt = dt
        self._t = 0.0

    def step(self) -> None:
        self.bus.publish("other_vehicle_status", [self.script(self._t)])
        self._t += self.dt
