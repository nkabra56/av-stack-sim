"""Tick-based executor for H1 (ACC): owns the Bus, builds the four longitudinal-mode
nodes, drives them in a fixed order each tick, records ground truth for evaluation.
Mirrors harness.py's tick-based structure but for the straight-line ACC mode -- kept
as a separate module rather than forcing a shared base class out of a single existing
pattern; worth consolidating once H3 (lane centering) shows what's actually common
between the two harnesses. See DESIGN.md's ACC section.
"""

from dataclasses import dataclass

import numpy as np

from auto_park.messaging.bus import Bus
from auto_park.messaging.messages import EgoLongitudinalStateMsg, LeadVehicleStateMsg
from auto_park.nodes.acc_controller_node import AccController, AccControllerNode
from auto_park.nodes.ego_longitudinal_node import EgoLongitudinalNode
from auto_park.nodes.lead_vehicle_node import LeadVehicleNode
from auto_park.nodes.radar_node import RadarNode


@dataclass
class AccSimulationResult:
    times: np.ndarray  # (N,)
    ego_position: np.ndarray  # (N,)
    ego_speed: np.ndarray  # (N,)
    ego_accel: np.ndarray  # (N,)
    lead_position: np.ndarray  # (N,)
    lead_speed: np.ndarray  # (N,)
    gap: np.ndarray  # (N,) true bumper-to-bumper gap, meters
    min_gap: float
    collided: bool  # true gap ever reached zero


class AccHarness:
    def __init__(
        self,
        lead_position: np.ndarray,
        lead_speed: np.ndarray,
        lead_length: float,
        controller: AccController,
        ego_initial_speed: float,
        ego_initial_gap: float,
        dt: float = 0.1,
        seed: int = 42,
        a_min: float = -9.0,
        a_max: float = 3.0,
        range_std: float = 0.5,
        range_rate_std: float = 0.3,
    ):
        self.dt = dt
        self.lead_length = lead_length
        self.bus = Bus()
        rng = np.random.default_rng(seed)

        self.lead_node = LeadVehicleNode(self.bus, lead_position, lead_speed)
        ego_start_position = (lead_position[0] - lead_length) - ego_initial_gap
        self.ego_node = EgoLongitudinalNode(
            self.bus, ego_start_position, ego_initial_speed, dt, a_min=a_min, a_max=a_max
        )
        self.radar_node = RadarNode(
            self.bus, rng, lead_length, range_std=range_std, range_rate_std=range_rate_std
        )
        self.acc_node = AccControllerNode(self.bus, controller)

        self._latest_ego: EgoLongitudinalStateMsg | None = None
        self._latest_lead: LeadVehicleStateMsg | None = None
        self.bus.subscribe("ego_state", self._on_ego_state)
        self.bus.subscribe("lead_state", self._on_lead_state)

    def _on_ego_state(self, msg: EgoLongitudinalStateMsg) -> None:
        self._latest_ego = msg

    def _on_lead_state(self, msg: LeadVehicleStateMsg) -> None:
        self._latest_lead = msg

    def run(self, max_steps: int | None = None) -> AccSimulationResult:
        n_available = len(self.lead_node)
        n = min(max_steps, n_available) if max_steps is not None else n_available

        times, ego_pos, ego_speed, ego_accel = [], [], [], []
        lead_pos, lead_speed_hist, gaps = [], [], []
        min_gap = float("inf")

        for tick in range(n):
            if not self.lead_node.step(tick):
                break
            self.ego_node.step()
            self.radar_node.step()
            self.acc_node.step()

            gap = (self._latest_lead.position - self.lead_length) - self._latest_ego.position
            min_gap = min(min_gap, gap)

            times.append(tick * self.dt)
            ego_pos.append(self._latest_ego.position)
            ego_speed.append(self._latest_ego.speed)
            ego_accel.append(self._latest_ego.accel)
            lead_pos.append(self._latest_lead.position)
            lead_speed_hist.append(self._latest_lead.speed)
            gaps.append(gap)

        return AccSimulationResult(
            times=np.array(times),
            ego_position=np.array(ego_pos),
            ego_speed=np.array(ego_speed),
            ego_accel=np.array(ego_accel),
            lead_position=np.array(lead_pos),
            lead_speed=np.array(lead_speed_hist),
            gap=np.array(gaps),
            min_gap=min_gap,
            collided=min_gap <= 0,
        )
