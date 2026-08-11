"""Tick-based executor for H1/H2 (ACC + fused ego speed): owns the Bus, builds the
longitudinal-mode nodes, drives them in a fixed order each tick, records ground truth
for evaluation. Mirrors harness.py's tick-based structure but for the straight-line
ACC mode -- kept as a separate module rather than forcing a shared base class out of a
single existing pattern; worth consolidating once H3 (lane centering) shows what's
actually common between the two harnesses. See DESIGN.md's ACC section.
"""

from dataclasses import dataclass

import numpy as np

from auto_park.estimation.ekf import ExtendedKalmanFilter
from auto_park.messaging.bus import Bus
from auto_park.messaging.messages import EgoLongitudinalStateMsg, EgoSpeedEstimateMsg, LeadVehicleStateMsg
from auto_park.nodes.acc_controller_node import AccController, AccControllerNode
from auto_park.nodes.ego_longitudinal_node import EgoLongitudinalNode
from auto_park.nodes.lead_vehicle_node import LeadVehicleNode
from auto_park.nodes.radar_node import RadarNode
from auto_park.nodes.speed_estimator_node import SpeedEstimatorNode


@dataclass
class AccSimulationResult:
    times: np.ndarray  # (N,)
    ego_position: np.ndarray  # (N,)
    ego_speed: np.ndarray  # (N,) true
    ego_speed_estimate: np.ndarray  # (N,) fused (H2)
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
        accel_odom_std: float = 0.15,
        speedometer_std: float = 0.2,
    ):
        self.dt = dt
        self.lead_length = lead_length
        self.bus = Bus()
        rng = np.random.default_rng(seed)

        self.lead_node = LeadVehicleNode(self.bus, lead_position, lead_speed)
        ego_start_position = (lead_position[0] - lead_length) - ego_initial_gap
        self.ego_node = EgoLongitudinalNode(
            self.bus, ego_start_position, ego_initial_speed, dt, rng,
            a_min=a_min, a_max=a_max, accel_odom_std=accel_odom_std, speedometer_std=speedometer_std,
        )
        self.radar_node = RadarNode(
            self.bus, rng, lead_length, range_std=range_std, range_rate_std=range_rate_std
        )

        ekf = ExtendedKalmanFilter(
            x0=np.array([ego_start_position, 0.0, 0.0, ego_initial_speed]),
            p0=np.diag([1.0, 1.0, 0.1, 0.5]),
            wheelbase=2.7,  # unused for H1 (delta always 0); kept for API consistency with H3
            odom_v_std=0.0,
            odom_delta_std=0.0,
            r_heading=1.0,  # unused (update_heading never called in this mode)
            r_position=np.eye(2),  # unused (update_position never called in this mode)
            r_landmark=np.eye(2),  # unused (update_landmark never called in this mode)
            r_speed=speedometer_std**2,
            accel_std=accel_odom_std,
        )
        self.speed_estimator_node = SpeedEstimatorNode(self.bus, ekf, dt)
        self.acc_node = AccControllerNode(self.bus, controller)

        self._latest_ego: EgoLongitudinalStateMsg | None = None
        self._latest_lead: LeadVehicleStateMsg | None = None
        self._latest_speed_estimate: EgoSpeedEstimateMsg | None = None
        self.bus.subscribe("ego_state", self._on_ego_state)
        self.bus.subscribe("lead_state", self._on_lead_state)
        self.bus.subscribe("ego_speed_estimate", self._on_speed_estimate)

    def _on_ego_state(self, msg: EgoLongitudinalStateMsg) -> None:
        self._latest_ego = msg

    def _on_lead_state(self, msg: LeadVehicleStateMsg) -> None:
        self._latest_lead = msg

    def _on_speed_estimate(self, msg: EgoSpeedEstimateMsg) -> None:
        self._latest_speed_estimate = msg

    def run(self, max_steps: int | None = None) -> AccSimulationResult:
        n_available = len(self.lead_node)
        n = min(max_steps, n_available) if max_steps is not None else n_available

        times, ego_pos, ego_speed, ego_speed_est, ego_accel = [], [], [], [], []
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
            ego_speed_est.append(self._latest_speed_estimate.speed)
            ego_accel.append(self._latest_ego.accel)
            lead_pos.append(self._latest_lead.position)
            lead_speed_hist.append(self._latest_lead.speed)
            gaps.append(gap)

        return AccSimulationResult(
            times=np.array(times),
            ego_position=np.array(ego_pos),
            ego_speed=np.array(ego_speed),
            ego_speed_estimate=np.array(ego_speed_est),
            ego_accel=np.array(ego_accel),
            lead_position=np.array(lead_pos),
            lead_speed=np.array(lead_speed_hist),
            gap=np.array(gaps),
            min_gap=min_gap,
            collided=min_gap <= 0,
        )
