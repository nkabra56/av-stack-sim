"""Tick-based executor for the full closed-loop highway drive: H1 (ACC) controls speed,
H3 (Stanley) controls steering, H2's EKF fuses both, all on one real Vehicle. Uses real
NGSIM US-101 lane 2 data throughout: see core/data/ngsim/ATTRIBUTION.md and DESIGN.md
section 12."""

from dataclasses import dataclass

import numpy as np

from core.control.intersection import IntersectionNavigator, IntersectionState
from core.control.lane_centering import StanleyController
from core.control.lane_geometry import build_arc_length_table, geometric_following_gap, pose_at_arc_length
from core.estimation.ekf import ExtendedKalmanFilter
from core.intersection_harness import no_other_vehicle
from core.messaging.bus import Bus
from core.messaging.messages import (
    EgoHighwayStateMsg,
    EgoLongitudinalStateMsg,
    EgoSpeedEstimateMsg,
    LeadVehicleStateMsg,
)
from core.nodes.acc_controller_node import AccController, AccControllerNode
from core.nodes.highway_vehicle_node import HighwayVehicleNode
from core.nodes.intersection_controller_node import IntersectionControllerNode
from core.nodes.lane_centering_node import LaneCenteringControllerNode
from core.nodes.lead_vehicle_node import LeadVehicleNode
from core.nodes.longitudinal_arbiter_node import LongitudinalArbiterNode
from core.nodes.merging_vehicle_node import MergingVehicleNode
from core.nodes.other_vehicle_script_node import OtherVehicleScript, OtherVehicleScriptNode
from core.nodes.radar_node import IN_LANE_TOLERANCE, RadarNode
from core.nodes.speed_estimator_node import SpeedEstimatorNode
from core.vehicle import Vehicle


@dataclass(frozen=True)
class MergingVehicleSpec:
    """Initial state and gap-acceptance trigger for an optional reactive merging vehicle
    in the lane beside the ego. See core/nodes/merging_vehicle_node.py."""

    initial_gap_ahead_of_ego: float  # meters; its starting front-bumper position minus the ego's
    speed: float  # m/s, its own free-flow speed
    length: float = 4.5  # meters, for radar bumper-to-bumper range
    gap_threshold: float = 20.0  # meters; gap ahead of the ego it waits for before merging
    initial_lane_offset: float = 3.7  # meters, real US lane width
    merge_duration: float = 3.0  # seconds to complete the lane change once accepted


@dataclass
class FullHighwaySimulationResult:
    times: np.ndarray
    ego_x: np.ndarray
    ego_y: np.ndarray
    ego_theta: np.ndarray
    ego_arc_length: np.ndarray
    ego_speed: np.ndarray  # true
    ego_speed_estimate: np.ndarray  # fused (H2)
    ego_accel: np.ndarray
    ego_delta: np.ndarray
    cross_track_error: np.ndarray  # true, signed, same metric as lane_centering_validation.py
    lead_position: np.ndarray
    lead_speed: np.ndarray
    gap: np.ndarray  # true bumper-to-bumper, meters (arc-length based)
    min_gap: float
    collided: bool  # true gap ever reached zero, against the recorded lead OR the merging vehicle
    # Set only when merging_vehicle is given, None/empty otherwise:
    merging_position: np.ndarray | None = None
    merging_speed: np.ndarray | None = None
    merging_lane_offset: np.ndarray | None = None
    merging_gap: np.ndarray | None = None  # true bumper-to-bumper to the merging vehicle
    min_merging_gap: float | None = None
    collided_with_merging: bool | None = None
    # Set only in Phase B (intersection_navigator given), None otherwise:
    states: list[IntersectionState] | None = None
    ego_stop_time: float | None = None
    ran_stop_sign: bool | None = None  # crossed the stop line without stopping (see intersection_harness.py)
    proceed_time: float | None = None  # first tick state became PROCEEDING
    # True bumper-to-bumper gap, free of the ego arc-length quantization above (DESIGN.md section 9):
    geometric_gap: np.ndarray | None = None
    geometric_min_gap: float | None = None
    # Set only when merging_vehicle is given, None otherwise; same in-lane filter as min_merging_gap:
    geometric_merging_gap: np.ndarray | None = None
    min_geometric_merging_gap: float | None = None


class FullHighwayHarness:
    def __init__(
        self,
        centerline: np.ndarray,
        lead_position: np.ndarray,
        lead_speed: np.ndarray,
        lead_length: float,
        acc_controller: AccController,
        stanley_k: float = 0.5,
        wheelbase: float = 2.7,
        max_steer: float = 0.6,
        ego_initial_gap: float = 15.0,
        ego_initial_lateral_offset: float = 1.5,
        dt: float = 0.1,
        seed: int = 42,
        a_min: float = -9.0,
        a_max: float = 3.0,
        range_std: float = 0.5,
        range_rate_std: float = 0.3,
        accel_odom_std: float = 0.15,
        speedometer_std: float = 0.2,
        steering_odom_std: float = 0.01,
        compass_std: float = 0.02,
        position_std: float = 0.3,
        position_fix_period: int = 10,
        intersection_navigator: IntersectionNavigator | None = None,
        other_vehicle_script: OtherVehicleScript | None = None,
        merging_vehicle: MergingVehicleSpec | None = None,
    ):
        self.dt = dt
        self.has_intersection = intersection_navigator is not None
        self.centerline = centerline
        self.arc_length_table = build_arc_length_table(centerline)
        self.lead_length = lead_length
        self.bus = Bus()
        rng = np.random.default_rng(seed)

        self.lead_node = LeadVehicleNode(self.bus, lead_position, lead_speed)

        ego_start_s = lead_position[0] - lead_length - ego_initial_gap
        x0, y0, theta0 = pose_at_arc_length(ego_start_s, centerline, self.arc_length_table)
        # Offsetting straight in y approximates a lateral offset well on this centerline's
        # gentle curvature: same shortcut lane_centering_validation.py takes.
        vehicle = Vehicle(
            x=x0, y=y0 + ego_initial_lateral_offset, theta=theta0, wheelbase=wheelbase, max_steer=max_steer
        )

        self.ego_node = HighwayVehicleNode(
            self.bus, vehicle, lead_speed[0], dt, rng, centerline, self.arc_length_table,
            a_min=a_min, a_max=a_max, accel_odom_std=accel_odom_std,
            speedometer_std=speedometer_std, steering_odom_std=steering_odom_std,
            compass_std=compass_std, position_std=position_std, position_fix_period=position_fix_period,
        )
        self.merging_vehicle_node = None
        self.merging_vehicle_length = None
        if merging_vehicle is not None:
            merging_start_position = ego_start_s + merging_vehicle.initial_gap_ahead_of_ego
            self.merging_vehicle_node = MergingVehicleNode(
                self.bus, merging_start_position, merging_vehicle.speed, dt,
                gap_threshold=merging_vehicle.gap_threshold,
                initial_lane_offset=merging_vehicle.initial_lane_offset,
                merge_duration=merging_vehicle.merge_duration,
            )
            self.merging_vehicle_length = merging_vehicle.length

        self.radar_node = RadarNode(
            self.bus, rng, lead_length, range_std=range_std, range_rate_std=range_rate_std,
            merging_length=self.merging_vehicle_length,
        )

        ekf = ExtendedKalmanFilter(
            x0=np.array([vehicle.x, vehicle.y, vehicle.theta, lead_speed[0]]),
            p0=np.diag([1.0, 1.0, 0.1, 0.5]),
            wheelbase=wheelbase,
            odom_v_std=0.0,  # unused: predict() is parking-only, this mode uses predict_with_speed_state
            odom_delta_std=steering_odom_std,  # load-bearing here: see ekf.py's predict_with_speed_state
            r_heading=compass_std**2,
            r_position=np.eye(2) * position_std**2,
            r_landmark=np.eye(2),  # unused (update_landmark never called in this mode)
            r_speed=speedometer_std**2,
            accel_std=accel_odom_std,
        )
        self.speed_estimator_node = SpeedEstimatorNode(self.bus, ekf, dt)
        self.lane_centering_node = LaneCenteringControllerNode(
            self.bus, StanleyController(wheelbase=wheelbase, k=stanley_k, max_steer=max_steer), centerline
        )

        # Phase B: with an IntersectionNavigator, ACC's accel becomes a candidate composed
        # with the intersection's via LongitudinalArbiterNode's min() (see its docstring).
        self.intersection_node = None
        self.other_vehicle_node = None
        self.arbiter_node = None
        if self.has_intersection:
            self.acc_node = AccControllerNode(self.bus, acc_controller, output_topic="acc_cmd_candidate")
            self.intersection_node = IntersectionControllerNode(
                self.bus, intersection_navigator, centerline, self.arc_length_table, dt,
                output_topic="intersection_cmd_candidate",
            )
            self.other_vehicle_node = OtherVehicleScriptNode(
                self.bus, other_vehicle_script or no_other_vehicle, dt
            )
            self.arbiter_node = LongitudinalArbiterNode(
                self.bus, sources=["acc_cmd_candidate", "intersection_cmd_candidate"]
            )
        else:
            self.acc_node = AccControllerNode(self.bus, acc_controller)

        self._latest_ego_state: EgoLongitudinalStateMsg | None = None
        self._latest_ego_highway: EgoHighwayStateMsg | None = None
        self._latest_lead: LeadVehicleStateMsg | None = None
        self._latest_merging: LeadVehicleStateMsg | None = None
        self._latest_estimate: EgoSpeedEstimateMsg | None = None
        self.bus.subscribe("ego_state", self._on_ego_state)
        self.bus.subscribe("ego_highway_state", self._on_ego_highway_state)
        self.bus.subscribe("lead_state", self._on_lead_state)
        self.bus.subscribe("merging_vehicle_state", self._on_merging_state)
        self.bus.subscribe("ego_speed_estimate", self._on_estimate)

    def _on_ego_state(self, msg: EgoLongitudinalStateMsg) -> None:
        self._latest_ego_state = msg

    def _on_ego_highway_state(self, msg: EgoHighwayStateMsg) -> None:
        self._latest_ego_highway = msg

    def _on_lead_state(self, msg: LeadVehicleStateMsg) -> None:
        self._latest_lead = msg

    def _on_merging_state(self, msg: LeadVehicleStateMsg) -> None:
        self._latest_merging = msg

    def _on_estimate(self, msg: EgoSpeedEstimateMsg) -> None:
        self._latest_estimate = msg

    def _cross_track_error(self) -> float:
        ts = self._latest_ego_highway
        nearest = int(np.argmin(np.hypot(self.centerline[:, 0] - ts.x, self.centerline[:, 1] - ts.y)))
        return ts.y - self.centerline[nearest, 1]

    def run(self, max_steps: int | None = None) -> FullHighwaySimulationResult:
        n_available = len(self.lead_node)
        n = min(max_steps, n_available) if max_steps is not None else n_available

        times, ego_x, ego_y, ego_theta, ego_arc, ego_speed, ego_speed_est = [], [], [], [], [], [], []
        ego_accel, ego_delta, ctes, lead_pos, lead_speed_hist, gaps = [], [], [], [], [], []
        merging_pos, merging_spd, merging_offset, merging_gaps = [], [], [], []
        min_gap = float("inf")
        min_merging_gap = float("inf")
        collided_with_lead = False
        collided_with_merging = False
        states = [] if self.has_intersection else None
        ran_stop_sign = False if self.has_intersection else None
        proceed_time = None
        crossed_line = False

        for tick in range(n):
            if not self.lead_node.step(tick):
                break
            if self.other_vehicle_node is not None:
                self.other_vehicle_node.step()
            self.ego_node.step()
            if self.merging_vehicle_node is not None:
                self.merging_vehicle_node.step()
            self.radar_node.step()
            self.lane_centering_node.step()
            self.acc_node.step()
            if self.intersection_node is not None:
                self.intersection_node.step()
                self.arbiter_node.step()

            ts, th = self._latest_ego_state, self._latest_ego_highway
            gap = (self._latest_lead.position - self.lead_length) - ts.position
            min_gap = min(min_gap, gap)
            if gap <= 0:
                collided_with_lead = True

            times.append(tick * self.dt)
            ego_x.append(th.x)
            ego_y.append(th.y)
            ego_theta.append(th.theta)
            ego_arc.append(ts.position)
            ego_speed.append(ts.speed)
            ego_speed_est.append(self._latest_estimate.speed if self._latest_estimate else ts.speed)
            ego_accel.append(th.accel)
            ego_delta.append(th.delta)
            ctes.append(self._cross_track_error())
            lead_pos.append(self._latest_lead.position)
            lead_speed_hist.append(self._latest_lead.speed)
            gaps.append(gap)

            if self.merging_vehicle_node is not None:
                mv = self._latest_merging
                merging_pos.append(mv.position)
                merging_spd.append(mv.speed)
                merging_offset.append(mv.lane_offset)
                merging_gap = (mv.position - self.merging_vehicle_length) - ts.position
                merging_gaps.append(merging_gap)
                # Only in-lane overlap is a real collision (same convention RadarNode uses for
                # visibility): side-by-side with lane_offset != 0 is a safe pass, not a crash.
                if abs(mv.lane_offset) < IN_LANE_TOLERANCE:
                    min_merging_gap = min(min_merging_gap, merging_gap)
                    if merging_gap <= 0:
                        collided_with_merging = True

            if self.has_intersection:
                navigator = self.intersection_node.navigator
                states.append(navigator.state)
                if not crossed_line and ts.position >= navigator.stop_line_position:
                    crossed_line = True
                    if navigator.stop_time is None:
                        ran_stop_sign = True
                if proceed_time is None and navigator.state == IntersectionState.PROCEEDING:
                    proceed_time = tick * self.dt

            if collided_with_lead or collided_with_merging or ts.position >= self.arc_length_table[-1]:
                break

        ego_stop_time = self.intersection_node.navigator.stop_time if self.has_intersection else None

        ego_x_a, ego_y_a, ego_theta_a = np.array(ego_x), np.array(ego_y), np.array(ego_theta)
        geometric_gap = geometric_following_gap(
            ego_x_a, ego_y_a, ego_theta_a, np.array(lead_pos), self.lead_length,
            self.centerline, self.arc_length_table,
        )
        geometric_min_gap = float(geometric_gap.min()) if geometric_gap.size else float("inf")

        geometric_merging_gap = None
        min_geometric_merging_gap = None
        if self.merging_vehicle_node is not None:
            # Only meaningful while the merging vehicle's arc length stays within centerline's
            # own recorded range (see pose_at_arc_length_interpolated's docstring); a scenario
            # that drives it past the end gets a silently clamped, meaningless pose instead.
            geometric_merging_gap = geometric_following_gap(
                ego_x_a, ego_y_a, ego_theta_a, np.array(merging_pos), self.merging_vehicle_length,
                self.centerline, self.arc_length_table,
            )
            # Same in-lane filter min_merging_gap uses: side-by-side during the lane change isn't a near-miss.
            in_lane = np.abs(np.array(merging_offset)) < IN_LANE_TOLERANCE
            min_geometric_merging_gap = (
                float(geometric_merging_gap[in_lane].min()) if np.any(in_lane) else float("inf")
            )

        return FullHighwaySimulationResult(
            times=np.array(times),
            ego_x=np.array(ego_x),
            ego_y=np.array(ego_y),
            ego_theta=np.array(ego_theta),
            ego_arc_length=np.array(ego_arc),
            ego_speed=np.array(ego_speed),
            ego_speed_estimate=np.array(ego_speed_est),
            ego_accel=np.array(ego_accel),
            ego_delta=np.array(ego_delta),
            cross_track_error=np.array(ctes),
            lead_position=np.array(lead_pos),
            lead_speed=np.array(lead_speed_hist),
            gap=np.array(gaps),
            min_gap=min_gap,
            collided=collided_with_lead or collided_with_merging,
            merging_position=np.array(merging_pos) if self.merging_vehicle_node is not None else None,
            merging_speed=np.array(merging_spd) if self.merging_vehicle_node is not None else None,
            merging_lane_offset=np.array(merging_offset) if self.merging_vehicle_node is not None else None,
            merging_gap=np.array(merging_gaps) if self.merging_vehicle_node is not None else None,
            min_merging_gap=min_merging_gap if self.merging_vehicle_node is not None else None,
            collided_with_merging=collided_with_merging if self.merging_vehicle_node is not None else None,
            states=states,
            ego_stop_time=ego_stop_time,
            ran_stop_sign=ran_stop_sign,
            proceed_time=proceed_time,
            geometric_gap=geometric_gap,
            geometric_min_gap=geometric_min_gap,
            geometric_merging_gap=geometric_merging_gap,
            min_geometric_merging_gap=min_geometric_merging_gap,
        )
