"""Tick-based executor: owns the Bus, builds all 5 nodes, and drives them in a fixed
order each tick. See DESIGN.md's architecture section for the node/topic diagram."""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from core.environment import VEHICLE_RADIUS, Environment
from core.estimation.ekf import ExtendedKalmanFilter
from core.interfaces import Controller, Planner
from core.messaging.bus import Bus
from core.messaging.messages import ObstacleRangeMsg, PathMsg, PoseEstimateMsg, TrueStateMsg
from core.nodes.controller_node import ControllerNode
from core.nodes.estimator_node import EstimatorNode
from core.nodes.planner_node import PlannerNode
from core.nodes.sensor_node import SensorNode
from core.nodes.vehicle_node import VehicleNode
from core.sensors import UltrasonicArray
from core.vehicle import Vehicle

# Front cone plus a mirrored rear cone, so ControllerNode's speed governor can see
# obstacles behind the vehicle during reverse maneuvers too.
_FRONT_SENSOR_ANGLES = [-0.6, -0.3, 0.0, 0.3, 0.6]
DEFAULT_SENSOR_ANGLES = _FRONT_SENSOR_ANGLES + [angle + np.pi for angle in _FRONT_SENSOR_ANGLES]


@dataclass
class SimulationResult:
    true_history: np.ndarray  # (N, 3) ground truth x, y, theta
    estimated_history: np.ndarray  # (N, 3) EKF pose estimate
    covariance_history: np.ndarray  # (N, 3, 3) EKF covariance at each step
    controls: np.ndarray  # (N, 2) v, delta actually applied
    success: bool
    collision: bool
    path: np.ndarray = field(repr=False, default=None)  # planned (M, 3) path, for plotting
    sensor_ranges: np.ndarray = field(repr=False, default=None)  # (N, num_beams); max_range = no hit
    sensor_angles: np.ndarray = field(repr=False, default=None)  # (num_beams,) rad, relative to heading
    dt: float = 0.1  # seconds per tick


class ParkingHarness:
    def __init__(
        self,
        vehicle: Vehicle,
        environment: Environment,
        planner: Planner,
        controller: Controller,
        seed: int = 42,
        dt: float = 0.1,
        v_max: float = 1.5,
        a_max: float = 0.8,
        k_acc: float = 2.0,
        tol: float = 0.4,  # allows for pose-estimate error, not just controller error
        stopping_buffer: float = 0.5,  # see ControllerNode._safe_speed's docstring
        max_replans: int = 3,  # see PlannerNode's docstring
        tracked_buffer_extra: float = 0.4,  # margin above the tracked planner's safety_margin;
        # see KNOWN_BUGS.md entry 3 for the seed-sweep history behind this value.
        tracking_threshold: float = 0.03,  # see ControllerNode's constructor docstring
        sensor_dropout_prob: float = 0.0,  # see SensorNode's module docstring; 0.0 keeps old behavior
        sensor_latency_ticks: int = 0,
    ):
        self.environment = environment
        self.tol = tol
        self.dt = dt
        self.bus = Bus()
        rng = np.random.default_rng(seed)

        self.vehicle_node = VehicleNode(self.bus, vehicle, dt, rng, v_max=v_max, a_max=a_max, k_acc=k_acc)
        ultrasonic = UltrasonicArray(angles=DEFAULT_SENSOR_ANGLES, max_range=8.0)
        self.sensor_node = SensorNode(
            self.bus, ultrasonic, environment, rng,
            dropout_prob=sensor_dropout_prob, latency_ticks=sensor_latency_ticks,
        )
        self.sensor_angles = np.array(ultrasonic.angles)
        self._sensor_max_range = ultrasonic.max_range

        # Only planners with a `safety_margin` (HybridAStarPlanner) earn a smaller tracked
        # buffer; obstacle-blind planners (Dubins/ReedsShepp) get None, disabling the feature.
        planner_margin = getattr(planner, "safety_margin", None)
        tracked_stopping_buffer = planner_margin + tracked_buffer_extra if planner_margin is not None else None

        ekf = ExtendedKalmanFilter(
            x0=np.array([vehicle.x, vehicle.y, vehicle.theta]),
            p0=np.diag([0.25, 0.25, 0.05]),
            wheelbase=vehicle.wheelbase,
            odom_v_std=self.vehicle_node.odom_v_std,
            odom_delta_std=self.vehicle_node.odom_delta_std,
            r_heading=self.sensor_node.compass_std**2,
            r_position=np.eye(2) * self.sensor_node.position_std**2,
            r_landmark=np.diag(
                [self.sensor_node.landmark_range_std**2, self.sensor_node.landmark_bearing_std**2]
            ),
        )
        self.estimator_node = EstimatorNode(self.bus, ekf, environment)
        self.planner_node = PlannerNode(self.bus, planner, environment, vehicle.turning_radius, max_replans=max_replans)
        # Worst-case gap erosion during an in-flight sensor reading; see ControllerNode's
        # latency_margin docstring. Zero unless sensor_latency_ticks is enabled.
        latency_margin = sensor_latency_ticks * dt * v_max
        self.controller_node = ControllerNode(
            self.bus, controller, a_max=a_max, stopping_buffer=stopping_buffer,
            tracked_stopping_buffer=tracked_stopping_buffer, tracking_threshold=tracking_threshold,
            latency_margin=latency_margin,
        )

        # A controller with its own rollout model must predict using the harness's actual
        # dt, not whatever default it was constructed with.
        if hasattr(controller, "dt"):
            controller.dt = dt

        self._latest_true: TrueStateMsg | None = None
        self._latest_est: PoseEstimateMsg | None = None
        self._latest_path: np.ndarray | None = None
        self._latest_ranges: ObstacleRangeMsg | None = None
        self.bus.subscribe("true_state", self._on_true_state)
        self.bus.subscribe("pose_estimate", self._on_pose_estimate)
        self.bus.subscribe("path", self._on_path)
        self.bus.subscribe("obstacle_ranges", self._on_obstacle_ranges)

    def _on_true_state(self, msg: TrueStateMsg) -> None:
        self._latest_true = msg

    def _on_pose_estimate(self, msg: PoseEstimateMsg) -> None:
        self._latest_est = msg

    def _on_path(self, msg: PathMsg) -> None:
        self._latest_path = msg.path

    def _on_obstacle_ranges(self, msg: ObstacleRangeMsg) -> None:
        self._latest_ranges = msg

    def _collided(self, ts: TrueStateMsg) -> bool:
        for obstacle in self.environment.obstacles:
            if np.hypot(ts.x - obstacle.x, ts.y - obstacle.y) < obstacle.radius + VEHICLE_RADIUS:
                return True
        return False

    def run(self, max_steps: int = 500, on_tick: Callable[[int], None] | None = None) -> SimulationResult:
        """`on_tick(tick)`, if given, runs before each tick: used by re-planning tests to
        mutate `self.environment.obstacles` mid-run (KNOWN_BUGS.md entry 3)."""
        true_history: list[tuple[float, float, float]] = []
        est_history: list[tuple[float, float, float]] = []
        cov_history: list[np.ndarray] = []
        controls: list[tuple[float, float]] = []
        ranges_history: list[list[float]] = []
        collision = False

        for tick in range(max_steps):
            if on_tick is not None:
                on_tick(tick)
            self.vehicle_node.step()
            self.sensor_node.step()
            self.controller_node.step()

            ts = self._latest_true
            true_history.append((ts.x, ts.y, ts.theta))
            controls.append((ts.v, ts.delta))
            if self._latest_est is not None:
                est_history.append((self._latest_est.x, self._latest_est.y, self._latest_est.theta))
                cov_history.append(self._latest_est.covariance.copy())
            else:
                est_history.append((ts.x, ts.y, ts.theta))
                cov_history.append(np.zeros((3, 3)))
            readings = self._latest_ranges.readings if self._latest_ranges is not None else {}
            ranges_history.append([readings.get(a, self._sensor_max_range) for a in self.sensor_angles])

            if self._collided(ts):
                collision = True
                break
            if np.hypot(self.environment.spot.x - ts.x, self.environment.spot.y - ts.y) < self.tol:
                break

        true_arr = np.array(true_history) if true_history else np.zeros((0, 3))
        est_arr = np.array(est_history) if est_history else np.zeros((0, 3))
        cov_arr = np.array(cov_history) if cov_history else np.zeros((0, 3, 3))
        controls_arr = np.array(controls) if controls else np.zeros((0, 2))
        ranges_arr = np.array(ranges_history) if ranges_history else np.zeros((0, len(self.sensor_angles)))

        success = (
            not collision
            and len(true_arr) > 0
            and np.hypot(self.environment.spot.x - true_arr[-1, 0], self.environment.spot.y - true_arr[-1, 1])
            < self.tol
        )
        return SimulationResult(
            true_arr, est_arr, cov_arr, controls_arr, success, collision, self._latest_path,
            sensor_ranges=ranges_arr, sensor_angles=self.sensor_angles, dt=self.dt,
        )
