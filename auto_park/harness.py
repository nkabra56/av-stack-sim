"""Tick-based executor: owns the Bus, builds all 5 nodes, and drives them in a fixed
order each tick. Replaces ParkingSimulation's direct-call loop. See DESIGN.md's
architecture section for the node/topic diagram and why tick order is what it is.

Per tick: VehicleNode applies the previous tick's control and publishes true_state +
odometry (which synchronously triggers an EKF predict, which -- on the very first tick
-- also triggers PlannerNode to plan once). Then SensorNode publishes obstacle ranges
and whichever of compass/position_fix/landmark_bearings are due this tick, each
synchronously triggering further EKF corrections. Only then does ControllerNode
compute a single control_cmd for the *next* tick, off the freshest pose_estimate
available. Success/collision are evaluated against true_state only -- the harness is
the one place, besides SensorNode, allowed to see ground truth.
"""

from dataclasses import dataclass, field

import numpy as np

from auto_park.environment import Environment
from auto_park.estimation.ekf import ExtendedKalmanFilter
from auto_park.interfaces import Controller, Planner
from auto_park.messaging.bus import Bus
from auto_park.messaging.messages import PathMsg, PoseEstimateMsg, TrueStateMsg
from auto_park.nodes.controller_node import ControllerNode
from auto_park.nodes.estimator_node import EstimatorNode
from auto_park.nodes.planner_node import PlannerNode
from auto_park.nodes.sensor_node import SensorNode
from auto_park.nodes.vehicle_node import VehicleNode
from auto_park.sensors import UltrasonicArray
from auto_park.vehicle import Vehicle

VEHICLE_RADIUS = 1.0  # ego vehicle's own collision-circle radius; must be on the same scale as
# the ~1.3m obstacle radii used for parked cars in scenarios/*.yaml (see DESIGN.md section 8),
# not an arbitrary small buffer -- a real car is comparable in size to the cars it's driving
# among, so its own collision footprint has to be too, or _collided() under-reports real hits.
DEFAULT_SENSOR_ANGLES = [-0.6, -0.3, 0.0, 0.3, 0.6]


@dataclass
class SimulationResult:
    true_history: np.ndarray  # (N, 3) ground truth x, y, theta
    estimated_history: np.ndarray  # (N, 3) EKF pose estimate
    covariance_history: np.ndarray  # (N, 3, 3) EKF covariance at each step
    controls: np.ndarray  # (N, 2) v, delta actually applied
    success: bool
    collision: bool
    path: np.ndarray = field(repr=False, default=None)  # planned (M, 3) path, for plotting


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
        tol: float = 0.4,  # looser than the pre-estimation baseline (0.3): the vehicle now
        # only ever knows its NOISY pose estimate, not ground truth, so "close enough to
        # call it parked" has to allow for realistic estimation error, not just controller error
        # >= v_max^2/(2*a_max) stopping distance (~1.41m) + VEHICLE_RADIUS (the vehicle brakes
        # based on a sensor reading to the obstacle SURFACE, but collision is checked between
        # vehicle CENTER and obstacle center, so the vehicle's own radius has to be part of the
        # margin too, not just its stopping distance) + a safety margin.
        brake_distance: float = 3.0,
    ):
        self.environment = environment
        self.tol = tol
        self.bus = Bus()
        rng = np.random.default_rng(seed)

        self.vehicle_node = VehicleNode(self.bus, vehicle, dt, rng, v_max=v_max, a_max=a_max, k_acc=k_acc)
        ultrasonic = UltrasonicArray(angles=DEFAULT_SENSOR_ANGLES, max_range=8.0)
        self.sensor_node = SensorNode(self.bus, ultrasonic, environment, rng)

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
        self.planner_node = PlannerNode(self.bus, planner, environment, vehicle.turning_radius)
        self.controller_node = ControllerNode(self.bus, controller, brake_distance=brake_distance)

        # A controller with its own internal rollout model (e.g. MPCController) has to predict
        # forward using the *actual* tick length, or its predictions silently desync from the
        # real simulation step -- the harness's dt is the single source of truth, not whatever
        # default the controller happened to be constructed with.
        if hasattr(controller, "dt"):
            controller.dt = dt

        self._latest_true: TrueStateMsg | None = None
        self._latest_est: PoseEstimateMsg | None = None
        self._latest_path: np.ndarray | None = None
        self.bus.subscribe("true_state", self._on_true_state)
        self.bus.subscribe("pose_estimate", self._on_pose_estimate)
        self.bus.subscribe("path", self._on_path)

    def _on_true_state(self, msg: TrueStateMsg) -> None:
        self._latest_true = msg

    def _on_pose_estimate(self, msg: PoseEstimateMsg) -> None:
        self._latest_est = msg

    def _on_path(self, msg: PathMsg) -> None:
        self._latest_path = msg.path

    def _collided(self, ts: TrueStateMsg) -> bool:
        for obstacle in self.environment.obstacles:
            if np.hypot(ts.x - obstacle.x, ts.y - obstacle.y) < obstacle.radius + VEHICLE_RADIUS:
                return True
        return False

    def run(self, max_steps: int = 500) -> SimulationResult:
        true_history: list[tuple[float, float, float]] = []
        est_history: list[tuple[float, float, float]] = []
        cov_history: list[np.ndarray] = []
        controls: list[tuple[float, float]] = []
        collision = False

        for _ in range(max_steps):
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

            if self._collided(ts):
                collision = True
                break
            if np.hypot(self.environment.spot.x - ts.x, self.environment.spot.y - ts.y) < self.tol:
                break

        true_arr = np.array(true_history) if true_history else np.zeros((0, 3))
        est_arr = np.array(est_history) if est_history else np.zeros((0, 3))
        cov_arr = np.array(cov_history) if cov_history else np.zeros((0, 3, 3))
        controls_arr = np.array(controls) if controls else np.zeros((0, 2))

        success = (
            not collision
            and len(true_arr) > 0
            and np.hypot(self.environment.spot.x - true_arr[-1, 0], self.environment.spot.y - true_arr[-1, 1])
            < self.tol
        )
        return SimulationResult(true_arr, est_arr, cov_arr, controls_arr, success, collision, self._latest_path)
