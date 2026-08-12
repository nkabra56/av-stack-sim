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

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from core.environment import VEHICLE_RADIUS, Environment
from core.estimation.ekf import ExtendedKalmanFilter
from core.interfaces import Controller, Planner
from core.messaging.bus import Bus
from core.messaging.messages import PathMsg, PoseEstimateMsg, TrueStateMsg
from core.nodes.controller_node import ControllerNode
from core.nodes.estimator_node import EstimatorNode
from core.nodes.planner_node import PlannerNode
from core.nodes.sensor_node import SensorNode
from core.nodes.vehicle_node import VehicleNode
from core.sensors import UltrasonicArray
from core.vehicle import Vehicle

# Front cone (+/-0.6 rad either side of heading) plus a mirrored rear cone (same fan,
# rotated 180 degrees by construction rather than separately hand-typed, so the two
# cones can't drift out of symmetry) -- added so ControllerNode's speed governor
# (controller_node.py's module docstring) can actually see obstacles behind the
# vehicle during reverse-gear maneuvers, not just in front of it.
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
        stopping_buffer: float = 0.5,  # see ControllerNode._safe_speed's docstring
        max_replans: int = 3,  # see PlannerNode's docstring
        tracked_buffer_extra: float = 0.4,  # see ControllerNode's "Tracking-aware buffer"
        # docstring entry (KNOWN_BUGS.md entry 3): real stopping margin *above* whatever
        # clearance the active planner guarantees while accurately tracked, not the
        # planner's raw safety_margin itself (that would leave zero margin for the same
        # sense-decide-act latency stopping_buffer already has to absorb). Paired with
        # ControllerNode's default `tracking_threshold=0.03` in a real parameter sweep
        # (tests/test_replanning.py). Raised from 0.3 (the original smallest/safest value
        # found) to 0.4 after `_effective_buffer`'s cross-track measurement was fixed to
        # use true perpendicular-to-segment distance instead of nearest-waypoint distance
        # (a code-review finding, not an entry-3 finding): the more accurate, generally
        # *smaller* measurement classifies more ticks as "tracking," using the smaller
        # buffer more often -- and 0.3 turned out to have essentially no margin left, a
        # 25-seed re-sweep found 1/10 real collisions at 0.3 that a narrower 5-seed check
        # didn't surface (and 0.2 fails outright, 4/5 collisions -- confirming this
        # constant sits right at a real safety cliff, not a comfortable margin above one).
        # 0.4 held 0/25 collisions with 23/25 full completions across seeds 1-25 (the 2
        # non-completions are the already-documented max_replans residual, not unsafe).
        tracking_threshold: float = 0.03,  # see ControllerNode's constructor docstring
    ):
        self.environment = environment
        self.tol = tol
        self.bus = Bus()
        rng = np.random.default_rng(seed)

        self.vehicle_node = VehicleNode(self.bus, vehicle, dt, rng, v_max=v_max, a_max=a_max, k_acc=k_acc)
        ultrasonic = UltrasonicArray(angles=DEFAULT_SENSOR_ANGLES, max_range=8.0)
        self.sensor_node = SensorNode(self.bus, ultrasonic, environment, rng)

        # Only planners that actually guarantee an obstacle clearance while being tracked
        # (HybridAStarPlanner's `safety_margin`) earn a smaller tracked-buffer; planners with
        # no such attribute (DubinsPlanner/ReedsSheppPlanner, both obstacle-blind) get None,
        # which disables the feature entirely and keeps ControllerNode's fully conservative
        # `stopping_buffer` for any proximity, tracked or not -- see ControllerNode's
        # "Tracking-aware buffer" docstring entry.
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
        self.controller_node = ControllerNode(
            self.bus, controller, a_max=a_max, stopping_buffer=stopping_buffer,
            tracked_stopping_buffer=tracked_stopping_buffer, tracking_threshold=tracking_threshold,
        )

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

    def run(self, max_steps: int = 500, on_tick: Callable[[int], None] | None = None) -> SimulationResult:
        """`on_tick(tick)`, if given, runs before each tick's nodes step -- its only use
        so far is tests exercising re-planning (KNOWN_BUGS.md entry 3): mutating
        `self.environment.obstacles` mid-run to simulate an obstacle that wasn't there
        when `PlannerNode` made its first (and, for every scenario this project ships,
        only) plan. Every node reads `self.environment` live, not a snapshot, so this is
        the only hook needed -- no separate "inject an obstacle" API on the harness
        itself."""
        true_history: list[tuple[float, float, float]] = []
        est_history: list[tuple[float, float, float]] = []
        cov_history: list[np.ndarray] = []
        controls: list[tuple[float, float]] = []
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
