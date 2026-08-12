"""Ground-truth plant node for the full closed-loop highway drive: owns a real 2D
kinematic bicycle Vehicle (unmodified, same class parking uses), applies the last
commanded (accel, delta) each tick, and publishes both the highway-mode topics H1/H2
already expect and a new full-pose topic for evaluation. See DESIGN.md section 12's
closed-loop drive entry.

Deliberately NOT a modification of EgoLongitudinalNode: H1/H2 standalone keep their
exact scalar point-mass model, tests, and validated numbers untouched (same "two
methods/nodes for two genuinely different situations" precedent as ekf.py's
predict()/predict_with_speed_state() split). This node reuses EgoLongitudinalNode's
exact accel-integration physics (same a_min/a_max clip, same "can't reverse" floor) so
H1/H2's already-NGSIM-validated closed-loop dynamics carry over unchanged -- adding
steering doesn't change how speed itself is computed, it only adds what Vehicle.update
does with that speed once computed.

Publishes on the SAME "ego_state"/EgoLongitudinalStateMsg topic H1's RadarNode already
consumes, so RadarNode needs zero code changes -- what changes is what `position`
means (a projected arc-length along the lane centerline, core/control/lane_geometry.py,
instead of a raw scalar), which RadarNode never needed to know about even before.

Also publishes an always-on noisy compass and a low-rate noisy position fix (reusing
CompassMsg/PositionFixMsg unchanged, same types/semantics/defaults SensorNode already
uses for parking) -- a real gap found while building this: with no absolute
correction at all, the 4-state EKF's x/y/theta is pure dead reckoning, and realistic
steering-odometry noise accumulates into meters of real heading/position drift over a
~600m run once Stanley is actually closing the loop on it (see ekf.py's
predict_with_speed_state docstring for the two related fixes this also needed).
Bundling these into the plant node rather than a separate highway SensorNode matches
this project's own existing highway-mode precedent (EgoLongitudinalNode already
publishes its own noisy accel_odometry/speedometer directly, no separate sensor node
exists for H1/H2 either) -- parking's stricter plant/sensor split isn't violated by a
mode that never had it to begin with.
"""

import numpy as np

from core.control.lane_geometry import project_to_arc_length
from core.messaging.bus import Bus
from core.messaging.messages import (
    AccelOdometryMsg,
    CompassMsg,
    EgoHighwayStateMsg,
    EgoLongitudinalStateMsg,
    LateralCmdMsg,
    LongitudinalCmdMsg,
    PositionFixMsg,
    SpeedometerMsg,
    SteeringOdometryMsg,
)
from core.vehicle import Vehicle, wrap_angle


class HighwayVehicleNode:
    def __init__(
        self,
        bus: Bus,
        vehicle: Vehicle,
        speed: float,
        dt: float,
        rng: np.random.Generator,
        centerline: np.ndarray,
        arc_length_table: np.ndarray,
        a_min: float = -9.0,
        a_max: float = 3.0,
        accel_odom_std: float = 0.15,
        speedometer_std: float = 0.2,
        steering_odom_std: float = 0.01,
        compass_std: float = 0.02,
        position_std: float = 0.3,
        position_fix_period: int = 10,
    ):
        self.bus = bus
        self.vehicle = vehicle
        self.speed = speed
        self.dt = dt
        self.rng = rng
        self.centerline = centerline
        self.arc_length_table = arc_length_table
        self.a_min = a_min
        self.a_max = a_max
        self.accel_odom_std = accel_odom_std
        self.speedometer_std = speedometer_std
        self.steering_odom_std = steering_odom_std
        self.compass_std = compass_std
        self.position_std = position_std
        self.position_fix_period = position_fix_period
        self._tick = 0

        self._last_accel_cmd = LongitudinalCmdMsg(0.0)
        self._last_lateral_cmd = LateralCmdMsg(0.0)
        bus.subscribe("longitudinal_cmd", self._on_accel_cmd)
        bus.subscribe("lateral_cmd", self._on_lateral_cmd)

    def _on_accel_cmd(self, msg: LongitudinalCmdMsg) -> None:
        self._last_accel_cmd = msg

    def _on_lateral_cmd(self, msg: LateralCmdMsg) -> None:
        self._last_lateral_cmd = msg

    def step(self) -> None:
        accel = float(np.clip(self._last_accel_cmd.accel, self.a_min, self.a_max))
        self.speed = max(0.0, self.speed + accel * self.dt)  # can't reverse under ACC,
        # identical to EgoLongitudinalNode's integration -- see module docstring
        delta = float(np.clip(self._last_lateral_cmd.delta, -self.vehicle.max_steer, self.vehicle.max_steer))
        self.vehicle.update(self.speed, delta, self.dt)

        arc_length = project_to_arc_length(self.vehicle.x, self.vehicle.y, self.centerline, self.arc_length_table)
        self.bus.publish("ego_state", EgoLongitudinalStateMsg(arc_length, self.speed, accel))
        self.bus.publish(
            "ego_highway_state",
            EgoHighwayStateMsg(self.vehicle.x, self.vehicle.y, self.vehicle.theta, self.speed, accel, delta),
        )

        # Steering published before accel_odometry: SpeedEstimatorNode's predict fires
        # synchronously off accel_odometry and needs this tick's noisy delta already
        # cached (see speed_estimator_node.py's _on_steering_odometry).
        self.bus.publish("steering_odometry", SteeringOdometryMsg(delta + self.rng.normal(0.0, self.steering_odom_std)))
        self.bus.publish("accel_odometry", AccelOdometryMsg(accel + self.rng.normal(0.0, self.accel_odom_std)))
        self.bus.publish("speedometer", SpeedometerMsg(self.speed + self.rng.normal(0.0, self.speedometer_std)))

        compass_meas = wrap_angle(self.vehicle.theta + self.rng.normal(0.0, self.compass_std))
        self.bus.publish("compass", CompassMsg(compass_meas))

        self._tick += 1
        if self._tick % self.position_fix_period == 0:
            x_meas = self.vehicle.x + self.rng.normal(0.0, self.position_std)
            y_meas = self.vehicle.y + self.rng.normal(0.0, self.position_std)
            self.bus.publish("position_fix", PositionFixMsg(x_meas, y_meas))
