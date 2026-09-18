"""Wraps the 4-state [x, y, theta, v] EKF (H2) for the highway mode: predicts on noisy
accel odometry, corrects on speedometer/compass/position_fix. See DESIGN.md section 12."""

from core.estimation.ekf import ExtendedKalmanFilter
from core.messaging.bus import Bus
from core.messaging.messages import (
    AccelOdometryMsg,
    CompassMsg,
    EgoSpeedEstimateMsg,
    PositionFixMsg,
    SpeedometerMsg,
    SteeringOdometryMsg,
)


class SpeedEstimatorNode:
    def __init__(self, bus: Bus, ekf: ExtendedKalmanFilter, dt: float):
        self.bus = bus
        self.ekf = ekf
        self.dt = dt
        self._last_delta = 0.0  # stays 0.0 unless steering_odometry is published (only
        # the full closed-loop drive's HighwayVehicleNode does; H1-standalone never does)
        bus.subscribe("steering_odometry", self._on_steering_odometry)
        bus.subscribe("accel_odometry", self._on_accel_odometry)
        bus.subscribe("speedometer", self._on_speedometer)
        bus.subscribe("compass", self._on_compass)
        bus.subscribe("position_fix", self._on_position_fix)

    def _on_steering_odometry(self, msg: SteeringOdometryMsg) -> None:
        self._last_delta = msg.delta

    def _on_accel_odometry(self, msg: AccelOdometryMsg) -> None:
        self.ekf.predict_with_speed_state(msg.accel, delta=self._last_delta, dt=self.dt)
        self._publish()

    def _on_speedometer(self, msg: SpeedometerMsg) -> None:
        self.ekf.update_speed(msg.speed)
        self._publish()

    def _on_compass(self, msg: CompassMsg) -> None:
        self.ekf.update_heading(msg.theta)
        self._publish()

    def _on_position_fix(self, msg: PositionFixMsg) -> None:
        self.ekf.update_position(msg.x, msg.y)
        self._publish()

    def _publish(self) -> None:
        x, y, theta, v = self.ekf.x
        self.bus.publish("ego_speed_estimate", EgoSpeedEstimateMsg(x, y, theta, v, self.ekf.p.copy()))
