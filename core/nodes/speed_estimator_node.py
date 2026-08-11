"""Wraps the 4-state [x, y, theta, v] EKF (H2, estimation/ekf.py's
predict_with_speed_state/update_speed) for the highway mode: predicts on noisy
acceleration odometry, corrects on the noisy speedometer, republishes the fused speed
estimate. See DESIGN.md section 12.

x/y/theta are along for the ride and stay near zero for H1 (straight-line, no
steering) -- they become meaningful once H3 (lane centering) reintroduces real
lateral motion. Reusing the same 4-state filter now, even though two of its
dimensions are degenerate for H1, avoids building a separate linear speed-only
filter that would just get thrown away once H3 needs the full state anyway.
"""

from core.estimation.ekf import ExtendedKalmanFilter
from core.messaging.bus import Bus
from core.messaging.messages import AccelOdometryMsg, EgoSpeedEstimateMsg, SpeedometerMsg


class SpeedEstimatorNode:
    def __init__(self, bus: Bus, ekf: ExtendedKalmanFilter, dt: float):
        self.bus = bus
        self.ekf = ekf
        self.dt = dt
        bus.subscribe("accel_odometry", self._on_accel_odometry)
        bus.subscribe("speedometer", self._on_speedometer)

    def _on_accel_odometry(self, msg: AccelOdometryMsg) -> None:
        self.ekf.predict_with_speed_state(msg.accel, delta=0.0, dt=self.dt)
        self._publish()

    def _on_speedometer(self, msg: SpeedometerMsg) -> None:
        self.ekf.update_speed(msg.speed)
        self._publish()

    def _publish(self) -> None:
        self.bus.publish("ego_speed_estimate", EgoSpeedEstimateMsg(self.ekf.x[3], self.ekf.p.copy()))
