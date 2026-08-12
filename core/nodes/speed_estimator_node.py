"""Wraps the 4-state [x, y, theta, v] EKF (H2, estimation/ekf.py's
predict_with_speed_state/update_speed) for the highway mode: predicts on noisy
acceleration odometry, corrects on the noisy speedometer, republishes the fused pose
+ speed estimate. See DESIGN.md section 12.

x/y/theta stay near zero and degenerate for H1-only use (straight-line, no steering)
-- they become meaningful once H3 (lane centering) reintroduces real lateral motion,
which is exactly why the full closed-loop drive feeds this node a real steering
reading (see _on_steering_odometry) instead of leaving `delta` hardcoded at 0.0 the
way H1-standalone use does. Reusing the same 4-state filter now, even though two of
its dimensions are degenerate for H1 alone, avoids building a separate linear
speed-only filter that would just get thrown away once H3 needs the full state
anyway.

Also corrects on compass/position_fix when published (H1-standalone never publishes
either, so these handlers are simply never called there) -- a real gap found while
building the full closed-loop drive: with no absolute correction at all, x/y/theta is
pure dead reckoning, and realistic steering-odometry noise accumulates into real,
meaningful drift over a long run once Stanley is actually closing the loop on the
estimate (see ekf.py's predict_with_speed_state docstring for the two related EKF
fixes this also needed). Reuses update_heading/update_position unchanged -- exactly
the reuse H2's own original design already anticipated by generalizing them to
len(self.x) instead of a hardcoded 3.
"""

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
        self._last_delta = 0.0  # H1-standalone never publishes steering_odometry, so
        # this stays 0.0 (the old hardcoded behavior) unless something does -- the full
        # closed-loop drive's HighwayVehicleNode publishes a real noisy reading each tick.
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
