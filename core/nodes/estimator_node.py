"""Wraps the EKF: predicts on odometry, corrects on compass/position_fix/landmark
readings, republishes pose_estimate after every update. Landmark ids are resolved
against Environment.obstacles -- the filter treats their positions as known/mapped,
not something it also has to estimate (localization only, not SLAM).
"""

from core.environment import Environment
from core.estimation.ekf import ExtendedKalmanFilter
from core.messaging.bus import Bus
from core.messaging.messages import (
    CompassMsg,
    LandmarkBearingMsg,
    OdometryMsg,
    PoseEstimateMsg,
    PositionFixMsg,
)


class EstimatorNode:
    def __init__(self, bus: Bus, ekf: ExtendedKalmanFilter, environment: Environment):
        self.bus = bus
        self.ekf = ekf
        self.environment = environment

        bus.subscribe("odometry", self._on_odometry)
        bus.subscribe("compass", self._on_compass)
        bus.subscribe("position_fix", self._on_position_fix)
        bus.subscribe("landmark_bearings", self._on_landmark_bearings)

    def _on_odometry(self, msg: OdometryMsg) -> None:
        self.ekf.predict(msg.v, msg.delta, msg.dt)
        self._publish()

    def _on_compass(self, msg: CompassMsg) -> None:
        self.ekf.update_heading(msg.theta)
        self._publish()

    def _on_position_fix(self, msg: PositionFixMsg) -> None:
        self.ekf.update_position(msg.x, msg.y)
        self._publish()

    def _on_landmark_bearings(self, msg: LandmarkBearingMsg) -> None:
        for reading in msg.readings:
            landmark = self.environment.obstacles[reading.landmark_id]
            self.ekf.update_landmark(reading.range, reading.bearing, (landmark.x, landmark.y))
        self._publish()

    def _publish(self) -> None:
        x, y, theta = self.ekf.x
        self.bus.publish("pose_estimate", PoseEstimateMsg(x, y, theta, self.ekf.p.copy()))
