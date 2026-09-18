"""Typed messages passed over the Bus. TrueStateMsg models the perception/reality boundary
-- only SensorNode and the harness may see it, never Estimator/Planner/Controller. See DESIGN.md section 2."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TrueStateMsg:
    x: float
    y: float
    theta: float
    v: float
    delta: float


@dataclass(frozen=True)
class OdometryMsg:
    """Noisy wheel/steering-encoder reading of the *actually applied* v, delta."""

    v: float
    delta: float
    dt: float


@dataclass(frozen=True)
class CompassMsg:
    theta: float


@dataclass(frozen=True)
class PositionFixMsg:
    """Low-rate absolute (x, y) fix -- e.g. a garage RTLS/UWB-anchor system. Does not
    observe heading."""

    x: float
    y: float


@dataclass(frozen=True)
class LandmarkReading:
    landmark_id: int
    range: float
    bearing: float  # relative to vehicle heading, radians


@dataclass(frozen=True)
class LandmarkBearingMsg:
    readings: list[LandmarkReading] = field(default_factory=list)


@dataclass(frozen=True)
class ObstacleRangeMsg:
    """Ultrasonic beam readings, angle (rad, relative to heading) -> range (m)."""

    readings: dict[float, float]


@dataclass(eq=False)
class PoseEstimateMsg:
    x: float
    y: float
    theta: float
    covariance: np.ndarray  # (3, 3)


@dataclass(eq=False)
class PathMsg:
    path: np.ndarray  # (N, 3) x, y, theta waypoints


@dataclass(frozen=True)
class ReplanRequestMsg:
    """Published by ControllerNode when the speed governor has been binding long enough to
    look like a stall -- see ControllerNode's docstring and PlannerNode's `_on_replan_request`."""


@dataclass(frozen=True)
class ControlCmdMsg:
    v: float
    delta: float


# --- Highway/ACC mode (H1): longitudinal-only, straight-line following. LeadVehicleStateMsg/
# EgoLongitudinalStateMsg are TrueStateMsg's highway analogs (ground truth). See DESIGN.md's ACC section.


@dataclass(frozen=True)
class LeadVehicleStateMsg:
    position: float  # meters, along-road, ground truth
    speed: float  # m/s


@dataclass(frozen=True)
class EgoLongitudinalStateMsg:
    position: float  # meters, along-road, ground truth
    speed: float  # m/s
    accel: float  # m/s^2, last applied


@dataclass(frozen=True)
class RadarMsg:
    """Noisy forward-radar reading: bumper-to-bumper range and closing range-rate."""

    range: float  # meters
    range_rate: float  # m/s, positive = closing (ego faster than lead)


@dataclass(frozen=True)
class LongitudinalCmdMsg:
    accel: float  # m/s^2


# --- H2: fused ego speed. AccelOdometryMsg/SpeedometerMsg are noisy sensor readings;
# EgoSpeedEstimateMsg is what AccControllerNode acts on -- ground truth stays elsewhere.


@dataclass(frozen=True)
class AccelOdometryMsg:
    accel: float  # m/s^2, noisy reading of the actually-applied acceleration


@dataclass(frozen=True)
class SpeedometerMsg:
    speed: float  # m/s, noisy


@dataclass(eq=False)
class EgoSpeedEstimateMsg:
    x: float  # near 0 for H1 (straight-line only); meaningful once H3 adds lateral motion
    y: float
    theta: float
    speed: float
    covariance: np.ndarray  # (4, 4), the full [x,y,theta,v] state covariance


# --- H3/full closed-loop: the 2D kinematic Vehicle back in highway mode. SteeringOdometryMsg
# is OdometryMsg's highway analog; EgoHighwayStateMsg is TrueStateMsg's (ground truth).


@dataclass(frozen=True)
class LateralCmdMsg:
    delta: float  # rad


@dataclass(frozen=True)
class SteeringOdometryMsg:
    delta: float  # rad, noisy reading of the actually-applied steering angle


@dataclass(frozen=True)
class EgoHighwayStateMsg:
    x: float
    y: float
    theta: float
    speed: float
    accel: float  # m/s^2, last applied
    delta: float  # rad, last applied
