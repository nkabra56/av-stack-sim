"""Typed messages passed over the Bus. See DESIGN.md section 2.

`TrueStateMsg` is the one topic that models the perception/reality boundary: only
SensorNode and the harness's own evaluation logic may subscribe to it. Estimator,
Planner, and Controller nodes must never see it -- they only ever see what SensorNode
and EstimatorNode derive from it, exactly like a real vehicle never has direct access
to its own ground-truth pose.
"""

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
class ControlCmdMsg:
    v: float
    delta: float
