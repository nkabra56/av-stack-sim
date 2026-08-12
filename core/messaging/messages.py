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
class ReplanRequestMsg:
    """Published by ControllerNode when the speed governor has been binding for long
    enough to look like a genuine stall rather than a momentary slowdown -- see
    ControllerNode's docstring and PlannerNode's `_on_replan_request`."""


@dataclass(frozen=True)
class ControlCmdMsg:
    v: float
    delta: float


# --- Highway/ACC mode (H1) -- longitudinal-only, straight-line following. See
# DESIGN.md's ACC section. LeadVehicleStateMsg/EgoLongitudinalStateMsg are the
# highway-mode analogs of TrueStateMsg: ground truth, visible to RadarNode and the
# harness's own evaluation logic, never to AccControllerNode directly.


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


# --- H2: fused ego speed for the highway mode. AccelOdometryMsg/SpeedometerMsg are
# noisy sensor readings (like OdometryMsg/CompassMsg for parking); EgoSpeedEstimateMsg
# is what AccControllerNode actually acts on -- the ego's own true speed
# (EgoLongitudinalStateMsg) stays visible only to RadarNode and the harness's
# evaluation logic, same ground-truth boundary as everywhere else in this project.


@dataclass(frozen=True)
class AccelOdometryMsg:
    accel: float  # m/s^2, noisy reading of the actually-applied acceleration


@dataclass(frozen=True)
class SpeedometerMsg:
    speed: float  # m/s, noisy


@dataclass(eq=False)
class EgoSpeedEstimateMsg:
    x: float  # degenerate (near 0) for H1 straight-line-only use; meaningful once H3
    y: float  # lane centering reintroduces real lateral motion -- see DESIGN.md
    theta: float  # section 12's H2 entry, which anticipates exactly this extension.
    speed: float
    covariance: np.ndarray  # (4, 4), the full [x,y,theta,v] state covariance


# --- H3/full closed-loop: bringing the 2D kinematic Vehicle back into the highway
# mode, alongside H1/H2's longitudinal machinery, on one ego. SteeringOdometryMsg is
# the highway-mode analog of parking's OdometryMsg.delta_meas -- a noisy reading of
# the actually-applied steering angle, needed once predict_with_speed_state() gets a
# real (non-zero) delta each tick. EgoHighwayStateMsg is TrueStateMsg's highway-mode
# analog: full ground-truth pose, visible only to the harness's own evaluation logic.


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
