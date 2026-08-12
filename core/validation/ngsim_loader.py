"""Parses NGSIM vehicle-trajectory excerpts into leader/follower trajectory pairs for
ACC validation. See DESIGN.md's ACC section.

Uses the standard library `csv` module, not pandas -- a plain filter/sort over a few
hundred rows doesn't need a dataframe library, and the project has deliberately stayed
dependency-light (see IMPLEMENTATION.md's dependency notes for the EKF/pub-sub and
MPC milestones, neither of which needed a new dependency either).

NGSIM's native units are feet/feet-per-second; converted to SI (meters, m/s, m/s^2)
here so the committed CSV stays verbatim from the source (see
core/data/ngsim/ATTRIBUTION.md) while everything downstream uses the project's
usual units. `local_y` is the along-road (forward) coordinate. `vehicle_id`/`frame_id`
reset across NGSIM's recording sub-periods, so `global_time` (a genuinely monotonic
millisecond timestamp) is what identifies a single contiguous trajectory -- not
frame_id, which this project's data-extraction step for the committed excerpt learned
the hard way (see IMPLEMENTATION.md's known-issues log).
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEET_TO_METERS = 0.3048
DEFAULT_EXCERPT_PATH = Path(__file__).parent.parent / "data" / "ngsim" / "excerpt_trajectories.csv"
# vehicle_id 2896 (leader) / 2903 (follower), NGSIM US-101 lane 2 -- re-extracted to close
# KNOWN_BUGS.md entry 6 (the excerpt used to be lane 1, geometrically overlapping but not
# lane-coherent with lane_centerline.csv below, which is lane 2). See ATTRIBUTION.md.
DEFAULT_LEADER_ID = 2896
DEFAULT_FOLLOWER_ID = 2903
DEFAULT_LANE_CENTERLINE_PATH = Path(__file__).parent.parent / "data" / "ngsim" / "lane_centerline.csv"


@dataclass
class NgsimTrajectory:
    times: np.ndarray  # (N,) seconds, relative to the excerpt's start
    position: np.ndarray  # (N,) meters, along-road
    speed: np.ndarray  # (N,) m/s
    accel: np.ndarray  # (N,) m/s^2
    length: float  # meters


@dataclass
class NgsimFollowingPair:
    leader: NgsimTrajectory
    follower: NgsimTrajectory
    real_space_headway: np.ndarray  # (N,) meters -- NGSIM's own recorded gap for the follower
    real_time_headway: np.ndarray  # (N,) seconds


def _read_rows(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _vehicle_trajectory(rows: list[dict], vehicle_id: int) -> NgsimTrajectory:
    vrows = sorted(
        (r for r in rows if int(r["vehicle_id"]) == vehicle_id), key=lambda r: int(r["global_time"])
    )
    t0 = int(vrows[0]["global_time"])
    times = np.array([(int(r["global_time"]) - t0) / 1000.0 for r in vrows])
    position = np.array([float(r["local_y"]) * FEET_TO_METERS for r in vrows])
    speed = np.array([float(r["v_vel"]) * FEET_TO_METERS for r in vrows])
    accel = np.array([float(r["v_acc"]) * FEET_TO_METERS for r in vrows])
    length = float(vrows[0]["v_length"]) * FEET_TO_METERS
    return NgsimTrajectory(times=times, position=position, speed=speed, accel=accel, length=length)


def load_following_pair(
    path: str | Path = DEFAULT_EXCERPT_PATH,
    leader_id: int = DEFAULT_LEADER_ID,
    follower_id: int = DEFAULT_FOLLOWER_ID,
) -> NgsimFollowingPair:
    rows = _read_rows(path)
    leader = _vehicle_trajectory(rows, leader_id)
    follower = _vehicle_trajectory(rows, follower_id)

    follower_rows = sorted(
        (r for r in rows if int(r["vehicle_id"]) == follower_id), key=lambda r: int(r["global_time"])
    )
    real_space_headway = np.array([float(r["space_headway"]) * FEET_TO_METERS for r in follower_rows])
    real_time_headway = np.array([float(r["time_headway"]) for r in follower_rows])

    return NgsimFollowingPair(
        leader=leader, follower=follower, real_space_headway=real_space_headway, real_time_headway=real_time_headway
    )


def load_lane_centerline(path: str | Path = DEFAULT_LANE_CENTERLINE_PATH) -> np.ndarray:
    """Returns an (N, 3) x/y/theta path along a real, NGSIM-derived lane centerline
    (see core/data/ngsim/ATTRIBUTION.md for how it was derived -- aggregated from
    ~10,400 real vehicle positions, not hand-authored), in the same (N, 3) format
    Planner.plan() returns for the parking mode, so Stanley control (control/
    lane_centering.py) can be validated the same way parking's path-tracking
    controllers are: given a real path, does the controller track it.
    """
    rows = _read_rows(path)
    position = np.array([float(r["position_m"]) for r in rows])
    lateral = np.array([float(r["lateral_offset_m"]) for r in rows])
    heading = np.arctan2(np.diff(lateral), np.diff(position))
    heading = np.append(heading, heading[-1])
    return np.column_stack([position, lateral, heading])
