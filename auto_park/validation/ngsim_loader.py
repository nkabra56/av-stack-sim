"""Parses NGSIM vehicle-trajectory excerpts into leader/follower trajectory pairs for
ACC validation. See DESIGN.md's ACC section.

Uses the standard library `csv` module, not pandas -- a plain filter/sort over a few
hundred rows doesn't need a dataframe library, and the project has deliberately stayed
dependency-light (see IMPLEMENTATION.md's dependency notes for the EKF/pub-sub and
MPC milestones, neither of which needed a new dependency either).

NGSIM's native units are feet/feet-per-second; converted to SI (meters, m/s, m/s^2)
here so the committed CSV stays verbatim from the source (see
auto_park/data/ngsim/ATTRIBUTION.md) while everything downstream uses the project's
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
DEFAULT_LEADER_ID = 9
DEFAULT_FOLLOWER_ID = 12


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
