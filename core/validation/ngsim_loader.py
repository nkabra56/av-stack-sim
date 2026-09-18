"""Parses NGSIM vehicle-trajectory excerpts into leader/follower trajectory pairs for ACC
validation. Converts native feet/fps to SI units. See DESIGN.md's ACC section."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEET_TO_METERS = 0.3048
DEFAULT_EXCERPT_PATH = Path(__file__).parent.parent / "data" / "ngsim" / "excerpt_trajectories.csv"
# vehicle_id 2896 (leader) / 2903 (follower), NGSIM US-101 lane 2 -- re-extracted to be
# lane-coherent with lane_centerline.csv (KNOWN_BUGS.md entry 6). See ATTRIBUTION.md.
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
    real_space_headway: np.ndarray  # (N,) meters -- TRUE bumper-to-bumper gap (see load_following_pair)
    real_time_headway: np.ndarray  # (N,) seconds


def _read_rows(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _vehicle_trajectory(rows: list[dict], vehicle_id: int) -> NgsimTrajectory:
    # Sorted by global_time, not frame_id, which resets across NGSIM's recording sub-periods.
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
    # NGSIM's own space_headway is front-center-to-front-center, not the bumper-to-bumper
    # "gap" acc.py/AccHarness mean -- subtract the leader's length to convert, once, here.
    real_space_headway = (
        np.array([float(r["space_headway"]) * FEET_TO_METERS for r in follower_rows]) - leader.length
    )
    real_time_headway = np.array([float(r["time_headway"]) for r in follower_rows])

    return NgsimFollowingPair(
        leader=leader, follower=follower, real_space_headway=real_space_headway, real_time_headway=real_time_headway
    )


def load_lane_centerline(path: str | Path = DEFAULT_LANE_CENTERLINE_PATH) -> np.ndarray:
    """Returns an (N, 3) x/y/theta path along a real NGSIM-derived lane centerline (see
    ATTRIBUTION.md), in the same format Planner.plan() returns."""
    rows = _read_rows(path)
    position = np.array([float(r["position_m"]) for r in rows])
    lateral = np.array([float(r["lateral_offset_m"]) for r in rows])
    heading = np.arctan2(np.diff(lateral), np.diff(position))
    heading = np.append(heading, heading[-1])
    return np.column_stack([position, lateral, heading])
