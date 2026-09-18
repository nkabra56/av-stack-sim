"""Real 2D geometry for a 4-way intersection: perpendicular approaches, a real conflict-zone
box, and heading-based right-of-way -- `control/intersection.py`'s single-conflict-point
model can't represent this. See KNOWN_BUGS.md entry 4 / DESIGN.md section 12."""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from core.planning.dubins import DubinsPlanner

Turn = Literal["straight", "left", "right"]

HIGHWAY_VEHICLE_RADIUS = 2.5  # bounding-circle radius for a real car's footprint (~4.5x2m).
# Deliberately not named VEHICLE_RADIUS -- that's environment.py's parking-scale (1.0m) constant.


def wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Approach:
    """One lane feeding the intersection: travels in a straight line along `heading`, offset
    from the road centerline by `lane_offset` to its own right (right-hand traffic)."""

    name: str  # e.g. "N" for "coming from North" (heading south) -- descriptive only
    heading: float  # radians, direction of travel; core.vehicle's convention (0=+x east)

    def position(self, longitudinal: float, lane_offset: float, center: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
        """(x, y) at signed distance `longitudinal` from `center` along this approach's
        heading, offset into this approach's own lane."""
        right_dir = self.heading - math.pi / 2  # direction from centerline to this lane
        cx, cy = center
        x = cx + lane_offset * math.cos(right_dir) + longitudinal * math.cos(self.heading)
        y = cy + lane_offset * math.sin(right_dir) + longitudinal * math.sin(self.heading)
        return x, y


# The four standard cardinal approaches, named by where the vehicle is coming *from*.
NORTH = Approach("N", heading=-math.pi / 2)  # coming from North, heading South
EAST = Approach("E", heading=math.pi)  # coming from East, heading West
SOUTH = Approach("S", heading=math.pi / 2)  # coming from South, heading North
WEST = Approach("W", heading=0.0)  # coming from West, heading East


def is_to_the_right(self_heading: float, other_heading: float, tol: float = 1e-3) -> bool:
    """True if a vehicle traveling `other_heading` is to the right of one traveling
    `self_heading` (other_heading == self_heading + 90deg). Verified against all N/E/S/W pairs."""
    return abs(wrap_angle(other_heading - self_heading - math.pi / 2)) < tol


def in_conflict_zone(x: float, y: float, half_width: float, center: tuple[float, float] = (0.0, 0.0)) -> bool:
    """The conflict zone two perpendicular roads share, as a square -- a single shared
    point (H4's original model) is this square shrunk to zero width."""
    cx, cy = center
    return abs(x - cx) <= half_width and abs(y - cy) <= half_width


def is_opposite(heading_a: float, heading_b: float, tol: float = 1e-3) -> bool:
    """True if two approaches directly face each other (e.g. NORTH/SOUTH) -- the pairing
    left-turns yield to while crossing oncoming traffic (KNOWN_BUGS.md entry 4)."""
    return abs(wrap_angle(heading_b - heading_a - math.pi)) < tol


def turn_exit_heading(entry_heading: float, turn: Turn) -> float:
    """New heading after `turn` from `entry_heading`. Real-driving convention: facing South
    and turning left ends up heading East. In this module's math convention (East=0,
    counterclockwise positive): left = +90deg, right = -90deg."""
    if turn == "straight":
        return entry_heading
    if turn == "left":
        return wrap_angle(entry_heading + math.pi / 2)
    if turn == "right":
        return wrap_angle(entry_heading - math.pi / 2)
    # A typo used to silently fall through to the right-turn branch instead of failing
    # loudly -- caught in code review, not by any test.
    raise ValueError(f"turn must be 'straight', 'left', or 'right', got {turn!r}")


TURN_LEAD_RATIO = 2.5  # turn_lead = TURN_LEAD_RATIO * turning_radius. A too-short lead makes
# DubinsPlanner's CSC solve find a long looping path instead of a direct one for right turns.


def build_turn_path(
    entry: Approach, turn: Turn, lane_offset: float, turning_radius: float, center: tuple[float, float] = (0.0, 0.0)
) -> tuple[Approach, np.ndarray]:
    """The exit `Approach` and a curvature-respecting (x, y, theta) path connecting entry to
    exit via DubinsPlanner (forward-only CSC; an intersection turn never needs reverse)."""
    exit_heading = turn_exit_heading(entry.heading, turn)
    exit_approach = Approach(f"{entry.name}_{turn}", exit_heading)
    if turn == "straight":
        return exit_approach, np.array([[*entry.position(0.0, lane_offset, center), entry.heading]])

    turn_lead = TURN_LEAD_RATIO * turning_radius
    start_x, start_y = entry.position(-turn_lead, lane_offset, center)
    end_x, end_y = exit_approach.position(turn_lead, lane_offset, center)
    # Computed once per vehicle at setup, not per search node, so a generous fixed point
    # count is simpler than predicting the curve's exact length in advance.
    path = DubinsPlanner().plan(
        (start_x, start_y, entry.heading), (end_x, end_y, exit_heading), [], turning_radius, npts=200
    )
    return exit_approach, path
