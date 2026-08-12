"""Real 2D geometry for a 4-way intersection: straight approaches on two perpendicular
roads, a genuine conflict-zone box where they overlap, and the "who's on whose right"
relation as an actual function of travel heading rather than a hand-set boolean. See
KNOWN_BUGS.md entry 4 / DESIGN.md section 12's "Full 2D intersection geometry" note --
`control/intersection.py`'s `IntersectionNavigator` models right-of-way as mutual
exclusion + arrival-order priority at a single point two approaches share, which is
correct reasoning but has no way to represent real crossing paths, a third or fourth
approach, or verify geometrically that right-of-way was actually enough to avoid a
collision. This module supplies the geometry; `intersection2d_harness.py` uses it to run
several real vehicles -- each still just an unmodified `IntersectionNavigator` instance,
same reuse principle as H4 reusing H1's `IDMController` -- through one shared
intersection.

Deliberately straight-through movements only, no turning: turning movements need real
per-turn path geometry (a curved connector from one approach's lane to another's), a
separable further increment from "more than two approaches with real crossing paths",
which is the specific gap this module closes. Noted as still-open in KNOWN_BUGS.md.
"""

import math
from dataclasses import dataclass

VEHICLE_RADIUS = 2.5  # bounding-circle approximation for a real car's footprint (~4.5m
# long x ~2m wide, the same scale highway_harness.py's ACC validation already uses via
# `lead_length=4.5`) -- conservatively above the exact half-diagonal (~2.46m), needed
# here for genuine 2D crossing-path collision checking, something neither the 1D ACC
# gap check nor H4's single-conflict-point model ever had to do.


def wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Approach:
    """One lane feeding the intersection: a vehicle on this approach travels in a
    straight line along `heading`, offset from the road centerline by `lane_offset` --
    to its own right, matching right-hand-traffic lane placement, so opposing traffic on
    the same road (e.g. "coming from North" and "coming from South") never share a line.
    """

    name: str  # e.g. "N" for "coming from North" (heading south) -- descriptive only
    heading: float  # radians, direction of travel; core.vehicle's convention (0=+x east)

    def position(self, longitudinal: float, lane_offset: float, center: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
        """(x, y) at signed distance `longitudinal` from `center` along this approach's
        heading (negative = before the intersection), offset into this approach's own
        lane."""
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
    """True if a vehicle traveling `other_heading` is positioned to the right of one
    traveling `self_heading` at a shared intersection -- e.g. facing South, your right
    hand points West, so the vehicle "to your right" is the one coming *from* the West,
    i.e. heading East. Geometrically: the other vehicle's heading is `self_heading` +
    90 degrees. Verified against all 4 cardinal pairs (N/E/S/W) by
    tests/test_intersection_geometry.py; a real 4-way-stop yield-to-the-right rule isn't
    well-defined for approaches that aren't roughly perpendicular, which every use in
    this module's own `NORTH/EAST/SOUTH/WEST` constants already satisfies."""
    return abs(wrap_angle(other_heading - self_heading - math.pi / 2)) < tol


def in_conflict_zone(x: float, y: float, half_width: float, center: tuple[float, float] = (0.0, 0.0)) -> bool:
    """The real conflict zone two perpendicular roads share is the square where their
    roadways overlap -- a single shared *point* (H4's original model) is the degenerate
    case of this square shrunk to zero width."""
    cx, cy = center
    return abs(x - cx) <= half_width and abs(y - cy) <= half_width
