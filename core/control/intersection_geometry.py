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

**Turning movements** (KNOWN_BUGS.md entry 4): `turn_exit_heading` gives the new travel
heading after a left/right turn; `build_turn_path` connects the entry lane to the exit
lane with a real curvature-respecting curve, reusing `planning/dubins.py`'s
`DubinsPlanner` (forward-only CSC, exactly right for an ordinary intersection turn --
no reverse gear involved) rather than inventing new curve math, the same "the geometry
problem is already solved on the parking side" reasoning KNOWN_BUGS.md itself suggested.
`intersection2d_harness.py` walks entry-straight -> curve -> exit-straight as one
continuous route per turning vehicle.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from core.planning.dubins import DubinsPlanner

Turn = Literal["straight", "left", "right"]

HIGHWAY_VEHICLE_RADIUS = 2.5  # bounding-circle approximation for a real car's footprint
# (~4.5m long x ~2m wide, the same scale highway_harness.py's ACC validation already
# uses via `lead_length=4.5`) -- conservatively above the exact half-diagonal (~2.46m),
# needed here for genuine 2D crossing-path collision checking, something neither the 1D
# ACC gap check nor H4's single-conflict-point model ever had to do.
#
# Deliberately NOT named `VEHICLE_RADIUS`: that name is already `core/environment.py`'s
# parking-scale collision radius (1.0m -- a real car's radius against ~1.3m parked-car
# obstacles at parking-lot scale, not this module's highway-lane scale). Found in code
# review: two same-named constants with a 2.5x different value, in modules that don't
# import from each other today but easily could in the future, is exactly the kind of
# thing that turns into a silent wrong-scale collision-radius bug the moment someone
# reaches for the "obviously" correct import.


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


def is_opposite(heading_a: float, heading_b: float, tol: float = 1e-3) -> bool:
    """True if two approaches are directly facing each other (e.g. NORTH and SOUTH) --
    the pairing "yield to oncoming through traffic while turning left" (KNOWN_BUGS.md
    entry 4) applies to."""
    return abs(wrap_angle(heading_b - heading_a - math.pi)) < tol


def turn_exit_heading(entry_heading: float, turn: Turn) -> float:
    """New travel heading after `turn` ("left"/"right"/"straight") from `entry_heading`.
    Real-driving convention: facing South and turning left, your left hand points East,
    so you end up heading East -- the same direction a vehicle continuing straight from
    the West approach already travels (turning right from North instead lands you
    heading West, matching East's straight-through traffic). In this module's math
    convention (East=0, counterclockwise positive), left = +90 degrees, right =
    -90 degrees."""
    if turn == "straight":
        return entry_heading
    if turn == "left":
        return wrap_angle(entry_heading + math.pi / 2)
    if turn == "right":
        return wrap_angle(entry_heading - math.pi / 2)
    # A typo (e.g. "Left", "rihgt") used to silently fall through to the right-turn
    # branch (the old `else` had no third case to fall to) instead of failing loudly --
    # caught in code review, not by any test, since nothing had exercised an invalid
    # value.
    raise ValueError(f"turn must be 'straight', 'left', or 'right', got {turn!r}")


TURN_LEAD_RATIO = 2.5  # turn_lead = TURN_LEAD_RATIO * turning_radius. Not independently
# settable: a real parameter sweep (start/goal turning circles vs. turning_radius,
# exactly the kind of geometry bug 5's CCC investigation already mapped out) found that
# a short lead relative to turning_radius makes DubinsPlanner's CSC solve find a long,
# looping connection instead of the short, direct one for RIGHT turns specifically (a
# tight turn's start/goal turning circles end up close together, the same "CCC would be
# shorter" regime bug 5 found -- but DubinsPlanner is deliberately forward-only/CSC-only,
# so the fix here is geometric, not a different solver). ratio=1.5 still produced a ~5x
# too-long right-turn path; ratio=2.0 was the smallest value that gave a short, direct
# connection for both left and right turns across turning_radius 4-8m; 2.5 keeps margin
# above that boundary rather than sitting right on it.


def build_turn_path(
    entry: Approach, turn: Turn, lane_offset: float, turning_radius: float, center: tuple[float, float] = (0.0, 0.0)
) -> tuple[Approach, np.ndarray]:
    """The exit `Approach` (synthetic -- not necessarily one of NORTH/EAST/SOUTH/WEST,
    though for a 90-degree turn at a 4-way intersection it always coincides with one)
    and a real curvature-respecting (x, y, theta) path connecting entry to exit, via
    `planning/dubins.py`'s `DubinsPlanner` (forward-only CSC -- an intersection turn
    never needs reverse gear, so the fuller Reeds-Shepp machinery isn't needed here).

    The curve runs from `TURN_LEAD_RATIO * turning_radius` meters before the
    intersection center (on the entry lane) to the same distance past it (on the exit
    lane). `turn == "straight"` is a degenerate 1-point "curve" (never actually walked;
    `intersection2d_harness.py` uses `Approach.position` directly for straight-through
    vehicles, unchanged from before this function existed) so every `VehicleSpec` can go
    through one code path regardless of turn."""
    exit_heading = turn_exit_heading(entry.heading, turn)
    exit_approach = Approach(f"{entry.name}_{turn}", exit_heading)
    if turn == "straight":
        return exit_approach, np.array([[*entry.position(0.0, lane_offset, center), entry.heading]])

    turn_lead = TURN_LEAD_RATIO * turning_radius
    start_x, start_y = entry.position(-turn_lead, lane_offset, center)
    end_x, end_y = exit_approach.position(turn_lead, lane_offset, center)
    # Computed once per vehicle at scenario setup, not per search node -- unlike
    # Hybrid A*'s heuristic calls, oversampling here costs nothing that matters, so a
    # generous fixed point count is simpler than trying to predict the curve's actual
    # length in advance just to hit `step` exactly.
    path = DubinsPlanner().plan(
        (start_x, start_y, entry.heading), (end_x, end_y, exit_heading), [], turning_radius, npts=200
    )
    return exit_approach, path
