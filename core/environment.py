"""Parking lot geometry: spots and obstacles. See DESIGN.md section 1/8."""

from dataclasses import dataclass, field

VEHICLE_RADIUS = 1.0  # ego vehicle's own collision-circle radius; must be on the same scale as
# the ~1.3m obstacle radii used for parked cars in scenarios/*.yaml (see DESIGN.md section 8),
# not an arbitrary small buffer -- a real car is comparable in size to the cars it's driving
# among, so its own collision footprint has to be too. Shared by harness.py's _collided() and
# planning/hybrid_astar.py's obstacle-avoidance predicate, so both independently-written
# collision checks stay against the exact same threshold rather than silently drifting apart.


@dataclass(frozen=True)
class Obstacle:
    """A circular obstacle (bounding-circle approximation, see DESIGN.md section 8)."""

    x: float
    y: float
    radius: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.radius)


@dataclass(frozen=True)
class Spot:
    """Target parking spot center, plus its heading for perpendicular vs. parallel spots."""

    x: float
    y: float
    theta: float = 0.0
    size: float = 1.0


@dataclass
class Environment:
    spot: Spot
    obstacles: list[Obstacle] = field(default_factory=list)

    def obstacle_tuples(self) -> list[tuple[float, float, float]]:
        return [o.as_tuple() for o in self.obstacles]
