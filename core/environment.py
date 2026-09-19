"""Parking lot geometry: spots and obstacles. See DESIGN.md section 1/8."""

from dataclasses import dataclass, field

VEHICLE_RADIUS = 1.0  # ego collision-circle radius, same scale as scenario obstacle radii.
# Shared by harness.py's _collided() and hybrid_astar.py's avoidance check (DESIGN.md section 8).


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
    """Target parking stall: center, heading, and a `length` x `width` rectangle (length along the heading)."""

    x: float
    y: float
    theta: float = 0.0
    width: float = 2.75
    length: float = 5.5


@dataclass
class Environment:
    spot: Spot
    obstacles: list[Obstacle] = field(default_factory=list)

    def obstacle_tuples(self) -> list[tuple[float, float, float]]:
        return [o.as_tuple() for o in self.obstacles]
