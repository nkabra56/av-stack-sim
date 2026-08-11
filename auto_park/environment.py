"""Parking lot geometry: spots and obstacles. See DESIGN.md section 1/7."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Obstacle:
    """A circular obstacle (bounding-circle approximation, see DESIGN.md section 7)."""

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
