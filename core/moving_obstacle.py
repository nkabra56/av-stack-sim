"""A moving obstacle (pedestrian/car crossing the drive aisle): constant-velocity motion,
computed fresh from elapsed simulation time rather than stepped/integrated (DESIGN.md's
future-extensions gap: re-planning previously only ever saw a static obstacle appear once)."""

from dataclasses import dataclass

from core.environment import Obstacle


@dataclass(frozen=True)
class MovingObstacleSpec:
    """Starts at (start_x, start_y) at `start_time`, moves at constant (vx, vy) m/s, and
    freezes in place once `duration` seconds have elapsed (None = moves for the whole run)."""

    start_x: float
    start_y: float
    vx: float
    vy: float
    radius: float
    start_time: float = 0.0  # seconds; not present in the environment before this
    duration: float | None = None  # seconds of motion after start_time


def obstacle_at(spec: MovingObstacleSpec, t: float) -> Obstacle | None:
    """The spec's `Obstacle` at simulation time `t`, or None if it hasn't appeared yet.
    Pure function of `t`: no hidden state, so re-plans always see the true current position."""
    if t < spec.start_time:
        return None
    elapsed = t - spec.start_time
    if spec.duration is not None:
        elapsed = min(elapsed, spec.duration)
    return Obstacle(x=spec.start_x + spec.vx * elapsed, y=spec.start_y + spec.vy * elapsed, radius=spec.radius)
