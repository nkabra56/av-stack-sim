"""Kinematic bicycle model. See DESIGN.md section 3."""

import numpy as np


def wrap_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class Vehicle:
    """Kinematic bicycle model. All angles (theta, delta) are radians."""

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        wheelbase: float = 2.7,
        max_steer: float = 0.6,
    ):
        self.x = x
        self.y = y
        self.theta = theta
        self.wheelbase = wheelbase
        self.max_steer = max_steer  # rad; ~34 deg, realistic for a passenger car

    @property
    def pose(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])

    @property
    def turning_radius(self) -> float:
        """Minimum turning radius at max_steer: L / tan(max_steer). See DESIGN.md section 3."""
        return self.wheelbase / np.tan(self.max_steer)

    def update(self, v: float, delta: float, dt: float) -> None:
        """Advance state by speed v (m/s, signed) and steering delta (rad) over dt seconds."""
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta = wrap_angle(self.theta + (v / self.wheelbase) * np.tan(delta) * dt)
