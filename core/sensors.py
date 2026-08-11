"""Simulated ultrasonic ray-cast sensor array. See DESIGN.md section 4."""

import numpy as np

from core.environment import Obstacle
from core.vehicle import Vehicle


class UltrasonicArray:
    """A fixed set of beams, each cast against circular obstacles.

    Beam angles are relative to the vehicle heading. Each beam returns the distance
    to the nearest ray/circle intersection, or max_range if nothing is hit.
    """

    def __init__(self, angles: list[float], max_range: float = 5.0):
        self.angles = angles
        self.max_range = max_range

    def sense(self, vehicle: Vehicle, obstacles: list[Obstacle]) -> dict[float, float]:
        readings: dict[float, float] = {}
        for beam_angle in self.angles:
            ray_theta = vehicle.theta + beam_angle
            dmin = self.max_range
            dx, dy = np.cos(ray_theta), np.sin(ray_theta)
            for obstacle in obstacles:
                ex, ey = vehicle.x - obstacle.x, vehicle.y - obstacle.y
                a = dx * dx + dy * dy
                b = 2 * (dx * ex + dy * ey)
                c = ex * ex + ey * ey - obstacle.radius * obstacle.radius
                disc = b * b - 4 * a * c
                if disc < 0:
                    continue
                sqrt_disc = np.sqrt(disc)
                for t in ((-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)):
                    if 0 <= t < dmin:
                        dmin = t
            readings[beam_angle] = dmin
        return readings
