"""Shared structural types for planners and controllers. See IMPLEMENTATION.md section 2.

Planner and Controller are Protocols (not base classes) on purpose: new algorithms
are added by writing a class with the right method signature, no shared inheritance
required, so `simulation.py` never needs to change when a new planner/controller is added.
"""

from typing import Protocol

import numpy as np

from auto_park.environment import Obstacle
from auto_park.vehicle import Vehicle

Pose = tuple[float, float, float]  # (x, y, theta)


class Planner(Protocol):
    def plan(
        self, start: Pose, goal: Pose, obstacles: list[Obstacle], turning_radius: float
    ) -> np.ndarray:
        """Return an (N, 3) array of x, y, theta waypoints from start to goal."""
        ...


class Controller(Protocol):
    def control(self, vehicle: Vehicle, path: np.ndarray) -> tuple[float, float]:
        """Return (v_desired, delta) given the current vehicle state and a path to track."""
        ...
