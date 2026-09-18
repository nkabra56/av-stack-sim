"""Shared structural types for planners and controllers. Protocols, not base classes, so
new algorithms need no shared inheritance (see IMPLEMENTATION.md section 2)."""

from typing import Protocol

import numpy as np

from core.environment import Obstacle

Pose = tuple[float, float, float]  # (x, y, theta)


class Planner(Protocol):
    def plan(
        self, start: Pose, goal: Pose, obstacles: list[Obstacle], turning_radius: float
    ) -> np.ndarray:
        """Return an (N, 3) array of x, y, theta waypoints from start to goal."""
        ...


class HasPose(Protocol):
    """Anything with .x/.y/.theta -- a real Vehicle, or a PoseEstimateMsg, since
    controllers only ever act on the estimate, never ground truth."""

    x: float
    y: float
    theta: float


class Controller(Protocol):
    def control(self, pose: HasPose, path: np.ndarray) -> tuple[float, float]:
        """Return (v_desired, delta) given the current pose (estimate) and a path to track."""
        ...
