"""Shared structural types for planners and controllers. See IMPLEMENTATION.md section 2.

Planner and Controller are Protocols (not base classes) on purpose: new algorithms
are added by writing a class with the right method signature, no shared inheritance
required, so PlannerNode/ControllerNode (auto_park/nodes/) never need to change when a
new planner/controller is added -- they wrap whatever satisfies these protocols.
"""

from typing import Protocol

import numpy as np

from auto_park.environment import Obstacle

Pose = tuple[float, float, float]  # (x, y, theta)


class Planner(Protocol):
    def plan(
        self, start: Pose, goal: Pose, obstacles: list[Obstacle], turning_radius: float
    ) -> np.ndarray:
        """Return an (N, 3) array of x, y, theta waypoints from start to goal."""
        ...


class HasPose(Protocol):
    """Anything with .x/.y/.theta -- a real Vehicle, or (in the node architecture)
    a PoseEstimateMsg, since controllers only ever act on the estimate, never on
    ground truth."""

    x: float
    y: float
    theta: float


class Controller(Protocol):
    def control(self, pose: HasPose, path: np.ndarray) -> tuple[float, float]:
        """Return (v_desired, delta) given the current pose (estimate) and a path to track."""
        ...
