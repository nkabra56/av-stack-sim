"""Geometric Pure Pursuit path tracker, with automatic forward/reverse selection.
See DESIGN.md section 6.
"""

import numpy as np

from auto_park.interfaces import HasPose
from auto_park.vehicle import wrap_angle


class PurePursuitAdaptive:
    """Pure Pursuit that reverses automatically when the lookahead point lies behind
    the vehicle (needed for parking maneuvers, which require backing up)."""

    def __init__(self, wheelbase: float, lookahead: float = 3.5, v_max: float = 1.0, max_steer: float = 0.6):
        self.wheelbase = wheelbase
        self.lookahead = lookahead
        self.v_max = v_max
        self.max_steer = max_steer

    def control(self, pose: HasPose, path: np.ndarray) -> tuple[float, float]:
        dists = np.hypot(path[:, 0] - pose.x, path[:, 1] - pose.y)
        nearest = int(np.argmin(dists))

        # Search forward from the nearest point, not from index 0: scanning the whole
        # array for "distance >= lookahead" re-selects the path's start point once the
        # vehicle is more than `lookahead` past it (it's far away again, just behind
        # instead of ahead), yanking the target backward. This was the root cause of
        # the original prototype's "makes bigger circles" bug.
        ahead = np.where(dists[nearest:] >= self.lookahead)[0]
        target_idx = nearest + ahead[0] if len(ahead) else len(path) - 1
        target_x, target_y = path[target_idx, :2]

        dx, dy = target_x - pose.x, target_y - pose.y
        alpha = wrap_angle(np.arctan2(dy, dx) - pose.theta)
        delta = np.clip(
            np.arctan2(2 * self.wheelbase * np.sin(alpha), self.lookahead), -self.max_steer, self.max_steer
        )

        direction = -1.0 if abs(alpha) > np.pi / 2 else 1.0
        rho = np.hypot(dx, dy)
        v_desired = direction * self.v_max * np.tanh(rho)
        return v_desired, delta
