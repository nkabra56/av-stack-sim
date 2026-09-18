"""Stanley lane-centering controller: the classical lateral-control law, same role
Pure Pursuit (parking) and IDM (ACC) play elsewhere. See DESIGN.md section 12's H3 entry."""

import numpy as np

from core.interfaces import HasPose
from core.vehicle import wrap_angle


class StanleyController:
    def __init__(self, wheelbase: float, k: float = 0.5, max_steer: float = 0.6):
        self.wheelbase = wheelbase
        self.k = k
        self.max_steer = max_steer

    def control(self, pose: HasPose, path: np.ndarray, speed: float) -> float:
        # Cross-track error is measured at the front axle, not the rear-axle reference
        # point, steering corrects what's about to leave the lane.
        front_x = pose.x + self.wheelbase * np.cos(pose.theta)
        front_y = pose.y + self.wheelbase * np.sin(pose.theta)

        dists = np.hypot(path[:, 0] - front_x, path[:, 1] - front_y)
        idx = int(np.argmin(dists))
        path_x, path_y, path_theta = path[idx]

        # Signed cross-track error: positive means the front axle is right of the path's
        # direction of travel (position error projected onto the path's right-normal).
        dx, dy = front_x - path_x, front_y - path_y
        cross_track_error = dx * np.sin(path_theta) - dy * np.cos(path_theta)

        heading_error = wrap_angle(path_theta - pose.theta)
        # atan2(k*cte, speed) blends toward zero correction as speed -> 0; a speed floor
        # avoids chattering near a near-stop.
        correction = np.arctan2(self.k * cross_track_error, max(abs(speed), 0.5))
        delta = heading_error + correction
        return float(np.clip(delta, -self.max_steer, self.max_steer))
