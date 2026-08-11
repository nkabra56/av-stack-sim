"""Stanley lane-centering controller -- the classical lateral-control law, playing
the same "geometric baseline" role Pure Pursuit (parking) and IDM (ACC) play
elsewhere. See DESIGN.md section 12's H3 entry.

Unlike this project's other controllers (which act on HasPose alone), Stanley's
formula divides by speed, so `control()` takes it as an explicit third argument
rather than requiring every pose object to carry one -- HasPose stays a minimal
x/y/theta contract, and callers that have a speed (true, estimated, or assumed
constant) pass it explicitly.
"""

import numpy as np

from auto_park.interfaces import HasPose
from auto_park.vehicle import wrap_angle


class StanleyController:
    def __init__(self, wheelbase: float, k: float = 0.5, max_steer: float = 0.6):
        self.wheelbase = wheelbase
        self.k = k
        self.max_steer = max_steer

    def control(self, pose: HasPose, path: np.ndarray, speed: float) -> float:
        # Stanley's cross-track error is measured at the front axle, not the vehicle's
        # own (rear-axle) reference point -- steering corrects what's actually about
        # to leave the lane, not where the car's center happens to be right now.
        front_x = pose.x + self.wheelbase * np.cos(pose.theta)
        front_y = pose.y + self.wheelbase * np.sin(pose.theta)

        dists = np.hypot(path[:, 0] - front_x, path[:, 1] - front_y)
        idx = int(np.argmin(dists))
        path_x, path_y, path_theta = path[idx]

        # Signed cross-track error: positive means the front axle is to the RIGHT of
        # the path's direction of travel (project the position error onto the path's
        # right-normal (sin theta, -cos theta)) -- this sign convention is what makes
        # `correction` below steer *toward* the path rather than away from it; getting
        # this backwards doesn't error, it just diverges (caught by a standalone
        # convergence test before this controller was wired into anything else).
        dx, dy = front_x - path_x, front_y - path_y
        cross_track_error = dx * np.sin(path_theta) - dy * np.cos(path_theta)

        heading_error = wrap_angle(path_theta - pose.theta)
        # atan2(k*cte, speed) blends toward zero correction as speed -> 0 rather than
        # exploding (a raw division would); using |speed| plus a small floor keeps the
        # correction meaningful even at a near-stop instead of chattering near v=0.
        correction = np.arctan2(self.k * cross_track_error, max(abs(speed), 0.5))
        delta = heading_error + correction
        return float(np.clip(delta, -self.max_steer, self.max_steer))
