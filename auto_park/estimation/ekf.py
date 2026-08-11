"""Extended Kalman Filter for [x, y, theta] pose estimation. See DESIGN.md section
"EKF design" for the math and why each measurement type is modeled the way it is.

Standard "odometry + periodic absolute correction" mobile-robot localization pattern:
predict every tick from noisy odometry (dead reckoning, using the same nonlinear
bicycle-model equations as Vehicle.update, linearized via its Jacobian), then correct
with whichever measurements are available this tick -- an always-on compass (heading
only), a low-rate position fix (x, y only, rank-deficient by design), and opportunistic
landmark range-bearing readings (fully nonlinear, both in prediction and measurement,
which is what makes this a genuine EKF rather than a linear Kalman filter in disguise).

Process noise uses the control-dependent "velocity motion model" formulation (Thrun,
Burgard & Fox, *Probabilistic Robotics*, ch. 5) rather than a fixed, arbitrarily-chosen
Q: Q = V @ M @ Vᵀ, where M is the odometry noise covariance (how noisy the v/delta
measurement itself is) and V = df/d(v,delta) is the motion model's Jacobian with
respect to its *inputs*. This ties how fast position/heading uncertainty grows during
prediction directly to how noisy the odometry actually is, instead of guessing a
growth rate independently of the sensor model that's supposedly driving it.
"""

import numpy as np

from auto_park.vehicle import wrap_angle


class ExtendedKalmanFilter:
    def __init__(
        self,
        x0: np.ndarray,
        p0: np.ndarray,
        wheelbase: float,
        odom_v_std: float,
        odom_delta_std: float,
        r_heading: float,
        r_position: np.ndarray,
        r_landmark: np.ndarray,
    ):
        self.x = np.array(x0, dtype=float)
        self.p = np.array(p0, dtype=float)
        self.wheelbase = wheelbase
        self.odom_v_std = odom_v_std
        self.odom_delta_std = odom_delta_std
        self.r_heading = r_heading
        self.r_position = r_position
        self.r_landmark = r_landmark

    def predict(self, v: float, delta: float, dt: float) -> None:
        x, y, theta = self.x
        dtheta = (v / self.wheelbase) * np.tan(delta) * dt
        self.x = np.array(
            [x + v * np.cos(theta) * dt, y + v * np.sin(theta) * dt, wrap_angle(theta + dtheta)]
        )
        f = np.array(
            [
                [1.0, 0.0, -v * np.sin(theta) * dt],
                [0.0, 1.0, v * np.cos(theta) * dt],
                [0.0, 0.0, 1.0],
            ]
        )
        cos_delta = np.cos(delta)
        v_jacobian = np.array(
            [
                [np.cos(theta) * dt, 0.0],
                [np.sin(theta) * dt, 0.0],
                [np.tan(delta) * dt / self.wheelbase, v * dt / (self.wheelbase * cos_delta**2)],
            ]
        )
        m = np.diag([self.odom_v_std**2, self.odom_delta_std**2])
        q = v_jacobian @ m @ v_jacobian.T
        self.p = f @ self.p @ f.T + q

    def _apply_update(self, innovation: np.ndarray, h: np.ndarray, r: np.ndarray) -> None:
        s = h @ self.p @ h.T + r
        k = self.p @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ innovation
        self.x[2] = wrap_angle(self.x[2])
        self.p = (np.eye(3) - k @ h) @ self.p

    def update_heading(self, theta_meas: float) -> None:
        h = np.array([[0.0, 0.0, 1.0]])
        innovation = np.array([wrap_angle(theta_meas - self.x[2])])
        self._apply_update(innovation, h, np.array([[self.r_heading]]))

    def update_position(self, x_meas: float, y_meas: float) -> None:
        h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        innovation = np.array([x_meas, y_meas]) - self.x[:2]
        self._apply_update(innovation, h, self.r_position)

    def update_landmark(self, range_meas: float, bearing_meas: float, landmark_xy: tuple[float, float]) -> None:
        lx, ly = landmark_xy
        x, y, theta = self.x
        dx, dy = lx - x, ly - y
        q = dx * dx + dy * dy
        range_pred = np.sqrt(q)
        bearing_pred = wrap_angle(np.arctan2(dy, dx) - theta)

        h = np.array([[-dx / range_pred, -dy / range_pred, 0.0], [dy / q, -dx / q, -1.0]])
        innovation = np.array([range_meas - range_pred, wrap_angle(bearing_meas - bearing_pred)])
        self._apply_update(innovation, h, self.r_landmark)
