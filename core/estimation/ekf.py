"""Extended Kalman Filter for [x, y, theta] pose estimation, with an optional 4th state
(v, for H2 speed estimation). Predicts from odometry, corrects with whichever of
compass/position-fix/landmark readings are available. See DESIGN.md's "EKF design" section."""

import numpy as np

from core.vehicle import wrap_angle


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
        r_speed: float | None = None,
        accel_std: float = 0.0,
    ):
        self.x = np.array(x0, dtype=float)
        self.p = np.array(p0, dtype=float)
        self.wheelbase = wheelbase
        self.odom_v_std = odom_v_std
        self.odom_delta_std = odom_delta_std
        self.r_heading = r_heading
        self.r_position = r_position
        self.r_landmark = r_landmark
        self.r_speed = r_speed  # only required if update_speed() is used (4-state mode)
        self.accel_std = accel_std  # process noise input for predict_with_speed_state()

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

    def predict_with_speed_state(self, accel: float, delta: float, dt: float) -> None:
        """4-state predict (H2): control input is acceleration and v is an estimated state, not odometry.
        Propagates with v_new = v + accel*dt, as Vehicle.update does; using the prior v caused a heading-bias bug."""
        x, y, theta, v = self.x
        v_new = v + accel * dt
        dtheta = (v_new / self.wheelbase) * np.tan(delta) * dt
        self.x = np.array(
            [x + v_new * np.cos(theta) * dt, y + v_new * np.sin(theta) * dt, wrap_angle(theta + dtheta), v_new]
        )
        f = np.array(
            [
                [1.0, 0.0, -v_new * np.sin(theta) * dt, np.cos(theta) * dt],
                [0.0, 1.0, v_new * np.cos(theta) * dt, np.sin(theta) * dt],
                [0.0, 0.0, 1.0, (np.tan(delta) * dt) / self.wheelbase],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        # Input-Jacobian process noise (same principle as predict()'s V @ M @ Vᵀ), modeling
        # accel and steering uncertainty's effect on the state: u = d(x,y,theta,v)/d(accel,delta).
        cos_delta = np.cos(delta)
        u = np.array(
            [
                [np.cos(theta) * dt**2, 0.0],
                [np.sin(theta) * dt**2, 0.0],
                [(np.tan(delta) * dt**2) / self.wheelbase, (v_new * dt) / (self.wheelbase * cos_delta**2)],
                [dt, 0.0],
            ]
        )
        m = np.diag([self.accel_std**2, self.odom_delta_std**2])
        q = u @ m @ u.T
        self.p = f @ self.p @ f.T + q

    def _apply_update(self, innovation: np.ndarray, h: np.ndarray, r: np.ndarray) -> None:
        s = h @ self.p @ h.T + r
        k = self.p @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ innovation
        self.x[2] = wrap_angle(self.x[2])
        self.p = (np.eye(len(self.x)) - k @ h) @ self.p

    def update_heading(self, theta_meas: float) -> None:
        h = np.zeros((1, len(self.x)))
        h[0, 2] = 1.0
        innovation = np.array([wrap_angle(theta_meas - self.x[2])])
        self._apply_update(innovation, h, np.array([[self.r_heading]]))

    def update_position(self, x_meas: float, y_meas: float) -> None:
        h = np.zeros((2, len(self.x)))
        h[0, 0] = 1.0
        h[1, 1] = 1.0
        innovation = np.array([x_meas, y_meas]) - self.x[:2]
        self._apply_update(innovation, h, self.r_position)

    def update_landmark(self, range_meas: float, bearing_meas: float, landmark_xy: tuple[float, float]) -> None:
        lx, ly = landmark_xy
        x, y, theta = self.x[0], self.x[1], self.x[2]
        dx, dy = lx - x, ly - y
        q = dx * dx + dy * dy
        if q < 1e-6:
            # Degenerate geometry: the estimate sits on the landmark, so both Jacobians
            # blow up toward Inf/NaN. Skip this update; next tick's motion recovers it.
            return
        range_pred = np.sqrt(q)
        bearing_pred = wrap_angle(np.arctan2(dy, dx) - theta)

        h = np.zeros((2, len(self.x)))
        h[0, 0], h[0, 1] = -dx / range_pred, -dy / range_pred
        h[1, 0], h[1, 1], h[1, 2] = dy / q, -dx / q, -1.0
        innovation = np.array([range_meas - range_pred, wrap_angle(bearing_meas - bearing_pred)])
        self._apply_update(innovation, h, self.r_landmark)

    def update_speed(self, v_meas: float) -> None:
        """4-state only (H2): fuse a noisy speedometer reading into the v state."""
        h = np.zeros((1, len(self.x)))
        h[0, 3] = 1.0
        innovation = np.array([v_meas - self.x[3]])
        self._apply_update(innovation, h, np.array([[self.r_speed]]))
