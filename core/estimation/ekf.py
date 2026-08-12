"""Extended Kalman Filter for [x, y, theta] pose estimation, with an optional 4th
[x, y, theta, v] speed-estimating mode (H2, highway/ACC use). See DESIGN.md section
"EKF design" for the 3-state math and DESIGN.md section 12 for why/how the speed
state was added, and why it's a second method pair rather than a rewrite of predict().

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

`update_heading`/`update_position`/`update_landmark`/`_apply_update` are sized off
`len(self.x)` rather than a hardcoded 3, specifically so they keep working unchanged
for a 4-state instance too -- these corrections (heading, position, landmark bearing)
apply the same way regardless of whether speed is also being estimated. `predict()`
(3-state, speed as a perfect odometry *input*) and `predict_with_speed_state()`
(4-state, speed as an estimated, filtered *state*) are two separate methods rather
than one branching method: they represent genuinely different process models (the
control input itself changes -- speed vs. acceleration -- not just the state size),
and keeping the original `predict()` completely untouched was the point, given how
much the parking mode already depends on and has validated it (see
IMPLEMENTATION.md's MV milestone: 83% RMSE reduction against real KITTI data).
"""

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
        """4-state predict (H2): control input is acceleration, not speed -- v is now
        an estimated state, propagated forward and later corrected by update_speed(),
        rather than a perfect pass-through odometry input the way the 3-state
        predict() treats it. Requires the filter to have been constructed with a
        4-element x0/p0 ([x, y, theta, v]).

        Position/heading are propagated using v_new = v + accel*dt (this tick's
        already-updated speed), NOT the prior v -- matching Vehicle.update()'s own
        convention (every plant node computes the new speed first, then calls
        Vehicle.update with it, so x/y/theta integrate against the post-accel speed).
        Using the prior v here instead is a real bug this project shipped and didn't
        notice for a while: with H1's straight-line-only delta=0, dtheta is zero
        either way, so the mismatch was completely invisible until real steering (H3,
        the full closed-loop drive) started exercising a nonzero delta -- at which
        point it showed up as a small but systematic per-tick heading bias that
        compounded into meters of lateral drift the estimator itself couldn't see
        (Stanley, tracking the estimate, saw a small stable error and barely
        corrected, while the true vehicle drifted steadily away)."""
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
        # Input-Jacobian process noise (Thrun/Burgard/Fox's control-dependent
        # formulation, the same principle predict()'s V @ M @ Vᵀ already uses) --
        # NOT just a bare q[3,3] = (accel_std*dt)^2, which was a real gap: it modeled
        # acceleration uncertainty's effect on v (and, via F, everything F propagates
        # v's uncertainty into next step) but never modeled *steering* uncertainty's
        # direct effect on theta/x/y this same step. With H1's delta=0 that gap was
        # invisible (tan(delta)-dependent terms are all zero); once real steering
        # noise is actually driving dtheta each tick (H3), omitting it made the filter
        # overconfident in its own heading estimate -- P underestimated true
        # uncertainty, so later corrections (compass/position fixes) were weighted too
        # lightly to keep up with real drift. u = d(x_new,y_new,theta_new,v_new)/d(accel,delta).
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
            # Degenerate geometry: the estimate sits on (or within 1mm of) the landmark,
            # so both the range Jacobian (1/range_pred) and bearing Jacobian (1/q) blow
            # up toward Inf/NaN and would permanently poison self.x with no recovery
            # path. There's no real information in a bearing to a point you're standing
            # on anyway, so skip this update rather than risk it -- next tick's motion
            # moves the estimate off the landmark and updates resume normally.
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
