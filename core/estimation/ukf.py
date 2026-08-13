"""Unscented Kalman Filter for [x, y, theta] pose estimation -- an alternative to
ekf.py's EKF, built to answer DESIGN.md section 10's future-extensions question
directly rather than just asserting it: is the EKF's linearization (a first-order
Taylor expansion of the bicycle model / measurement functions around the current
estimate, via `ekf.py`'s Jacobians) actually a fine approximation at the turning
rates this project exercises, or does it just look fine because nobody compared it
against something that doesn't linearize at all?

The unscented transform answers that without linearizing: instead of propagating one
mean + one Jacobian-derived covariance through a first-order approximation of f/h, it
deterministically picks 2n+1 "sigma points" that exactly capture (x, P)'s mean and
covariance, propagates each of them through the *exact* nonlinear f/h (no Taylor
expansion at all), and reconstitutes the mean/covariance from the transformed points.
This is a genuinely different algorithm, not a relabeled EKF -- see `_sigma_points`/
`predict`/`_correct` -- while sharing the exact same process model (bicycle
kinematics) and measurement models (heading/position/landmark range-bearing) `ekf.py`
uses, so any accuracy difference measured between them is attributable to the
propagation method, not to the two filters modeling different physics.

**Scope choice**: control-input noise (odometry uncertainty in v, delta) is folded in
as an additive process-noise term using the same input-Jacobian formula `ekf.py`'s
`predict()` already uses (V @ M @ Vᵀ), rather than also augmenting the sigma-point
state with the input-noise dimensions (the "fully unscented" treatment). The point
of this comparison is whether sigma-point propagation of *existing state uncertainty*
through the nonlinear model handles curvature better than Jacobian linearization does
-- augmenting control noise into the sigma points too would improve both filters'
handling of input uncertainty roughly equally, so it wouldn't change what this
comparison is actually measuring, while meaningfully complicating the implementation.

See `core/validation/ukf_comparison.py` for the actual head-to-head numbers.
"""

import numpy as np

from core.vehicle import wrap_angle

ANGLE_INDEX = 2  # theta's position in the [x, y, theta] state -- the one component
# that needs circular (not linear) mean/difference handling throughout.


def _circular_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean of angles, correct across the -pi/pi wrap -- a naive weighted
    average of e.g. [3.1, -3.1] would give ~0 (wrong; the true mean is near pi/-pi),
    exactly the failure mode sigma points spanning the wrap boundary can trigger."""
    return float(np.arctan2(np.sum(weights * np.sin(angles)), np.sum(weights * np.cos(angles))))


class UnscentedKalmanFilter:
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
        alpha: float = 1e-3,
        beta: float = 2.0,  # optimal for Gaussian priors (Wan & van der Merwe, 2000)
        kappa: float = 0.0,
    ):
        self.x = np.array(x0, dtype=float)
        self.p = np.array(p0, dtype=float)
        self.wheelbase = wheelbase
        self.odom_v_std = odom_v_std
        self.odom_delta_std = odom_delta_std
        self.r_heading = r_heading
        self.r_position = r_position
        self.r_landmark = r_landmark

        n = len(self.x)
        self._n = n
        self._lambda = alpha**2 * (n + kappa) - n
        self._wm = np.full(2 * n + 1, 1.0 / (2 * (n + self._lambda)))
        self._wc = self._wm.copy()
        self._wm[0] = self._lambda / (n + self._lambda)
        self._wc[0] = self._wm[0] + (1 - alpha**2 + beta)

    def _sigma_points(self) -> np.ndarray:
        n = self._n
        # Cholesky, not a generic sqrtm: P is a covariance matrix (symmetric PSD by
        # construction every tick), and Cholesky is the standard, cheaper choice for
        # exactly that case -- same assumption the rest of this project already makes
        # about P (e.g. ekf.py's own eigendecomposition-based ellipse rendering).
        sqrt_p = np.linalg.cholesky((n + self._lambda) * self.p)
        points = np.empty((2 * n + 1, n))
        points[0] = self.x
        points[1 : n + 1] = self.x + sqrt_p.T
        points[n + 1 :] = self.x - sqrt_p.T
        return points

    def predict(self, v: float, delta: float, dt: float) -> None:
        sigma = self._sigma_points()
        propagated = np.empty_like(sigma)
        for i, (x, y, theta) in enumerate(sigma):
            dtheta = (v / self.wheelbase) * np.tan(delta) * dt
            propagated[i] = [x + v * np.cos(theta) * dt, y + v * np.sin(theta) * dt, wrap_angle(theta + dtheta)]

        # Same input-Jacobian process noise ekf.py's predict() uses -- see this
        # module's docstring for why control-input noise isn't also folded into the
        # sigma points themselves.
        theta = self.x[2]
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

        mean = np.sum(self._wm[:, None] * propagated, axis=0)
        mean[ANGLE_INDEX] = _circular_mean(propagated[:, ANGLE_INDEX], self._wm)
        diff = propagated - mean
        diff[:, ANGLE_INDEX] = wrap_angle(diff[:, ANGLE_INDEX])
        cov = (self._wc[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0) + q

        self.x = mean
        self.p = cov

    def _correct(self, h, z_meas: np.ndarray, r: np.ndarray, angle_indices: tuple[int, ...] = ()) -> None:
        """Shared correction step for all three measurement types below: propagate the
        current sigma points through measurement function `h` (state -> predicted
        measurement), reconstruct the predicted measurement's mean/covariance and its
        cross-covariance with state (all angle-aware where `angle_indices` marks a
        circular measurement component -- heading and landmark bearing), then apply
        the standard UKF gain update."""
        sigma = self._sigma_points()
        z_sigma = np.array([h(point) for point in sigma])

        z_mean = np.sum(self._wm[:, None] * z_sigma, axis=0)
        for idx in angle_indices:
            z_mean[idx] = _circular_mean(z_sigma[:, idx], self._wm)

        z_diff = z_sigma - z_mean
        x_diff = sigma - self.x
        for idx in angle_indices:
            z_diff[:, idx] = wrap_angle(z_diff[:, idx])
        x_diff[:, ANGLE_INDEX] = wrap_angle(x_diff[:, ANGLE_INDEX])

        p_zz = (self._wc[:, None, None] * z_diff[:, :, None] * z_diff[:, None, :]).sum(axis=0) + r
        p_xz = (self._wc[:, None, None] * x_diff[:, :, None] * z_diff[:, None, :]).sum(axis=0)
        k = p_xz @ np.linalg.inv(p_zz)

        innovation = z_meas - z_mean
        for idx in angle_indices:
            innovation[idx] = wrap_angle(innovation[idx])

        self.x = self.x + k @ innovation
        self.x[ANGLE_INDEX] = wrap_angle(self.x[ANGLE_INDEX])
        self.p = self.p - k @ p_zz @ k.T

    def update_heading(self, theta_meas: float) -> None:
        self._correct(lambda state: np.array([state[2]]), np.array([theta_meas]), np.array([[self.r_heading]]), angle_indices=(0,))

    def update_position(self, x_meas: float, y_meas: float) -> None:
        self._correct(lambda state: state[:2], np.array([x_meas, y_meas]), self.r_position)

    def update_landmark(self, range_meas: float, bearing_meas: float, landmark_xy: tuple[float, float]) -> None:
        lx, ly = landmark_xy

        def h(state):
            x, y, theta = state
            dx, dy = lx - x, ly - y
            return np.array([np.hypot(dx, dy), wrap_angle(np.arctan2(dy, dx) - theta)])

        self._correct(h, np.array([range_meas, bearing_meas]), self.r_landmark, angle_indices=(1,))
