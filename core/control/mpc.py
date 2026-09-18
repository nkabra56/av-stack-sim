"""Short-horizon nonlinear MPC path tracker: rolls out the true kinematic bicycle model
(no linearization) and solves with SLSQP, receding-horizon, warm-started. See DESIGN.md section 7."""

import numpy as np
from scipy.optimize import minimize

from core.interfaces import HasPose
from core.vehicle import wrap_angle


class MPCController:
    def __init__(
        self,
        wheelbase: float,
        dt: float = 0.1,
        horizon: int = 6,
        v_max: float = 1.5,
        delta_max: float = 0.6,
        w_pos: float = 1.0,
        w_heading: float = 0.3,
        w_effort: float = 0.01,
        w_smooth: float = 0.05,
        maxiter: int = 20,
    ):
        self.wheelbase = wheelbase
        self.dt = dt
        self.horizon = horizon
        self.v_max = v_max
        self.delta_max = delta_max
        self.w_pos = w_pos
        self.w_heading = w_heading
        self.w_effort = w_effort
        self.w_smooth = w_smooth
        self.maxiter = maxiter
        self._warm_start: np.ndarray | None = None

    def _rollout(self, x: float, y: float, theta: float, u: np.ndarray):
        xs = np.empty(self.horizon)
        ys = np.empty(self.horizon)
        thetas = np.empty(self.horizon)
        for k in range(self.horizon):
            v, delta = u[2 * k], u[2 * k + 1]
            x = x + v * np.cos(theta) * self.dt
            y = y + v * np.sin(theta) * self.dt
            theta = theta + (v / self.wheelbase) * np.tan(delta) * self.dt
            xs[k], ys[k], thetas[k] = x, y, theta
        return xs, ys, thetas

    def _reference(self, pose: HasPose, path: np.ndarray) -> np.ndarray:
        dists = np.hypot(path[:, 0] - pose.x, path[:, 1] - pose.y)
        nearest = int(np.argmin(dists))

        seg = np.diff(path[:, :2], axis=0)
        avg_spacing = np.hypot(seg[:, 0], seg[:, 1]).mean() if len(seg) else 1.0
        stride = max(1, round((self.v_max * self.dt) / max(avg_spacing, 1e-6)))

        idxs = np.clip(nearest + stride * np.arange(1, self.horizon + 1), 0, len(path) - 1)
        return path[idxs]

    def _cost(self, u: np.ndarray, x0: float, y0: float, theta0: float, ref: np.ndarray) -> float:
        xs, ys, thetas = self._rollout(x0, y0, theta0, u)
        pos_err = (xs - ref[:, 0]) ** 2 + (ys - ref[:, 1]) ** 2
        heading_err = wrap_angle(thetas - ref[:, 2]) ** 2

        cost = self.w_pos * pos_err.sum() + self.w_heading * heading_err.sum()
        cost += self.w_effort * np.sum(u**2)
        du = np.diff(u.reshape(-1, 2), axis=0)
        cost += self.w_smooth * np.sum(du**2)
        return cost

    def control(self, pose: HasPose, path: np.ndarray) -> tuple[float, float]:
        ref = self._reference(pose, path)
        u0 = self._warm_start if self._warm_start is not None else np.zeros(2 * self.horizon)
        bounds = [(-self.v_max, self.v_max), (-self.delta_max, self.delta_max)] * self.horizon

        result = minimize(
            self._cost,
            u0,
            args=(pose.x, pose.y, pose.theta, ref),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": self.maxiter, "ftol": 1e-4},
        )
        # SLSQP's bounds are structurally enforced even on non-convergence, but an
        # unconverged solve can still be poor: fall back to last tick's warm-started plan.
        u = result.x if result.success else u0

        shifted = np.roll(u, -2)
        shifted[-2:] = u[-2:]
        self._warm_start = shifted

        return float(u[0]), float(u[1])
