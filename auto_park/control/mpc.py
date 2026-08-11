"""Short-horizon nonlinear MPC path tracker. See DESIGN.md section 6.

Each call optimizes a control sequence (v_k, delta_k) for k=0..horizon-1 by rolling
out the (nonlinear) kinematic bicycle model and minimizing tracking error against a
reference slice of the path, plus control-effort and smoothness penalties, subject to
speed/steering bounds -- a direct-shooting nonlinear program solved with SLSQP. Only
the first control of the optimized sequence is applied (receding horizon); the next
call re-optimizes from the new state, warm-started from the previous solution shifted
by one step.

Rolling out the true nonlinear model (rather than linearizing around the reference,
as a linear MPC would) avoids deriving/maintaining a Jacobian, at the cost of a
slightly more expensive per-step solve -- an acceptable tradeoff at this horizon
length (6-8 steps) and control rate (10 Hz).
"""

import numpy as np
from scipy.optimize import minimize

from auto_park.vehicle import Vehicle, wrap_angle


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

    def _reference(self, vehicle: Vehicle, path: np.ndarray) -> np.ndarray:
        dists = np.hypot(path[:, 0] - vehicle.x, path[:, 1] - vehicle.y)
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

    def control(self, vehicle: Vehicle, path: np.ndarray) -> tuple[float, float]:
        ref = self._reference(vehicle, path)
        u0 = self._warm_start if self._warm_start is not None else np.zeros(2 * self.horizon)
        bounds = [(-self.v_max, self.v_max), (-self.delta_max, self.delta_max)] * self.horizon

        result = minimize(
            self._cost,
            u0,
            args=(vehicle.x, vehicle.y, vehicle.theta, ref),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": self.maxiter, "ftol": 1e-4},
        )
        u = result.x

        shifted = np.roll(u, -2)
        shifted[-2:] = u[-2:]
        self._warm_start = shifted

        return float(u[0]), float(u[1])
