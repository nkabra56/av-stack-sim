"""Adaptive cruise control: two longitudinal controllers, the same classical-vs-optimization
comparison used for parking (Pure Pursuit vs. MPC), applied to car-following. See DESIGN.md's
ACC section. Both share `control(ego_speed, gap, lead_speed) -> accel`, gap bumper-to-bumper."""

import numpy as np
from scipy.optimize import minimize


class IDMController:
    """Intelligent Driver Model (Treiber, Hennecke & Helbing, 2000) -- literature-standard,
    closed-form car-following law. Default parameters are from the original paper / SUMO."""

    def __init__(
        self,
        v0: float = 30.0,  # desired free-flow speed, m/s (~108 km/h)
        a_max: float = 1.5,  # max acceleration, m/s^2
        b_comfortable: float = 2.0,  # comfortable braking deceleration, m/s^2
        s0: float = 2.0,  # minimum standstill gap, m
        time_headway: float = 1.5,  # desired time headway, s
        delta: float = 4.0,  # acceleration exponent (standard IDM value)
        a_min: float = -9.0,  # physical emergency-braking floor, ~1g
    ):
        self.v0 = v0
        self.a_max = a_max
        self.b = b_comfortable
        self.s0 = s0
        self.time_headway = time_headway
        self.delta = delta
        self.a_min = a_min

    def control(self, ego_speed: float, gap: float, lead_speed: float) -> float:
        # The raw IDM term is unbounded as gap -> 0; clip to what a real actuator can do.
        gap = max(gap, 1e-3)
        closing_speed = ego_speed - lead_speed
        s_star = self.s0 + max(
            0.0, ego_speed * self.time_headway + (ego_speed * closing_speed) / (2 * np.sqrt(self.a_max * self.b))
        )
        accel = self.a_max * (1 - (ego_speed / self.v0) ** self.delta - (s_star / gap) ** 2)
        return float(np.clip(accel, self.a_min, self.a_max))


class MpcAccController:
    """Short-horizon MPC for ACC. Unlike the parking MPC (box bounds only), this adds a
    hard nonlinear gap(t) >= min_gap constraint via SLSQP, re-solved every tick against a
    constant-velocity lead prediction from the latest radar reading."""

    def __init__(
        self,
        dt: float = 0.1,
        horizon: int = 10,
        v0: float = 30.0,
        a_max: float = 1.5,
        a_min: float = -9.0,  # matches IDMController's emergency-braking floor (~1g)
        min_gap: float = 3.0,  # extra cushion above the more natural 2.0m, against sensor noise
        time_headway: float = 1.5,
        w_speed: float = 1.0,
        w_gap: float = 1.0,
        w_effort: float = 0.05,
        w_jerk: float = 0.1,
        maxiter: int = 30,
    ):
        self.dt = dt
        self.horizon = horizon
        self.v0 = v0
        self.a_max = a_max
        self.a_min = a_min
        self.min_gap = min_gap
        self.time_headway = time_headway
        self.w_speed = w_speed
        self.w_gap = w_gap
        self.w_effort = w_effort
        self.w_jerk = w_jerk
        self.maxiter = maxiter
        self._warm_start: np.ndarray | None = None

    def _rollout(self, ego_speed: float, lead_positions: np.ndarray, a_seq: np.ndarray):
        speeds = np.empty(self.horizon)
        positions = np.empty(self.horizon)
        speed, pos = ego_speed, 0.0
        for k in range(self.horizon):
            speed = max(0.0, speed + a_seq[k] * self.dt)
            pos = pos + speed * self.dt
            speeds[k], positions[k] = speed, pos
        gaps = lead_positions - positions
        return speeds, gaps

    def _cost(self, a_seq: np.ndarray, ego_speed: float, lead_positions: np.ndarray) -> float:
        speeds, gaps = self._rollout(ego_speed, lead_positions, a_seq)
        desired_gap = self.min_gap + self.time_headway * speeds
        cost = self.w_gap * np.sum((gaps - desired_gap) ** 2) + self.w_speed * np.sum((speeds - self.v0) ** 2)
        cost += self.w_effort * np.sum(a_seq**2)
        cost += self.w_jerk * np.sum(np.diff(a_seq) ** 2)
        return cost

    def _effective_min_gap(self, ego_speed: float, lead_positions: np.ndarray) -> np.ndarray:
        """Per-step gap floor that's always achievable (braking at a_min from now), so the
        SLSQP constraint stays feasible instead of silently violated. See KNOWN_BUGS.md entry 1."""
        _, floor = self._rollout(ego_speed, lead_positions, np.full(self.horizon, self.a_min))
        return np.minimum(self.min_gap, floor)

    def _gap_constraint(
        self, a_seq: np.ndarray, ego_speed: float, lead_positions: np.ndarray, effective_min_gap: np.ndarray
    ) -> np.ndarray:
        _, gaps = self._rollout(ego_speed, lead_positions, a_seq)
        return gaps - effective_min_gap  # scipy 'ineq': feasible when >= 0

    def control(self, ego_speed: float, gap: float, lead_speed: float) -> float:
        lead_positions = gap + lead_speed * self.dt * np.arange(1, self.horizon + 1)
        effective_min_gap = self._effective_min_gap(ego_speed, lead_positions)
        u0 = self._warm_start if self._warm_start is not None else np.zeros(self.horizon)
        bounds = [(self.a_min, self.a_max)] * self.horizon
        constraints = [
            {"type": "ineq", "fun": self._gap_constraint, "args": (ego_speed, lead_positions, effective_min_gap)}
        ]

        result = minimize(
            self._cost,
            u0,
            args=(ego_speed, lead_positions),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.maxiter, "ftol": 1e-4},
        )
        a_seq = result.x

        # SLSQP can silently return a constraint-violating point on non-convergence; verify
        # directly and fall back to braking at a_min (an always-feasible witness) if so.
        if np.any(self._gap_constraint(a_seq, ego_speed, lead_positions, effective_min_gap) < -1e-3):
            a_seq = np.full(self.horizon, self.a_min)

        shifted = np.roll(a_seq, -1)
        shifted[-1] = a_seq[-1]
        self._warm_start = shifted

        return float(a_seq[0])
