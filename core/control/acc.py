"""Adaptive cruise control: two longitudinal controllers, the same "classical/reactive
vs. optimization-based" comparison used for parking (Pure Pursuit vs. MPC) and applied
here to car-following. See DESIGN.md's ACC section.

Both share the call signature `control(ego_speed, gap, lead_speed) -> accel` -- gap is
the bumper-to-bumper distance to the lead vehicle (meters), not center-to-center.
"""

import numpy as np
from scipy.optimize import minimize


class IDMController:
    """Intelligent Driver Model (Treiber, Hennecke & Helbing, 2000) -- the
    literature-standard car-following law, closed-form and reactive like Pure Pursuit
    was for parking. Reference parameter ranges (v0, a_max, b, s0, time_headway) are
    from the original paper and widely-used traffic-simulation defaults (e.g. SUMO's),
    not hand-tuned for this project specifically.
    """

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
        # The raw IDM interaction term (s*/gap)^2 is unbounded as gap -> 0 -- a real car
        # can't actually decelerate at whatever multiple of a_max that implies, so the
        # output has to be clipped to a physical limit, same as any other actuator.
        gap = max(gap, 1e-3)
        closing_speed = ego_speed - lead_speed
        s_star = self.s0 + max(
            0.0, ego_speed * self.time_headway + (ego_speed * closing_speed) / (2 * np.sqrt(self.a_max * self.b))
        )
        accel = self.a_max * (1 - (ego_speed / self.v0) ** self.delta - (s_star / gap) ** 2)
        return float(np.clip(accel, self.a_min, self.a_max))


class MpcAccController:
    """Short-horizon MPC for ACC. Unlike control/mpc.py's parking MPC, which only uses
    box bounds on the controls, this adds a genuine nonlinear inequality constraint
    (gap(t) >= min_gap at every step in the horizon, via scipy.optimize.minimize's
    `constraints` argument) -- a hard safety constraint enforced by the optimizer
    itself, not a soft cost penalty that can be traded off against tracking accuracy.

    The lead vehicle is assumed to hold constant velocity over the horizon -- the
    standard simplifying prediction used in real ACC/MPC literature; re-solved every
    tick from the latest radar reading, so it's continuously corrected, not a
    long-range forecast.

    `min_gap` default is 3.0 m, not the more natural-looking 2.0 m, because of a real
    finding from validating against NGSIM (real congested stop-and-go traffic, see
    DESIGN.md's ACC section): if the ego ever ends up closer than `min_gap` while both
    vehicles are stopped -- which can happen during the approach to a standstill,
    since the ego can't reverse -- there is no feasible acceleration sequence that
    satisfies the constraint from there (moving apart from a standstill would require
    driving backward), so the constrained optimization becomes locally infeasible and
    SLSQP silently returns its best constraint-violating attempt rather than failing
    loudly. This is a genuine limitation of *nominal* (non-robust) MPC under model
    mismatch, not a bug to paper over: across a range of `min_gap` values the realized
    minimum gap consistently landed about 0.5 m below the nominal target, so 3.0 m
    keeps the realized worst case comfortably positive (~2.5 m) rather than picking a
    value that looks right on paper but erodes to a near-miss in practice.
    """

    def __init__(
        self,
        dt: float = 0.1,
        horizon: int = 10,
        v0: float = 30.0,
        a_max: float = 1.5,
        a_min: float = -3.0,
        min_gap: float = 3.0,
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

    def _gap_constraint(self, a_seq: np.ndarray, ego_speed: float, lead_positions: np.ndarray) -> np.ndarray:
        _, gaps = self._rollout(ego_speed, lead_positions, a_seq)
        return gaps - self.min_gap  # scipy 'ineq': feasible when >= 0

    def control(self, ego_speed: float, gap: float, lead_speed: float) -> float:
        lead_positions = gap + lead_speed * self.dt * np.arange(1, self.horizon + 1)
        u0 = self._warm_start if self._warm_start is not None else np.zeros(self.horizon)
        bounds = [(self.a_min, self.a_max)] * self.horizon
        constraints = [{"type": "ineq", "fun": self._gap_constraint, "args": (ego_speed, lead_positions)}]

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

        shifted = np.roll(a_seq, -1)
        shifted[-1] = a_seq[-1]
        self._warm_start = shifted

        return float(a_seq[0])
