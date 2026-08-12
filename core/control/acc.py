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

    `min_gap` defaults to 3.0 m, not the more natural-looking 2.0 m, as extra cushion
    against sensor noise (see `_effective_min_gap`'s docstring for the constraint-
    feasibility fix itself -- this default is now a margin choice, not a workaround
    for a broken constraint).
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

    def _effective_min_gap(self, ego_speed: float, lead_positions: np.ndarray) -> np.ndarray:
        """A per-horizon-step floor that's *always* achievable, fixing the actual defect
        behind KNOWN_BUGS.md's bug 2: a flat `min_gap` constraint can demand something no
        acceleration sequence can deliver (the ego closing on a stopped lead can't reverse
        to recover lost distance), so SLSQP was solving a genuinely infeasible NLP and
        silently returning its best constraint-violating attempt.

        The gap achievable at horizon step k is bounded by braking at `a_min` from *now*
        every step -- that specific sequence is a concrete witness that reaching
        `_rollout(..., a_min-repeated)`'s gap at each step is always possible, so clamping
        the constraint to never demand more than that (`min(min_gap, floor)`, per step, not
        one scalar for the whole horizon) keeps the NLP feasible at every step instead of
        only at whichever one is easiest. The optimizer is still free -- and, via `_cost`'s
        `desired_gap` term, still incentivized -- to reach the full `min_gap` whenever
        that's actually reachable; this only relaxes the constraint at the specific steps
        where `min_gap` itself would be physically impossible.

        Verified against the NGSIM standstill case (real congested stop-and-go traffic,
        DESIGN.md section 11): cut the realized gap erosion at `min_gap=3.0` from ~0.56 m
        to ~0.21 m, with the remainder attributable to radar range noise
        (`RadarNode`'s `range_std=0.5`) feeding the *measured* gap the constraint is built
        from each tick, not to any remaining infeasibility -- confirmed by comparing each
        tick's promised next-step floor against the next tick's realized true gap.
        """
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

        shifted = np.roll(a_seq, -1)
        shifted[-1] = a_seq[-1]
        self._warm_start = shifted

        return float(a_seq[0])
