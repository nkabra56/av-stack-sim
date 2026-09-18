"""Fixed-time signalized 4-way intersection: straight-through IDM cars that queue and stop on red.
Same 2D geometry as intersection2d_harness; no turns, pedestrians, or actuated signals."""

from dataclasses import dataclass, field

import numpy as np

from core.control.acc import IDMController
from core.control.intersection_geometry import HIGHWAY_VEHICLE_RADIUS, Approach, in_conflict_zone

GREEN, YELLOW, RED = 0, 1, 2
CAR_LENGTH = 4.5


@dataclass(frozen=True)
class SignalPlan:
    """North-south goes green first, then east-west; every switch passes through yellow and all-red."""

    green: float = 14.0
    yellow: float = 3.0
    all_red: float = 2.0

    @property
    def cycle(self) -> float:
        return 2 * (self.green + self.yellow + self.all_red)

    def state(self, t: float, axis: str) -> int:
        u = (t - (0.0 if axis == "NS" else self.cycle / 2)) % self.cycle
        if u < self.green:
            return GREEN
        return YELLOW if u < self.green + self.yellow else RED

    def yellow_remaining(self, t: float, axis: str) -> float:
        u = (t - (0.0 if axis == "NS" else self.cycle / 2)) % self.cycle
        return self.green + self.yellow - u


@dataclass(frozen=True)
class SignalCar:
    approach: Approach
    spawn_time: float
    v0: float = 13.0  # desired speed, m/s
    lane_offset: float = 3.0


@dataclass
class CarTrace:
    approach: str
    active: np.ndarray  # (N,) bool, False before spawn and after exit
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    speed: np.ndarray
    line_time: float | None = None  # when the front bumper crossed the stop line
    line_signal: int | None = None  # the signal state at that moment


@dataclass
class SignalizedResult:
    times: np.ndarray
    cars: list[CarTrace]
    ns_state: np.ndarray
    ew_state: np.ndarray
    collided: bool = False
    red_entries: list[int] = field(default_factory=list)
    min_gap: float = float("inf")


def _axis(approach: Approach) -> str:
    return "NS" if approach.name in ("N", "S") else "EW"


def run_signalized_scenario(
    cars: list[SignalCar],
    plan: SignalPlan | None = None,
    duration: float = 90.0,
    dt: float = 0.1,
    start_distance: float = 90.0,  # meters from spawn to the intersection center
    half: float = 9.5,  # conflict-zone half width
    stop_margin: float = 1.5,
    b_max: float = 4.5,  # hardest braking a car will choose to stop for a yellow
) -> SignalizedResult:
    plan = plan or SignalPlan()
    n_cars, n = len(cars), round(duration / dt)
    line = start_distance - half - stop_margin  # stop line, in distance from spawn
    exit_at = start_distance + 60.0
    idm = [IDMController(v0=c.v0, a_max=2.0, b_comfortable=2.5, time_headway=1.2) for c in cars]
    d = np.zeros(n_cars)  # front-bumper distance from spawn
    v = np.array([c.v0 for c in cars], dtype=float)
    state = ["waiting"] * n_cars  # waiting -> active -> gone
    trace = [CarTrace(c.approach.name, np.zeros(n, bool), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n))
             for c in cars]
    ns_state, ew_state = np.zeros(n, int), np.zeros(n, int)
    result = SignalizedResult(np.arange(n) * dt, trace, ns_state, ew_state)

    def pose(i: int, front: float) -> tuple[float, float]:
        return cars[i].approach.position(front - CAR_LENGTH / 2 - start_distance, cars[i].lane_offset)

    for k in range(n):
        t = k * dt
        ns_state[k], ew_state[k] = plan.state(t, "NS"), plan.state(t, "EW")
        for i, c in enumerate(cars):
            if state[i] == "waiting" and t >= c.spawn_time:
                state[i] = "active"
        active = [i for i in range(n_cars) if state[i] == "active"]

        accel = {}
        for i in active:
            lead = min((j for j in active if j != i and cars[j].approach.name == cars[i].approach.name and d[j] > d[i]),
                       key=lambda j: d[j], default=None)
            gap, lead_v = (d[lead] - CAR_LENGTH - d[i], v[lead]) if lead is not None else (1e3, v[i])
            a = idm[i].control(v[i], gap, lead_v)
            if d[i] < line:
                dist, sig = line - d[i], plan.state(t, _axis(cars[i].approach))
                if sig != GREEN:
                    needs = v[i] ** 2 / (2 * max(dist, 1e-3))
                    reaches = v[i] > 0.1 and dist / v[i] <= plan.yellow_remaining(t, _axis(cars[i].approach))
                    go = (sig == YELLOW and (reaches or needs > b_max)) or (sig == RED and needs > b_max)
                    if not go:
                        a = min(a, idm[i].control(v[i], max(dist, 0.01), 0.0))
            accel[i] = a

        for i in active:
            v[i] = max(0.0, v[i] + accel[i] * dt)
            was = d[i]
            d[i] += v[i] * dt
            if was < line <= d[i]:
                trace[i].line_time, trace[i].line_signal = t, plan.state(t, _axis(cars[i].approach))
                if trace[i].line_signal == RED:
                    result.red_entries.append(i)
            if d[i] - CAR_LENGTH / 2 > exit_at:
                state[i] = "gone"

        for i in range(n_cars):
            live = state[i] == "active"
            x, y = pose(i, d[i] if live else CAR_LENGTH)
            trace[i].active[k], trace[i].x[k], trace[i].y[k] = live, x, y
            trace[i].theta[k], trace[i].speed[k] = cars[i].approach.heading, v[i] if live else 0.0

        live_now = [i for i in range(n_cars) if trace[i].active[k]]
        for a_i, a in enumerate(live_now):
            for b in live_now[a_i + 1:]:
                if cars[a].approach.name == cars[b].approach.name:
                    result.min_gap = min(result.min_gap, abs(d[a] - d[b]) - CAR_LENGTH)
                    continue
                pa, pb = (trace[a].x[k], trace[a].y[k]), (trace[b].x[k], trace[b].y[k])
                if (in_conflict_zone(*pa, half) or in_conflict_zone(*pb, half)) and (
                    np.hypot(pa[0] - pb[0], pa[1] - pb[1]) < 2 * HIGHWAY_VEHICLE_RADIUS
                ):
                    result.collided = True
    return result


def demo_cars() -> list[SignalCar]:
    """A busy two-cycle scenario: cross-street cars arrive during red, queue, then discharge on green."""
    from core.control.intersection_geometry import EAST, NORTH, SOUTH, WEST

    spawns = {
        NORTH: [0.0, 3.5, 7.0, 25.0, 28.5, 32.0, 50.0],
        SOUTH: [1.5, 5.0, 26.5, 30.0, 33.5, 52.0],
        EAST: [4.0, 7.5, 11.0, 14.0, 40.0, 44.0],
        WEST: [2.5, 8.5, 12.5, 15.5, 42.0, 46.0],
    }
    speeds = [13.0, 12.0, 14.0, 13.5]
    cars = [SignalCar(a, t, v0=speeds[i % 4]) for a, times in spawns.items() for i, t in enumerate(times)]
    return sorted(cars, key=lambda c: c.spawn_time)
