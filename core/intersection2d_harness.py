"""Runs several real IntersectionNavigator instances, one per approach, through one shared
2D intersection, with real circle-to-circle collision checks. See KNOWN_BUGS.md entry 4."""

from dataclasses import dataclass, field

import numpy as np

from core.control.intersection import IntersectionNavigator, IntersectionState, OtherVehicleStatus
from core.control.intersection_geometry import (
    HIGHWAY_VEHICLE_RADIUS,
    TURN_LEAD_RATIO,
    Approach,
    Turn,
    build_turn_path,
    in_conflict_zone,
    is_opposite,
    is_to_the_right,
)


@dataclass(frozen=True)
class VehicleSpec:
    approach: Approach
    lane_offset: float = 3.0  # meters from the road centerline to this approach's lane
    start_distance: float = 100.0  # meters behind the conflict zone at t=0
    initial_speed: float | None = None  # defaults to v_cruise if None
    turn: Turn = "straight"
    turning_radius: float = 4.0  # only used if turn != "straight"; must satisfy
    # TURN_LEAD_RATIO * turning_radius <= conflict_half_width + stop_margin (KNOWN_BUGS.md entry 4)


@dataclass(frozen=True)
class _Route:
    """Precomputed per vehicle: (x, y, theta) at any cumulative distance `d` traveled,
    across up to 3 phases (entry-straight, curve, exit-straight)."""

    exit_approach: Approach
    pre_curve_length: float | None  # None for straight-through (no curve phase at all)
    curve_path: np.ndarray | None
    curve_arc_length: np.ndarray | None
    curve_length: float


def _build_route(spec: VehicleSpec) -> _Route:
    exit_approach, curve_path = build_turn_path(spec.approach, spec.turn, spec.lane_offset, spec.turning_radius)
    if spec.turn == "straight":
        return _Route(exit_approach, None, None, None, 0.0)
    seg = np.hypot(np.diff(curve_path[:, 0]), np.diff(curve_path[:, 1]))
    arc_length = np.concatenate([[0.0], np.cumsum(seg)])
    turn_lead = TURN_LEAD_RATIO * spec.turning_radius
    return _Route(exit_approach, spec.start_distance - turn_lead, curve_path, arc_length, float(arc_length[-1]))


def _pose_at(spec: VehicleSpec, route: _Route, d: float) -> tuple[float, float, float]:
    if spec.turn == "straight":
        x, y = spec.approach.position(d - spec.start_distance, spec.lane_offset)
        return x, y, spec.approach.heading

    if d < route.pre_curve_length:
        x, y = spec.approach.position(d - spec.start_distance, spec.lane_offset)
        return x, y, spec.approach.heading

    if d < route.pre_curve_length + route.curve_length:
        s = d - route.pre_curve_length
        idx = int(np.argmin(np.abs(route.curve_arc_length - s)))
        x, y, theta = route.curve_path[idx]
        return float(x), float(y), float(theta)

    turn_lead = TURN_LEAD_RATIO * spec.turning_radius
    s_past_curve = d - (route.pre_curve_length + route.curve_length)
    x, y = route.exit_approach.position(turn_lead + s_past_curve, spec.lane_offset)
    return x, y, route.exit_approach.heading


def _clear_distance(spec: VehicleSpec, route: _Route, conflict_half_width: float, clear_margin: float) -> float:
    if spec.turn == "straight":
        return spec.start_distance + conflict_half_width + clear_margin
    # A turning vehicle isn't "cleared" until past the curve's real endpoint, and that
    # alone isn't always far enough past the box edge either: take whichever is farther.
    turn_lead = TURN_LEAD_RATIO * spec.turning_radius
    extra_past_curve = max(clear_margin, conflict_half_width + clear_margin - turn_lead)
    return route.pre_curve_length + route.curve_length + extra_past_curve


@dataclass
class VehicleTrace:
    name: str
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    speed: np.ndarray
    states: list[IntersectionState]
    stop_time: float | None
    ran_stop_sign: bool


@dataclass
class MultiIntersectionResult:
    times: np.ndarray
    vehicles: list[VehicleTrace] = field(default_factory=list)
    collided: bool = False
    collision_pairs: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ran_stop_sign(self) -> list[str]:
        return [v.name for v in self.vehicles if v.ran_stop_sign]


def run_multi_approach_scenario(
    specs: list[VehicleSpec],
    conflict_half_width: float = 9.5,  # half-width of the box where the two roads overlap
    stop_margin: float = 1.5,  # stop this far short of the conflict zone's edge
    clear_margin: float = 3.0,  # must be this far past the far edge to count as "cleared"
    v_cruise: float = 15.0,
    dt: float = 0.1,
    max_steps: int = 3000,
    navigators: list[IntersectionNavigator] | None = None,  # override, one per spec, lets tests
    # substitute a non-compliant navigator to verify the collision check is a real safety net.
) -> MultiIntersectionResult:
    # conflict_half_width/stop_margin interact with VehicleSpec.lane_offset and
    # HIGHWAY_VEHICLE_RADIUS; the checks below guard the turning-vehicle geometry (KNOWN_BUGS.md entry 4).
    for spec in specs:
        if spec.turn == "straight":
            continue
        turn_lead = TURN_LEAD_RATIO * spec.turning_radius
        if turn_lead > conflict_half_width + stop_margin:
            raise ValueError(
                f"{spec.approach.name} turn={spec.turn}: turn_lead ({turn_lead:.1f}m, from "
                f"turning_radius={spec.turning_radius}) exceeds conflict_half_width+stop_margin "
                f"({conflict_half_width + stop_margin:.1f}m), a stopped vehicle would already be "
                "mid-curve. Increase conflict_half_width/stop_margin or decrease turning_radius."
            )
        # Also guard against pre_curve_length going negative via too-small start_distance:
        # the same mid-curve-at-t=0 bug, from a different cause.
        if turn_lead >= spec.start_distance - 1.0:
            raise ValueError(
                f"{spec.approach.name} turn={spec.turn}: turn_lead ({turn_lead:.1f}m) leaves less "
                f"than 1m of straight entry lane before start_distance ({spec.start_distance:.1f}m) "
                "-- the vehicle would start already mid-curve. Increase start_distance or decrease "
                "turning_radius."
            )
    n = len(specs)
    if navigators is None:
        navigators = [
            IntersectionNavigator(
                stop_line_position=spec.start_distance - conflict_half_width - stop_margin, v_cruise=v_cruise
            )
            for spec in specs
        ]
    routes = [_build_route(spec) for spec in specs]
    d = [0.0] * n  # distance traveled since t=0, along each vehicle's own route
    v = [spec.initial_speed if spec.initial_speed is not None else v_cruise for spec in specs]
    clear_distance = [_clear_distance(specs[a], routes[a], conflict_half_width, clear_margin) for a in range(n)]
    crossed_center = [False] * n
    ran_stop_sign = [False] * n

    times: list[float] = []
    xs: list[list[float]] = [[] for _ in range(n)]
    ys: list[list[float]] = [[] for _ in range(n)]
    thetas: list[list[float]] = [[] for _ in range(n)]
    speeds: list[list[float]] = [[] for _ in range(n)]
    states: list[list[IntersectionState]] = [[] for _ in range(n)]
    collided = False
    collision_pairs: list[tuple[str, str]] = []

    for step in range(max_steps):
        t = round(step * dt, 4)

        others_per_vehicle = []
        for a in range(n):
            others = []
            for b in range(n):
                if a == b:
                    continue
                # A straight vehicle never yields to an opposing left-turner via arrival-order
                # bookkeeping: real right-of-way gives it unconditional precedence (paired with
                # the phantom rule below); this avoided a circular deadlock (KNOWN_BUGS.md entry 4).
                opposing_left_turner = (
                    specs[a].turn == "straight"
                    and specs[b].turn == "left"
                    and is_opposite(specs[a].approach.heading, specs[b].approach.heading)
                )
                if not opposing_left_turner:
                    others.append(
                        OtherVehicleStatus(
                            stopped=navigators[b].state != IntersectionState.APPROACHING,
                            stop_time=navigators[b].stop_time,
                            cleared=d[b] >= clear_distance[b],
                            is_to_the_right=is_to_the_right(specs[a].approach.heading, specs[b].approach.heading),
                        )
                    )
                # Left turns yield to oncoming straight traffic that hasn't cleared yet,
                # regardless of arrival order: modeled as a phantom "other" guaranteed to
                # have arrived first. Gated on position, not navigator state (KNOWN_BUGS.md entry 4).
                if (
                    specs[a].turn == "left"
                    and specs[b].turn == "straight"
                    and is_opposite(specs[a].approach.heading, specs[b].approach.heading)
                    and d[b] < clear_distance[b]
                ):
                    others.append(OtherVehicleStatus(stopped=True, stop_time=-1e9, cleared=False, is_to_the_right=False))
            others_per_vehicle.append(others)

        accels = [navigators[a].control(d[a], v[a], t, others_per_vehicle[a]) for a in range(n)]

        positions = []
        for a in range(n):
            v[a] = max(0.0, v[a] + accels[a] * dt)
            d[a] += v[a] * dt
            x, y, theta = _pose_at(specs[a], routes[a], d[a])
            xs[a].append(x)
            ys[a].append(y)
            thetas[a].append(theta)
            speeds[a].append(v[a])
            states[a].append(navigators[a].state)
            positions.append((x, y))

            if not crossed_center[a] and d[a] - specs[a].start_distance >= 0.0:
                crossed_center[a] = True
                if navigators[a].stop_time is None:
                    ran_stop_sign[a] = True

        for a in range(n):
            for b in range(a + 1, n):
                if not (in_conflict_zone(*positions[a], conflict_half_width) or in_conflict_zone(*positions[b], conflict_half_width)):
                    continue
                dist = float(np.hypot(positions[a][0] - positions[b][0], positions[a][1] - positions[b][1]))
                if dist < 2 * HIGHWAY_VEHICLE_RADIUS:
                    collided = True
                    pair = (specs[a].approach.name, specs[b].approach.name)
                    if pair not in collision_pairs:
                        collision_pairs.append(pair)

        times.append(t)
        if all(d[a] >= clear_distance[a] for a in range(n)):
            break

    vehicles = [
        VehicleTrace(
            name=specs[a].approach.name,
            x=np.array(xs[a]),
            y=np.array(ys[a]),
            theta=np.array(thetas[a]),
            speed=np.array(speeds[a]),
            states=states[a],
            stop_time=navigators[a].stop_time,
            ran_stop_sign=ran_stop_sign[a],
        )
        for a in range(n)
    ]
    return MultiIntersectionResult(
        times=np.array(times), vehicles=vehicles, collided=collided, collision_pairs=collision_pairs
    )
