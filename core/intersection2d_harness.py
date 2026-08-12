"""Runs several real vehicles -- each an unmodified `IntersectionNavigator` instance
(control/intersection.py), one per approach, same reuse principle H4 itself already used
for `IDMController` -- through one shared, real 2D intersection. See KNOWN_BUGS.md entry
4 / intersection_geometry.py's module docstring: `intersection_harness.py` (H4's own
standalone harness) only ever drives one ego against a single *scripted*
`OtherVehicleStatus`, never real multi-vehicle 2D geometry -- this is what actually
closes that gap, by deriving every `OtherVehicleStatus` other vehicles see from another
vehicle's own real, independently-simulated navigator and position instead of a hand-
authored script, and checking real circle-to-circle proximity between vehicles' actual
(x, y) positions instead of trusting the arrival-order bookkeeping alone.

**Turning movements**: a vehicle with `turn != "straight"` drives its entry approach's
straight lane, then a real curved connector (`intersection_geometry.build_turn_path`),
then the exit approach's straight lane -- one continuous route parametrized by
cumulative distance traveled, walked exactly like a straight-through vehicle's simpler
one-phase route. Left turns additionally yield to oncoming (opposite-approach)
straight-through traffic that hasn't cleared yet, regardless of arrival order -- the one
real-world right-of-way rule `IntersectionNavigator`'s arrival-order model can't express
on its own, added here as an extra `OtherVehicleStatus` entry with a guaranteed-earliest
`stop_time` rather than by modifying `IntersectionNavigator` itself (same "compose, don't
modify a validated component" principle the rest of this project already follows).
"""

from dataclasses import dataclass, field

import numpy as np

from core.control.intersection import IntersectionNavigator, IntersectionState, OtherVehicleStatus
from core.control.intersection_geometry import (
    TURN_LEAD_RATIO,
    HIGHWAY_VEHICLE_RADIUS,
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
    turning_radius: float = 4.0  # only used if turn != "straight" -- see build_turn_path. Must
    # satisfy TURN_LEAD_RATIO * turning_radius <= conflict_half_width + stop_margin (see
    # run_multi_approach_scenario) or a stopped turning vehicle ends up geometrically mid-curve
    # instead of on its straight entry lane -- KNOWN_BUGS.md entry 4.


@dataclass(frozen=True)
class _Route:
    """Precomputed once per vehicle at scenario setup: where its (x, y, theta) is for
    any cumulative distance `d` it's traveled, in up to 3 phases (entry-straight,
    curve, exit-straight -- straight-through vehicles only ever use phase 1)."""

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
    # The curve is genuinely longer than the straight-line chord it replaces (see
    # intersection_geometry.py's TURN_LEAD_RATIO comment), so the straight-through
    # formula above would declare a turning vehicle "cleared" before it's actually
    # finished the curve -- use the curve's own real endpoint instead.
    #
    # But the curve's endpoint alone isn't necessarily far enough: at that point the
    # vehicle is only `turn_lead` past the conflict-zone *center* along the exit
    # approach, not necessarily past the box's edge (`conflict_half_width`) at all --
    # for a small enough turning_radius, `turn_lead < conflict_half_width` and the
    # curve's own endpoint still sits inside the box. Require the same real physical
    # margin past the box's edge the straight-through formula requires
    # (`conflict_half_width + clear_margin`, measured from center along the exit
    # direction), not just "finished the curve" -- whichever is farther.
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
    navigators: list[IntersectionNavigator] | None = None,  # override, one per spec -- lets a
    # test substitute a deliberately non-compliant navigator (e.g. one that never yields) to
    # verify the collision check below is a real safety net and not vacuously true. Production
    # callers never pass this; every real vehicle gets an unmodified IntersectionNavigator.
) -> MultiIntersectionResult:
    # `conflict_half_width`/`stop_margin` vs. `VehicleSpec.lane_offset`/`HIGHWAY_VEHICLE_RADIUS`
    # aren't independent: a vehicle waiting at its own stop line has to clear the *perpendicular*
    # approach's through-lane by at least 2*HIGHWAY_VEHICLE_RADIUS, not just be outside the box.
    # Closest approach (verified against a real simulated run while building this) is
    # `conflict_half_width + stop_margin + IntersectionNavigator's own stop_gap(1.0) -
    # lane_offset`; the defaults above keep that at 6.5m against the default 3.0m `lane_offset`
    # and 2.5m `HIGHWAY_VEHICLE_RADIUS` (5.0m needed) -- a real, physical spacing requirement a
    # single-conflict-point model never had to reason about at all.
    #
    # A second, independent constraint applies to turning vehicles: `_build_route` starts the
    # curved connector `turn_lead = TURN_LEAD_RATIO * turning_radius` before the conflict zone,
    # while `IntersectionNavigator`'s 1D stop line sits at `conflict_half_width + stop_margin`
    # before it. If the curve starts earlier (further back) than the stop line, a *stopped*
    # turning vehicle is already partway around the curve instead of cleanly on its straight
    # entry lane -- a real bug found via a random mixed-turn sweep (KNOWN_BUGS.md entry 4) that
    # produced a nonsensical waiting position and a resulting collision. Guard against silently
    # reintroducing it.
    for spec in specs:
        if spec.turn == "straight":
            continue
        turn_lead = TURN_LEAD_RATIO * spec.turning_radius
        if turn_lead > conflict_half_width + stop_margin:
            raise ValueError(
                f"{spec.approach.name} turn={spec.turn}: turn_lead ({turn_lead:.1f}m, from "
                f"turning_radius={spec.turning_radius}) exceeds conflict_half_width+stop_margin "
                f"({conflict_half_width + stop_margin:.1f}m) -- a stopped vehicle would already be "
                "mid-curve. Increase conflict_half_width/stop_margin or decrease turning_radius."
            )
        # The check above only relates turn_lead to the *box*, not to this vehicle's own
        # start_distance -- `_build_route`'s `pre_curve_length = start_distance -
        # turn_lead` can still go negative (or non-positive) if start_distance itself is
        # too small, which reproduces the identical mid-curve-at-t=0 bug the check above
        # exists to prevent, just via a different dimension. Require at least a nominal
        # 1m of real straight lane before the curve begins.
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
                # A straight vehicle never yields to an *opposing* left-turner via ordinary
                # arrival-order bookkeeping -- real-world right-of-way already gives it
                # unconditional precedence over that specific pairing (see the
                # phantom-blocker comment below). Omitting b's real status here (rather
                # than including it and letting arrival order decide) is what makes that
                # one-directional: without this, a left-turner that happened to arrive
                # first would make the straight vehicle yield to *it* via real
                # arrival-order, while the phantom rule below simultaneously makes the
                # left-turner yield to the straight vehicle -- a live circular deadlock
                # found via a random mixed-turn sweep (KNOWN_BUGS.md entry 4), fixed by
                # ensuring the relation between an opposing left/straight pair can only
                # ever constrain the left-turner, never the straight vehicle. Deliberately
                # NOT extended to a=="right": the phantom rule below only ever fires for
                # b=="straight", so a right-turner has no substitute protection against an
                # opposing left-turner -- exempting it too (tried first) reproduced real
                # collisions in the same sweep, since it left that pairing with no mutual-
                # exclusion mechanism at all. Right-turners keep the normal, unmodified,
                # already-deadlock-safe arrival-order relation against every other vehicle.
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
                # Left turns yield to oncoming (opposite-approach) straight-through
                # traffic that hasn't cleared yet, regardless of arrival order -- a real
                # right-of-way rule IntersectionNavigator's own arrival-order model has no
                # way to express on its own. Modeled as an extra phantom "other"
                # guaranteed to count as having arrived first (stop_time far in the
                # past), rather than by changing IntersectionNavigator itself.
                #
                # Gated on `d[b] < clear_distance[b]` alone -- deliberately NOT also
                # requiring `navigators[b].state != APPROACHING` (tried first). That
                # extra gate let a left-turner treat a still-APPROACHING oncoming vehicle
                # as no threat and launch immediately, even though "still approaching"
                # only means "hasn't reached its own stop line yet," not "far away" --
                # it can be moving at full v_cruise and reach the conflict zone well
                # before the left-turner finishes crossing it. Found via the same random
                # mixed-turn sweep (KNOWN_BUGS.md entry 4): both vehicles were already
                # PROCEEDING when they collided, because the left-turner's stop_time
                # arrived before the oncoming vehicle had even stopped, so the state gate
                # let it go. Position (`d[b]` vs. `clear_distance[b]`) is what actually
                # matters, not the oncoming vehicle's own state-machine phase; this is
                # conservative (a left-turner waits out the oncoming vehicle's entire
                # approach, not just its time in the box) but that trade favors safety
                # over throughput, consistent with the rest of this project.
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
