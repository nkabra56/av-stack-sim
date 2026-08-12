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
"""

from dataclasses import dataclass, field

import numpy as np

from core.control.intersection import IntersectionNavigator, IntersectionState, OtherVehicleStatus
from core.control.intersection_geometry import VEHICLE_RADIUS, Approach, in_conflict_zone, is_to_the_right


@dataclass(frozen=True)
class VehicleSpec:
    approach: Approach
    lane_offset: float = 3.0  # meters from the road centerline to this approach's lane
    start_distance: float = 100.0  # meters behind the conflict zone at t=0
    initial_speed: float | None = None  # defaults to v_cruise if None


@dataclass
class VehicleTrace:
    name: str
    x: np.ndarray
    y: np.ndarray
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
    conflict_half_width: float = 7.0,  # half-width of the box where the two roads overlap
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
    # `conflict_half_width`/`stop_margin` vs. `VehicleSpec.lane_offset`/`VEHICLE_RADIUS` aren't
    # independent: a vehicle waiting at its own stop line has to clear the *perpendicular*
    # approach's through-lane by at least 2*VEHICLE_RADIUS, not just be outside the box. Closest
    # approach (verified against a real simulated run while building this) is
    # `conflict_half_width + stop_margin + IntersectionNavigator's own stop_gap(1.0) -
    # lane_offset`; the defaults above keep that at 6.5m against the default 3.0m `lane_offset`
    # and 2.5m `VEHICLE_RADIUS` (5.0m needed) -- a real, physical spacing requirement a single-
    # conflict-point model never had to reason about at all.
    n = len(specs)
    if navigators is None:
        navigators = [
            IntersectionNavigator(
                stop_line_position=spec.start_distance - conflict_half_width - stop_margin, v_cruise=v_cruise
            )
            for spec in specs
        ]
    d = [0.0] * n  # distance traveled since t=0, along each vehicle's own approach
    v = [spec.initial_speed if spec.initial_speed is not None else v_cruise for spec in specs]
    clear_distance = [spec.start_distance + conflict_half_width + clear_margin for spec in specs]
    crossed_center = [False] * n
    ran_stop_sign = [False] * n

    times: list[float] = []
    xs: list[list[float]] = [[] for _ in range(n)]
    ys: list[list[float]] = [[] for _ in range(n)]
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
                others.append(
                    OtherVehicleStatus(
                        stopped=navigators[b].state != IntersectionState.APPROACHING,
                        stop_time=navigators[b].stop_time,
                        cleared=d[b] >= clear_distance[b],
                        is_to_the_right=is_to_the_right(specs[a].approach.heading, specs[b].approach.heading),
                    )
                )
            others_per_vehicle.append(others)

        accels = [navigators[a].control(d[a], v[a], t, others_per_vehicle[a]) for a in range(n)]

        positions = []
        for a in range(n):
            v[a] = max(0.0, v[a] + accels[a] * dt)
            d[a] += v[a] * dt
            longitudinal = d[a] - specs[a].start_distance
            x, y = specs[a].approach.position(longitudinal, specs[a].lane_offset)
            xs[a].append(x)
            ys[a].append(y)
            speeds[a].append(v[a])
            states[a].append(navigators[a].state)
            positions.append((x, y))

            if not crossed_center[a] and longitudinal >= 0.0:
                crossed_center[a] = True
                if navigators[a].stop_time is None:
                    ran_stop_sign[a] = True

        for a in range(n):
            for b in range(a + 1, n):
                if not (in_conflict_zone(*positions[a], conflict_half_width) or in_conflict_zone(*positions[b], conflict_half_width)):
                    continue
                dist = float(np.hypot(positions[a][0] - positions[b][0], positions[a][1] - positions[b][1]))
                if dist < 2 * VEHICLE_RADIUS:
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
