"""Runs an IntersectionNavigator against a scripted other-vehicle scenario. See
DESIGN.md section 12's H4 entry.

Simpler than harness.py/highway_harness.py's pub/sub node structure on purpose: there's
no sensor noise or multi-component estimation happening here (the point of H4 is the
right-of-way reasoning, not sensor fusion), so a direct simulation loop is honest about
what's actually being tested, the same call made for H3's lane-centering validation.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from core.control.intersection import IntersectionNavigator, IntersectionState, OtherVehicleStatus

OtherVehicleScript = Callable[[float], OtherVehicleStatus]


@dataclass
class IntersectionResult:
    times: np.ndarray
    ego_position: np.ndarray
    ego_speed: np.ndarray
    states: list[IntersectionState]
    ego_stop_time: float | None
    ran_stop_sign: bool  # crossed the stop line above the stop-speed threshold, never having stopped
    proceed_time: float | None = field(default=None)  # first tick state became PROCEEDING


def run_intersection_scenario(
    other_vehicle_script: OtherVehicleScript,
    stop_line_position: float = 100.0,
    v_cruise: float = 15.0,
    ego_initial_speed: float | None = None,
    dt: float = 0.1,
    max_steps: int = 2000,
) -> IntersectionResult:
    nav = IntersectionNavigator(stop_line_position=stop_line_position, v_cruise=v_cruise)
    x, v = 0.0, ego_initial_speed if ego_initial_speed is not None else v_cruise

    times, xs, vs, states = [], [], [], []
    ran_stop_sign = False
    proceed_time = None
    crossed_line = False

    for i in range(max_steps):
        t = round(i * dt, 4)
        others = [other_vehicle_script(t)]
        accel = nav.control(x, v, t, others)
        v = max(0.0, v + accel * dt)
        x += v * dt

        if not crossed_line and x >= stop_line_position:
            crossed_line = True
            if nav.stop_time is None:
                ran_stop_sign = True
        if proceed_time is None and nav.state == IntersectionState.PROCEEDING:
            proceed_time = t

        times.append(t)
        xs.append(x)
        vs.append(v)
        states.append(nav.state)

        if crossed_line and x > stop_line_position + 30.0:
            break

    return IntersectionResult(
        times=np.array(times),
        ego_position=np.array(xs),
        ego_speed=np.array(vs),
        states=states,
        ego_stop_time=nav.stop_time,
        ran_stop_sign=ran_stop_sign,
        proceed_time=proceed_time,
    )


def no_other_vehicle(_t: float) -> OtherVehicleStatus:
    return OtherVehicleStatus(stopped=False, stop_time=None, cleared=False, is_to_the_right=False)


def other_vehicle_present_from(
    arrival_time: float, clear_time: float, is_to_the_right: bool = False
) -> OtherVehicleScript:
    def script(t: float) -> OtherVehicleStatus:
        if t < arrival_time:
            return OtherVehicleStatus(stopped=False, stop_time=None, cleared=False, is_to_the_right=is_to_the_right)
        if t < clear_time:
            return OtherVehicleStatus(
                stopped=True, stop_time=arrival_time, cleared=False, is_to_the_right=is_to_the_right
            )
        return OtherVehicleStatus(stopped=True, stop_time=arrival_time, cleared=True, is_to_the_right=is_to_the_right)

    return script
