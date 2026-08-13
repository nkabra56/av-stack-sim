"""Direct convergence unit coverage for the two path-tracking controllers, decoupled
from a real planner -- given a straight-line path and no obstacles, does each
controller's closed loop actually reach the goal (see IMPLEMENTATION.md section 4).
test_simulation.py already covers this end-to-end (planner + controller + noisy
estimation, across real scenarios); this isolates the controller alone against a
trivial, hand-built path, so a controller regression shows up here without needing a
planner or the estimation stack to also be working."""

import numpy as np
import pytest

from core.control.mpc import MPCController
from core.control.pure_pursuit import PurePursuitAdaptive
from core.vehicle import Vehicle

WHEELBASE = 2.7
GOAL = (20.0, 0.0)
TOL = 0.4
MAX_STEPS = 400
DT = 0.1

CONTROLLERS = {
    "pure_pursuit": lambda: PurePursuitAdaptive(wheelbase=WHEELBASE, v_max=1.5, max_steer=0.6),
    "mpc": lambda: MPCController(wheelbase=WHEELBASE, v_max=1.5, delta_max=0.6),
}


def _straight_path() -> np.ndarray:
    xs = np.linspace(0.0, GOAL[0], 400)
    return np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])


def _run(controller_name: str, start_y: float) -> Vehicle:
    controller = CONTROLLERS[controller_name]()
    path = _straight_path()
    vehicle = Vehicle(x=0.0, y=start_y, theta=0.0, wheelbase=WHEELBASE)
    for _ in range(MAX_STEPS):
        v, delta = controller.control(vehicle, path)
        vehicle.update(v, delta, DT)
        if np.hypot(GOAL[0] - vehicle.x, GOAL[1] - vehicle.y) < TOL:
            break
    return vehicle


@pytest.mark.parametrize("controller_name", list(CONTROLLERS))
def test_converges_to_the_goal_from_directly_on_the_path(controller_name):
    vehicle = _run(controller_name, start_y=0.0)
    assert np.hypot(GOAL[0] - vehicle.x, GOAL[1] - vehicle.y) < TOL


@pytest.mark.parametrize("controller_name", list(CONTROLLERS))
def test_converges_to_the_goal_from_a_lateral_offset(controller_name):
    """A real tracking test, not just "drive straight ahead": start 1.5m off the
    path's line and confirm the controller actually steers back onto it and still
    reaches the goal, not just moves forward."""
    vehicle = _run(controller_name, start_y=1.5)
    assert np.hypot(GOAL[0] - vehicle.x, GOAL[1] - vehicle.y) < TOL


@pytest.mark.parametrize("controller_name", list(CONTROLLERS))
def test_final_heading_matches_the_straight_path(controller_name):
    vehicle = _run(controller_name, start_y=1.5)
    assert vehicle.theta == pytest.approx(0.0, abs=0.2)
