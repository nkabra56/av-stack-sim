import numpy as np
import pytest

from auto_park.control.mpc import MPCController
from auto_park.control.pure_pursuit import PurePursuitAdaptive
from auto_park.planning.dubins import DubinsPlanner
from auto_park.scenario_loader import list_scenarios, load_scenario
from auto_park.sensors import UltrasonicArray
from auto_park.simulation import ParkingSimulation

# Scenarios with a clear path from start to spot: both controllers must actually reach
# the spot. The remaining scenarios (obstacle-obstructed) are only required to be safe
# (no collision) -- the M1 baseline planner doesn't route around obstacles by design
# (see IMPLEMENTATION.md section 6), so stalling short of the goal is expected there.
OPEN_SCENARIOS = ["perpendicular_open", "parallel_open"]

CONTROLLERS = {
    "pure_pursuit": lambda v: PurePursuitAdaptive(wheelbase=v.wheelbase, v_max=1.5, max_steer=v.max_steer),
    "mpc": lambda v: MPCController(wheelbase=v.wheelbase, delta_max=v.max_steer, v_max=1.5),
}


def _run(scenario_name: str, controller_name: str, max_steps: int = 400):
    scenario = load_scenario(scenario_name)
    planner = DubinsPlanner()
    controller = CONTROLLERS[controller_name](scenario.vehicle)
    sensor = UltrasonicArray(angles=[-0.6, -0.3, 0.0, 0.3, 0.6], max_range=8.0)
    sim = ParkingSimulation(scenario.vehicle, scenario.environment, planner, controller, sensor)
    return sim.run(max_steps=max_steps)


@pytest.mark.parametrize("scenario_name", OPEN_SCENARIOS)
@pytest.mark.parametrize("controller_name", list(CONTROLLERS))
def test_open_scenarios_reach_the_spot(scenario_name, controller_name):
    result = _run(scenario_name, controller_name)
    assert result.success
    assert not result.collision


@pytest.mark.parametrize("scenario_name", list_scenarios())
@pytest.mark.parametrize("controller_name", list(CONTROLLERS))
def test_no_scenario_ever_collides(scenario_name, controller_name):
    result = _run(scenario_name, controller_name)
    assert not result.collision


@pytest.mark.parametrize("scenario_name", list_scenarios())
def test_all_scenario_headings_are_radians(scenario_name):
    """Regression guard for the prototype bug where some scenarios passed degrees
    (e.g. theta=90.0) for a model that only accepts radians. Any valid heading must
    fall within [-pi, pi]; a value like 90.0 is ~14 full rotations out of range.
    """
    scenario = load_scenario(scenario_name)
    assert -np.pi <= scenario.vehicle.theta <= np.pi
    assert -np.pi <= scenario.environment.spot.theta <= np.pi
