import numpy as np

from core.control.intersection_geometry import EAST, NORTH
from core.signalized_intersection import GREEN, RED, SignalCar, SignalPlan, demo_cars, run_signalized_scenario


def test_signal_plan_never_shows_conflicting_greens_or_yellows():
    plan = SignalPlan()
    for t in np.arange(0.0, 2 * plan.cycle, 0.1):
        ns, ew = plan.state(t, "NS"), plan.state(t, "EW")
        assert ns == RED or ew == RED


def test_signal_plan_gives_each_axis_a_green_and_an_all_red_gap():
    plan = SignalPlan()
    states = [(plan.state(t, "NS"), plan.state(t, "EW")) for t in np.arange(0.0, plan.cycle, 0.1)]
    assert (GREEN, RED) in states and (RED, GREEN) in states and (RED, RED) in states


def test_no_car_crosses_on_red_and_none_collide():
    result = run_signalized_scenario(demo_cars())
    assert not result.collided
    assert result.red_entries == []
    assert result.min_gap > 0


def test_cars_queue_on_red_and_every_car_eventually_crosses():
    cars = demo_cars()
    result = run_signalized_scenario(cars)
    assert all(c.line_time is not None for c in result.cars)
    stopped = [c for c in result.cars if ((c.speed < 0.2) & c.active).any()]
    assert len(stopped) >= 8  # cross-street cars really did wait at the line


def test_a_car_facing_red_stops_short_of_the_line():
    result = run_signalized_scenario([SignalCar(EAST, 0.0)], duration=20.0)  # east-west is red until 19 s
    car = result.cars[0]
    assert car.speed[150] < 0.2
    assert car.line_time is None or car.line_time > 19.0


def test_simulation_is_deterministic():
    a = run_signalized_scenario(demo_cars())
    b = run_signalized_scenario(demo_cars())
    assert all(np.array_equal(x.x, y.x) for x, y in zip(a.cars, b.cars, strict=True))


def test_green_light_car_does_not_stop():
    result = run_signalized_scenario([SignalCar(NORTH, 0.0)], duration=20.0)
    assert result.cars[0].speed[result.cars[0].active].min() > 5.0
