import numpy as np
import pytest

from core.vehicle import Vehicle


def test_straight_line_motion():
    vehicle = Vehicle(0.0, 0.0, 0.0, wheelbase=2.5)
    for _ in range(50):
        vehicle.update(v=1.0, delta=0.0, dt=0.1)
    assert vehicle.x == pytest.approx(5.0, abs=1e-9)
    assert vehicle.y == pytest.approx(0.0, abs=1e-9)
    assert vehicle.theta == pytest.approx(0.0, abs=1e-9)


def test_turning_radius_matches_bicycle_model():
    """A constant (v, delta) should trace a circle of radius L / tan(delta)."""
    wheelbase = 2.5
    delta = 0.3
    v = 1.0
    dt = 0.01  # coarse enough to stay fast; Euler error is still small over a quarter turn
    radius = wheelbase / np.tan(delta)

    vehicle = Vehicle(0.0, 0.0, 0.0, wheelbase)
    quarter_turn = np.pi / 2
    n_steps = int((quarter_turn * radius / v) / dt)
    for _ in range(n_steps):
        vehicle.update(v, delta, dt)

    expected_x = radius * np.sin(quarter_turn)
    expected_y = radius * (1 - np.cos(quarter_turn))
    assert vehicle.theta == pytest.approx(quarter_turn, abs=1e-2)
    assert vehicle.x == pytest.approx(expected_x, abs=0.1)
    assert vehicle.y == pytest.approx(expected_y, abs=0.1)


def test_turning_radius_property():
    vehicle = Vehicle(wheelbase=2.7, max_steer=0.6)
    assert vehicle.turning_radius == pytest.approx(2.7 / np.tan(0.6))
