import numpy as np
import pytest

from core.control.lane_centering import StanleyController
from core.vehicle import Vehicle


def test_converges_from_a_lateral_offset_on_a_straight_lane():
    """Regression guard for a real sign-convention bug found while building this:
    the cross-track error's sign has to make the correction steer *toward* the path,
    not away from it -- getting it backwards doesn't error, it silently diverges."""
    path = np.column_stack([np.linspace(0, 200, 201), np.zeros(201), np.zeros(201)])
    vehicle = Vehicle(x=0.0, y=2.0, theta=0.0, wheelbase=2.7)
    controller = StanleyController(wheelbase=2.7, k=0.5)

    for _ in range(300):
        delta = controller.control(vehicle, path, speed=15.0)
        vehicle.update(15.0, delta, 0.1)

    assert abs(vehicle.y) < 0.05


def test_converges_from_the_opposite_lateral_offset_too():
    """Same check, offset to the other side -- catches a sign bug that only
    happens to look correct from one direction."""
    path = np.column_stack([np.linspace(0, 200, 201), np.zeros(201), np.zeros(201)])
    vehicle = Vehicle(x=0.0, y=-2.0, theta=0.0, wheelbase=2.7)
    controller = StanleyController(wheelbase=2.7, k=0.5)

    for _ in range(300):
        delta = controller.control(vehicle, path, speed=15.0)
        vehicle.update(15.0, delta, 0.1)

    assert abs(vehicle.y) < 0.05


def test_steering_stays_within_max_steer():
    path = np.column_stack([np.linspace(0, 200, 201), np.zeros(201), np.zeros(201)])
    vehicle = Vehicle(x=0.0, y=5.0, theta=0.0, wheelbase=2.7)
    controller = StanleyController(wheelbase=2.7, k=0.5, max_steer=0.6)

    delta = controller.control(vehicle, path, speed=15.0)
    assert -0.6 <= delta <= 0.6


def test_near_zero_speed_does_not_blow_up():
    """Stanley's correction term divides by speed; without a floor this would
    produce an enormous (or NaN/inf at exactly zero) correction near a stop."""
    path = np.column_stack([np.linspace(0, 200, 201), np.zeros(201), np.zeros(201)])
    vehicle = Vehicle(x=0.0, y=2.0, theta=0.0, wheelbase=2.7)
    controller = StanleyController(wheelbase=2.7, k=0.5)

    delta = controller.control(vehicle, path, speed=0.0)
    assert np.isfinite(delta)
    assert -0.6 <= delta <= 0.6
