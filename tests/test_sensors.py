"""Hand-computed ray/circle intersection cases for UltrasonicArray.sense -- direct unit
coverage for the geometry itself, on top of the integration-level coverage
test_simulation.py already provides (see IMPLEMENTATION.md section 4)."""

import numpy as np
import pytest

from core.environment import Obstacle
from core.sensors import UltrasonicArray
from core.vehicle import Vehicle


def test_obstacle_dead_ahead_returns_the_near_edge_distance():
    """Vehicle at the origin facing +x, obstacle centered at (5, 0) r=1 -- the ray hits
    the circle's near edge at x=4, so the reading is 4.0, not the 5.0 center distance."""
    sensor = UltrasonicArray(angles=[0.0], max_range=10.0)
    vehicle = Vehicle(x=0.0, y=0.0, theta=0.0)
    readings = sensor.sense(vehicle, [Obstacle(x=5.0, y=0.0, radius=1.0)])
    assert readings[0.0] == pytest.approx(4.0)


def test_obstacle_out_of_range_reads_max_range():
    """The near edge (9.0) is still farther than max_range (5.0), so the beam reports
    max_range, not the true (out-of-range) distance."""
    sensor = UltrasonicArray(angles=[0.0], max_range=5.0)
    vehicle = Vehicle(x=0.0, y=0.0, theta=0.0)
    readings = sensor.sense(vehicle, [Obstacle(x=10.0, y=0.0, radius=1.0)])
    assert readings[0.0] == pytest.approx(5.0)


def test_obstacle_behind_the_beam_direction_is_ignored():
    """Obstacle at (-5, 0), beam pointing along +x -- both ray/circle intersection
    parameters are negative (behind the ray's origin), so they're excluded by the
    `0 <= t` check and the beam reports max_range, not a negative/behind hit."""
    sensor = UltrasonicArray(angles=[0.0], max_range=10.0)
    vehicle = Vehicle(x=0.0, y=0.0, theta=0.0)
    readings = sensor.sense(vehicle, [Obstacle(x=-5.0, y=0.0, radius=1.0)])
    assert readings[0.0] == pytest.approx(10.0)


def test_obstacle_off_to_the_side_of_a_straight_beam_is_a_true_miss():
    """Obstacle centered well off the ray's line entirely (discriminant < 0) -- a
    genuine geometric miss, not just an out-of-range or behind-the-beam case."""
    sensor = UltrasonicArray(angles=[0.0], max_range=10.0)
    vehicle = Vehicle(x=0.0, y=0.0, theta=0.0)
    readings = sensor.sense(vehicle, [Obstacle(x=5.0, y=5.0, radius=1.0)])
    assert readings[0.0] == pytest.approx(10.0)


def test_tangent_obstacle_touches_at_exactly_one_point():
    """Obstacle at (5, 1) r=1, beam along +x -- the ray is exactly tangent to the
    circle (discriminant == 0), touching at x=5. Both quadratic roots coincide, so the
    reading is the single tangent-point distance, not a NaN or a double-counted miss."""
    sensor = UltrasonicArray(angles=[0.0], max_range=10.0)
    vehicle = Vehicle(x=0.0, y=0.0, theta=0.0)
    readings = sensor.sense(vehicle, [Obstacle(x=5.0, y=1.0, radius=1.0)])
    assert readings[0.0] == pytest.approx(5.0)


def test_nearest_of_several_obstacles_wins():
    """Two obstacles on the same beam -- the reading is the nearer one's distance, not
    the farther one's or some combination."""
    sensor = UltrasonicArray(angles=[0.0], max_range=10.0)
    vehicle = Vehicle(x=0.0, y=0.0, theta=0.0)
    readings = sensor.sense(
        vehicle, [Obstacle(x=8.0, y=0.0, radius=1.0), Obstacle(x=4.0, y=0.0, radius=1.0)]
    )
    assert readings[0.0] == pytest.approx(3.0)


def test_beam_angle_is_relative_to_vehicle_heading():
    """A beam_angle of +pi/2, with the vehicle already facing +pi/2 (north), points the
    ray along -x (west) -- beam angles compose with theta, they aren't absolute."""
    sensor = UltrasonicArray(angles=[np.pi / 2], max_range=10.0)
    vehicle = Vehicle(x=0.0, y=0.0, theta=np.pi / 2)
    readings = sensor.sense(vehicle, [Obstacle(x=-5.0, y=0.0, radius=1.0)])
    assert readings[np.pi / 2] == pytest.approx(4.0)


def test_no_obstacles_reads_max_range_on_every_beam():
    sensor = UltrasonicArray(angles=[-0.3, 0.0, 0.3], max_range=6.0)
    vehicle = Vehicle(x=0.0, y=0.0, theta=0.0)
    readings = sensor.sense(vehicle, [])
    assert readings == {-0.3: 6.0, 0.0: 6.0, 0.3: 6.0}
