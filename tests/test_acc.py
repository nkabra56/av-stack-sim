import numpy as np
import pytest

from auto_park.control.acc import IDMController, MpcAccController
from auto_park.highway_harness import AccHarness


def test_idm_accelerates_toward_desired_speed_with_no_lead_constraint():
    idm = IDMController(v0=30.0, a_max=1.5)
    accel = idm.control(ego_speed=20.0, gap=1000.0, lead_speed=20.0)
    assert 0.0 < accel <= 1.5


def test_idm_clips_extreme_deceleration_to_a_physical_limit():
    """Raw IDM's (s*/gap)^2 term is unbounded as gap -> 0; a real car can't actually
    decelerate at whatever multiple of a_max that implies."""
    idm = IDMController(a_min=-9.0)
    accel = idm.control(ego_speed=25.0, gap=5.0, lead_speed=10.0)
    assert accel == pytest.approx(-9.0)


def test_mpc_accel_stays_within_bounds():
    mpc = MpcAccController(a_min=-3.0, a_max=1.5)
    accel = mpc.control(ego_speed=25.0, gap=5.0, lead_speed=10.0)
    assert -3.0 <= accel <= 1.5


@pytest.mark.parametrize("controller_factory", [IDMController, MpcAccController])
def test_follows_a_braking_lead_without_collision(controller_factory):
    dt = 0.1
    n = 300
    lead_speed = np.full(n, 25.0)
    lead_speed[100:150] = np.linspace(25.0, 15.0, 50)
    lead_speed[150:] = 15.0
    lead_position = np.cumsum(lead_speed) * dt + 100.0

    harness = AccHarness(
        lead_position, lead_speed, lead_length=4.5, controller=controller_factory(),
        ego_initial_speed=25.0, ego_initial_gap=40.0, seed=1,
    )
    result = harness.run()

    assert not result.collided
    assert result.min_gap > 0
    assert result.ego_speed[-1] == pytest.approx(15.0, abs=1.0)
