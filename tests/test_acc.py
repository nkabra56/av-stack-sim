from unittest.mock import patch

import numpy as np
import pytest

from core.control.acc import IDMController, MpcAccController
from core.highway_harness import AccHarness


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


def test_mpc_a_min_defaults_to_idms_physical_emergency_floor():
    """Code-review finding: MpcAccController used to default a_min=-3.0 while
    IDMController defaults a_min=-9.0, even though both implement the identical
    control(ego_speed, gap, lead_speed) contract for the same plant -- an untested
    asymmetry that let MPC-ACC brake materially less hard than IDM under an identical
    hard-braking lead event."""
    assert MpcAccController().a_min == IDMController().a_min == -9.0


def test_mpc_falls_back_to_safe_braking_if_solver_returns_a_constraint_violating_point():
    """Code-review finding: neither result.success nor the gap constraint itself used
    to be checked before applying result.x as the real command -- a solver that
    silently returns a constraint-violating point (SLSQP can do this even while
    reporting success, near a boundary within its own tolerance) would have applied an
    unsafe accelerate-through-the-gap command. Force minimize to return a deliberately
    bad point and confirm the fallback (braking at a_min every step -- the same
    feasibility witness `_effective_min_gap` already relies on) engages instead."""
    mpc = MpcAccController(a_min=-9.0, a_max=1.5, min_gap=3.0)

    class _BadResult:
        success = True
        x = np.full(mpc.horizon, mpc.a_max)  # accelerating hard would blow through the gap

    with patch("core.control.acc.minimize", return_value=_BadResult()):
        accel = mpc.control(ego_speed=25.0, gap=3.0, lead_speed=0.0)
    assert accel == pytest.approx(mpc.a_min)


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
