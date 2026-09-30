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
    """Code-review finding: MpcAccController used to default a_min=-3.0 while IDMController
    defaults a_min=-9.0, an untested asymmetry that let MPC-ACC brake less hard under identical events."""
    assert MpcAccController().a_min == IDMController().a_min == -9.0


def test_mpc_falls_back_to_safe_braking_if_solver_returns_a_constraint_violating_point():
    """Code-review finding: neither result.success nor the gap constraint was checked before
    applying result.x. Force minimize to return a bad point and confirm the fallback engages."""
    mpc = MpcAccController(a_min=-9.0, a_max=1.5, min_gap=3.0)

    class _BadResult:
        success = True
        x = np.full(mpc.horizon, mpc.a_max)  # accelerating hard would blow through the gap

    with patch("core.control.acc.minimize", return_value=_BadResult()):
        accel = mpc.control(ego_speed=25.0, gap=3.0, lead_speed=0.0)
    assert accel == pytest.approx(mpc.a_min)


def test_mpc_uncertainty_gain_zero_is_bit_for_bit_identical_to_pre_feature_behavior():
    """Regression pin: uncertainty_gain defaults to 0.0, so this must reproduce the
    pre-feature MpcAccController's output exactly on this scripted sequence, confirmed by
    running that version directly and hardcoding its (bit-for-bit identical) result below."""
    dt = 0.1
    n = 20
    t = np.arange(n) * dt
    lead_speed_seq = 18.0 + 3.0 * np.sin(2 * np.pi * 0.7 * t)
    gap_seq = 25.0 + 3.0 * np.cos(2 * np.pi * 0.3 * t)
    ego_speed_seq = 17.0 + 1.5 * np.sin(2 * np.pi * 0.5 * t + 1.0)

    mpc = MpcAccController()
    out = np.array([mpc.control(ego_speed_seq[i], gap_seq[i], lead_speed_seq[i]) for i in range(n)])

    expected = np.array([
        1.4999999999999944, 1.499999999999998, 1.4999999999999973, 1.4999999999999998, 1.5,
        1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
        0.6794041937901282, -1.5786801828018955, -3.622592658105083,
    ])
    assert np.array_equal(out, expected)


def _simulate_min_gap(controller, lead_speed_seq, dt=0.1, ego_initial_speed=20.0, initial_gap=15.0):
    """Minimal closed-loop rollout (no radar/EKF noise) so the effect of `uncertainty_gain`
    is measured directly, without AccHarness's sensor noise as a confound."""
    ego_speed = ego_initial_speed
    gap = initial_gap
    min_gap = gap
    for lead_speed in lead_speed_seq:
        accel = controller.control(ego_speed, gap, lead_speed)
        gap += lead_speed * dt - ego_speed * dt
        ego_speed = max(0.0, ego_speed + accel * dt)
        min_gap = min(min_gap, gap)
    return min_gap


def _jerky_braking_lead_speed(dt=0.1, n=150):
    # A real hard-braking event (~5.3 m/s^2) with a jerky oscillation riding on top, so the
    # implied lead-acceleration signal the uncertainty estimate tracks is large and noisy.
    t = np.arange(n) * dt
    lead_speed = np.full(n, 20.0)
    lead_speed[30:60] = np.linspace(20.0, 4.0, 30)
    lead_speed[60:] = 4.0
    lead_speed += 1.0 * np.sin(2 * np.pi * 2.0 * t)
    return np.clip(lead_speed, 0.0, None)


def test_mpc_uncertainty_gain_grows_realized_min_gap_under_a_jerky_braking_lead():
    """Point of the feature: a nonzero uncertainty_gain should keep a measurably bigger
    realized minimum gap than gain=0.0 under the same jerky/braking scenario."""
    lead_speed_seq = _jerky_braking_lead_speed()

    min_gap_off = _simulate_min_gap(MpcAccController(uncertainty_gain=0.0), lead_speed_seq)
    min_gap_on = _simulate_min_gap(MpcAccController(uncertainty_gain=1.0), lead_speed_seq)

    assert min_gap_on > min_gap_off + 1.0  # a real, not marginal, difference


@pytest.mark.parametrize("uncertainty_gain", [0.0, 0.5, 1.0, 5.0, 50.0, 1000.0])
def test_effective_min_gap_never_exceeds_the_emergency_braking_floor(uncertainty_gain):
    """The uncertainty margin can only tighten the constraint, never push it past what's
    still achievable via full emergency braking; see KNOWN_BUGS.md entry 1."""
    mpc = MpcAccController(uncertainty_gain=uncertainty_gain)
    ego_speed = 25.0
    lead_positions = 5.0 + 2.0 * mpc.dt * np.arange(1, mpc.horizon + 1)  # a slow, close lead

    _, floor = mpc._rollout(ego_speed, lead_positions, np.full(mpc.horizon, mpc.a_min))
    for sigma in [0.0, 1.0, 10.0, 100.0]:
        effective = mpc._effective_min_gap(ego_speed, lead_positions, sigma)
        assert np.all(effective <= floor + 1e-9)


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
