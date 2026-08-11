import numpy as np
import pytest

from auto_park.estimation.ekf import ExtendedKalmanFilter


def _ekf(x0=(0.0, 0.0, 0.0), p0_scale=0.05, odom_v_std=0.03, odom_delta_std=0.01):
    return ExtendedKalmanFilter(
        x0=np.array(x0),
        p0=np.eye(3) * p0_scale,
        wheelbase=2.7,
        odom_v_std=odom_v_std,
        odom_delta_std=odom_delta_std,
        r_heading=0.02**2,
        r_position=np.eye(2) * 0.3**2,
        r_landmark=np.diag([0.2**2, 0.03**2]),
    )


def test_predict_only_matches_closed_form_arc():
    """With zero odometry noise, the predicted mean should exactly match the
    closed-form bicycle-model arc -- same check as test_vehicle's turning-radius test,
    applied to the EKF's prediction step instead of Vehicle.update directly."""
    wheelbase = 2.7
    delta = 0.3
    v = 1.0
    dt = 0.01
    radius = wheelbase / np.tan(delta)

    ekf = ExtendedKalmanFilter(
        x0=np.array([0.0, 0.0, 0.0]),
        p0=np.eye(3) * 0.01,
        wheelbase=wheelbase,
        odom_v_std=0.0,
        odom_delta_std=0.0,
        r_heading=0.01,
        r_position=np.eye(2) * 0.09,
        r_landmark=np.diag([0.04, 0.01]),
    )

    quarter_turn = np.pi / 2
    n_steps = int((quarter_turn * radius / v) / dt)
    for _ in range(n_steps):
        ekf.predict(v, delta, dt)

    expected_x = radius * np.sin(quarter_turn)
    expected_y = radius * (1 - np.cos(quarter_turn))
    assert ekf.x[2] == pytest.approx(quarter_turn, abs=1e-2)
    assert ekf.x[0] == pytest.approx(expected_x, abs=0.1)
    assert ekf.x[1] == pytest.approx(expected_y, abs=0.1)


@pytest.mark.parametrize(
    "update_name, apply_update",
    [
        ("heading", lambda ekf: ekf.update_heading(0.05)),
        ("position", lambda ekf: ekf.update_position(0.1, -0.1)),
        ("landmark", lambda ekf: ekf.update_landmark(5.0, 0.2, (5.0, 1.0))),
    ],
)
def test_update_reduces_covariance(update_name, apply_update):
    ekf = _ekf(p0_scale=1.0)
    trace_before = np.trace(ekf.p)
    apply_update(ekf)
    assert np.trace(ekf.p) < trace_before


def test_predict_grows_covariance():
    ekf = _ekf(p0_scale=0.01)
    trace_before = np.trace(ekf.p)
    ekf.predict(v=1.0, delta=0.2, dt=0.1)
    assert np.trace(ekf.p) > trace_before


def test_filter_stays_bounded_near_truth_over_many_cycles():
    """Simulate a vehicle driving straight with noisy odometry (dead reckoning) and
    periodic compass + position corrections; the filter's estimate should track true
    state within a modest bound throughout, not just "run without crashing"."""
    rng = np.random.default_rng(7)
    wheelbase = 2.7
    v, delta, dt = 1.0, 0.0, 0.1
    odom_v_std, odom_delta_std = 0.03, 0.01
    compass_std, position_std = 0.02, 0.3

    true_x, true_y, true_theta = 0.0, 0.0, 0.0
    ekf = _ekf(x0=(0.0, 0.0, 0.0), p0_scale=0.05, odom_v_std=odom_v_std, odom_delta_std=odom_delta_std)

    max_error = 0.0
    for step in range(200):
        true_x += v * np.cos(true_theta) * dt
        true_y += v * np.sin(true_theta) * dt
        true_theta += (v / wheelbase) * np.tan(delta) * dt

        v_meas = v + rng.normal(0, odom_v_std)
        delta_meas = delta + rng.normal(0, odom_delta_std)
        ekf.predict(v_meas, delta_meas, dt)

        ekf.update_heading(true_theta + rng.normal(0, compass_std))
        if step % 10 == 0:
            ekf.update_position(true_x + rng.normal(0, position_std), true_y + rng.normal(0, position_std))

        error = np.hypot(ekf.x[0] - true_x, ekf.x[1] - true_y)
        max_error = max(max_error, error)

    assert max_error < 1.0


def _speed_ekf(x0=(0.0, 0.0, 0.0, 20.0), p0_scale=0.05, accel_std=0.15, r_speed=0.2**2):
    return ExtendedKalmanFilter(
        x0=np.array(x0),
        p0=np.eye(4) * p0_scale,
        wheelbase=2.7,
        odom_v_std=0.0,
        odom_delta_std=0.0,
        r_heading=1.0,
        r_position=np.eye(2),
        r_landmark=np.eye(2),
        r_speed=r_speed,
        accel_std=accel_std,
    )


def test_3state_ekf_behavior_is_unchanged_by_the_4state_extension():
    """The generalization of _apply_update/update_heading/update_position/
    update_landmark to be dimension-agnostic (len(self.x) instead of a hardcoded 3)
    must produce numerically identical results for the original 3-state case --
    this is the regression guard for that refactor, on top of the unchanged
    KITTI-validation RMSE already checked manually."""
    ekf = _ekf(p0_scale=1.0)
    ekf.predict(v=1.0, delta=0.2, dt=0.1)
    ekf.update_heading(0.05)
    ekf.update_position(0.3, -0.2)
    assert len(ekf.x) == 3
    assert ekf.p.shape == (3, 3)


def test_predict_with_speed_state_matches_closed_form_constant_acceleration():
    """v update (v += accel*dt each step) is exact for constant acceleration -- Euler
    integration of a linear ODE has no discretization error. x update (x += v*dt using
    v from the *start* of the step) is a left-Riemann-sum approximation of the true
    integral though, so it systematically undershoots for accelerating motion -- same
    Euler-discretization tolerance test_vehicle.py's turning-radius test already needs
    for the same underlying reason, not a bug in either place."""
    dt = 0.1
    accel = 2.0
    ekf = _speed_ekf(x0=(0.0, 0.0, 0.0, 10.0), accel_std=0.0)
    for _ in range(50):
        ekf.predict_with_speed_state(accel=accel, delta=0.0, dt=dt)

    t = 50 * dt
    expected_v = 10.0 + accel * t
    expected_x = 10.0 * t + 0.5 * accel * t**2
    assert ekf.x[3] == pytest.approx(expected_v, abs=1e-6)
    assert ekf.x[0] == pytest.approx(expected_x, abs=1.0)
    assert ekf.x[1] == pytest.approx(0.0, abs=1e-9)  # straight line, no lateral motion


def test_update_speed_reduces_covariance():
    ekf = _speed_ekf(p0_scale=1.0)
    trace_before = np.trace(ekf.p)
    ekf.update_speed(21.0)
    assert np.trace(ekf.p) < trace_before


def test_speed_state_stays_bounded_near_truth_over_many_cycles():
    """Same style of check as test_filter_stays_bounded_near_truth_over_many_cycles,
    for the 4-state speed-estimating mode: predict on noisy acceleration, correct on
    a noisy speedometer, verify the fused estimate tracks true speed."""
    rng = np.random.default_rng(11)
    dt = 0.1
    accel_std, speedometer_std = 0.15, 0.2

    true_v = 20.0
    ekf = _speed_ekf(x0=(0.0, 0.0, 0.0, 20.0), accel_std=accel_std, r_speed=speedometer_std**2)

    max_error = 0.0
    for step in range(200):
        true_accel = 0.5 if step < 100 else -0.5
        true_v = max(0.0, true_v + true_accel * dt)

        accel_meas = true_accel + rng.normal(0, accel_std)
        ekf.predict_with_speed_state(accel_meas, delta=0.0, dt=dt)
        ekf.update_speed(true_v + rng.normal(0, speedometer_std))

        max_error = max(max_error, abs(ekf.x[3] - true_v))

    assert max_error < 2.0
