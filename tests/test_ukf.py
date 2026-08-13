import numpy as np
import pytest

from core.estimation.ukf import UnscentedKalmanFilter


def _ukf(x0=(0.0, 0.0, 0.0), p0_scale=0.05, odom_v_std=0.03, odom_delta_std=0.01):
    return UnscentedKalmanFilter(
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
    """Same check as test_ekf.py's version, applied to the UKF: with zero odometry
    noise, sigma-point propagation of a deterministic bicycle-model turn should land
    on the same closed-form arc the EKF's linearized version does -- both are
    supposed to be unbiased for the mean, they just handle covariance differently."""
    wheelbase = 2.7
    delta = 0.3
    v = 1.0
    dt = 0.01
    radius = wheelbase / np.tan(delta)

    ukf = UnscentedKalmanFilter(
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
        ukf.predict(v, delta, dt)

    expected_x = radius * np.sin(quarter_turn)
    expected_y = radius * (1 - np.cos(quarter_turn))
    assert ukf.x[2] == pytest.approx(quarter_turn, abs=1e-2)
    assert ukf.x[0] == pytest.approx(expected_x, abs=0.1)
    assert ukf.x[1] == pytest.approx(expected_y, abs=0.1)


@pytest.mark.parametrize(
    "update_name, apply_update",
    [
        ("heading", lambda ukf: ukf.update_heading(0.05)),
        ("position", lambda ukf: ukf.update_position(0.1, -0.1)),
        ("landmark", lambda ukf: ukf.update_landmark(5.0, 0.2, (5.0, 1.0))),
    ],
)
def test_update_reduces_covariance(update_name, apply_update):
    ukf = _ukf(p0_scale=1.0)
    trace_before = np.trace(ukf.p)
    apply_update(ukf)
    assert np.trace(ukf.p) < trace_before


def test_predict_grows_covariance():
    ukf = _ukf(p0_scale=0.01)
    trace_before = np.trace(ukf.p)
    ukf.predict(v=1.0, delta=0.2, dt=0.1)
    assert np.trace(ukf.p) > trace_before


def test_update_landmark_on_top_of_the_landmark_does_not_raise_or_produce_nan():
    """Unlike ekf.py's update_landmark, this measurement function is only ever
    evaluated at sigma points, not differentiated -- so there's no 1/range or 1/range^2
    Jacobian term to blow up. Confirms that's actually true (no special-casing needed
    here, unlike the EKF), not just assumed: landing exactly on the landmark should
    stay finite and well-behaved."""
    ukf = _ukf()
    ukf.x[:2] = [5.0, 1.0]
    ukf.p = np.eye(3) * 0.05  # nonzero spread, so sigma points aren't all coincident with the landmark
    ukf.update_landmark(0.0, 0.0, (5.0, 1.0))
    assert np.all(np.isfinite(ukf.x))
    assert np.all(np.isfinite(ukf.p))


def test_filter_stays_bounded_near_truth_over_many_cycles():
    """Same scenario as test_ekf.py's version: dead reckoning + periodic compass/
    position corrections while driving straight, tracked within a modest bound."""
    rng = np.random.default_rng(7)
    wheelbase = 2.7
    v, delta, dt = 1.0, 0.0, 0.1
    odom_v_std, odom_delta_std = 0.03, 0.01
    compass_std, position_std = 0.02, 0.3

    true_x, true_y, true_theta = 0.0, 0.0, 0.0
    ukf = _ukf(x0=(0.0, 0.0, 0.0), p0_scale=0.05, odom_v_std=odom_v_std, odom_delta_std=odom_delta_std)

    max_error = 0.0
    for step in range(200):
        true_x += v * np.cos(true_theta) * dt
        true_y += v * np.sin(true_theta) * dt
        true_theta += (v / wheelbase) * np.tan(delta) * dt

        v_meas = v + rng.normal(0, odom_v_std)
        delta_meas = delta + rng.normal(0, odom_delta_std)
        ukf.predict(v_meas, delta_meas, dt)

        ukf.update_heading(true_theta + rng.normal(0, compass_std))
        if step % 10 == 0:
            ukf.update_position(true_x + rng.normal(0, position_std), true_y + rng.normal(0, position_std))

        error = np.hypot(ukf.x[0] - true_x, ukf.x[1] - true_y)
        max_error = max(max_error, error)

    assert max_error < 1.0


def test_heading_wraps_correctly_across_the_pi_boundary():
    """A state estimate near +pi, corrected by a compass reading just past -pi (the
    same real heading, wrapped) should pull toward the short way around, not spin the
    long way through 0 -- exercises _circular_mean/the angle-aware innovation, not
    just the ordinary in-range case every other test already uses."""
    ukf = _ukf(x0=(0.0, 0.0, 3.05), p0_scale=0.02)
    ukf.update_heading(-3.05)  # ~0.18 rad away the short way, ~6.1 rad the long way
    assert abs(ukf.x[2]) > 3.0  # moved toward +/-pi (the short way), not toward 0
