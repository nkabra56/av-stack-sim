"""Pins the real, measured EKF-vs-UKF comparison (core/validation/ukf_comparison.py)
as a regression test, not just a script output someone has to remember to re-run.
See DESIGN.md section 10 / core/estimation/ukf.py's module docstring for what
question this is actually answering: is the EKF's linearization "a fine
approximation... but it's an approximation" actually fine at this project's real
operating parameters?
"""

from core.validation.kitti_ekf_validation import DEFAULT_POSES_PATH
from core.validation.kitti_loader import load_kitti_poses
from core.validation.ukf_comparison import validate_against_kitti, validate_tight_turn_stress


def test_ukf_and_ekf_agree_closely_on_real_kitti_data():
    """The headline finding: on a real driven trajectory, at this project's real
    dt=0.1s, the two propagation methods produce nearly identical accuracy (measured:
    0.89m RMSE each, differing by ~0.03% of that) -- not asserted, measured, with a
    generous 10% margin so this isn't a flaky pin on the exact figure."""
    sequence = load_kitti_poses(DEFAULT_POSES_PATH)
    result = validate_against_kitti(sequence, seed=0)
    assert result.ekf_rmse < 5.0  # sanity bound, matches test_kitti_ekf_validation.py's
    assert result.ukf_rmse < 5.0
    assert abs(result.ekf_rmse - result.ukf_rmse) < 0.10 * result.ekf_rmse


def test_ukf_and_ekf_agree_closely_even_at_the_vehicles_tightest_turning_radius():
    """The stronger claim: even at the most nonlinear regime this project's own
    kinematic model can physically produce (Vehicle's own minimum turning radius, not
    an arbitrarily chosen extreme), the two still agree closely (measured: ~0.138m
    RMSE each, differing by ~0.1%) -- real evidence the EKF's linearization isn't
    quietly losing accuracy just because nobody compared it against something that
    doesn't linearize, the actual question DESIGN.md section 10 posed."""
    result = validate_tight_turn_stress(seed=0)
    assert abs(result.ekf_rmse - result.ukf_rmse) < 0.10 * result.ekf_rmse


def test_result_is_deterministic_for_a_fixed_seed():
    a = validate_tight_turn_stress(seed=5)
    b = validate_tight_turn_stress(seed=5)
    assert a.ekf_rmse == b.ekf_rmse
    assert a.ukf_rmse == b.ukf_rmse
