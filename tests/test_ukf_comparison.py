"""Pins the real, measured EKF-vs-UKF comparison as a regression test: is the EKF's
linearization actually fine at this project's real operating parameters? See DESIGN.md section 10."""

from core.validation.kitti_ekf_validation import DEFAULT_POSES_PATH
from core.validation.kitti_loader import load_kitti_poses
from core.validation.ukf_comparison import validate_against_kitti, validate_tight_turn_stress


def test_ukf_and_ekf_agree_closely_on_real_kitti_data():
    """The headline finding: on a real driven trajectory, the two propagation methods
    produce nearly identical accuracy: measured, with a generous 10% margin, not a flaky pin."""
    sequence = load_kitti_poses(DEFAULT_POSES_PATH)
    result = validate_against_kitti(sequence, seed=0)
    assert result.ekf_rmse < 5.0  # sanity bound, matches test_kitti_ekf_validation.py's
    assert result.ukf_rmse < 5.0
    assert abs(result.ekf_rmse - result.ukf_rmse) < 0.10 * result.ekf_rmse


def test_ukf_and_ekf_agree_closely_even_at_the_vehicles_tightest_turning_radius():
    """The stronger claim: even at the vehicle's own most nonlinear regime (its minimum
    turning radius), the two still agree closely: real evidence the EKF isn't quietly losing accuracy."""
    result = validate_tight_turn_stress(seed=0)
    assert abs(result.ekf_rmse - result.ukf_rmse) < 0.10 * result.ekf_rmse


def test_result_is_deterministic_for_a_fixed_seed():
    a = validate_tight_turn_stress(seed=5)
    b = validate_tight_turn_stress(seed=5)
    assert a.ekf_rmse == b.ekf_rmse
    assert a.ukf_rmse == b.ukf_rmse
