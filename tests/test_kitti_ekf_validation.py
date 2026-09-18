from pathlib import Path

import numpy as np

from core.validation.kitti_ekf_validation import DEFAULT_POSES_PATH, validate
from core.validation.kitti_loader import load_kitti_poses


def test_excerpt_data_is_present():
    assert Path(DEFAULT_POSES_PATH).exists()


def test_ekf_beats_dead_reckoning_on_real_kitti_data():
    """The core claim: fusing periodic corrections against a real driven trajectory
    reduces error vs. odometry alone. No arbitrary threshold: strict improvement over dead reckoning."""
    sequence = load_kitti_poses(DEFAULT_POSES_PATH)
    result = validate(sequence, seed=0)
    assert result.ekf_rmse < result.dr_rmse


def test_ekf_error_stays_bounded():
    """Sanity check against divergence/blowup, not a tight accuracy claim."""
    sequence = load_kitti_poses(DEFAULT_POSES_PATH)
    result = validate(sequence, seed=0)
    assert result.ekf_rmse < 5.0


def test_result_is_deterministic_for_a_fixed_seed():
    sequence = load_kitti_poses(DEFAULT_POSES_PATH)
    result_a = validate(sequence, seed=3)
    result_b = validate(sequence, seed=3)
    assert result_a.ekf_rmse == result_b.ekf_rmse


def test_ekf_advantage_holds_across_noise_draws_not_just_one_seed():
    """Dead-reckoning error swings widely with the noise draw (about 1 to 11 m), so the claim is
    checked over 20 draws: the EKF stays near 1 m and beats dead reckoning on nearly all of them."""
    sequence = load_kitti_poses(DEFAULT_POSES_PATH)
    runs = [validate(sequence, seed=seed) for seed in range(20)]
    ekf = np.array([r.ekf_rmse for r in runs])
    dr = np.array([r.dr_rmse for r in runs])
    assert (ekf < dr).sum() >= 18
    assert np.median(ekf) < 1.1
    assert ekf.max() < 1.3
    assert np.median(dr) > 2 * np.median(ekf)
