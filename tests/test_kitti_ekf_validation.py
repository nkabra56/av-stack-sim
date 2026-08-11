from pathlib import Path

from core.validation.kitti_ekf_validation import DEFAULT_POSES_PATH, validate
from core.validation.kitti_loader import load_kitti_poses


def test_excerpt_data_is_present():
    assert Path(DEFAULT_POSES_PATH).exists()


def test_ekf_beats_dead_reckoning_on_real_kitti_data():
    """The core claim: fusing periodic corrections against a real driven trajectory
    actually reduces error versus integrating noisy odometry alone -- not just on the
    synthetic noise the project generates for itself (test_ekf.py), but on real KITTI
    ground truth. No arbitrary accuracy threshold to pick; a strict improvement over
    the dead-reckoning-only baseline is the right thing to assert.
    """
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
