import json

import numpy as np

from core.validation.ngsim_loader import load_following_pair
from core.visualization.web_export import (
    _four_way_scene,
    _kitti_noisy_scene,
    _kitti_scene,
    _lane_scene,
    _left_turn_scene,
    _parking_scene,
    _stop_sign_scene,
)


def _assert_consistent(scene):
    n = scene["n"]
    for track in scene["tracks"].values():
        assert len(track["x"]) == len(track["y"]) == len(track["th"]) == n
    for values in scene["signals"].values():
        assert len(values) == n
    for stat in scene["hud"]:
        assert stat["key"] in scene["signals"]
        if "labels" in stat:
            assert set(scene["signals"][stat["key"]]) <= set(range(len(stat["labels"])))
    for vehicle in scene["vehicles"]:
        assert vehicle["track"] in scene["tracks"]
    json.dumps(scene)


def test_parking_scene_export_is_consistent_and_complete():
    scene = _parking_scene("parallel_open", "pure_pursuit", "t", "b")
    _assert_consistent(scene)
    assert len(scene["rays"]["ranges"]) == scene["n"]
    assert len(scene["uncertainty"]["ellipse"]) == scene["n"]
    assert scene["collision_radius"] > 0


def test_kitti_export_flips_the_frame_without_changing_errors():
    """The left-handed KITTI frame is mirrored for display; distances must be unchanged."""
    scene = _kitti_scene()
    _assert_consistent(scene)
    truth, ekf = scene["tracks"]["truth"], scene["tracks"]["ekf"]
    err = np.hypot(np.subtract(ekf["x"], truth["x"]), np.subtract(ekf["y"], truth["y"]))
    assert np.allclose(err, scene["signals"]["ekf_err"], atol=0.01)
    assert truth["x"][0] == 0.0 and truth["y"][0] == 0.0


def test_noisier_position_fixes_leave_the_dead_reckoning_baseline_unchanged():
    """The comparison is controlled only if the odometry noise draw is identical between scenes."""
    base, noisy = _kitti_scene(), _kitti_noisy_scene()
    _assert_consistent(noisy)
    assert noisy["signals"]["dr_err"] == base["signals"]["dr_err"]
    assert max(noisy["signals"]["ekf_err"]) > max(base["signals"]["ekf_err"])


def test_intersection_scenes_are_safe_and_follow_right_of_way():
    four, left = _four_way_scene(), _left_turn_scene()
    for scene in (four, left):
        _assert_consistent(scene)
        assert scene["outcome"]["label"] == "No collision"
    assert four["outcome"]["detail"].startswith("Proceed order: N")
    assert left["outcome"]["detail"].startswith("Proceed order: S")  # N arrived first but yields


def test_stop_sign_scene_shows_cross_traffic_only_inside_its_window():
    scene = _stop_sign_scene()
    _assert_consistent(scene)
    zone = scene["zones"][0]
    assert 0 < zone["t0"] < zone["t1"] < scene["n"] * scene["dt"]
    assert scene["outcome"]["label"] == "Yielded, then went"
    assert min(scene["signals"]["to_line"]) >= 0


def test_lane_scene_records_heading_and_steering():
    scene = _lane_scene()
    _assert_consistent(scene)
    assert max(abs(c) for c in scene["signals"]["cte"]) > 2.5  # starts 3 m off center
    assert abs(scene["signals"]["cte"][-1]) < 0.2


def test_recorded_follower_aligns_with_the_recorded_leader():
    """The follower scene indexes leader, follower and headway by tick, so they must share a clock."""
    pair = load_following_pair()
    n = min(len(pair.leader.position), len(pair.follower.position))
    implied_gap = pair.leader.position[:n] - pair.leader.length - pair.follower.position[:n]
    assert np.allclose(implied_gap, pair.real_space_headway[:n], atol=0.01)
