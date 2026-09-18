import json

import numpy as np

from core.visualization.web_export import _kitti_scene, _parking_scene


def _assert_consistent(scene):
    n = scene["n"]
    for track in scene["tracks"].values():
        assert len(track["x"]) == len(track["y"]) == len(track["th"]) == n
    for values in scene["signals"].values():
        assert len(values) == n
    for stat in scene["hud"]:
        assert stat["key"] in scene["signals"]
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
