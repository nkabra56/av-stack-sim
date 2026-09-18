import numpy as np

from core.control.intersection_geometry import EAST, NORTH, SOUTH, WEST
from core.visualization.scenery import _slab_rects, box_rect, urban_props

APPROACHES = {"N": NORTH, "E": EAST, "S": SOUTH, "W": WEST}


def _overlap(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def test_box_shrinks_toward_missing_legs():
    assert box_rect("NESW", 4.5, 9.5) == (-9.5, 9.5, -9.5, 9.5)
    assert box_rect("NEW", 4.5, 9.5) == (-9.5, 9.5, -4.5, 9.5)


def test_sidewalk_blocks_never_cover_the_paved_box_or_an_existing_road():
    corridors = {"N": (-4.5, 4.5, 0.0, 75.0), "S": (-4.5, 4.5, -75.0, 0.0),
                 "E": (0.0, 75.0, -4.5, 4.5), "W": (-75.0, 0.0, -4.5, 4.5)}
    for legs in ("NESW", "NEW"):
        box = box_rect(legs, 4.5, 9.5)
        for rect in _slab_rects(legs, 4.5, 9.5, 75.0):
            assert not _overlap(rect, box)
            assert not any(_overlap(rect, corridors[leg]) for leg in legs)


def test_every_leg_gets_a_stop_sign_and_a_crosswalk():
    props = urban_props("NEW", APPROACHES, 9.5, 9.0, 75.0, seed=1)
    kinds = [p["type"] for p in props]
    assert kinds.count("stop_sign") == 3 and kinds.count("crosswalk") == 3
    assert kinds.count("signal") == 0


def test_signal_heads_replace_stop_signs_when_keys_are_given():
    keys = {"N": "a", "S": "a", "E": "b", "W": "b"}
    props = urban_props("NESW", APPROACHES, 9.5, 9.0, 75.0, seed=1, signal_keys=keys)
    assert [p["key"] for p in props if p["type"] == "signal"].count("a") == 2
    assert not [p for p in props if p["type"] == "stop_sign"]


def test_scenery_is_deterministic():
    a = urban_props("NESW", APPROACHES, 9.5, 9.0, 75.0, seed=4)
    b = urban_props("NESW", APPROACHES, 9.5, 9.0, 75.0, seed=4)
    assert a == b and len(a) > 50
    assert np.isfinite([p["x"] for p in a if "x" in p]).all()
