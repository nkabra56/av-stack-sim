from pathlib import Path

import pytest

from core.validation.acc_validation import validate
from core.validation.ngsim_loader import DEFAULT_EXCERPT_PATH


def test_excerpt_data_is_present():
    assert Path(DEFAULT_EXCERPT_PATH).exists()


@pytest.mark.parametrize("controller_name", ["idm", "mpc"])
def test_never_collides_on_real_ngsim_data(controller_name):
    """Safety is a hard pass/fail, evaluated against a real recorded car-following
    trajectory (including a full stop in congested traffic), not a synthetic one."""
    result = validate(controller_name, seed=0)
    assert not result.sim.collided
    assert result.sim.min_gap > 0


@pytest.mark.parametrize("controller_name", ["idm", "mpc"])
def test_gap_is_plausible_relative_to_the_real_follower(controller_name):
    """Not a strict match (the real driver isn't assumed optimal), but our controller
    shouldn't be wildly divergent from real following behavior on the same data."""
    result = validate(controller_name, seed=0)
    assert 0.2 * result.mean_real_gap < result.mean_gap < 3.0 * result.mean_real_gap


def test_deterministic_for_a_fixed_seed():
    a = validate("idm", seed=3)
    b = validate("idm", seed=3)
    assert a.sim.min_gap == b.sim.min_gap
