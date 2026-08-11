from pathlib import Path

import pytest

from core.validation.lane_centering_validation import REAL_LATERAL_STD_M, validate
from core.validation.ngsim_loader import DEFAULT_LANE_CENTERLINE_PATH


def test_lane_centerline_data_is_present():
    assert Path(DEFAULT_LANE_CENTERLINE_PATH).exists()


@pytest.mark.parametrize("initial_offset", [1.5, -1.5, 3.0])
def test_converges_and_stays_within_real_driver_scatter(initial_offset):
    """The plausibility bar: after settling, tracking error should stay within the
    range real drivers' own lateral positioning naturally varies by on this lane --
    not a strict target (no single "correct" position within a lane), but a sanity
    check that the controller isn't doing something a real driver never would."""
    result = validate(initial_offset=initial_offset)
    assert result.max_cte_after_settling < REAL_LATERAL_STD_M


def test_deterministic():
    a = validate(initial_offset=1.5)
    b = validate(initial_offset=1.5)
    assert a.rms_cte == b.rms_cte
