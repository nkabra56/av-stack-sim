import numpy as np

from core.background_traffic import simulate_lane
from core.validation.ngsim_loader import load_following_pair


def test_followers_never_reach_the_car_ahead_through_the_recorded_stop():
    flow = load_following_pair().leader.speed  # includes a full stop
    lengths = [4.5, 4.5, 4.5]
    pos, spd = simulate_lane(flow, [200.0, 170.0, 140.0], lengths)
    for i in (1, 2):
        assert (pos[i - 1] - lengths[i - 1] - pos[i]).min() > 0.5
    assert spd.min() >= 0.0


def test_lane_leader_replays_the_recorded_flow_with_the_requested_lag():
    flow = load_following_pair().leader.speed
    _, spd = simulate_lane(flow, [100.0], [4.5], lag_ticks=30)
    assert np.allclose(spd[0, 30:], flow[:-30])
