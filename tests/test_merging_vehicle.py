import numpy as np
import pytest

from core.control.acc import IDMController, MpcAccController
from core.full_highway_harness import FullHighwayHarness, MergingVehicleSpec
from core.messaging.bus import Bus
from core.messaging.messages import EgoLongitudinalStateMsg, LeadVehicleStateMsg
from core.nodes.radar_node import RadarNode
from core.validation.ngsim_loader import load_lane_centerline

# A synthetic, non-blocking lead (matches ACC's own v0) isolates the merging vehicle's effect:
# the recorded NGSIM leader's own stop-and-go would otherwise mask the merge's contribution.
CENTERLINE = load_lane_centerline()
V0 = 20.0
DT = 0.1
N = 400
NON_BLOCKING_LEAD_POSITION = 300.0 + V0 * DT * np.arange(N)
NON_BLOCKING_LEAD_SPEED = np.full(N, V0)


def _run(merging_vehicle: MergingVehicleSpec, controller, seed: int = 1):
    harness = FullHighwayHarness(
        centerline=CENTERLINE,
        lead_position=NON_BLOCKING_LEAD_POSITION,
        lead_speed=NON_BLOCKING_LEAD_SPEED,
        lead_length=4.5,
        acc_controller=controller,
        seed=seed,
        ego_initial_gap=200.0,
        merging_vehicle=merging_vehicle,
    )
    return harness.run(max_steps=N)


def _merge_start_tick(result) -> int:
    started = np.where(result.merging_lane_offset < 3.7)[0]
    assert len(started) > 0, "merging vehicle never crossed its gap-acceptance threshold"
    return int(started[0])


def test_merge_forces_a_measurable_acc_response():
    """Once the merging vehicle cuts in, the radar's nearest in-lane target switches from
    the distant non-blocking lead to it, and the ACC must actually brake: not a decoration."""
    spec = MergingVehicleSpec(initial_gap_ahead_of_ego=5.0, speed=22.0, gap_threshold=15.0)
    result = _run(spec, IDMController(v0=V0))
    mt = _merge_start_tick(result)

    assert not result.collided
    # Measured: ~0.002 m/s^2 (steady cruise) before, peaking at ~-2.39 m/s^2 and a ~1.4 m/s
    # speed drop shortly after the merge completes.
    assert np.abs(result.ego_accel[max(0, mt - 20):mt]).max() < 0.5
    assert result.ego_accel[mt:mt + 80].min() < -1.0
    assert result.ego_speed[mt] - result.ego_speed[mt:mt + 80].min() > 0.5


def test_merge_timing_tracks_the_real_gap_not_a_fixed_script_time():
    """The gap-acceptance decision is reactive: a higher closing speed reaches the same
    acceptance threshold sooner, so the merge tick must shift with it, not stay fixed."""
    slower_closing = MergingVehicleSpec(initial_gap_ahead_of_ego=5.0, speed=22.0, gap_threshold=15.0)
    faster_closing = MergingVehicleSpec(initial_gap_ahead_of_ego=5.0, speed=24.0, gap_threshold=15.0)

    slow_result = _run(slower_closing, IDMController(v0=V0))
    fast_result = _run(faster_closing, IDMController(v0=V0))

    assert _merge_start_tick(fast_result) < _merge_start_tick(slow_result)


@pytest.mark.parametrize("controller_name", ["idm", "mpc"])
@pytest.mark.parametrize("seed", [1, 2, 3, 17, 42])
def test_never_collides_with_the_merging_vehicle_worst_case(controller_name, seed):
    """Synthetic worst case: a small gap and a high closing speed at the moment of merge.
    Measured: IDM keeps ~11.8m, MPC ~2.6m: tight but never zero, across every seed tried."""
    controllers = {"idm": IDMController(v0=V0), "mpc": MpcAccController(v0=V0)}
    worst_case = MergingVehicleSpec(initial_gap_ahead_of_ego=2.0, speed=24.0, gap_threshold=6.0)
    result = _run(worst_case, controllers[controller_name], seed=seed)

    assert not result.collided
    assert not result.collided_with_merging
    assert result.min_merging_gap > 0.5


def test_deterministic_for_a_fixed_seed():
    spec = MergingVehicleSpec(initial_gap_ahead_of_ego=5.0, speed=22.0, gap_threshold=15.0)
    a = _run(spec, MpcAccController(v0=V0), seed=7)
    b = _run(spec, MpcAccController(v0=V0), seed=7)
    assert a.collided == b.collided
    assert a.min_merging_gap == b.min_merging_gap
    assert np.array_equal(a.merging_gap, b.merging_gap)
    assert np.array_equal(a.cross_track_error, b.cross_track_error)


def test_merging_vehicle_absent_reproduces_the_original_single_lead_radar_reading():
    """Regression guard: with no merging vehicle, RadarNode's output must be byte-for-byte
    identical to its pre-merge-feature single-lead formula (see radar_node.py's docstring)."""
    bus = Bus()
    received = []
    bus.subscribe("radar", lambda m: received.append(m))
    rng = np.random.default_rng(0)
    radar = RadarNode(bus, rng, lead_length=4.5, range_std=0.0, range_rate_std=0.0)

    bus.publish("ego_state", EgoLongitudinalStateMsg(position=100.0, speed=20.0, accel=0.0))
    bus.publish("lead_state", LeadVehicleStateMsg(position=150.0, speed=18.0))
    radar.step()

    assert received[-1].range == pytest.approx(150.0 - 4.5 - 100.0)
    assert received[-1].range_rate == pytest.approx(20.0 - 18.0)


def test_radar_only_locks_onto_the_merging_vehicle_once_it_is_actually_in_lane():
    """Two-candidates-in-one-tick edge case: a closer vehicle still beside the ego (not
    in-lane yet) must be ignored in favor of the farther in-lane lead, until it merges."""
    bus = Bus()
    received = []
    bus.subscribe("radar", lambda m: received.append(m))
    rng = np.random.default_rng(0)
    radar = RadarNode(bus, rng, lead_length=4.5, range_std=0.0, range_rate_std=0.0, merging_length=4.0)

    bus.publish("ego_state", EgoLongitudinalStateMsg(position=100.0, speed=20.0, accel=0.0))
    bus.publish("lead_state", LeadVehicleStateMsg(position=150.0, speed=20.0, lane_offset=0.0))

    # Closer, but still fully in the adjacent lane: radar must keep tracking the far lead.
    bus.publish("merging_vehicle_state", LeadVehicleStateMsg(position=120.0, speed=18.0, lane_offset=3.7))
    radar.step()
    assert received[-1].range == pytest.approx(150.0 - 4.5 - 100.0)

    # Now in-lane and nearer: radar must switch to it.
    bus.publish("merging_vehicle_state", LeadVehicleStateMsg(position=120.0, speed=18.0, lane_offset=0.0))
    radar.step()
    assert received[-1].range == pytest.approx(120.0 - 4.0 - 100.0)
