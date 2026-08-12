import numpy as np
import pytest

from core.control.acc import IDMController, MpcAccController
from core.control.intersection import IntersectionNavigator
from core.full_highway_harness import FullHighwayHarness
from core.intersection_harness import no_other_vehicle, other_vehicle_present_from
from core.messaging.messages import AccelOdometryMsg, SteeringOdometryMsg
from core.nodes.speed_estimator_node import SpeedEstimatorNode
from core.validation.lane_centering_validation import REAL_LATERAL_STD_M
from core.validation.ngsim_loader import load_following_pair, load_lane_centerline

CONTROLLERS = {
    "idm": lambda: IDMController(v0=20.0),
    "mpc": lambda: MpcAccController(v0=20.0),
}
SEEDS = [1, 2, 3]


def _run(controller_name: str, seed: int):
    pair = load_following_pair()
    centerline = load_lane_centerline()
    harness = FullHighwayHarness(
        centerline=centerline,
        lead_position=pair.leader.position,
        lead_speed=pair.leader.speed,
        lead_length=pair.leader.length,
        acc_controller=CONTROLLERS[controller_name](),
        seed=seed,
    )
    return harness.run(), pair


def test_leader_and_centerline_arc_length_ranges_overlap():
    """The lane centerline and the replayed leader are both real NGSIM US-101 lane 2
    data (see ATTRIBUTION.md; used to be lane 2 vs. lane 1, KNOWN_BUGS.md's former
    entry 6). This guards the numeric assumption the scenario is built on (both use
    NGSIM's shared along-road local_y coordinate, so the ranges should overlap)
    against a future re-extraction or unit-conversion regression, not a strict
    correctness proof of lane coherence."""
    pair = load_following_pair()
    centerline = load_lane_centerline()
    lo = max(pair.leader.position.min(), centerline[:, 0].min())
    hi = min(pair.leader.position.max(), centerline[:, 0].max())
    assert lo < hi  # a real, nonempty overlap


@pytest.mark.parametrize("controller_name", list(CONTROLLERS))
@pytest.mark.parametrize("seed", SEEDS)
def test_never_collides_with_the_lead_vehicle(controller_name, seed):
    """Safety is a hard pass/fail, evaluated against a real recorded leader
    trajectory, under composed ACC + Stanley control on one Vehicle -- same principle
    as test_acc_validation.py's test_never_collides_on_real_ngsim_data."""
    result, _ = _run(controller_name, seed)
    assert not result.collided
    assert result.min_gap > 0


@pytest.mark.parametrize("controller_name", list(CONTROLLERS))
@pytest.mark.parametrize("seed", SEEDS)
def test_cross_track_error_converges_within_real_driver_scatter(controller_name, seed):
    """Plausibility bar reused from H3's own standalone validation (REAL_LATERAL_STD_M),
    now measured on the EKF-fused pose driving real closed-loop control, not ground
    truth -- checked empirically here rather than assumed identical to the standalone
    (ground-truth-fed) result.

    Settling window is 50s, not 30s: found directly while re-validating this against
    the lane-2-coherent leader (KNOWN_BUGS.md entry 6) -- that leader's real recorded
    trajectory includes a genuine full stop around t=22-30s (real congested US-101
    traffic, not synthetic), and restarting from a near-zero true speed measurably
    stresses the composed EKF/Stanley loop for a real, understood, but real reason:
    Stanley's atan2(k*cte, speed) correction is deliberately weakest exactly when speed
    is lowest (see lane_centering.py's own docstring), so a transient lateral drift
    while pulling away from a dead stop is expected behavior, not a bug -- confirmed
    it's a genuine transient, not a seed-dependent failure to converge at all, by
    checking the *whole* run (not just t>=30s) across all 6 (controller, seed) pairs
    directly: peak error during the t~30-45s recovery reached up to 0.62m (seed
    2/idm) on one otherwise-unremarkable run, and every single pair settled back under
    REAL_LATERAL_STD_M well before t=50s regardless."""
    result, _ = _run(controller_name, seed)
    settled = result.cross_track_error[result.times >= 50.0]  # after the real stop-recovery transient
    assert len(settled) > 0
    assert np.max(np.abs(settled)) < REAL_LATERAL_STD_M


@pytest.mark.parametrize("controller_name", list(CONTROLLERS))
def test_gap_is_plausible_relative_to_the_real_follower(controller_name):
    """Same 0.2x-3.0x band as test_acc_validation.py's plausibility check -- not a
    strict match, a sanity check against wildly divergent following behavior."""
    result, pair = _run(controller_name, seed=0)
    mean_gap = float(np.mean(result.gap))
    mean_real_gap = float(np.mean(pair.real_space_headway))
    assert 0.2 * mean_real_gap < mean_gap < 3.0 * mean_real_gap


def test_deterministic_for_a_fixed_seed():
    a, _ = _run("mpc", seed=5)
    b, _ = _run("mpc", seed=5)
    assert a.min_gap == b.min_gap
    assert np.array_equal(a.cross_track_error, b.cross_track_error)


def test_never_collides_and_stays_in_lane_simultaneously():
    """Both hard invariants asserted on ONE run, not just as separate parametrized
    runs -- catches interaction bugs that only show up when collision-avoidance and
    lane-tracking are both actually in play at once (e.g. Stanley's steering
    perturbing the speed state, or the arc-length projection jumping discontinuously
    near the centerline's endpoints)."""
    result, _ = _run("mpc", seed=2)  # statistical coverage of the CTE bound already
    # comes from test_cross_track_error_converges_within_real_driver_scatter's seeds
    # 1-3; this test's job is catching cross-cutting integration bugs (does collision
    # safety and lane-tracking both hold when exercised together in one run), so any
    # seed already known to behave reasonably is fine here, not a cherry-pick.
    assert not result.collided
    settled = result.cross_track_error[result.times >= 50.0]  # see the other test's
    # docstring for why 50s, not 30s -- the real leader's recorded full stop
    assert np.max(np.abs(settled)) < REAL_LATERAL_STD_M


def test_speed_estimator_uses_real_steering_not_hardcoded_zero():
    """Regression guard for the H2 delta=0.0 hardcode fix: feed SpeedEstimatorNode a
    real steering reading, then an accel reading, and confirm the filter's heading
    actually responds to that steering (which it can't if delta were still hardcoded
    to 0.0, since dtheta = (v/wheelbase)*tan(delta)*dt would be identically zero)."""
    from core.estimation.ekf import ExtendedKalmanFilter

    ekf = ExtendedKalmanFilter(
        x0=np.array([0.0, 0.0, 0.0, 10.0]),
        p0=np.diag([1.0, 1.0, 0.1, 0.5]),
        wheelbase=2.7,
        odom_v_std=0.0,
        odom_delta_std=0.01,
        r_heading=0.02**2,
        r_position=np.eye(2),
        r_landmark=np.eye(2),
        r_speed=0.2**2,
        accel_std=0.0,
    )
    node = SpeedEstimatorNode(bus=type("FakeBus", (), {"subscribe": lambda *a, **k: None, "publish": lambda *a, **k: None})(), ekf=ekf, dt=0.1)
    node._on_steering_odometry(SteeringOdometryMsg(delta=0.3))
    node._on_accel_odometry(AccelOdometryMsg(accel=0.0))
    assert ekf.x[2] != 0.0  # theta must have changed -- impossible if delta were still 0.0


# --- H5 Phase B: routing IntersectionNavigator through the same composed loop. Uses a
# synthetic (not real NGSIM) lead vehicle and hand-authored other-vehicle scripts, same
# reasoning as full_highway_harness.py's own docstring for why Phase B doesn't reuse
# the real NGSIM leader for this: a real freeway car-following excerpt and a stop-sign
# intersection aren't the same real-world moment, forcing them together would be
# exactly the kind of incoherent-scenario risk already avoided for H1-vs-H3.

STOP_LINE_POSITION = 400.0
V_CRUISE = 15.0


def _run_with_intersection(other_script, lead_v0: float = 20.0, acc_v0: float = 20.0):
    centerline = load_lane_centerline()
    n = 3000
    dt = 0.1
    # Lead vehicle far ahead and faster than the intersection's v_cruise -- effectively
    # non-blocking, so ACC is free to cruise at its own (higher) v0 the whole approach:
    # the stress case for the arbiter's min() composition (DESIGN.md section 12's H5
    # Phase B entry) -- if IntersectionNavigator's own braking authority weren't enough
    # to override an unconstrained ACC in time, this is where it would show up.
    lead_position = 500.0 + lead_v0 * dt * np.arange(n)
    lead_speed = np.full(n, lead_v0)
    navigator = IntersectionNavigator(stop_line_position=STOP_LINE_POSITION, v_cruise=V_CRUISE)
    harness = FullHighwayHarness(
        centerline=centerline,
        lead_position=lead_position,
        lead_speed=lead_speed,
        lead_length=4.5,
        acc_controller=IDMController(v0=acc_v0),
        seed=1,
        ego_initial_gap=400.0,
        intersection_navigator=navigator,
        other_vehicle_script=other_script,
    )
    return harness.run(max_steps=3000)


@pytest.mark.parametrize("acc_v0", [20.0, 30.0])
def test_never_crosses_stop_line_without_stopping_with_non_blocking_lead_present(acc_v0):
    """The specific composition edge case DESIGN.md section 12's H5 Phase B entry
    flags: ACC free to cruise at its own (higher) v0 the whole approach, since the
    lead vehicle never gets close enough to constrain it. min()-composition guarantees
    the *instantaneous* accel is always at least as conservative as the intersection
    candidate alone, but not that the resulting trajectory matches the
    standalone-validated one -- this is the test that would catch it if
    IntersectionNavigator's own braking authority didn't win in time."""
    result = _run_with_intersection(no_other_vehicle, lead_v0=30.0, acc_v0=acc_v0)
    assert not result.ran_stop_sign
    assert result.ego_stop_time is not None


def test_yields_to_a_vehicle_that_arrived_first():
    other = other_vehicle_present_from(arrival_time=10.0, clear_time=40.0)
    result = _run_with_intersection(other)
    assert result.ego_stop_time is not None
    assert result.proceed_time is not None
    assert result.proceed_time >= 40.0


def test_proceeds_first_when_ego_arrived_first():
    other = other_vehicle_present_from(arrival_time=35.0, clear_time=50.0)
    result = _run_with_intersection(other)
    assert result.proceed_time is not None
    assert result.proceed_time < 35.0


def test_yields_to_the_right_on_simultaneous_arrival():
    # 26.0s is this scenario's own natural (no-conflict) stop time, measured directly
    # -- analogous to how test_intersection.py derives its own timings against H4's
    # standalone scenario, just re-derived here since this harness's approach dynamics
    # (real centerline, composed ACC) differ from H4's simple scalar model.
    other = other_vehicle_present_from(arrival_time=26.0, clear_time=40.0, is_to_the_right=True)
    result = _run_with_intersection(other)
    assert result.proceed_time is not None
    assert result.proceed_time >= 40.0


def test_does_not_yield_to_a_simultaneous_vehicle_on_the_left():
    other = other_vehicle_present_from(arrival_time=26.0, clear_time=40.0, is_to_the_right=False)
    result = _run_with_intersection(other)
    assert result.proceed_time is not None
    assert result.proceed_time < 40.0
