# Implementation Plan

This is the build spec for the architecture described in [DESIGN.md](DESIGN.md). Parking-mode
milestones M1 (correct baseline), the control half of M3 (MPC), ME (state estimation + pub/sub
architecture), MV (real-data EKF validation against KITTI), **M2 (Hybrid A* + Reeds-Shepp
obstacle-aware planning)**, and **M4 (sensing & re-planning)** are done. All four highway-mode
milestones — **H1 (adaptive cruise control)**, **H2 (fused ego speed via an extended EKF)**, **H3
(lane centering)**, and **H4 (intersection navigation)** — are also done, each originally validated
standalone, and **H5 (the full closed-loop highway drive, H1+H2+H3+H4 all composed onto one
Vehicle) is now done too**, built in two deliberately sequenced phases — see DESIGN.md section 12's
H5 entry. The highway side is now fully integrated; on the parking side, M5/M6 (visualization
polish, CI) are what's next — see Section 3 for the full roadmap.

## 1. Directory structure

```
core/
  vehicle.py           # kinematic bicycle model + turning_radius from max_steer
  sensors.py            # ultrasonic ray-cast sensor array (used internally by SensorNode)
  environment.py         # parking lot, spots, obstacles, boundaries, VEHICLE_RADIUS (shared by
                       # harness.py and planning/hybrid_astar.py, one collision threshold)
  interfaces.py          # Planner / Controller structural protocols, HasPose, Pose type
  scenario_loader.py       # loads scenarios/*.yaml into Vehicle + Environment (+ seed)
  messaging/
    __init__.py
    bus.py             # Bus: synchronous publish/subscribe
    messages.py          # parking: TrueStateMsg, OdometryMsg, CompassMsg, PositionFixMsg,
                       # LandmarkBearingMsg, ObstacleRangeMsg, PoseEstimateMsg, PathMsg,
                       # ControlCmdMsg -- highway: LeadVehicleStateMsg, EgoLongitudinalStateMsg,
                       # RadarMsg, LongitudinalCmdMsg, AccelOdometryMsg, SpeedometerMsg,
                       # EgoSpeedEstimateMsg (H2, now carries x/y/theta too) -- H5:
                       # LateralCmdMsg, SteeringOdometryMsg, EgoHighwayStateMsg -- see
                       # DESIGN.md section 2
  estimation/
    __init__.py
    ekf.py             # ExtendedKalmanFilter: predict + 3 correction types (3-state, parking) +
                       # predict_with_speed_state/update_speed (4-state, H2/highway) -- H2's
                       # process model got two correctness fixes while building H5 (propagate
                       # against the *new* speed each tick, not the prior one; model steering-
                       # reading uncertainty in the process noise, not just accel's), both
                       # invisible under H1's delta=0 straight-line-only use, see DESIGN.md
                       # section 5 and section 12's H2/H5 entries
  validation/
    __init__.py
    kitti_loader.py       # parses KITTI poses.txt -> KittiSequence(times,x,y,theta,v,yaw_rate)
    kitti_ekf_validation.py  # runs the (unmodified) EKF against a real trajectory, EKF vs.
                       # dead-reckoning-only RMSE comparison + plot, see DESIGN.md section 5
    ngsim_loader.py       # parses NGSIM CSV -> NgsimFollowingPair, and (H3) load_lane_centerline()
                       # -> a real (N,3) lane path from aggregated real vehicle positions
    acc_validation.py      # runs ACC controllers vs. a real leader trajectory, safety/comfort/
                       # plausibility metrics + plot, see DESIGN.md section 11
    lane_centering_validation.py  # runs Stanley along the real lane centerline, checks
                       # convergence against real driver lateral scatter, see section 12's H3 entry
  data/
    kitti/
      excerpt_poses.txt      # committed 300-frame excerpt (KITTI seq 09, frames 840-1139)
      ATTRIBUTION.md       # license/citation for the redistributed KITTI excerpt
    ngsim/
      excerpt_trajectories.csv  # committed 780-frame (78s) real leader/follower pair, US-101
      lane_centerline.csv     # derived real lane centerline (H3), ~10,400 positions aggregated
      ATTRIBUTION.md       # license/citation for the redistributed + derived NGSIM data
  nodes/
    __init__.py
    vehicle_node.py       # ground truth + odometry publisher; owns accel/steering limits
    sensor_node.py        # obstacle_ranges, compass, position_fix, landmark_bearings publisher
    estimator_node.py      # wraps ekf.py, publishes pose_estimate
    planner_node.py       # wraps Planner, plans once off the first pose_estimate
    controller_node.py     # wraps Controller, one control_cmd per tick via explicit step()
    lead_vehicle_node.py    # replays a real recorded lead-vehicle trajectory tick by tick
    ego_longitudinal_node.py # 1D point-mass ego state; also publishes noisy accel_odometry/
                       # speedometer readings (H2) for SpeedEstimatorNode to fuse
    radar_node.py         # noisy bumper-to-bumper range + range-rate to the lead vehicle
    speed_estimator_node.py  # wraps the EKF's 4-state mode, publishes ego_speed_estimate (H2);
                       # H5: also corrects on compass/position_fix (see section 12's H5 entry)
    acc_controller_node.py   # wraps an ACC controller; acts on the fused speed estimate
                       # (not true speed), one accel command per tick
    highway_vehicle_node.py  # H5: real 2D Vehicle plant for the closed-loop drive -- accel-in
                       # (not desired-speed-in like VehicleNode), publishes on H1's ego_state
                       # topic unchanged plus a new full-pose topic and steering/compass/
                       # position-fix readings, see section 12's H5 entry
    lane_centering_node.py   # H5: wraps StanleyController, acts on the fused pose+speed estimate
    longitudinal_arbiter_node.py  # H5 Phase B: min() over accel candidate topics -- ACC's real-
                       # lead-vehicle accel vs. IntersectionNavigator's stop-line accel
    intersection_controller_node.py  # H5 Phase B: wraps IntersectionNavigator, feeds it the
                       # fused estimate (not ground truth, a first for H4), publishes an accel
                       # candidate for the arbiter
    other_vehicle_script_node.py  # H5 Phase B: thin Bus adapter for H4's existing
                       # OtherVehicleScript/other_vehicle_present_from, unchanged
  harness.py            # tick-based executor (parking mode): owns the Bus, builds all 5 nodes
  highway_harness.py       # tick-based executor (ACC/H1+H2 mode): owns the Bus, builds the
                       # longitudinal nodes -- mirrors harness.py's structure, kept separate
                       # rather than forcing a shared base class before H3 shows what's common
  intersection_harness.py    # direct simulation loop (no Bus -- H4 has no sensor noise/fusion
                       # to decouple) for IntersectionNavigator scenarios, see section 12's H4 entry
  full_highway_harness.py    # H5: tick-based executor combining H1+H2+H3 on one real Vehicle --
                       # kept separate from harness.py/highway_harness.py, same reasoning as
                       # highway_harness.py's own docstring for staying separate from harness.py,
                       # see section 12's H5 entry
  planning/
    __init__.py
    dubins.py            # M1 baseline: curvature-feasible fixed path, no obstacle avoidance.
                       # Kept as a documented reference (demo.py --planner dubins); also the
                       # single source of truth for the CSC formulas reeds_shepp.py reuses
    reeds_shepp.py        # M2: Dubins + reverse gear (CSC family only, see DESIGN.md section 6)
    hybrid_astar.py        # M2: obstacle-aware search over (x,y,theta), uses reeds_shepp.py as
                       # heuristic + analytic-expansion connector; the default planner now
  control/
    __init__.py
    pure_pursuit.py        # adaptive Pure Pursuit, acts on HasPose (a Vehicle or a pose estimate)
    mpc.py              # nonlinear MPC (direct shooting, SLSQP)
    acc.py              # IDMController + MpcAccController (longitudinal, see DESIGN.md section 11)
    lane_centering.py       # StanleyController (lateral, see DESIGN.md section 12's H3 entry)
    lane_geometry.py       # H5: arc-length along a curved centerline (build_arc_length_table,
                       # project_to_arc_length, pose_at_arc_length), see section 12's H5 entry
  visualization/
    __init__.py
    animate.py            # true vs. estimated trajectory + covariance ellipse + planned path
  scenarios/
    perpendicular_open.yaml
    perpendicular_flanked.yaml
    perpendicular_obstructed_lane.yaml
    parallel_open.yaml
    parallel_between_cars.yaml
  demo.py              # CLI entry point: run a named scenario + controller/planner (+ seed),
                       # show/save; --planner {hybrid_astar (default), reeds_shepp, dubins}
tests/
  test_vehicle.py
  test_bus.py
  test_sensors.py         # UltrasonicArray: hand-computed ray/circle intersection cases
                       # (dead ahead, out of range, behind the beam, tangent, nearest-of-several)
  test_control.py         # Pure Pursuit/MPC convergence on a straight line + lateral offset,
                       # decoupled from any planner or the estimation stack
  test_ekf.py
  test_ukf.py             # DESIGN.md section 10: UKF alternative to the EKF -- same closed-form-arc
                       # and filter-consistency checks as test_ekf.py, plus circular-mean handling
  test_ukf_comparison.py    # the actual EKF-vs-UKF head-to-head numbers, pinned as a regression
  test_kitti_ekf_validation.py  # EKF vs. dead-reckoning-only on the committed real KITTI excerpt
  test_planning.py        # endpoint + curvature checks (both planners), obstacle-clearance
                       # check (Hybrid A* only) -- see DESIGN.md section 6's M2 entry
  test_mpc.py            # parking MPCController: falls back to the warm-started plan, not a
                       # fresh unconverged solve, when SLSQP doesn't converge
  test_sensor_node.py      # SensorNode dropout/latency modeling (DESIGN.md section 10,
                       # KNOWN_BUGS.md entry 7): delivery gating and delayed-message timing, direct
  test_sensor_robustness.py  # closed-loop collision safety under dropout/latency across real
                       # scenarios; pins the verified-safe range and the latency_margin fix itself
  test_parking_env.py      # DESIGN.md section 10: ParkingEnv's Gym contract (gymnasium's own
                       # check_env), reward shape, collision/success termination -- fast, no training
  test_rl_training.py      # PPO trains end to end on ParkingEnv without crashing (a smoke test,
                       # not a convergence claim -- see core/validation/rl_comparison.py for that)
  test_rl_comparison.py    # real, measured RL-vs-baseline numbers against a committed trained
                       # policy (core/data/rl/) -- fast, evaluates only, no training
  test_ros2_bridge.py      # message-conversion functions + Ros2Bridge's wiring, against fakes --
                       # see ros2_bridge.py's own docstring for what is/isn't verified and why
  test_simulation.py       # integration tests, harness-based, across scenarios x controllers x seeds
  test_acc.py            # IDM/MPC-ACC unit + synthetic braking-lead scenario checks
  test_acc_validation.py    # IDM/MPC-ACC vs. real NGSIM data: safety, plausibility, determinism
  test_lane_centering.py    # Stanley convergence (both directions) + steering/speed edge cases
  test_lane_centering_validation.py  # Stanley vs. the real derived lane centerline
  test_intersection.py     # H4 state-machine + right-of-way branch coverage
  test_intersection_geometry.py  # H4 real 2D geometry + turning movements (KNOWN_BUGS.md entry 4):
                       # N-way real crossing-path collision checks, turn exit heading/position,
                       # left-yields-to-oncoming-despite-earlier-arrival, a random mixed-turn sweep
  test_controller_node.py   # ControllerNode unit coverage: direction-aware governor, tracking-
                       # aware buffer selection
  test_replanning.py      # KNOWN_BUGS.md entry 3: PlannerNode/ControllerNode re-plan wiring,
                       # closed-loop recovery across seeds
  test_full_highway.py     # H5 Phase A: collision safety, real-data coherence, cross-track-error
                       # and gap plausibility, determinism, a direct regression test for the H2
                       # delta=0.0 hardcode fix -- Phase B: stop-line compliance under the
                       # composed accel arbiter (incl. the non-blocking-lead stress case) and
                       # the four right-of-way branches, re-derived against this harness's own
                       # approach dynamics -- see section 12's H5 entry
pyproject.toml
DESIGN.md
IMPLEMENTATION.md
README.md
```

`simulation.py`/`ParkingSimulation` (the M1 direct-call loop) is retired — `harness.py` is now the
one way `demo.py` and the tests run a scenario, so there's a single execution path rather than two.

`test_planning.py` now exists (added with M2, once `planning/` had more than one planner to
compare against). `test_sensors.py` and `test_control.py` (unit-level, per Section 4) still
haven't been split out — current coverage for those is integration-level via `test_simulation.py`,
which has been enough to catch the regressions that mattered so far.

## 2. Key interfaces

Keeping these consistent is what lets planners and controllers be swapped without touching
`harness.py` or any other node.

```python
# vehicle.py
class Vehicle:
    def __init__(self, x=0.0, y=0.0, theta=0.0, wheelbase=2.7, max_steer=0.6): ...
    def update(self, v: float, delta: float, dt: float) -> None: ...
    # theta always in radians; callers passing degrees was the #1 historical bug (see Section 6)
    @property
    def turning_radius(self) -> float: ...  # wheelbase / tan(max_steer) -- the single source
                                              # of truth the planner and both controllers use

# interfaces.py
class HasPose(Protocol):
    x: float; y: float; theta: float  # satisfied by both Vehicle and PoseEstimateMsg

class Planner(Protocol):
    def plan(
        self, start: Pose, goal: Pose, obstacles: list[Obstacle], turning_radius: float
    ) -> np.ndarray:  # (N, 3) array of x, y, theta waypoints
        ...

class Controller(Protocol):
    def control(self, pose: HasPose, path: np.ndarray) -> tuple[float, float]:
        # returns (v_desired, delta) -- delta is clipped to vehicle.max_steer by VehicleNode
        # regardless of what the controller returns; controllers may command anything,
        # VehicleNode is the one place that enforces what's physically achievable

# estimation/ekf.py
class ExtendedKalmanFilter:
    def __init__(
        self, x0, p0, wheelbase, odom_v_std, odom_delta_std, r_heading, r_position, r_landmark
    ): ...
    def predict(self, v: float, delta: float, dt: float) -> None: ...
    def update_heading(self, theta_meas: float) -> None: ...
    def update_position(self, x_meas: float, y_meas: float) -> None: ...
    def update_landmark(self, range_meas, bearing_meas, landmark_xy: tuple[float, float]) -> None: ...

# harness.py -- replaces simulation.py's ParkingSimulation
class ParkingHarness:
    def __init__(
        self, vehicle: Vehicle, environment: Environment,
        planner: Planner, controller: Controller,
        seed=42, dt=0.1, v_max=1.5, a_max=0.8, k_acc=2.0, tol=0.4, brake_distance=2.0,
    ): ...
    def run(self, max_steps: int = 500) -> SimulationResult:
        # SimulationResult: true_history, estimated_history, covariance_history,
        # controls, success flag, collision flag, path
        ...
```

`Planner` and `Controller` as structural-typing protocols (not required base classes) is
intentional: `reeds_shepp.ReedsSheppPlanner` and `planning.hybrid_astar.HybridAStarPlanner`
both satisfy `Planner` without a shared inheritance hierarchy, and the same for the two
controllers — new algorithms can be added later without editing existing ones. `HasPose` extends
that same principle to controllers' inputs: a controller doesn't need to know or care whether
it's tracking a real `Vehicle` or a `PoseEstimateMsg`, only that whatever it's given has
`.x/.y/.theta`.

## 3. Milestones

- **M1 — Correct baseline: done.** Extracted the original prototype into the module layout
  above and fixed the degrees/radians bug. The planner ended up being more than a straight port:
  the originally-planned fixed Bezier curve turned out to be kinematically infeasible (see
  DESIGN.md section 6), so M1 now ships a **Dubins path planner** instead — still a single fixed
  path with no obstacle avoidance (that's still M2's job), but one that's guaranteed drivable.
- **M3 — Control: done, pulled ahead of M2.** `control/mpc.py` (nonlinear MPC via SLSQP) is
  implemented and selectable per run via `demo.py --controller {pure_pursuit,mpc}`. Pulled ahead
  of M2 because getting the controllers actually converging reliably was higher-value than
  obstacle routing for a first working end-to-end demo — see DESIGN.md section 7 for the
  head-to-head comparison this produced.
- **ME — State estimation & pub/sub architecture: done.** The single biggest realism gap in M1
  was that the controller and planner acted on perfect ground-truth pose — no sensor noise, no
  localization uncertainty at all. This milestone rebuilt the execution model around a pub/sub
  node graph (`messaging/`, `nodes/`, `harness.py`, replacing `simulation.py` outright) and added
  an EKF (`estimation/ekf.py`) that fuses noisy odometry, a compass, a periodic position fix, and
  opportunistic landmark range-bearing readings into the pose estimate the controller and planner
  actually act on. See DESIGN.md section 5 for the estimator design and section 2 for the
  architecture. Net effect on outcomes: both controllers still succeed reliably on the two
  obstacle-free scenarios (now evaluated as a 5-seed success rate rather than a single
  deterministic run — Section 4), and every scenario still never collides, across every seed.
- **MV — Real-data EKF validation: done.** `test_ekf.py` only proves the filter is
  self-consistent against noise the project generates for itself. `core/validation/` adds an
  independent check: the same, unmodified EKF replayed against real KITTI Odometry ground-truth
  poses (a committed 300-frame excerpt with real turns, `core/data/kitti/`), using the same
  noise defaults as `SensorNode`/`VehicleNode`. Result: 0.85 m RMSE with corrections vs. 4.97 m
  dead-reckoning-only on the excerpt — an 83% reduction — see DESIGN.md section 5,
  "Validation against real data."
- **H1 — Adaptive cruise control: done.** First highway-mode milestone: `control/acc.py`
  (`IDMController`, `MpcAccController`), a longitudinal-only node set (`lead_vehicle_node.py`,
  `ego_longitudinal_node.py`, `radar_node.py`, `acc_controller_node.py`) and
  `highway_harness.py`, validated against a real 78s NGSIM leader/follower trajectory
  (`validation/ngsim_loader.py`, `acc_validation.py`). See DESIGN.md section 11 for the
  controller comparison and two real findings from building it: IDM's raw formula needs an
  explicit physical deceleration floor (caught by a standalone unit test before any node existed
  to catch it later), and nominal MPC's hard gap constraint can still be violated in a
  standstill-recovery edge case (fixed by picking `min_gap` from measured real-world erosion, not
  from what looks right on paper).
- **H2 — Sensor fusion, extend the EKF with a speed state: done.** `estimation/ekf.py` gained
  `predict_with_speed_state`/`update_speed` (4-state `[x,y,theta,v]` mode) as new methods
  alongside the original `predict` (left untouched); `_apply_update` and the three `update_*`
  correction methods were generalized to size off `len(self.x)` instead of a hardcoded 3, verified
  as a true no-behavior-change refactor by re-checking the KITTI validation's RMSE came back
  bit-for-bit identical (0.845 m / 4.966 m / 83.0%). New `speed_estimator_node.py` wraps the
  4-state filter; `AccControllerNode` now acts on the fused `EgoSpeedEstimateMsg`, not true speed.
  Effect on H1's outcomes: realized minimum gap shifted by only 4-8 cm — see DESIGN.md section 12
  for the full account, including why H1's use of the 4-state filter has two degenerate
  dimensions (x/y/theta don't do much until H3 adds real lateral motion) and why that's an
  accepted forward-compatibility tradeoff, not an oversight.
- **H3 — Lane centering: done.** `control/lane_centering.py`'s `StanleyController`, validated
  against `data/ngsim/lane_centerline.csv` — a real lane centerline derived from ~10,400 actual
  vehicle positions (not hand-authored), with genuine curvature (1.76m end-to-end lateral drift
  over 642m). `validation/lane_centering_validation.py` checks closed-loop convergence against
  real drivers' own lateral scatter on the same lane (the plausibility bar, since there's no
  single "correct" in-lane position to replay against the way H1 had a real leader trajectory).
  See DESIGN.md section 12's H3 entry for a real sign-convention bug caught by a standalone
  convergence test before any validation module existed to catch it later (the controller
  diverged from a 2m offset to 374m within 30 seconds when the cross-track-error sign was
  backwards), and for what's explicitly deferred: combining H1/H2's ACC with H3's Stanley control
  into one closed loop over a single `Vehicle` is real follow-up work, not done in this pass (see
  H5 below for where that landed).
- **M2 — Planning: done.** `planning/reeds_shepp.py` (Reeds-Shepp curves, CSC family only) and
  `planning/hybrid_astar.py` (obstacle-aware search using Reeds-Shepp as heuristic/analytic
  connector) replace the fixed Dubins path with `HybridAStarPlanner` as the default planner
  (`demo.py`/`test_simulation.py`), planning from the pose *estimate* as before — no `PlannerNode`
  changes needed, exactly as anticipated. Validated against the three scenarios that used to stall
  safely rather than reach the spot: `perpendicular_flanked` and `perpendicular_obstructed_lane`
  now succeed 5/5 seeds with both controllers; `parallel_between_cars` (the tightest, needing a
  genuine reverse-gear cusp) succeeds 5/5 with MPC but is a documented, measured *unsafe*
  combination with Pure Pursuit (5/5 collisions, not flaky — see DESIGN.md section 6's M2 entry and
  the "Found while building M2" entries below for the two real issues this surfaced: reactive
  braking defeating intentional close passes, and Pure Pursuit's pre-existing curvature-limit
  weakness becoming an actual safety failure rather than just an efficiency loss).
- **H5 — Full closed-loop highway drive: done, both phases.** **Phase A** (H1+H2+H3 on one
  Vehicle): `full_highway_harness.py`'s `FullHighwayHarness` + `highway_vehicle_node.py`'s
  `HighwayVehicleNode` (a new real-`Vehicle` plant node, accel-in like `EgoLongitudinalNode`, not
  desired-speed-in like `VehicleNode` — see DESIGN.md section 12's H5 entry for why that
  distinction matters) + `lane_geometry.py` (arc-length along the real centerline). Found and fixed
  three real bugs in `estimation/ekf.py`'s 4-state mode along the way, all masked until now by H1's
  `delta=0` straight-line-only use: a speed-propagation timing bug, a missing steering-uncertainty
  process-noise term, and a complete absence of absolute heading/position correction (fixed by
  reusing parking's `CompassMsg`/`PositionFixMsg`/`update_heading`/`update_position` unchanged).
  **Phase B** (routing H4's intersection logic through the same composed loop): built deliberately
  second, once Phase A was verified — `longitudinal_arbiter_node.py`'s `LongitudinalArbiterNode`
  composes ACC's and `IntersectionNavigator`'s accel candidates via `min()`. Directly stress-tested
  the one real, non-obvious edge case this composition raises (ACC free to cruise unconstrained the
  whole approach, in principle risking a later, faster crossing of the stop-line detection window
  than `IntersectionNavigator` was ever validated at standalone) with a synthetic worst-case
  scenario rather than just reasoning about it — held up cleanly, now pinned as a permanent
  regression test. Full account, including the measured before/after numbers for both phases, in
  DESIGN.md section 12's H5 entry.
- **Real 2D intersection geometry for H4, including turning movements** (KNOWN_BUGS.md entry 4, now
  closed): `control/intersection_geometry.py` (a real 2-road layout, right-hand-traffic lane offsets,
  a genuine conflict-zone box, `is_to_the_right` derived from actual travel headings, `build_turn_path`
  reusing `DubinsPlanner` for curved connectors between approaches) + `intersection2d_harness.py`
  (runs any number of vehicles — each still an unmodified `IntersectionNavigator` — through it,
  deriving every `OtherVehicleStatus` from another vehicle's real simulated state instead of a
  script, plus a left-yields-to-oncoming-straight-traffic rule for turning vehicles). Closes the "not
  full 2D... not more than two approaches... no notion of actual crossing paths... no turning
  movements" gap in full. **Real findings while building it**: (1) the geometric constants aren't
  independent of `VEHICLE_RADIUS` — the first version's `conflict_half_width=4.0` against
  `lane_offset=3.0` let a vehicle waiting at its own stop line land within collision range (2.97m
  apart, under the 5.0m two-radius threshold) of the perpendicular through-lane, caught directly by a
  3-way scenario test; (2) adding turning movements needed the conflict-zone size and the curve's own
  `turn_lead` distance to be mutually consistent too, or a stopped turning vehicle ended up
  geometrically mid-curve instead of on its straight lane (`conflict_half_width` widened again, to
  9.5, with a runtime guard against regressing the inequality); (3) the left-yield rule's first two
  versions each produced a genuine circular deadlock between vehicles (found via a random mixed-turn
  sweep, not the hand-picked scenarios that looked fine alone) — fixed by making the rule strictly
  one-directional and position- rather than state-gated. Full account, including the residual (safe
  but not always live) multi-vehicle wait-cycle limitation, in KNOWN_BUGS.md entry 4. Verified the
  collision check itself is a real safety net, not vacuously true, by substituting a navigator that
  always claims right-of-way and confirming it gets caught colliding on a pairing the compliant
  version already proved safe.
- **M4 — Sensing & re-planning: done.** Sensor is a multi-beam array (`[-0.6, -0.3, 0.0, 0.3, 0.6]`
  rad front cone -- a mirrored rear cone was added later, in code review, once the speed governor
  below needed to actually see behind the vehicle too; see `controller_node.py`'s module docstring)
  and braking already checked all beams, not just the front one. What was still open --
  wiring `PlannerNode` to re-plan (not just `ControllerNode` to brake) when `SensorNode` reports an
  obstacle the current path didn't account for -- is now built: `ControllerNode` publishes
  `replan_request` once its speed governor (KNOWN_BUGS.md entry 2's fix) has been binding for
  `STALL_TICKS` consecutive ticks, and `PlannerNode` re-plans from the latest pose estimate against
  `environment.obstacles` read live (not a snapshot from the first plan), capped at `max_replans`
  attempts. Verified directly (`tests/test_replanning.py`, since every scenario this project ships
  is static and fully known up front, so nothing exercises this in the 5 shipped scenarios
  themselves): an obstacle dropped into the environment mid-run, invisible to the original plan, is
  fully visible to and routed around by a re-plan, with a real measured clearance margin, not just
  "didn't crash." A real, separate residual found while building this -- a re-plan's resulting
  route can still be tight enough that `ControllerNode`'s fixed `stopping_buffer` throttles
  progress along it too -- was tracked as KNOWN_BUGS.md's (renumbered) entry 3, not treated as
  unresolved M4 scope, and has since been closed too: `ControllerNode` now uses a smaller,
  tracking-aware buffer while accurately following a path a planner already verified, derived from
  the planner's own `safety_margin` via `ParkingHarness` (see controller_node.py's "Tracking-aware
  buffer" docstring entry and KNOWN_BUGS.md entry 3 for the real parameter sweep this took).
- **M5 — Visualization polish: done.** `visualization/animate.py` shows true vs. estimated
  trajectory, a covariance ellipse, and the planned path on the top-down view (landed as part of
  ME, since the whole point of adding an estimator is visible directly in that comparison), plus
  two live telemetry panels added afterward: a speed-vs-time trace (the speed governor's
  throttling -- KNOWN_BUGS.md entries 2/3 -- is now visible as a real dip in the line, not just
  inferable from the vehicle slowing down on screen) and a polar ultrasonic-range display, one bar
  per beam at its angle relative to the vehicle's own heading, so the fan rotates rigidly with the
  vehicle the same way the real body-frame sensor does. `SimulationResult` gained `sensor_ranges`/
  `sensor_angles`/`dt` fields to carry the per-tick beam data out of `ParkingHarness.run()`
  (previously only `ControllerNode` ever saw obstacle_ranges; the harness now also subscribes to
  record a history, the same pattern it already used for true/estimated pose and covariance).
  Verified by rendering three real scenarios end to end (open lane, obstacle-flanked, and a full
  parallel-parking maneuver) and inspecting individual frames: the speed panel shows the governor's
  real oscillation during a tight reverse-forward sequence, and the sensor panel visibly shortens a
  beam exactly when the vehicle is close to an obstacle in that beam's direction, not just at rest.
- **M6 — Tests & CI**: 286 tests across both modes run in ~250s (with the optional `rl` extra
  installed; without it, `test_parking_env.py`/`test_rl_training.py`/`test_rl_comparison.py` skip
  via `pytest.importorskip`); no GitHub Actions workflow yet.

## 4. Testing strategy

Current (286 tests, ~250s -- up from ~100s pre-H5, almost entirely because `test_full_highway.py`
replays a real 78s/780-frame NGSIM trajectory through the full node graph, parametrized over
multiple controllers and seeds; the M2-era jump from ~20s to ~100s is explained in
IMPLEMENTATION.md's M2 section above):

- Kinematic checks: driving straight for N steps moves `x` by `v*N*dt` with `theta` unchanged; a
  fixed steering angle over time traces a circle of radius `L / tan(delta)`; `turning_radius`
  matches `wheelbase / tan(max_steer)`.
- Bus: publish delivers to all subscribers of a topic in order; unsubscribed topics don't error;
  a subscriber never receives messages published to a different topic.
- EKF: predict-only (zero odometry noise) matches the same closed-form bicycle-model arc used for
  `Vehicle` directly; each of the three correction types (heading/position/landmark) strictly
  reduces covariance trace; a predict-only step strictly grows it; over 200 predict+update cycles
  with fixed-seed bounded noise, estimation error stays bounded (a filter-consistency check, not
  just "it runs").
- Integration, run against `ParkingHarness` with `HybridAStarPlanner`: for every (scenario,
  controller) pair except the one documented exception (`parallel_between_cars` + Pure Pursuit,
  see M2's entry in Section 3), assert a **success rate ≥4/5 across 5 fixed seeds** (not
  single-run determinism — Section "Control" tradeoffs in DESIGN.md explains why a rate, not 100%,
  is the right thing to assert once real noise is in the loop) *and* assert not `.collision` on
  every seed — safety has to hold regardless of estimation noise. The one excluded combination gets
  its own pinned regression test asserting it *does* collide, so the exclusion can't silently go
  stale if a future change actually fixes it.
- Regression guard for the original degrees/radians bug: every scenario's `theta` values must
  fall in `[-pi, pi]`; a value like the old `90.0` is ~14 full rotations out of range and would
  fail this immediately.
- Real-data validation (parking/EKF): on the committed KITTI excerpt, the EKF's RMSE must be
  strictly lower than a dead-reckoning-only pass over the identical noisy odometry stream (the
  robust claim — no arbitrary accuracy number to pick), stays under a generous absolute bound
  (divergence guard), and is deterministic for a fixed seed.
- ACC unit checks: IDM accelerates toward `v0` with room ahead; IDM's output is clipped to a
  physical deceleration floor in a close/closing scenario (not the raw formula's unbounded
  value); MPC-ACC's output stays within its own bounds; both controllers follow a synthetic
  braking lead vehicle without the gap ever reaching zero.
- Real-data validation (ACC): on the committed NGSIM excerpt, both controllers must never collide
  (hard pass/fail, ground truth, same principle as parking's `test_never_collides`), land in a
  plausible gap range relative to the real recorded follower (not a strict match), and produce a
  deterministic result for a fixed seed.
- Lane-centering unit checks: converges from a lateral offset on both sides (not just one, since
  a sign bug can look correct from only one direction — this is exactly how one was found, see
  Section 6), steering stays within `max_steer`, near-zero speed doesn't blow up the correction.
- Real-data validation (lane centering): from three different initial offsets, Stanley's
  closed-loop tracking error must settle under real drivers' own lateral scatter on the same real
  NGSIM lane (the plausibility bar — there's no strict target, no single "correct" in-lane
  position) and produce a deterministic result.
- Planning unit checks (`test_planning.py`, added with M2, both `ReedsSheppPlanner` and
  `HybridAStarPlanner`, across all 5 scenarios): the returned path (a) starts at `start` and ends
  at `goal` within tight tolerance (both planners land exactly on the given poses by construction,
  not the controller's noisy `tol`), (b) never exceeds curvature `1/turning_radius` at any point
  (the same style of check that caught the infeasible-Bezier bug during M1 — see DESIGN.md section
  6), (c) for `HybridAStarPlanner` specifically, on the 3 obstacle scenarios, never comes within
  the vehicle's radius of any obstacle.
- Full closed-loop drive (H5, `test_full_highway.py`), against real NGSIM data throughout: never
  collides with the replayed real leader (hard pass/fail, ground truth); cross-track error settles
  under H3's own real-driver-scatter bar, now measured on the EKF-fused pose under real composed
  control rather than assumed identical to the ground-truth-fed standalone result; gap lands in a
  plausible range relative to the real follower (same 0.2x-3.0x band as H1's own check); one run
  asserts collision-safety and lane-tracking together (not just as separate parametrized checks) to
  catch interaction bugs neither alone would surface; deterministic for a fixed seed; a direct
  regression test for the H2 `delta=0.0` hardcode fix (feed a real nonzero steering reading, assert
  the filter's heading actually responds — impossible under the old hardcoded behavior). Phase B
  (intersection routing) adds: never crosses the stop line without having stopped, with a real
  non-blocking lead vehicle present and ACC free to cruise unconstrained the whole approach (the
  specific accel-arbiter edge case DESIGN.md section 12's H5 entry raises and stress-tests, not
  just reasons about); the four right-of-way branches H4's own standalone tests already cover
  (yield to first-arrived, proceed when ego arrived first, yield to the right on simultaneous
  arrival, don't yield to the left), re-derived against this harness's own real approach dynamics
  rather than reusing H4's exact timings verbatim.

Both `test_sensors.py` and `test_control.py` (below) closed this gap -- `test_simulation.py`'s
integration-level coverage remains, unchanged, as the end-to-end check on top of them.

## 5. Dependencies & running

```
numpy
scipy        # MPC (scipy.optimize.minimize, SLSQP), and Hybrid A* heuristic support later
matplotlib
pyyaml       # scenario file loading
pytest
```
`scipy.optimize.minimize(method="SLSQP")` turned out sufficient for `control/mpc.py` (and, with
the `constraints` argument, for `control/acc.py`'s MPC too); `cvxpy` was the planned fallback if
that proved awkward, but wasn't needed either time. No new dependency was needed for the
EKF/pub-sub milestone (`Bus` is ~15 lines of pure Python, the EKF is plain NumPy linear algebra),
and `validation/ngsim_loader.py` deliberately uses the standard library `csv` module rather than
pandas — a filter/sort over a few hundred rows doesn't need a dataframe library, and the project
has stayed dependency-light on purpose every time so far.

```
pip install -e .
pytest                                                    # run the test suite
python -m core.demo <scenario>                       # Pure Pursuit, show the animation
python -m core.demo <scenario> --controller mpc       # MPC instead
python -m core.demo <scenario> --seed 7               # override the scenario's RNG seed
python -m core.demo <scenario> --save out.gif         # save a GIF for the README
python -m core.validation.kitti_ekf_validation --plot out.png   # EKF vs. real KITTI data
python -m core.validation.acc_validation --controller mpc --plot out.png  # ACC vs. real NGSIM data
python -m core.validation.lane_centering_validation --plot out.png  # Stanley vs. real NGSIM lane geometry
```

## 6. Known issues

Resolved during M1:

- The original prototype constructed `Vehicle(theta=90.0)` in one scenario and
  `Vehicle(theta=np.pi/6)` in another — some scenarios passed degrees, some radians, into a
  model that only accepts radians. Root cause of the "barely works now just makes bigger circles"
  commit: a 90.0-*radian* heading is ~14 full rotations from what was intended. Fixed by making
  every scenario YAML radians-only, with a regression test enforcing it (Section 4).
- Pure Pursuit's lookahead-target search scanned the *entire* path array for "the first point at
  distance >= lookahead," from index 0 every time, instead of searching forward from the
  vehicle's current nearest point. Once the vehicle had traveled more than `lookahead` past the
  path's start, the start point became "far enough away" again and got re-selected as the
  target — steering the vehicle backward toward where it began. This was the actual mechanism
  behind the "makes bigger circles" symptom, inherited unchanged from the original prototype and
  not caused by the degrees/radians bug alone. Fixed in `control/pure_pursuit.py` by searching
  forward from the nearest-point index.
- Nothing clipped a controller's commanded steering angle to the vehicle's physical
  `max_steer` limit. Pure Pursuit does not self-limit by construction (its arctan2 formula can
  return arbitrarily large angles when the lookahead target is close and off-axis), so it was
  commanding ~65-degree steering on a car limited to ~34 — direct cause of an oscillating,
  looping trajectory. Fixed by clipping `delta` to `vehicle.max_steer` in `simulation.py` (the
  natural single enforcement point, since it's the actuator layer) and additionally in Pure
  Pursuit itself for correctness when used standalone.
- `brake_distance` (0.5 m) was picked arbitrarily, smaller than the vehicle's actual stopping
  distance at `v_max`/`a_max` (~1.4 m). The sensor detected obstacles in time, but the vehicle
  couldn't decelerate fast enough within the remaining gap — real collisions in scenarios that
  should have stalled safely. Fixed by sizing `brake_distance` from the stopping-distance formula
  plus margin (see DESIGN.md section 8).
- No obstacle-avoidance routing exists yet — the planner brakes to a stop when the sensor detects
  something close, it never routes around what it detects. This is by design for M1 (see DESIGN.md
  section 6) and is what M2 (Hybrid A*) resolves.
- Scenarios are now data (`scenarios/*.yaml`), not hardcoded Python dicts mixed with
  animation/plotting code.
- Automated tests now exist (Section 4); CI (GitHub Actions) is still open, tracked under M6.

Found and fixed during the state-estimation/pub-sub milestone (ME):

- The success tolerance (`tol=0.3`) was tuned against a *noise-free* baseline. Once real sensor
  noise entered the loop, Pure Pursuit's known near-goal limit cycle (its lookahead target snaps
  to the final path point once everything's within `lookahead`, so it orbits rather than
  converging exactly) settled at a radius comparable to that tolerance — runs that were clearly
  "parked" by any reasonable standard were failing the check by a few centimeters. Fixed by
  loosening `tol` to 0.4 (still tight, but consistent with acting on a noisy estimate rather than
  ground truth) and by evaluating success as a rate across 5 seeds instead of asserting a single
  run deterministically, since noise legitimately produces occasional misses even at a reasonable
  tolerance.
- The EKF's initial covariance and per-measurement noise values (`p0`, `r_heading`, `r_position`,
  `r_landmark`) are hand-picked plausible defaults (see `harness.py`'s `ParkingHarness.__init__`),
  not fit to any real sensor datasheet — reasonable for a simulation whose sensors are themselves
  synthetic, but worth stating explicitly rather than implying they're calibrated to something.

Found during the real-data validation milestone (MV):

- KITTI's poses-only download doesn't bundle per-frame timestamps, and its ground-plane axis
  convention isn't the obvious "position x/y" you'd assume without checking — the ground plane is
  camera **x**/**z** (not x/y) and heading is rotation about camera **y**, a direct consequence of
  KITTI's camera-frame convention (x-right, y-down, z-forward). Getting this wrong wouldn't have
  crashed anything — `kitti_loader.py` would have silently produced a plausible-looking but
  physically wrong trajectory. Caught by an explicit empirical check before trusting the loader:
  the extracted heading has to track the actual direction of travel between consecutive frames on
  a sequence with real turns, which it does (mean deviation ~0.12 rad, consistent with real
  vehicle slip/differencing noise) — not just "the code runs without an exception."

Found during pre-merge review (a full-diff pass against `main` before merging, plus a broader
200-run collision sweep across seeds beyond what the test suite covers):

- `VEHICLE_RADIUS` (the ego vehicle's own collision-circle radius, `harness.py`) was 0.3 m —
  roughly four times smaller than the ~1.3 m used for parked-car obstacles in every scenario,
  even though the ego vehicle is a comparably-sized real car. This doesn't fail loudly: it just
  makes `_collided()` under-report real collisions, silently passing safety checks it shouldn't.
  Fixed to 1.0 m, consistent with the obstacle scale (see DESIGN.md section 8).
- Fixing `VEHICLE_RADIUS` alone then broke `test_never_collides` on several obstacle scenarios:
  `brake_distance` (3.0 m as of this fix, was 2.0) has to cover stopping distance **plus the
  vehicle's own radius**, not stopping distance alone, since the sensor reading it's compared
  against measures to the obstacle's surface but collision is checked between the two circles'
  *centers*. The two fixes are coupled — sizing one constant realistically changes what the other
  needs to be (see DESIGN.md section 8). Re-verified with a 200-run sweep (20 seeds x 5 scenarios
  x 2 controllers), zero collisions.
- `visualization/animate.py`'s axis bounds were computed from true trajectory + obstacles + spot
  only, never from the *planned* path. For the three scenarios specifically designed to show a
  plan driving toward an obstacle before the vehicle safely stalls, the planned path extends well
  past where the truncated true trajectory does — so the gray "planned" line could render outside
  the visible axes in exactly the demos meant to showcase it. Fixed by including `result.path` in
  the bounds calculation. (Also fixed while in this file: the rendered vehicle rectangle was
  1.0m x 0.6m, comically small for a 2.7m-wheelbase car — bumped to 4.5m x 1.8m.)
- `MPCController` has its own internal rollout `dt` (default 0.1), independent of
  `ParkingHarness`'s `dt` (also default 0.1) — nothing wired them together. Both defaults happen
  to match today, so this wasn't causing an active failure, but it's a silent-desync footgun: a
  future `ParkingHarness(..., dt=0.05)` would leave the MPC predicting against the wrong step
  size with no error. Fixed by having `ParkingHarness.__init__` set `controller.dt = dt`
  whenever the controller has one, so the harness's `dt` is the single source of truth rather
  than something every caller has to remember to keep in sync by hand.
- `planning/dubins.py` and `validation/kitti_loader.py` each reimplemented the same
  wrap-to-`[-pi, pi]` formula already provided by `vehicle.wrap_angle` instead of importing it.
  Not a correctness bug (the duplicated formula was correct), but three copies of the same logic
  is three places a future change has to remember to touch. Both now import and use `wrap_angle`.

Found while building H1 (adaptive cruise control):

- `NgsimTrajectory` extraction initially grouped rows by `(vehicle_id, frame_id)` alone. NGSIM's
  `vehicle_id` and `frame_id` both reset across the dataset's separate recording sub-periods, so
  the same `(vehicle_id, frame_id)` pair can legitimately appear more than once, silently mixing
  two different real vehicles' data into what looked like one trajectory (caught because the
  resulting "trajectory" had more rows than the frame range should allow — an easy thing to miss
  if you don't sanity-check row counts). Fixed by keying on `global_time`, NGSIM's one genuinely
  monotonic, non-resetting timestamp, and verifying the extracted excerpt's consecutive timestamps
  are all exactly 100ms apart before committing it.
- `IDMController`'s raw formula returned -1309 m/s² for a plausible close-range, fast-closing
  scenario in a standalone sanity check, run *before* the controller was wired into any node.
  The model's `(s*/gap)^2` interaction term is mathematically unbounded as gap shrinks; nothing in
  the textbook formula itself imposes a physical floor. Fixed by clipping to `a_min` (default -9
  m/s², ~1g). Catching this via an isolated unit test rather than a failing integration test is
  the point of testing controllers standalone before wiring them into a harness — same lesson as
  M1's original Bezier-curve curvature bug, just one component earlier in the pipeline this time.
- `MpcAccController`'s hard `gap(t) >= min_gap` constraint was violated in the closed-loop
  simulation despite being enforced inside the optimizer — traced to a standstill-recovery
  scenario (real NGSIM data includes a full stop) where, once the ego is already closer than
  `min_gap` while both vehicles are stopped, no feasible acceleration sequence can satisfy the
  constraint (moving apart from a standstill would require reversing, which the ego can't do).
  SLSQP returns its best constraint-violating attempt rather than failing loudly in this case.
  Measured the actual erosion (~0.5 m below the nominal target, fairly consistent across several
  `min_gap` values tested) and set the default from that measurement (3.0 m) rather than picking
  a value that merely looked sufficient on paper. See DESIGN.md section 11 for the full
  explanation and section 12 for the proper long-term fix (robust/stochastic MPC).

Found while building H3 (lane centering):

- `StanleyController`'s first implementation defined cross-track error with the sign backwards —
  the correction term steered *away* from the path instead of toward it. This didn't raise an
  exception or look obviously wrong from reading the formula; it just diverged, from a 2 m initial
  offset to 374 m within 30 simulated seconds, caught by a direct standalone convergence check run
  before building the validation module on top of it (same lesson as H1's IDM deceleration-floor
  bug: test controllers in isolation before wiring them into anything that could mask the
  failure). `tests/test_lane_centering.py` now checks convergence from both directions
  specifically, since a flipped sign can look correct from only one side of the path.
- `lane_centering_validation.py`'s first `settle_distance` default (50 m) was calibrated against
  a small initial offset and didn't generalize: at highway speed, convergence distance scales with
  how large the initial offset is (measured directly: ~66 m for a 1.5 m offset, ~94 m for 3.0 m),
  so a 3.0 m offset test failed the real-driver-scatter plausibility check simply because it
  hadn't finished converging yet at the point the check was measured, not because tracking was
  actually bad. Fixed by measuring real convergence distance across the offsets the module is
  actually exercised with and picking a `settle_distance` (150 m) with real margin, rather than a
  round number that happened to work for the first offset tried.

Found while building M2 (Hybrid A* + Reeds-Shepp):

- `ControllerNode`'s reactive sensor-based braking (brakes whenever any obstacle reading is closer
  than `brake_distance`) was written for M1's Dubins planner, which never deliberately gets close
  to an obstacle -- so "sensor sees something within `brake_distance`" always meant "unplanned
  hazard, stop." Once `HybridAStarPlanner` started producing paths that *intentionally* pass within
  a vehicle length of a parked car as part of a valid avoidance maneuver, that same logic triggered
  on every such approach and stalled the vehicle before it ever reached the maneuver -- measured
  directly before assuming anything about *why* the new planner "wasn't working" (`perpendicular_flanked`:
  0/5 success, 0/5 collision, i.e. braking safely forever, not a planning or tracking failure). Fixed
  by `hybrid_astar.brake_distance_for(planner)`, deriving a smaller `brake_distance` from the
  planner's own guaranteed worst-case clearance (`vehicle_radius + safety_margin`, minus a buffer)
  so genuine avoidance maneuvers never falsely trigger it; `DubinsPlanner`/`ReedsSheppPlanner` keep
  `ParkingHarness`'s original `brake_distance=3.0` default, since braking is the *only* thing
  preventing a collision for planners that ignore obstacles.
- Even after that fix, `parallel_between_cars` (needing a genuine reverse-gear cusp between two
  close parked cars) still failed for Pure Pursuit -- but this turned out to be a real, consistent
  controller limitation, not a second planner bug: measured 5/5 collisions across seeds, unchanged
  across a wide sweep of `brake_distance` and the planner's `safety_margin` (a wider margin just
  traded the collision for Pure Pursuit never converging at all, ruling out "just needs more
  buffer" as the fix). This is DESIGN.md section 7's already-documented "no margin when curvature
  is already at the limit" weakness, concretely realized as an actual safety failure once a planner
  produces curvature-saturated, obstacle-hugging paths for it to track, rather than the mild
  efficiency loss it was previously only measured as. MPC's constraint-respecting rollout stays
  collision-free and converges reliably (5/5 seeds) given a higher step budget (up to ~880 of a
  raised 1000-step cap -- Hybrid A*'s avoidance routes are longer than M1's direct paths ever
  needed to be). Documented as a scoped, pinned exception (`tests/test_simulation.py`'s
  `UNSAFE_COMBINATIONS`, with its own regression test asserting the collision still happens) rather
  than papered over with a numeric hack -- the honest outcome of measuring rather than assuming.
- `dubins.py`'s `DubinsPlanner.plan()` was refactored to extract `_solve_csc`/`_csc_points` (family
  selection and segment-walking, previously inlined) so `reeds_shepp.py` could reuse the exact same
  CSC formulas rather than re-deriving them -- verified as a true no-behavior-change refactor by
  re-running the full suite before adding anything new on top of it (same discipline as H2's EKF
  generalization, re-checking the KITTI RMSE came back bit-for-bit identical).
- The backward-gear half of `reeds_shepp.py` almost became a second implementation of Dubins's CSC
  math with signs/headings flipped (the textbook reflect/timeflip transform), until hand-deriving
  the kinematics directly showed a simpler equivalence: a backward-gear CSC path from A to B is
  exactly the forward CSC solve from B to A with its point array reversed, headings untouched.
  Verified against `_arc_points`' actual formula for a turning primitive (two independent
  derivations landed on identical points) *before* building `hybrid_astar.py` on top of it -- the
  same "verify before you build on it" discipline as the KITTI axis-convention check and H3's
  Stanley sign-convention bug, applied one level earlier in the pipeline this time.

Found while building H5 (full closed-loop highway drive, Phase A) -- three bugs, all in
`estimation/ekf.py`'s 4-state mode, all invisible until now for the identical reason: H1's
straight-line-only use kept `delta` at exactly 0 forever, zeroing out every code path each one
lived in. Caught by direct empirical comparison (true vs. estimated pose, tick by tick, then a
zero-noise isolation run) rather than staring at the formulas -- the same "measure, don't assume"
discipline this project applies everywhere, just needed three rounds of it in a row here:

- `predict_with_speed_state` propagated x/y/theta using the *prior* tick's speed, then updated the
  speed state separately -- but every plant node (`EgoLongitudinalNode`, and the new
  `HighwayVehicleNode`) computes the *new* speed first and calls `Vehicle.update` with that. First
  symptom: cross-track error grew to tens of meters while the estimate itself looked fine and
  Stanley's commanded steering stayed near zero the whole time -- the controller wasn't broken, it
  was correctly holding the *estimate* on the path while the *true* vehicle drifted away underneath
  it, because the estimate's own process model didn't match the real plant it was supposed to be
  tracking. Fixed by computing `v_new = v + accel*dt` first and propagating x/y/theta with `v_new`
  throughout (Jacobian re-derived to match).
- Even after that fix, drift persisted -- smaller, but still meters over a full run. Isolated with
  a zero-noise experiment first (confirmed the process model itself was now exactly correct: near-
  zero drift with noise stds set to ~0) before looking further, which pointed squarely at *how*
  realistic noise was being handled rather than at the propagation formula again. `Q` only ever
  modeled acceleration uncertainty (`q[3,3]`), never steering-reading uncertainty's effect on
  theta/x/y the way the 3-state `predict()`'s full input-Jacobian `V @ M @ Vᵀ` already does --
  invisible under H1 (every `tan(delta)`-dependent term is zero there), and it made the filter
  overconfident, weighting later corrections too lightly. Fixed by generalizing to the same
  input-Jacobian approach, reusing the existing (previously-unread-by-this-method) `odom_delta_std`
  parameter.
- Residual drift *still* remained even with both fixes, isolated to: the highway EKF has no
  absolute heading or position correction at all -- no compass, position fix, or landmark
  equivalent, unlike parking's full three-sensor suite. Invisible under H1/H2 because nothing had
  ever consumed the estimate's x/y/theta before (only `.speed` was read) -- pure dead-reckoning
  drift is harmless when nobody's looking at where it drifts to. Once Stanley started steering off
  the estimate, realistic steering noise random-walked the heading estimate away from truth over
  the real ~600m/78s NGSIM scenario. Fixed by having `HighwayVehicleNode` also publish an always-on
  noisy compass and a low-rate noisy position fix, reusing `CompassMsg`/`PositionFixMsg`/
  `update_heading`/`update_position` completely unchanged -- exactly the reuse H2's original design
  already anticipated by generalizing those methods to `len(self.x)` instead of a hardcoded 3.

With all three fixed: cross-track error never exceeds its initial offset and settles to an RMS of
~0.27-0.33m across seeds (H3 standalone's own bar is 0.46m), and ACC's realized gap (~2.2m) matches
H1/H2's own standalone numbers -- confirming H1/H2's validated dynamics really did carry over
unchanged once `HighwayVehicleNode` reused `EgoLongitudinalNode`'s exact integration physics. See
DESIGN.md section 12's H5 entry for the full account.
