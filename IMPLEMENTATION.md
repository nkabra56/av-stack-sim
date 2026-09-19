# Implementation Plan

This is the build spec for the architecture described in [DESIGN.md](DESIGN.md). Parking-mode
milestones M1 (correct baseline), the control half of M3 (MPC), ME (state estimation + pub/sub
architecture), MV (real-data EKF validation against KITTI), **M2 (Hybrid A* + Reeds-Shepp
obstacle-aware planning)**, and **M4 (sensing & re-planning)** are done. All four highway-mode
milestones are also done: **H1 (adaptive cruise control)**, **H2 (fused ego speed via an extended EKF)**, **H3
(lane centering)**, and **H4 (intersection navigation)**, each originally validated
standalone, and **H5 (the full closed-loop highway drive, H1+H2+H3+H4 all composed onto one
Vehicle) is now done too**, built in two deliberately sequenced phases: see DESIGN.md section 12's
H5 entry. The highway side is now fully integrated; the parking side's M5 (visualization polish) and M6 (tests) are also done: see Section 3
for the full roadmap.

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
                       # ControlCmdMsg; highway: LeadVehicleStateMsg, EgoLongitudinalStateMsg,
                       # RadarMsg, LongitudinalCmdMsg, AccelOdometryMsg, SpeedometerMsg,
                       # EgoSpeedEstimateMsg (H2, now carries x/y/theta too); H5:
                       # LateralCmdMsg, SteeringOdometryMsg, EgoHighwayStateMsg; see
                       # DESIGN.md section 2
  estimation/
    __init__.py
    ekf.py             # ExtendedKalmanFilter: predict + 3 correction types (3-state, parking) +
                       # predict_with_speed_state/update_speed (4-state, H2/highway); H5's fixes to
                       # the 4-state model: DESIGN.md section 12
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
    highway_vehicle_node.py  # H5: real 2D Vehicle plant for the closed-loop drive, accel-in
                       # (not desired-speed-in like VehicleNode), publishes on H1's ego_state
                       # topic unchanged plus a new full-pose topic and steering/compass/
                       # position-fix readings, see section 12's H5 entry
    lane_centering_node.py   # H5: wraps StanleyController, acts on the fused pose+speed estimate
    longitudinal_arbiter_node.py  # H5 Phase B: min() over accel candidate topics, ACC's real-
                       # lead-vehicle accel vs. IntersectionNavigator's stop-line accel
    intersection_controller_node.py  # H5 Phase B: wraps IntersectionNavigator, feeds it the
                       # fused estimate (not ground truth, a first for H4), publishes an accel
                       # candidate for the arbiter
    other_vehicle_script_node.py  # H5 Phase B: thin Bus adapter for H4's existing
                       # OtherVehicleScript/other_vehicle_present_from, unchanged
  harness.py            # tick-based executor (parking mode): owns the Bus, builds all 5 nodes
  highway_harness.py       # tick-based executor (ACC/H1+H2 mode): owns the Bus, builds the
                       # longitudinal nodes: mirrors harness.py's structure, kept separate
                       # rather than forcing a shared base class before H3 shows what's common
  intersection_harness.py    # direct simulation loop (no Bus, since H4 has no sensor noise/fusion
                       # to decouple) for IntersectionNavigator scenarios; see section 12's H4 entry
  full_highway_harness.py    # H5: tick-based executor combining H1+H2+H3 on one real Vehicle:
                       # kept separate from harness.py/highway_harness.py, same reasoning as
                       # highway_harness.py's own docstring for staying separate from harness.py,
                       # see section 12's H5 entry
  planning/
    __init__.py
    dubins.py            # M1 baseline: curvature-feasible fixed path, no obstacle avoidance.
                       # Kept as a documented reference (demo.py --planner dubins); also the
                       # single source of truth for the CSC formulas reeds_shepp.py reuses
    reeds_shepp.py        # M2: Dubins + reverse gear (CSC plus optional CCC, see DESIGN.md section 6)
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
    foxglove_export.py    # 3D MCAP scene for Foxglove (optional `viz` extra)
    web_export.py         # scene data for the browser 3D viewer in docs/viewer/
    scenery.py            # signs, signals, crosswalks, sidewalks, buildings, trees, rails (visual only)
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
  test_ukf.py             # DESIGN.md section 10: UKF alternative to the EKF, same closed-form-arc
                       # and filter-consistency checks as test_ekf.py, plus circular-mean handling
  test_ukf_comparison.py    # the actual EKF-vs-UKF head-to-head numbers, pinned as a regression
  test_kitti_ekf_validation.py  # EKF vs. dead-reckoning-only on the committed real KITTI excerpt
  test_planning.py        # endpoint + curvature checks (both planners), obstacle-clearance
                       # check (Hybrid A* only); see DESIGN.md section 6's M2 entry
  test_mpc.py            # parking MPCController: falls back to the warm-started plan, not a
                       # fresh unconverged solve, when SLSQP doesn't converge
  test_sensor_node.py      # SensorNode dropout/latency modeling (DESIGN.md section 10,
                       # KNOWN_BUGS.md entry 7): delivery gating and delayed-message timing, direct
  test_sensor_robustness.py  # closed-loop collision safety under dropout/latency across real
                       # scenarios; pins the verified-safe range and the latency_margin fix itself
  test_parking_env.py      # DESIGN.md section 10: ParkingEnv's Gym contract (gymnasium's own
                       # check_env), reward shape, collision/success termination: fast, no training
  test_rl_training.py      # PPO trains end to end on ParkingEnv without crashing (a smoke test,
                       # not a convergence claim: see core/validation/rl_comparison.py for that)
  test_rl_comparison.py    # real, measured RL-vs-baseline numbers against a committed trained
                       # policy (core/data/rl/): fast, evaluates only, no training
  test_ros2_bridge.py      # message-conversion functions + Ros2Bridge's wiring, against fakes:
                       # see ros2_bridge.py's own docstring for what is/isn't verified and why
  test_simulation.py       # integration tests, harness-based, across scenarios x controllers x seeds
  test_foxglove_export.py   # MCAP export checks (need the `viz` extra)
  test_web_export.py        # viewer scene schema for every scene family, KITTI frame flip
  test_scenery.py           # sidewalk layout stays clear of the roads and box; signals vs stop signs
  test_signalized_intersection.py  # signal plan safety, no red-light entries, queues discharge
  test_background_traffic.py  # IDM followers keep a gap through the recorded stop
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
  test_full_highway.py     # H5 collision safety, real-data coherence, stop-line compliance: see section 4
pyproject.toml
DESIGN.md
IMPLEMENTATION.md
README.md
```

`simulation.py`/`ParkingSimulation` (the M1 direct-call loop) is retired: `harness.py` is now the
one way `demo.py` and the tests run a scenario, so there's a single execution path rather than two.

`test_planning.py` now exists (added with M2, once `planning/` had more than one planner to
compare against).

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
    def turning_radius(self) -> float: ...  # wheelbase / tan(max_steer), the single source
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
        # returns (v_desired, delta): delta is clipped to vehicle.max_steer by VehicleNode
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

# harness.py: replaces simulation.py's ParkingSimulation
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
both satisfy `Planner` without a shared inheritance hierarchy, and the same holds for the two
controllers, so new algorithms can be added later without editing existing ones. `HasPose` extends
that same principle to controllers' inputs: a controller doesn't need to know or care whether
it's tracking a real `Vehicle` or a `PoseEstimateMsg`, only that whatever it's given has
`.x/.y/.theta`.

## 3. Milestones

- **M1 (Correct baseline): done.** Extracted the original prototype into the module layout
  above and fixed the degrees/radians bug. The planner ended up being more than a straight port:
  the originally-planned fixed Bezier curve turned out to be kinematically infeasible (see
  DESIGN.md section 6), so M1 now ships a **Dubins path planner** instead, still a single fixed
  path with no obstacle avoidance (that's still M2's job), but one that's guaranteed drivable.
- **M3 (Control): done, pulled ahead of M2.** `control/mpc.py` (nonlinear MPC via SLSQP) is
  implemented and selectable per run via `demo.py --controller {pure_pursuit,mpc}`. Pulled ahead
  of M2 because getting the controllers actually converging reliably was higher-value than
  obstacle routing for a first working end-to-end demo: see DESIGN.md section 7 for the
  head-to-head comparison this produced.
- **ME (State estimation & pub/sub architecture): done.** The single biggest realism gap in M1
  was that the controller and planner acted on perfect ground-truth pose, no sensor noise, no
  localization uncertainty at all. This milestone rebuilt the execution model around a pub/sub
  node graph (`messaging/`, `nodes/`, `harness.py`, replacing `simulation.py` outright) and added
  an EKF (`estimation/ekf.py`) that fuses noisy odometry, a compass, a periodic position fix, and
  opportunistic landmark range-bearing readings into the pose estimate the controller and planner
  actually act on. See DESIGN.md section 5 for the estimator design and section 2 for the
  architecture. Net effect on outcomes: both controllers still succeed reliably on the two
  obstacle-free scenarios (now evaluated as a 5-seed success rate rather than a single
  deterministic run, Section 4), and every scenario still never collides, across every seed.
- **MV (Real-data EKF validation): done.** `test_ekf.py` only proves the filter is self-consistent against
  noise the project generates for itself. `core/validation/` adds an independent check: the same,
  unmodified EKF replayed against a committed 300-frame excerpt of real KITTI Odometry poses
  (`core/data/kitti/`), with the same noise defaults as `SensorNode`/`VehicleNode`. Results: DESIGN.md
  section 5, "Validation against real data."
- **H1 (Adaptive cruise control): done.** First highway-mode milestone: `control/acc.py`
  (`IDMController`, `MpcAccController`), a longitudinal-only node set (`lead_vehicle_node.py`,
  `ego_longitudinal_node.py`, `radar_node.py`, `acc_controller_node.py`) and
  `highway_harness.py`, validated against a real 78s NGSIM leader/follower trajectory
  (`validation/ngsim_loader.py`, `acc_validation.py`). See DESIGN.md section 11 for the
  controller comparison and the two findings from building it (IDM's deceleration floor, MPC's
  standstill infeasibility).
- **H2 (Sensor fusion, extend the EKF with a speed state): done.** `estimation/ekf.py` gained a 4-state
  `[x,y,theta,v]` mode (`predict_with_speed_state`/`update_speed`) and `speed_estimator_node.py` wraps
  it; `AccControllerNode` now acts on the fused `EgoSpeedEstimateMsg`, not true speed. Design rationale
  and measured effect: DESIGN.md section 12.
- **H3 (Lane centering): done.** `control/lane_centering.py`'s `StanleyController`, validated against a
  real NGSIM-derived lane centerline (`data/ngsim/lane_centerline.csv`). See DESIGN.md section 12's H3
  entry; combining it with H1/H2 on one `Vehicle` is H5 below.
- **M2 (Planning): done.** `planning/reeds_shepp.py` (Reeds-Shepp curves: CSC, with CCC added later, see
  KNOWN_BUGS.md entry 5) and `planning/hybrid_astar.py` (obstacle-aware search using Reeds-Shepp as
  heuristic/analytic connector) replace the fixed Dubins path with `HybridAStarPlanner` as the default
  planner, planning from the pose *estimate* as before, with no `PlannerNode` changes. `perpendicular_flanked` and
  `perpendicular_obstructed_lane` now succeed 5/5 seeds with both controllers; `parallel_between_cars`
  (a genuine reverse-gear cusp) succeeds 5/5 with MPC and fails safe with Pure Pursuit (0/5 success,
  0/5 collisions). See DESIGN.md section 6's M2 entry and "Found while building M2" below.
- **H5 (Full closed-loop highway drive): done, both phases.** **Phase A** (H1+H2+H3 on one `Vehicle`):
  `full_highway_harness.py`'s `FullHighwayHarness`, `highway_vehicle_node.py`'s `HighwayVehicleNode`
  (a real-`Vehicle` plant node that takes acceleration in, like `EgoLongitudinalNode`) and
  `lane_geometry.py` (arc-length along the real centerline). **Phase B** routes H4's intersection logic
  through the same loop: `longitudinal_arbiter_node.py`'s `LongitudinalArbiterNode` composes ACC's and
  `IntersectionNavigator`'s acceleration candidates via `min()`, and a synthetic worst-case scenario is
  pinned as a regression test. Bugs found, measured numbers and the stress-test reasoning: DESIGN.md
  section 12's H5 entry.
- **Real 2D intersection geometry for H4, including turning movements** (KNOWN_BUGS.md entry 4, now
  closed): `control/intersection_geometry.py` (a real 2-road layout, right-hand-traffic lane offsets, a
  genuine conflict-zone box, `is_to_the_right` derived from actual travel headings, `build_turn_path`
  reusing `DubinsPlanner` for curved connectors) + `intersection2d_harness.py` (runs any number of
  vehicles, each an unmodified `IntersectionNavigator`, deriving every `OtherVehicleStatus` from another
  vehicle's real simulated state, plus a left-yields-to-oncoming-straight-traffic rule). Three real bugs
  surfaced while building it (constants coupled to `VEHICLE_RADIUS`, `turn_lead` versus the stop line,
  a left-yield deadlock) and the collision check was confirmed non-vacuous with a deliberately
  non-compliant navigator; see KNOWN_BUGS.md entry 4.
- **M4 (Sensing & re-planning): done.** Sensor is a multi-beam array (`[-0.6, -0.3, 0.0, 0.3, 0.6]`
  rad front cone, plus a mirrored rear cone added later, in code review, once the speed governor
  below needed to actually see behind the vehicle too; see `controller_node.py`'s module docstring)
  and braking already checked all beams, not just the front one. What was still open,
  wiring `PlannerNode` to re-plan (not just `ControllerNode` to brake) when `SensorNode` reports an
  obstacle the current path didn't account for, is now built: `ControllerNode` publishes
  `replan_request` once its speed governor (KNOWN_BUGS.md entry 2's fix) has been binding for
  `STALL_TICKS` consecutive ticks, and `PlannerNode` re-plans from the latest pose estimate against
  `environment.obstacles` read live (not a snapshot from the first plan), capped at `max_replans`
  attempts. Verified directly (`tests/test_replanning.py`, since every scenario this project ships
  is static and fully known up front, so nothing exercises this in the 5 shipped scenarios
  themselves): an obstacle dropped into the environment mid-run, invisible to the original plan, is
  fully visible to and routed around by a re-plan, with a real measured clearance margin, not just
  "didn't crash." A real, separate residual found while building this (a re-plan's resulting
  route can still be tight enough that `ControllerNode`'s fixed `stopping_buffer` throttles
  progress along it too) was tracked as KNOWN_BUGS.md's (renumbered) entry 3, not treated as
  unresolved M4 scope, and has since been closed too: `ControllerNode` now uses a smaller,
  tracking-aware buffer while accurately following a path a planner already verified, derived from
  the planner's own `safety_margin` via `ParkingHarness` (see controller_node.py's "Tracking-aware
  buffer" docstring entry and KNOWN_BUGS.md entry 3 for the real parameter sweep this took).
- **M5 (Visualization polish): done.** `visualization/animate.py` shows true vs. estimated
  trajectory, a covariance ellipse, and the planned path on the top-down view (landed as part of
  ME, since the whole point of adding an estimator is visible directly in that comparison), plus
  two live telemetry panels added afterward: a speed-vs-time trace (the speed governor's
  throttling, KNOWN_BUGS.md entries 2/3, is now visible as a real dip in the line, not just
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
- **M6 (Tests): done.** 317 tests across both modes run in ~380s. The 13 tests in 4 modules that need the optional `rl`/`viz`
  extras skip via `pytest.importorskip`, so 304 run on a base install.

## 4. Testing strategy

Current (run time is dominated by `test_full_highway.py`, which replays a real 78s/780-frame NGSIM
trajectory through the full node graph, parametrized over multiple controllers and seeds):

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
  single-run determinism: Section "Control" tradeoffs in DESIGN.md explains why a rate, not 100%,
  is the right thing to assert once real noise is in the loop) *and* assert not `.collision` on
  every seed; safety has to hold regardless of estimation noise. The one excluded combination gets
  its own pinned regression test asserting it *does* collide, so the exclusion can't silently go
  stale if a future change actually fixes it.
- Regression guard for the M1 degrees/radians bug: every scenario's `theta` must fall in `[-pi, pi]`.
- Real-data validation (parking/EKF): on the committed KITTI excerpt, the EKF's RMSE must be
  strictly lower than a dead-reckoning-only pass over the identical noisy odometry stream (the
  robust claim, no arbitrary accuracy number to pick), stays under a generous absolute bound
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
  a sign bug can look correct from only one direction, which is exactly how one was found; see
  Section 6), steering stays within `max_steer`, near-zero speed doesn't blow up the correction.
- Real-data validation (lane centering): from three different initial offsets, Stanley's
  closed-loop tracking error must settle under real drivers' own lateral scatter on the same real
  NGSIM lane (the plausibility bar) and produce a deterministic result.
- Planning unit checks (`test_planning.py`, added with M2, both `ReedsSheppPlanner` and
  `HybridAStarPlanner`, across all 5 scenarios): the returned path (a) starts at `start` and ends
  at `goal` within tight tolerance (both planners land exactly on the given poses by construction,
  not the controller's noisy `tol`), (b) never exceeds curvature `1/turning_radius` at any point
  (the same style of check that caught the infeasible-Bezier bug during M1: see DESIGN.md section
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
  the filter's heading actually responds: impossible under the old hardcoded behavior). Phase B
  (intersection routing) adds: never crosses the stop line without having stopped, with a real
  non-blocking lead vehicle present and ACC free to cruise unconstrained the whole approach (the
  specific accel-arbiter edge case DESIGN.md section 12's H5 entry raises and stress-tests, not
  just reasons about); the four right-of-way branches H4's own standalone tests already cover
  (yield to first-arrived, proceed when ego arrived first, yield to the right on simultaneous
  arrival, don't yield to the left), re-derived against this harness's own real approach dynamics
  rather than reusing H4's exact timings verbatim.

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
pandas: a filter/sort over a few hundred rows doesn't need a dataframe library, and the project
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
  `Vehicle(theta=np.pi/6)` in another: some scenarios passed degrees, some radians, into a
  model that only accepts radians. Root cause of the "barely works now just makes bigger circles"
  commit: a 90.0-*radian* heading is ~14 full rotations from what was intended. Fixed by making
  every scenario YAML radians-only, with a regression test enforcing it (Section 4).
- Pure Pursuit's lookahead-target search scanned the *entire* path array for "the first point at
  distance >= lookahead," from index 0 every time, instead of searching forward from the
  vehicle's current nearest point. Once the vehicle had traveled more than `lookahead` past the
  path's start, the start point became "far enough away" again and got re-selected as the
  target, steering the vehicle backward toward where it began. This was the actual mechanism
  behind the "makes bigger circles" symptom, inherited unchanged from the original prototype and
  not caused by the degrees/radians bug alone. Fixed in `control/pure_pursuit.py` by searching
  forward from the nearest-point index.
- Nothing clipped a controller's commanded steering angle to the vehicle's physical
  `max_steer` limit. Pure Pursuit does not self-limit by construction (its arctan2 formula can
  return arbitrarily large angles when the lookahead target is close and off-axis), so it was
  commanding ~65-degree steering on a car limited to ~34: direct cause of an oscillating,
  looping trajectory. Fixed by clipping `delta` to `vehicle.max_steer` in `simulation.py` (the
  natural single enforcement point, since it's the actuator layer) and additionally in Pure
  Pursuit itself for correctness when used standalone.
- `brake_distance` (0.5 m) was picked arbitrarily, below the vehicle's real stopping distance, and caused
  real collisions; it is now sized from the stopping-distance formula (DESIGN.md section 8).
- No obstacle-avoidance routing exists yet: the planner brakes to a stop when the sensor detects
  something close, it never routes around what it detects. This is by design for M1 (see DESIGN.md
  section 6) and is what M2 (Hybrid A*) resolves.
- Scenarios are now data (`scenarios/*.yaml`), not hardcoded Python dicts mixed with
  animation/plotting code.
- Automated tests now exist (Section 4).

Found and fixed during the state-estimation/pub-sub milestone (ME):

- The success tolerance (`tol=0.3`) was tuned against a noise-free baseline and was loosened to 0.4 once
  noise entered the loop; success is now a 5-seed rate. See DESIGN.md section 7 for the Pure Pursuit
  near-goal limit cycle behind it.
- The EKF's initial covariance and per-measurement noise values (`p0`, `r_heading`, `r_position`,
  `r_landmark`) are hand-picked plausible defaults (see `harness.py`'s `ParkingHarness.__init__`),
  not fit to any real sensor datasheet: reasonable for a simulation whose sensors are themselves
  synthetic, but worth stating explicitly rather than implying they're calibrated to something.

Found during the real-data validation milestone (MV):

- KITTI's ground plane is camera x/z (not x/y) and its poses-only download has no timestamps; the loader's
  heading was checked empirically against the direction of travel. See DESIGN.md section 5.

Found during pre-merge review (a full-diff pass against `main` before merging, plus a broader
200-run collision sweep across seeds beyond what the test suite covers):

- `VEHICLE_RADIUS` (the ego collision-circle radius, `harness.py`) was 0.3 m and silently under-reported
  collisions. It is now 1.0 m, which forced `brake_distance` from 2.0 m to 3.0 m; rationale in DESIGN.md
  section 8. A 200-run sweep (20 seeds x 5 scenarios x 2 controllers) found zero collisions.
- `visualization/animate.py`'s axis bounds were computed from true trajectory + obstacles + spot
  only, never from the *planned* path. For the three scenarios specifically designed to show a
  plan driving toward an obstacle before the vehicle safely stalls, the planned path extends well
  past where the truncated true trajectory does, so the gray "planned" line could render outside
  the visible axes in exactly the demos meant to showcase it. Fixed by including `result.path` in
  the bounds calculation. (Also fixed while in this file: the rendered vehicle rectangle was
  1.0m x 0.6m, comically small for a 2.7m-wheelbase car, bumped to 4.5m x 1.8m.)
- `MPCController` has its own internal rollout `dt` (default 0.1), independent of
  `ParkingHarness`'s `dt` (also default 0.1), and nothing wired them together. Both defaults happen
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
  resulting "trajectory" had more rows than the frame range should allow: an easy thing to miss
  if you don't sanity-check row counts). Fixed by keying on `global_time`, NGSIM's one genuinely
  monotonic, non-resetting timestamp, and verifying the extracted excerpt's consecutive timestamps
  are all exactly 100ms apart before committing it.
- `IDMController`'s raw formula is unbounded at small gaps (-1309 m/s² in a standalone check, run before
  any node existed); it is clipped to `a_min`. See DESIGN.md section 11.
- `MpcAccController`'s gap constraint went infeasible at a real standstill and SLSQP returned a violating
  solution silently; its default `min_gap` came from the measured erosion. See DESIGN.md section 11.

Found while building H3 (lane centering):

- `StanleyController`'s first version had the cross-track-error sign backwards and diverged;
  `tests/test_lane_centering.py` now checks convergence from both sides. See DESIGN.md section 12's H3 entry.
- `lane_centering_validation.py`'s first `settle_distance` default (50 m) was calibrated against
  a small initial offset and didn't generalize: at highway speed, convergence distance scales with
  how large the initial offset is (measured directly: ~66 m for a 1.5 m offset, ~94 m for 3.0 m),
  so a 3.0 m offset test failed the real-driver-scatter plausibility check simply because it
  hadn't finished converging yet at the point the check was measured, not because tracking was
  actually bad. Fixed by measuring real convergence distance across the offsets the module is
  actually exercised with and picking a `settle_distance` (150 m) with real margin, rather than a
  round number that happened to work for the first offset tried.

Found while building M2 (Hybrid A* + Reeds-Shepp):

- `ControllerNode`'s reactive braking (it fires on any obstacle reading inside `brake_distance`) was
  written for M1's Dubins planner, which never approaches obstacles on purpose. Hybrid A* paths do,
  so every valid avoidance maneuver stalled (`perpendicular_flanked`: 0/5 success, 0/5 collisions).
  `hybrid_astar.brake_distance_for(planner)` now derives a smaller distance from the planner's own
  clearance, while the obstacle-blind planners keep the `brake_distance=3.0` default.
- `parallel_between_cars` (a reverse-gear cusp between two close cars) exposed Pure Pursuit's
  curvature-saturation weakness as a collision; it now fails safe while MPC succeeds 5/5. See DESIGN.md
  section 6 and KNOWN_BUGS.md entry 2.
- `dubins.py`'s `DubinsPlanner.plan()` was refactored to extract `_solve_csc`/`_csc_points` (family
  selection and segment-walking, previously inlined) so `reeds_shepp.py` could reuse the exact same
  CSC formulas rather than re-deriving them: verified as a true no-behavior-change refactor by
  re-running the full suite before adding anything new on top of it (same discipline as H2's EKF
  generalization, re-checking the KITTI RMSE came back bit-for-bit identical).
- The backward-gear half of `reeds_shepp.py` reuses the forward CSC solver with the point array reversed
  (derivation in DESIGN.md section 6), checked before `hybrid_astar.py` was built on it.

Found while building H5 (full closed-loop highway drive, Phase A): three bugs in
`estimation/ekf.py`'s 4-state mode (speed propagated with the prior tick's value, no steering
uncertainty in `Q`, no absolute heading/position correction) stayed hidden because H1 kept `delta`
at 0. They were found by comparing true and estimated pose tick by tick, then a zero-noise run.
See DESIGN.md section 12's H5 entry.
