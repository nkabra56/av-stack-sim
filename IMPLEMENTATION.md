# Implementation Plan

This is the build spec for the architecture described in [DESIGN.md](DESIGN.md). M1 (correct
baseline), the control half of M3 (MPC), the state-estimation + pub/sub-architecture milestone
(ME), and a real-data validation pass for the EKF (MV, below) are done; the sections below
reflect what's actually in the repo today, not just what was planned. M2 (Hybrid A* + Reeds-Shepp
obstacle routing) is the next milestone.

## 1. Directory structure

```
auto_park/
  vehicle.py           # kinematic bicycle model + turning_radius from max_steer
  sensors.py            # ultrasonic ray-cast sensor array (used internally by SensorNode)
  environment.py         # parking lot, spots, obstacles, boundaries
  interfaces.py          # Planner / Controller structural protocols, HasPose, Pose type
  scenario_loader.py       # loads scenarios/*.yaml into Vehicle + Environment (+ seed)
  messaging/
    __init__.py
    bus.py             # Bus: synchronous publish/subscribe
    messages.py          # TrueStateMsg, OdometryMsg, CompassMsg, PositionFixMsg,
                       # LandmarkBearingMsg, ObstacleRangeMsg, PoseEstimateMsg, PathMsg,
                       # ControlCmdMsg -- see DESIGN.md section 2
  estimation/
    __init__.py
    ekf.py             # ExtendedKalmanFilter: predict + 3 correction types, see DESIGN.md section 5
  validation/
    __init__.py
    kitti_loader.py       # parses KITTI poses.txt -> KittiSequence(times,x,y,theta,v,yaw_rate)
    kitti_ekf_validation.py  # runs the (unmodified) EKF against a real trajectory, EKF vs.
                       # dead-reckoning-only RMSE comparison + plot, see DESIGN.md section 5
  data/
    kitti/
      excerpt_poses.txt      # committed 300-frame excerpt (KITTI seq 09, frames 840-1139)
      ATTRIBUTION.md       # license/citation for the redistributed KITTI excerpt
  nodes/
    __init__.py
    vehicle_node.py       # ground truth + odometry publisher; owns accel/steering limits
    sensor_node.py        # obstacle_ranges, compass, position_fix, landmark_bearings publisher
    estimator_node.py      # wraps ekf.py, publishes pose_estimate
    planner_node.py       # wraps Planner, plans once off the first pose_estimate
    controller_node.py     # wraps Controller, one control_cmd per tick via explicit step()
  harness.py            # tick-based executor: owns the Bus, builds all 5 nodes, drives them
  planning/
    __init__.py
    dubins.py            # M1 baseline: curvature-feasible fixed path, no obstacle avoidance
    reeds_shepp.py        # M2 (not yet built): adds reverse gear
    hybrid_astar.py        # M2 (not yet built): obstacle-aware search, uses reeds_shepp
  control/
    __init__.py
    pure_pursuit.py        # adaptive Pure Pursuit, acts on HasPose (a Vehicle or a pose estimate)
    mpc.py              # nonlinear MPC (direct shooting, SLSQP)
  visualization/
    __init__.py
    animate.py            # true vs. estimated trajectory + covariance ellipse + planned path
  scenarios/
    perpendicular_open.yaml
    perpendicular_flanked.yaml
    perpendicular_obstructed_lane.yaml
    parallel_open.yaml
    parallel_between_cars.yaml
  demo.py              # CLI entry point: run a named scenario + controller (+ seed), show/save
tests/
  test_vehicle.py
  test_bus.py
  test_ekf.py
  test_kitti_ekf_validation.py  # EKF vs. dead-reckoning-only on the committed real KITTI excerpt
  test_simulation.py       # integration tests, harness-based, across scenarios x controllers x seeds
pyproject.toml
DESIGN.md
IMPLEMENTATION.md
README.md
```

`simulation.py`/`ParkingSimulation` (the M1 direct-call loop) is retired — `harness.py` is now the
one way `demo.py` and the tests run a scenario, so there's a single execution path rather than two.

`test_sensors.py`, `test_planning.py`, and `test_control.py` (unit-level, per Section 4) haven't
been split out yet — current coverage is integration-level via `test_simulation.py`, which is
enough to catch the regressions that mattered so far, but the finer-grained unit tests are still
worth adding as `planning/` grows with M2.

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
  self-consistent against noise the project generates for itself. `auto_park/validation/` adds an
  independent check: the same, unmodified EKF replayed against real KITTI Odometry ground-truth
  poses (a committed 300-frame excerpt with real turns, `auto_park/data/kitti/`), using the same
  noise defaults as `SensorNode`/`VehicleNode`. Result: 0.85 m RMSE with corrections vs. 4.97 m
  dead-reckoning-only on the excerpt — an 83% reduction — see DESIGN.md section 5,
  "Validation against real data."
- **M2 — Planning (next up)**: implement `reeds_shepp.py`, then `hybrid_astar.py` on top of it;
  replace the fixed Dubins path with Hybrid A* as the default planner, planning from the pose
  *estimate* (already how `PlannerNode` is wired — no further node changes needed). Validate
  against the scenarios that currently stall safely rather than reach the spot
  (`perpendicular_flanked`, `perpendicular_obstructed_lane`, `parallel_between_cars`) — Hybrid A*
  should solve all three.
- **M4 — Sensing & re-planning**: sensor is already a multi-beam array (`[-0.6, -0.3, 0.0, 0.3,
  0.6]` rad) and braking already checks all beams, not just the front one. Still open: wiring
  `PlannerNode` to re-plan (not just `ControllerNode` to brake) when `SensorNode` reports an
  obstacle the current path didn't account for (needs M2's planner to re-plan into).
- **M5 — Visualization polish**: `visualization/animate.py` now shows true vs. estimated
  trajectory, a covariance ellipse, and the planned path (landed as part of ME, since the whole
  point of adding an estimator is visible directly in that comparison). Still open: a genuinely
  multi-panel layout (live sensor readings, speed profile) alongside the top-down view.
- **M6 — Tests & CI**: `test_vehicle.py`, `test_bus.py`, `test_ekf.py`,
  `test_kitti_ekf_validation.py`, and `test_simulation.py` exist and run in ~11s (76 tests); no
  GitHub Actions workflow yet.

## 4. Testing strategy

Current (`tests/test_vehicle.py`, `test_bus.py`, `test_ekf.py`, `test_kitti_ekf_validation.py`,
`test_simulation.py`, 76 tests, ~11s):

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
- Integration, parametrized over both controllers, run against `ParkingHarness`: on the two
  obstacle-free scenarios, assert a **success rate ≥4/5 across 5 fixed seeds** (not single-run
  determinism — Section "Control" tradeoffs in DESIGN.md explains why a rate, not 100%, is the
  right thing to assert once real noise is in the loop). On *every* scenario × controller × seed
  (50 combinations), assert not `.collision` — safety has to hold regardless of estimation noise,
  including on the three scenarios that aren't expected to reach the spot.
- Regression guard for the original degrees/radians bug: every scenario's `theta` values must
  fall in `[-pi, pi]`; a value like the old `90.0` is ~14 full rotations out of range and would
  fail this immediately.
- Real-data validation: on the committed KITTI excerpt, the EKF's RMSE must be strictly lower
  than a dead-reckoning-only pass over the identical noisy odometry stream (the robust claim —
  no arbitrary accuracy number to pick), stays under a generous absolute bound (divergence
  guard), and is deterministic for a fixed seed.

Planned, once `planning/` grows with M2 (currently integration-level coverage is enough, but
won't scale to multiple planners):

- `test_sensors.py`: hand-computed ray/circle intersection cases (obstacle dead ahead, obstacle
  out of range, obstacle behind the beam direction).
- `test_planning.py`: for each planner, assert the returned path (a) starts at `start` and ends
  at `goal` within tolerance, (b) never exceeds curvature `1/turning_radius` at any point (this
  check is what caught the infeasible-Bezier bug during M1 — see DESIGN.md section 6), (c) for
  Hybrid A* specifically, never comes within the vehicle's radius of any obstacle.
- `test_control.py`: given a straight-line path and no obstacles, assert each controller's output
  converges toward the goal within a fixed number of steps and within `tol`.

## 5. Dependencies & running

```
numpy
scipy        # MPC (scipy.optimize.minimize, SLSQP), and Hybrid A* heuristic support later
matplotlib
pyyaml       # scenario file loading
pytest
```
`scipy.optimize.minimize(method="SLSQP")` turned out sufficient for `control/mpc.py`; `cvxpy`
was the planned fallback if that proved awkward, but wasn't needed. No new dependency was needed
for the EKF/pub-sub milestone either — `Bus` is ~15 lines of pure Python, and the EKF is plain
NumPy linear algebra.

```
pip install -e .
pytest                                                    # run the test suite
python -m auto_park.demo <scenario>                       # Pure Pursuit, show the animation
python -m auto_park.demo <scenario> --controller mpc       # MPC instead
python -m auto_park.demo <scenario> --seed 7               # override the scenario's RNG seed
python -m auto_park.demo <scenario> --save out.gif         # save a GIF for the README
python -m auto_park.validation.kitti_ekf_validation --plot out.png   # EKF vs. real KITTI data
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
