# Auto-Park Controller

A from-scratch autonomous-driving stack covering two regimes on one shared architecture:
**parking** (state estimation, Dubins path planning, Pure Pursuit/MPC control at <2 m/s) and
**highway driving** — **adaptive cruise control** (IDM and constrained-MPC longitudinal control)
and **lane centering** (Stanley lateral control), both validated against real freeway traffic
data. Both regimes run on the same pub/sub node graph (topics + typed messages, no ROS2
dependency) and the same `ExtendedKalmanFilter`, rather than one big direct-call loop per mode.

<!-- ![perpendicular parking demo](docs/perpendicular_demo.gif) -->
<!-- ![parallel parking demo](docs/parallel_demo.gif) -->
*(demo GIFs to be added — the underlying `demo.py --save` path is working, see Quickstart)*

## What this demonstrates

- **The controller and planner never see ground truth.** Everything they act on comes from an
  EKF (`estimation/ekf.py`) fusing noisy odometry (dead reckoning) with an always-on compass, a
  low-rate absolute position fix, and opportunistic landmark range-bearing readings — the classic
  "odometry + periodic correction" mobile-robot localization pattern, with control-dependent
  process noise (not an arbitrarily-picked fixed `Q`) tying the filter's uncertainty growth
  directly to how noisy the odometry actually is. Details and the math in
  [DESIGN.md](DESIGN.md#5-state-estimation).
- **A real pub/sub node architecture**: `VehicleNode`, `SensorNode`, `EstimatorNode`,
  `PlannerNode`, `ControllerNode` talk only through named topics and typed messages
  (`messaging/`), never direct references — the same decoupling principle a real ROS2 graph is
  built on. `true_state` (ground truth) is subscribed only by `SensorNode` and the test
  harness's own evaluation logic; the estimator, planner, and controller structurally cannot see
  it. See [DESIGN.md](DESIGN.md#2-system-architecture) for the node/topic diagram.
- **Curvature-feasible motion planning**: a Dubins path planner that plans the shortest path a
  real, turning-radius-limited car can actually drive — not just a smooth-looking curve (an
  earlier Bezier-curve version looked fine but demanded steering angles beyond the vehicle's
  physical limits; see [DESIGN.md](DESIGN.md#6-path-planning) for what that bug looked like and
  why Dubins paths fix it by construction).
- **A recurring "classical/reactive vs. optimization-based" controller comparison**, applied
  twice: Pure Pursuit vs. MPC for parking path-tracking, and IDM vs. constrained-MPC for highway
  car-following — same underlying design tradeoff (closed-form and cheap vs. predictive and
  constraint-aware), two different control problems. Lane centering (Stanley) plays the same
  "classical baseline" role a third time, once paired with a future optimization-based
  alternative. Details in [DESIGN.md](DESIGN.md#7-control) and
  [DESIGN.md](DESIGN.md#11-adaptive-cruise-control-h1).
- **A real lane geometry, not a hand-authored curve**: `lane_centerline.csv` aggregates ~10,400
  actual vehicle positions from real NGSIM freeway data into a centerline with genuine curvature
  (1.76m of real lateral drift over 642m) — the Stanley lane-centering controller is validated
  against this, tracking within real drivers' own lateral scatter on the same lane. Details in
  [DESIGN.md](DESIGN.md#12-highway-mode-roadmap-h2-h4).
- **Evaluated like a stochastic system, because it is one**: with real sensor noise in the loop,
  success is asserted as a rate across 5 fixed seeds, not a single deterministic run — and safety
  (no collision) is asserted on every scenario × controller × seed combination, always against
  ground truth, never against the filter's own (possibly optimistic) estimate of itself.
- **Validated against real-world data, twice, two different ways**: the unmodified EKF replayed
  against real **KITTI Odometry** ground truth (a committed excerpt with genuine turns) —
  **0.85 m RMSE with corrections vs. 4.97 m dead-reckoning-only, an 83% reduction**; and both ACC
  controllers replayed against a real 78-second **NGSIM** freeway car-following trajectory
  (congested US-101 traffic, including a full stop), evaluated on safety, comfort, and
  plausibility against what the real recorded driver actually did. Neither dataset needed
  registration — both were fetched directly from public, no-login sources. Details in
  [DESIGN.md](DESIGN.md#5-state-estimation) and [DESIGN.md](DESIGN.md#11-adaptive-cruise-control-h1).
- **Real bugs found by validating against real data, not just synthetic noise** — and documented,
  not hidden: a car-following model whose textbook formula has no physical deceleration floor (a
  standalone unit test caught it demanding -1309 m/s² before any node existed to catch it later);
  an MPC hard safety constraint that a real recorded traffic stop revealed could still be violated
  in a standstill-recovery edge case, fixed by measuring the actual erosion rather than guessing a
  safe-looking number. Full account in
  [IMPLEMENTATION.md](IMPLEMENTATION.md#6-known-issues)'s known-issues log.
- A tested, modular codebase — planners, controllers, and nodes are swappable behind common
  interfaces (see [IMPLEMENTATION.md](IMPLEMENTATION.md#2-key-interfaces)), not a single
  hardcoded pipeline. 100 tests, ~22s.
- Scenarios that are honest about current limits: two parking scenarios have a clear path and
  both controllers reach the spot reliably; three place an obstacle where the fixed Dubins path
  can't route around it, asserted safe (no collision) rather than pretending success — the gap
  the next milestone (Hybrid A*) closes.

For the algorithmic reasoning, tradeoffs, and what real bugs looked like along the way, see
**[DESIGN.md](DESIGN.md)**. For the module breakdown, milestones, and testing strategy, see
**[IMPLEMENTATION.md](IMPLEMENTATION.md)**.

## Tech stack

Python, NumPy, SciPy, Matplotlib, PyYAML, pytest.

## Quickstart

```bash
pip install -e .
pytest                                                     # run the test suite (~22s, 100 tests)

# Parking
python -m auto_park.demo perpendicular_open                # Pure Pursuit, show the animation
python -m auto_park.demo perpendicular_open --controller mpc   # MPC instead
python -m auto_park.demo perpendicular_open --seed 7             # different noise realization
python -m auto_park.demo perpendicular_open --save out.gif      # save a GIF
python -m auto_park.validation.kitti_ekf_validation --plot out.png   # EKF vs. real KITTI data

# Highway: adaptive cruise control
python -m auto_park.validation.acc_validation --controller idm --plot out.png   # IDM vs. real NGSIM data
python -m auto_park.validation.acc_validation --controller mpc --plot out.png   # MPC-ACC instead

# Highway: lane centering
python -m auto_park.validation.lane_centering_validation --plot out.png   # Stanley vs. a real NGSIM lane
```

Other parking scenarios: `perpendicular_flanked`, `perpendicular_obstructed_lane`,
`parallel_open`, `parallel_between_cars`.

## Project structure

```
auto_park/
  vehicle.py           # kinematic bicycle model + turning_radius
  environment.py         # parking lot, spots, obstacles
  scenario_loader.py       # loads scenarios/*.yaml
  messaging/             # Bus + typed messages (pub/sub backbone, shared by both modes)
  estimation/            # EKF: 3-state (parking, odometry+compass+position+landmark fusion)
                       # + 4-state speed-estimating mode (highway, H2)
  validation/            # EKF vs. real KITTI data; ACC + lane centering vs. real NGSIM data
  data/                # committed KITTI + NGSIM excerpts used by validation/ and its tests
  nodes/               # parking: VehicleNode, SensorNode, EstimatorNode, PlannerNode,
                       # ControllerNode -- highway: LeadVehicleNode, EgoLongitudinalNode,
                       # RadarNode, SpeedEstimatorNode, AccControllerNode
  harness.py            # tick-based executor (parking mode)
  highway_harness.py       # tick-based executor (ACC/highway mode)
  planning/             # dubins.py (built); reeds_shepp.py, hybrid_astar.py (next)
  control/              # pure_pursuit.py, mpc.py (parking) -- acc.py, lane_centering.py (highway:
                       # IDM + MPC-ACC, Stanley)
  visualization/          # true-vs-estimated trajectory + covariance ellipse animation
  scenarios/*.yaml         # parking scenario definitions
tests/                  # unit + integration tests
```

See [IMPLEMENTATION.md](IMPLEMENTATION.md#1-directory-structure) for the full layout and what
each file is responsible for.

## Status

**Parking**: M1 (correct baseline: Dubins planner, both controllers, realistic scenarios, tests),
M3 (MPC), state estimation + the pub/sub node architecture, and real-data EKF validation (KITTI)
are all done. M2 (Hybrid A* + Reeds-Shepp obstacle routing) is next.

**Highway**: H1 (adaptive cruise control: IDM + constrained-MPC, validated against real NGSIM
data), H2 (fused ego speed via the same EKF, extended without touching its already-validated
3-state parking path), and H3 (lane centering: Stanley control against a real derived NGSIM lane
geometry) are done. H4 (intersection navigation) is next. Combining H1/H2's ACC with H3's lane
centering into one closed loop over a single vehicle is real follow-up work, not done yet — see
IMPLEMENTATION.md's H3 milestone entry.

See the milestone list in [IMPLEMENTATION.md](IMPLEMENTATION.md#3-milestones) and DESIGN.md's
[highway-mode roadmap](DESIGN.md#12-highway-mode-roadmap-h2-h4) for full progress, and the
known-issues log for what was actually broken and fixed along the way on both sides.

## License

MIT
