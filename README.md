# Auto-Park Controller

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-274%20passing-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

A from-scratch autonomous-driving stack covering two regimes on one shared architecture:
**parking** (state estimation, Dubins/Hybrid A* path planning, Pure Pursuit/MPC control at
<2 m/s) and **highway driving** — **adaptive cruise control** (IDM and constrained-MPC
longitudinal control), **lane centering** (Stanley lateral control), and **intersection
navigation** — validated against real freeway traffic and driven-trajectory data, not just
synthetic noise. Both regimes run on the same pub/sub node graph (topics + typed messages, no
ROS2 dependency) and the same `ExtendedKalmanFilter`, rather than one big direct-call loop per
mode.

![perpendicular parking demo](docs/media/perpendicular_open.gif)

*Pure Pursuit tracking a Hybrid A* plan in a clear-lane perpendicular scenario — the top-down
view shows the planned path (gray), true trajectory (blue), EKF-estimated trajectory (orange,
with its 1σ uncertainty ellipse), and live ultrasonic sensor fan, alongside the live speed
profile. See [more scenarios below](#results), including the reverse-cusp parallel-parking case.*

**Contents:** [What this demonstrates](#what-this-demonstrates) ·
[Results](#results) · [Tech stack](#tech-stack) · [Quickstart](#quickstart) ·
[Docker](#docker) · [3D visualization](#3d-visualization-foxglove) ·
[Project structure](#project-structure) · [Status](#status)

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
  hardcoded pipeline. 278 tests (274 pass out of the box; the remaining 4 need the optional
  `rl`/`viz` extras — see [IMPLEMENTATION.md](IMPLEMENTATION.md#4-testing-strategy) for the
  breakdown).
- Scenarios that are honest about current limits: Hybrid A* routes around obstacles that a fixed
  Dubins path can't (both controllers reach the spot 5/5 seeds on all three obstacle scenarios),
  except `parallel_between_cars`'s reverse-gear cusp, which needs MPC — Pure Pursuit's known
  curvature-saturation weakness makes it fail *safe* there (0/5 success, 0/5 collisions) rather
  than pretending success. See [KNOWN_BUGS.md](KNOWN_BUGS.md) for the full account.

For the algorithmic reasoning, tradeoffs, and what real bugs looked like along the way, see
**[DESIGN.md](DESIGN.md)**. For the module breakdown, milestones, and testing strategy, see
**[IMPLEMENTATION.md](IMPLEMENTATION.md)**.

## Results

The obstacle-avoidance case that a fixed Dubins path can't solve — Hybrid A* plans a reverse-gear
cusp around both neighboring cars, MPC tracks it:

![parallel parking between two cars demo](docs/media/parallel_between_cars.gif)

Validated against real-world data, not just synthetic noise — generated by the commands in
[Quickstart](#quickstart):

| EKF vs. real KITTI Odometry | ACC vs. real NGSIM traffic | Stanley vs. a real NGSIM lane |
|---|---|---|
| ![KITTI EKF validation](docs/media/kitti_ekf_validation.png) | ![ACC validation against real NGSIM data](docs/media/acc_validation.png) | ![Lane centering validation against a real NGSIM lane](docs/media/lane_centering_validation.png) |
| 83% RMSE reduction vs. dead reckoning (0.85m vs. 4.97m) | MPC-ACC tracking a real 78s congested US-101 car-following trace | Cross-track error settles well inside real drivers' own 0.46m lateral scatter |

## Tech stack

**Core**: Python, NumPy, SciPy (`optimize.minimize`/SLSQP for MPC), Matplotlib, PyYAML, pytest.
**Optional**: Gymnasium + Stable-Baselines3 (PPO, `rl` extra), Foxglove SDK (3D export, `viz`
extra), ruff (linting), Docker.

## Quickstart

```bash
pip install -e ".[dev]"
pytest                                                # run the test suite (274 tests; 4 more need `rl`/`viz`)
ruff check core tests                                 # lint (clean by default)

# Parking
python -m core.demo perpendicular_open                # Pure Pursuit, show the animation
python -m core.demo parallel_between_cars --controller mpc  # the reverse-cusp obstacle case
python -m core.demo perpendicular_open --save out.gif      # save a GIF
python -m core.demo perpendicular_open --foxglove out/demo.mcap  # 3D scene, see below (needs `pip install -e ".[viz]"`)
python -m core.validation.kitti_ekf_validation --plot out.png   # EKF vs. real KITTI data

# Highway: adaptive cruise control
python -m core.validation.acc_validation --controller mpc --plot out.png   # MPC-ACC vs. real NGSIM data

# Highway: lane centering
python -m core.validation.lane_centering_validation --plot out.png   # Stanley vs. a real NGSIM lane

# Highway: RL vs. planner+controller baseline (needs `pip install -e ".[rl]"`)
python -m core.validation.rl_comparison core/data/rl/parking_policy_perpendicular_open.zip perpendicular_open
```

### Docker

```bash
docker build -t auto-park .          # base image: core sim + tests (numpy/scipy/matplotlib/pyyaml)
docker run --rm auto-park            # runs the test suite (the image's default command)

# Anything that writes a file (--save/--plot) needs an output volume mounted, since
# nothing else written inside the container persists past the run:
docker run --rm -v "$PWD/out:/app/out" auto-park \
  python -m core.demo perpendicular_open --save out/demo.gif

# The `rl` extra (gymnasium + stable-baselines3, pulling in torch -- a multi-GB
# addition, so it's a separate build target, not part of the base image) is only
# needed for core/rl/train.py and the RL comparison scripts:
docker build -t auto-park:rl --target rl .
docker run --rm -v "$PWD/out:/app/out" auto-park:rl \
  python -m core.rl.train perpendicular_open --timesteps 300000 --save out/policy.zip

# docker-compose.yml wraps the common cases above (test/demo/rl-train services):
docker compose run --rm test
docker compose run --rm demo
```

Build-verified: `docker build --target base .`, `docker run --rm auto-park` (the full test
suite), and `docker compose run --rm demo` (including the `./out` volume mount) all run clean
in a Linux container. One test needed fixing along the way for reasons unrelated to Docker
itself -- a razor-thin regression scenario turned out to be sensitive to which BLAS backend
`scipy.optimize`'s SLSQP solver picks per platform; see KNOWN_BUGS.md entry 8 for the full
account. The `rl` target (torch) wasn't separately build-verified.

Other parking scenarios: `perpendicular_flanked`, `perpendicular_obstructed_lane`,
`parallel_open`, `parallel_between_cars`.

### 3D visualization (Foxglove)

`core/visualization/foxglove_export.py` is an additive alternative to the default Matplotlib
animation: it writes an MCAP file (`--foxglove out/demo.mcap`) driving a 3D scene — the ego
vehicle, obstacles, the parking spot, true/EKF-estimated trajectories, the 1-sigma uncertainty
ellipse, and a live ultrasonic sensor fan — instead of a 2D top-down plot. Needs the optional
`viz` extra (`pip install -e ".[viz]"`, pulls in `foxglove-sdk`; the default install/Docker
image doesn't need it, same "opt-in" pattern as the `rl` extra below).

To turn this into a showcase video:

1. `pip install -e ".[viz]"` then `python -m core.demo perpendicular_open --foxglove out/demo.mcap`.
2. Install the free [Foxglove desktop app](https://foxglove.dev/download) (separate manual
   install, not part of this repo) and open `out/demo.mcap`.
3. Add a **3D** panel, and in its settings set the target/follow frame to `vehicle` with a
   Position + Rotation follow mode — the camera then tracks the car through the whole maneuver
   instead of sitting on a fixed view. Optionally add a **Plot** panel wired to `/speed.v` /
   `/speed.delta` and another to the `/sensors.*` fields for live telemetry alongside the 3D
   view, matching what the Matplotlib animation's side panels show. Save this as a layout (e.g.
   export it to `foxglove-layouts/parking_demo.json`) so it's a one-click load next time.
4. Hit play, and **screen-record the app window** (OBS Studio, Windows Game Bar, etc.) — Foxglove's
   built-in "Export video" command is Enterprise-plan only, so this is the practical path to an
   actual video file on the free tier. No ffmpeg needed for this path (ffmpeg is only relevant to
   the `--save out.gif` path above, and isn't required for that either — it's Pillow-based).

## Project structure

```
core/
  vehicle.py             # kinematic bicycle model + turning_radius
  environment.py          # parking lot, spots, obstacles
  scenario_loader.py        # loads scenarios/*.yaml
  messaging/              # Bus + typed messages (pub/sub backbone, shared by every mode)
  estimation/             # ekf.py (3-state parking / 4-state highway speed-estimating mode),
                        # ukf.py (sigma-point alternative, compared head-to-head)
  planning/              # dubins.py (M1 baseline), reeds_shepp.py (+ reverse gear), hybrid_astar.py
                        # (obstacle-aware search using Reeds-Shepp as heuristic/connector)
  control/               # pure_pursuit.py, mpc.py (parking) -- acc.py (IDM + MPC-ACC),
                        # lane_centering.py (Stanley), intersection.py + intersection_geometry.py
                        # (H4 right-of-way + real 2D intersection geometry)
  nodes/                # parking: VehicleNode, SensorNode, EstimatorNode, PlannerNode,
                        # ControllerNode -- highway: LeadVehicleNode/HighwayVehicleNode,
                        # RadarNode, SpeedEstimatorNode, AccControllerNode,
                        # LaneCenteringControllerNode, IntersectionControllerNode,
                        # LongitudinalArbiterNode (composes ACC + intersection candidates)
  harness.py             # tick-based executor (parking)
  highway_harness.py        # tick-based executor (H1/H2 ACC-only)
  intersection_harness.py     # H4 standalone (1D right-of-way)
  intersection2d_harness.py    # H4 over real 2D multi-approach geometry (KNOWN_BUGS.md entry 4)
  full_highway_harness.py     # H5: ACC + lane centering + intersection composed on one Vehicle
  rl/                   # ParkingEnv (Gymnasium) + PPO training, compared against the
                        # planner+controller baseline
  validation/             # EKF vs. real KITTI data; ACC/lane-centering/RL vs. real NGSIM data
                        # and the committed baseline; ukf_comparison.py (EKF vs. UKF)
  data/                 # committed KITTI + NGSIM excerpts, trained RL policies
  visualization/           # animate.py (Matplotlib) + foxglove_export.py (3D MCAP export)
  scenarios/*.yaml          # parking scenario definitions
tests/                   # unit + integration tests
```

See [IMPLEMENTATION.md](IMPLEMENTATION.md#1-directory-structure) for the full layout and what
each file is responsible for.

## Status

**Parking**: M1 (correct baseline: Dubins planner, both controllers, realistic scenarios, tests),
M2 (Hybrid A* + Reeds-Shepp obstacle-aware planning), M3 (MPC), M4 (sensing & re-planning), M5
(visualization), state estimation + the pub/sub node architecture, and real-data EKF validation
(KITTI) are all done. What's open: a documented sensor-latency gap beyond ~1-2s (KNOWN_BUGS.md
entry 7).

**Highway**: H1 (adaptive cruise control: IDM + constrained-MPC, validated against real NGSIM
data), H2 (fused ego speed via the same EKF, extended without touching its already-validated
3-state parking path), H3 (lane centering: Stanley control against a real derived NGSIM lane
geometry), H4 (intersection navigation, including turning movements), and H5 (the full closed-loop
drive composing H1-H4 onto one vehicle) are all done.

See the milestone list in [IMPLEMENTATION.md](IMPLEMENTATION.md#3-milestones) and DESIGN.md's
[highway-mode roadmap](DESIGN.md#12-highway-mode-roadmap-h2-h4) for full progress, and the
known-issues log for what was actually broken and fixed along the way on both sides.

## License

MIT
