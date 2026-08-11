# Auto-Park Controller

A from-scratch simulation of autonomous car parking: **state estimation** (an Extended Kalman
Filter fusing noisy odometry, compass, position-fix, and landmark sensors), **path planning**
(Dubins paths today, Hybrid A* + Reeds-Shepp next), and **path tracking control** (Pure Pursuit
and nonlinear MPC) — wired together as a small pub/sub node graph (topics + typed messages, no
ROS2 dependency) rather than one big direct-call loop, with matplotlib-animated top-down
visualizations of true vs. estimated trajectory.

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
- **Two path-tracking controllers, compared head-to-head on the same paths and the same noise**:
  geometric/reactive Pure Pursuit vs. a nonlinear MPC (direct-shooting, solved with
  `scipy.optimize`). Both converge reliably on open scenarios; MPC is measurably more robust both
  on curvature-saturated paths and under estimation noise, where Pure Pursuit's reactive steering
  has less margin. Details in [DESIGN.md](DESIGN.md#7-control).
- **Evaluated like a stochastic system, because it is one**: with real sensor noise in the loop,
  success is asserted as a rate across 5 fixed seeds, not a single deterministic run — and safety
  (no collision) is asserted on every scenario × controller × seed combination, always against
  ground truth, never against the filter's own (possibly optimistic) estimate of itself.
- A tested, modular codebase — planners, controllers, and nodes are swappable behind common
  interfaces (see [IMPLEMENTATION.md](IMPLEMENTATION.md#2-key-interfaces)), not a single
  hardcoded pipeline. 72 tests, ~10s.
- Scenarios that are honest about the current planner's limits: two scenarios have a clear path
  and both controllers reach the spot reliably; three place an obstacle where the fixed Dubins
  path can't route around it, and are asserted safe (no collision) rather than pretending the
  vehicle parks successfully. That gap is exactly what the next milestone (Hybrid A*) closes.

For the algorithmic reasoning, tradeoffs, and what real bugs looked like along the way, see
**[DESIGN.md](DESIGN.md)**. For the module breakdown, milestones, and testing strategy, see
**[IMPLEMENTATION.md](IMPLEMENTATION.md)**.

## Tech stack

Python, NumPy, SciPy, Matplotlib, PyYAML, pytest.

## Quickstart

```bash
pip install -e .
pytest                                                     # run the test suite (~10s, 72 tests)
python -m auto_park.demo perpendicular_open                # Pure Pursuit, show the animation
python -m auto_park.demo perpendicular_open --controller mpc   # MPC instead
python -m auto_park.demo perpendicular_open --seed 7             # different noise realization
python -m auto_park.demo perpendicular_open --save out.gif      # save a GIF
```

Other scenarios: `perpendicular_flanked`, `perpendicular_obstructed_lane`, `parallel_open`,
`parallel_between_cars`.

## Project structure

```
auto_park/
  vehicle.py           # kinematic bicycle model + turning_radius
  environment.py         # parking lot, spots, obstacles
  scenario_loader.py       # loads scenarios/*.yaml
  messaging/             # Bus + typed messages (pub/sub backbone)
  estimation/            # EKF (odometry + compass + position-fix + landmark fusion)
  nodes/               # VehicleNode, SensorNode, EstimatorNode, PlannerNode, ControllerNode
  harness.py            # tick-based executor tying the node graph together
  planning/             # dubins.py (built); reeds_shepp.py, hybrid_astar.py (next)
  control/              # pure_pursuit.py, mpc.py
  visualization/          # true-vs-estimated trajectory + covariance ellipse animation
  scenarios/*.yaml         # scenario definitions
tests/                  # unit + integration tests
```

See [IMPLEMENTATION.md](IMPLEMENTATION.md#1-directory-structure) for the full layout and what
each file is responsible for.

## Status

M1 (correct baseline: Dubins planner, both controllers, realistic scenarios, tests) and M3 (MPC)
are done. State estimation + the pub/sub node architecture (EKF, `messaging/`, `nodes/`,
`harness.py`) are also done, replacing what used to be a direct-call simulation loop over
ground-truth pose. M2 (Hybrid A* + Reeds-Shepp obstacle routing) is next — see the milestone list
in [IMPLEMENTATION.md](IMPLEMENTATION.md#3-milestones) for full progress and the known-issues log
of what was actually broken and fixed along the way.

## License

MIT
