# Auto-Park Controller

A from-scratch simulation of autonomous car parking: **path planning** (Dubins paths today,
Hybrid A* + Reeds-Shepp next) and **path tracking control** (Pure Pursuit and nonlinear MPC) for
a kinematic vehicle model, with a simulated ultrasonic sensor array and matplotlib-animated
top-down visualizations of perpendicular and parallel parking maneuvers.

<!-- ![perpendicular parking demo](docs/perpendicular_demo.gif) -->
<!-- ![parallel parking demo](docs/parallel_demo.gif) -->
*(demo GIFs to be added — the underlying `demo.py --save` path is working, see Quickstart)*

## What this demonstrates

- **Curvature-feasible motion planning**: a Dubins path planner that plans the shortest path a
  real, turning-radius-limited car can actually drive — not just a smooth-looking curve (an
  earlier Bezier-curve version looked fine but demanded steering angles beyond the vehicle's
  physical limits; see [DESIGN.md](DESIGN.md#5-path-planning) for what that bug looked like and
  why Dubins paths fix it by construction).
- **Two path-tracking controllers, compared head-to-head on the same paths**: geometric/reactive
  Pure Pursuit vs. a nonlinear MPC (direct-shooting, solved with `scipy.optimize`). Both converge
  reliably on open scenarios; MPC is measurably more robust on curvature-saturated paths, where
  Pure Pursuit's reactive steering has no margin left. Details and why in
  [DESIGN.md](DESIGN.md#6-control).
- **Sensor simulation**: a multi-beam ray-cast ultrasonic array; the simulation loop brakes
  (with a stopping distance actually sized from the vehicle's speed/accel limits) when it detects
  an obstacle the planner didn't know about.
- **A kinematic bicycle vehicle model** with a real `max_steer` limit driving a `turning_radius`
  property that the planner and both controllers all size themselves against — one source of
  truth, not three copies of the same constant.
- A tested, modular codebase — planners and controllers are swappable behind common interfaces
  (see [IMPLEMENTATION.md](IMPLEMENTATION.md#2-key-interfaces)), not a single hardcoded pipeline.
- Scenarios that are honest about the current planner's limits: two scenarios have a clear path
  and both controllers reach the spot; three place an obstacle where the fixed Dubins path can't
  route around it, and are asserted safe (no collision) rather than pretending the vehicle parks
  successfully. That gap is exactly what the next milestone (Hybrid A*) closes.

For the algorithmic reasoning, tradeoffs, and what real bugs looked like along the way, see
**[DESIGN.md](DESIGN.md)**. For the module breakdown, milestones, and testing strategy, see
**[IMPLEMENTATION.md](IMPLEMENTATION.md)**.

## Tech stack

Python, NumPy, SciPy, Matplotlib, PyYAML, pytest.

## Quickstart

```bash
pip install -e .
pytest                                                     # run the test suite (~2s, 22 tests)
python -m auto_park.demo perpendicular_open                # Pure Pursuit, show the animation
python -m auto_park.demo perpendicular_open --controller mpc   # MPC instead
python -m auto_park.demo perpendicular_open --save out.gif      # save a GIF
```

Other scenarios: `perpendicular_flanked`, `perpendicular_obstructed_lane`, `parallel_open`,
`parallel_between_cars`.

## Project structure

```
auto_park/
  vehicle.py           # kinematic bicycle model + turning_radius
  sensors.py            # ultrasonic ray-cast sensor array
  environment.py         # parking lot, spots, obstacles
  scenario_loader.py       # loads scenarios/*.yaml
  planning/             # dubins.py (built); reeds_shepp.py, hybrid_astar.py (next)
  control/              # pure_pursuit.py, mpc.py
  simulation.py          # ties planning + control + vehicle together each timestep
  visualization/          # matplotlib animation rendering
  scenarios/*.yaml         # scenario definitions
tests/                  # unit + integration tests
```

See [IMPLEMENTATION.md](IMPLEMENTATION.md#1-directory-structure) for the full layout and what
each file is responsible for.

## Status

M1 (correct baseline: Dubins planner, both controllers, realistic scenarios, tests) is done. M3
(MPC) landed alongside it. M2 (Hybrid A* + Reeds-Shepp obstacle routing) is next — see the
milestone list in [IMPLEMENTATION.md](IMPLEMENTATION.md#3-milestones) for full progress and the
known-issues log of what was actually broken and fixed along the way.

## License

MIT
