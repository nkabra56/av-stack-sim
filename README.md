# Auto-Park Controller

A from-scratch simulation of autonomous car parking: **path planning** (Reeds-Shepp curves and
Hybrid A*) and **path tracking control** (Pure Pursuit and linear MPC) for a kinematic vehicle
model, with a simulated ultrasonic sensor for obstacle detection and matplotlib-animated
top-down visualizations of perpendicular and parallel parking maneuvers.

<!-- ![perpendicular parking demo](docs/perpendicular_demo.gif) -->
<!-- ![parallel parking demo](docs/parallel_demo.gif) -->
*(demo GIFs to be added once the visualization milestone lands — see [IMPLEMENTATION.md](IMPLEMENTATION.md))*

## What this demonstrates

- **Classical motion planning**: Reeds-Shepp curves (the standard shortest-path solution for a
  car with reverse gear) and Hybrid A* search for planning around obstacles that block the
  direct approach into a spot.
- **Two path-tracking controllers**, run side by side on the same scenario: geometric/reactive
  Pure Pursuit vs. predictive/optimization-based linear MPC.
- **Sensor simulation**: a ray-cast ultrasonic array that can trigger a re-plan when it detects
  an obstacle the planner didn't already know about.
- **A kinematic bicycle vehicle model**, the standard low-speed vehicle model used throughout
  the planning/controls literature.
- A tested, modular codebase — planners and controllers are swappable behind common interfaces
  (see [IMPLEMENTATION.md](IMPLEMENTATION.md#2-key-interfaces)), not a single hardcoded pipeline.

For the algorithmic reasoning and tradeoffs (why Hybrid A* over RRT*, why two controllers, why a
kinematic rather than dynamic vehicle model), see **[DESIGN.md](DESIGN.md)**. For the module
breakdown, milestones, and testing strategy, see **[IMPLEMENTATION.md](IMPLEMENTATION.md)**.

## Tech stack

Python, NumPy, SciPy, Matplotlib, pytest.

## Quickstart

```bash
pip install -e .
pytest                                        # run the test suite
python -m auto_park.demo perpendicular_obstacle    # run a scenario, show the animation
python -m auto_park.demo perpendicular_obstacle --save out.gif   # save a GIF
```

## Project structure

```
auto_park/
  vehicle.py           # kinematic bicycle model
  sensors.py            # ultrasonic ray-cast sensor array
  environment.py         # parking lot, spots, obstacles
  planning/             # reeds_shepp.py, hybrid_astar.py
  control/              # pure_pursuit.py, mpc.py
  simulation.py          # ties planning + control + vehicle together each timestep
  visualization/          # matplotlib animation rendering
  scenarios/*.yaml         # scenario definitions
tests/                  # unit + integration tests
```

See [IMPLEMENTATION.md](IMPLEMENTATION.md#1-directory-structure) for the full layout and what
each file is responsible for.

## Status

This project is being rebuilt from an earlier prototype into the architecture described in
DESIGN.md/IMPLEMENTATION.md — see the milestone list in
[IMPLEMENTATION.md](IMPLEMENTATION.md#3-milestones) for current progress.

## License

MIT
