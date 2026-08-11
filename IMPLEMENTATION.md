# Implementation Plan

This is the build spec for the architecture described in [DESIGN.md](DESIGN.md). M1 (correct
baseline) and the control half of M3 (MPC) are done; the sections below reflect what's actually
in the repo today, not just what was planned. M2 (Hybrid A* + Reeds-Shepp obstacle routing) is
the next milestone.

## 1. Directory structure

```
auto_park/
  vehicle.py           # kinematic bicycle model + turning_radius from max_steer
  sensors.py            # ultrasonic ray-cast sensor array
  environment.py         # parking lot, spots, obstacles, boundaries
  interfaces.py          # Planner / Controller structural protocols, Pose type
  scenario_loader.py       # loads scenarios/*.yaml into Vehicle + Environment
  planning/
    __init__.py
    dubins.py            # M1 baseline: curvature-feasible fixed path, no obstacle avoidance
    reeds_shepp.py        # M2 (not yet built): adds reverse gear
    hybrid_astar.py        # M2 (not yet built): obstacle-aware search, uses reeds_shepp
  control/
    __init__.py
    pure_pursuit.py        # adaptive Pure Pursuit
    mpc.py              # nonlinear MPC (direct shooting, SLSQP), done in M1 alongside the baseline
  simulation.py          # orchestrates env + sensor + planner + controller + vehicle stepping
  visualization/
    __init__.py
    animate.py            # matplotlib animation/rendering
  scenarios/
    perpendicular_open.yaml
    perpendicular_flanked.yaml
    perpendicular_obstructed_lane.yaml
    parallel_open.yaml
    parallel_between_cars.yaml
  demo.py              # CLI entry point: run a named scenario + controller, show/save animation
tests/
  test_vehicle.py
  test_simulation.py       # integration tests across all scenarios x both controllers
pyproject.toml
DESIGN.md
IMPLEMENTATION.md
README.md
```

`test_sensors.py`, `test_planning.py`, and `test_control.py` (unit-level, per Section 4) haven't
been split out yet — current coverage is integration-level via `test_simulation.py`, which is
enough to catch the regressions that mattered so far, but the finer-grained unit tests are still
worth adding as `planning/` grows with M2.

## 2. Key interfaces

Keeping these consistent is what lets planners and controllers be swapped without touching
`simulation.py`.

```python
# vehicle.py
class Vehicle:
    def __init__(self, x=0.0, y=0.0, theta=0.0, wheelbase=2.7, max_steer=0.6): ...
    def update(self, v: float, delta: float, dt: float) -> None: ...
    # theta always in radians; callers passing degrees was the #1 historical bug (see Section 6)
    @property
    def turning_radius(self) -> float: ...  # wheelbase / tan(max_steer) -- the single source
                                              # of truth the planner and both controllers use

# sensors.py
class UltrasonicArray:
    def __init__(self, angles: list[float], max_range: float = 5.0): ...
    def sense(self, vehicle: Vehicle, obstacles: list[Obstacle]) -> dict[float, float]: ...

# interfaces.py
class Planner(Protocol):
    def plan(
        self, start: Pose, goal: Pose, obstacles: list[Obstacle], turning_radius: float
    ) -> np.ndarray:  # (N, 3) array of x, y, theta waypoints
        ...

class Controller(Protocol):
    def control(self, vehicle: Vehicle, path: np.ndarray) -> tuple[float, float]:
        # returns (v_desired, delta) -- delta is clipped to vehicle.max_steer by the
        # simulation loop regardless of what the controller returns, since nothing else
        # enforces that physical limit (see Section 6, "delta clipping")
        ...

# simulation.py
class ParkingSimulation:
    def __init__(
        self, vehicle: Vehicle, environment: Environment,
        planner: Planner, controller: Controller, sensor: UltrasonicArray,
        dt=0.1, v_max=1.5, a_max=0.8, k_acc=2.0, tol=0.3, brake_distance=2.0,
    ): ...
    def run(self, max_steps: int = 2000) -> SimulationResult:
        # SimulationResult: pose history, control history, success flag, collision flag, path
        ...
```

`Planner` and `Controller` as structural-typing protocols (not required base classes) is
intentional: `reeds_shepp.ReedsSheppPlanner` and `planning.hybrid_astar.HybridAStarPlanner`
both satisfy `Planner` without a shared inheritance hierarchy, and the same for the two
controllers — new algorithms can be added later without editing existing ones.

## 3. Milestones

- **M1 — Correct baseline: done.** Extracted the original prototype into the module layout
  above and fixed the degrees/radians bug. The planner ended up being more than a straight port:
  the originally-planned fixed Bezier curve turned out to be kinematically infeasible (see
  DESIGN.md section 5), so M1 now ships a **Dubins path planner** instead — still a single fixed
  path with no obstacle avoidance (that's still M2's job), but one that's guaranteed drivable.
- **M3 — Control: done, pulled ahead of M2.** `control/mpc.py` (nonlinear MPC via SLSQP) is
  implemented and selectable per run via `demo.py --controller {pure_pursuit,mpc}`. Pulled ahead
  of M2 because getting the controllers actually converging reliably was higher-value than
  obstacle routing for a first working end-to-end demo — see DESIGN.md section 6 for the
  head-to-head comparison this produced.
- **M2 — Planning (next up)**: implement `reeds_shepp.py`, then `hybrid_astar.py` on top of it;
  replace the fixed Dubins path with Hybrid A* as the default planner. Validate against the
  scenarios that currently stall safely rather than reach the spot
  (`perpendicular_flanked`, `perpendicular_obstructed_lane`, `parallel_between_cars`) — Hybrid A*
  should solve all three.
- **M4 — Sensing & re-planning**: sensor is already a multi-beam array (`[-0.6, -0.3, 0.0, 0.3,
  0.6]` rad) and braking already checks all beams, not just the front one. Still open: wiring
  `simulation.py` so a sensor detection of an unmapped obstacle triggers a re-plan call rather
  than only braking (needs M2's planner to re-plan into).
- **M5 — Visualization polish**: multi-panel animation (trajectory + live sensor readings +
  speed profile) is not yet built; current `visualization/animate.py` does single-panel
  trajectory + vehicle pose only, which is enough for the demo GIFs referenced in README.md but
  not the richer version originally scoped here.
- **M6 — Tests & CI**: `test_vehicle.py` and `test_simulation.py` exist and run in ~2s (22
  tests); no GitHub Actions workflow yet.

## 4. Testing strategy

Current (`tests/test_vehicle.py`, `tests/test_simulation.py`, 22 tests, ~2s):

- Kinematic checks: driving straight for N steps moves `x` by `v*N*dt` with `theta` unchanged; a
  fixed steering angle over time traces a circle of radius `L / tan(delta)`; `turning_radius`
  matches `wheelbase / tan(max_steer)`.
- Integration, parametrized over both controllers: on the two obstacle-free scenarios
  (`perpendicular_open`, `parallel_open`), assert `SimulationResult.success` and not
  `.collision`. On *every* scenario (including the three with obstacles), assert not `.collision`
  — the M1 baseline is expected to stall short of the goal there, not crash.
- Regression guard for the original degrees/radians bug: every scenario's `theta` values must
  fall in `[-pi, pi]`; a value like the old `90.0` is ~14 full rotations out of range and would
  fail this immediately.

Planned, once `planning/` grows with M2 (currently integration-level coverage is enough, but
won't scale to multiple planners):

- `test_sensors.py`: hand-computed ray/circle intersection cases (obstacle dead ahead, obstacle
  out of range, obstacle behind the beam direction).
- `test_planning.py`: for each planner, assert the returned path (a) starts at `start` and ends
  at `goal` within tolerance, (b) never exceeds curvature `1/turning_radius` at any point (this
  check is what caught the infeasible-Bezier bug during M1 — see DESIGN.md section 5), (c) for
  Hybrid A* specifically, never comes within the vehicle's radius of any obstacle.
- `test_control.py`: given a straight-line path and no obstacles, assert each controller's output
  converges the vehicle to the goal within a fixed number of steps and within `tol`.

## 5. Dependencies & running

```
numpy
scipy        # MPC (scipy.optimize.minimize, SLSQP), and Hybrid A* heuristic support later
matplotlib
pyyaml       # scenario file loading
pytest
```
`scipy.optimize.minimize(method="SLSQP")` turned out sufficient for `control/mpc.py`; `cvxpy`
was the planned fallback if that proved awkward, but wasn't needed.

```
pip install -e .
pytest                                                    # run the test suite
python -m auto_park.demo <scenario>                       # Pure Pursuit, show the animation
python -m auto_park.demo <scenario> --controller mpc       # MPC instead
python -m auto_park.demo <scenario> --save out.gif         # save a GIF for the README
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
  plus margin (see DESIGN.md section 7).
- No obstacle-avoidance routing exists yet — the planner brakes to a stop when the sensor detects
  something close, it never routes around what it detects. This is by design for M1 (see DESIGN.md
  section 5) and is what M2 (Hybrid A*) resolves.
- Scenarios are now data (`scenarios/*.yaml`), not hardcoded Python dicts mixed with
  animation/plotting code.
- Automated tests now exist (Section 4); CI (GitHub Actions) is still open, tracked under M6.
