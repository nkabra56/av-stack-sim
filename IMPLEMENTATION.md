# Implementation Plan

This is the build spec for the architecture described in [DESIGN.md](DESIGN.md). It targets a
rewrite of the current prototype (`controller.py`, `test_parking.py`) into the module layout
below. The rewrite itself is tracked as follow-up work (milestones M1–M6); this document is the
spec that work builds against.

## 1. Directory structure

```
auto_park/
  vehicle.py           # kinematic bicycle model
  sensors.py            # ultrasonic ray-cast sensor array
  environment.py         # parking lot, spots, obstacles, boundaries
  planning/
    __init__.py
    reeds_shepp.py        # classical forward/reverse parking curves
    hybrid_astar.py        # obstacle-aware global planner (uses reeds_shepp as heuristic)
  control/
    __init__.py
    pure_pursuit.py        # adaptive Pure Pursuit (ported from current controller.py)
    mpc.py              # linear MPC path tracker
  simulation.py          # orchestrates env + sensor + planner + controller + vehicle stepping
  visualization/
    __init__.py
    animate.py            # matplotlib animation/rendering (ported from test_parking.py)
  scenarios/
    perpendicular_clear.yaml
    perpendicular_obstacle.yaml
    parallel_clear.yaml
    parallel_obstacle.yaml
    ...
  demo.py              # CLI entry point: run a named scenario, show/save animation
tests/
  test_vehicle.py
  test_sensors.py
  test_planning.py
  test_control.py
  test_simulation.py       # integration test across all scenarios
pyproject.toml
DESIGN.md
IMPLEMENTATION.md
README.md
```

## 2. Key interfaces

Keeping these consistent is what lets planners and controllers be swapped without touching
`simulation.py`.

```python
# vehicle.py
class Vehicle:
    def __init__(self, x: float, y: float, theta: float, wheelbase: float = 2.5): ...
    def update(self, v: float, delta: float, dt: float) -> None: ...
    # theta always in radians; callers passing degrees is the #1 historical bug (see Section 6)

# sensors.py
class UltrasonicArray:
    def __init__(self, angles: list[float], max_range: float = 5.0): ...
    def sense(self, vehicle: Vehicle, obstacles: list[Obstacle]) -> dict[float, float]: ...

# planning/*.py
class Planner(Protocol):
    def plan(
        self, start: Pose, goal: Pose, obstacles: list[Obstacle], turning_radius: float
    ) -> np.ndarray:  # (N, 3) array of x, y, theta waypoints
        ...

# control/*.py
class Controller(Protocol):
    def control(self, vehicle: Vehicle, path: np.ndarray) -> tuple[float, float]:
        # returns (v_desired, delta)
        ...

# simulation.py
class ParkingSimulation:
    def __init__(
        self, vehicle: Vehicle, environment: Environment,
        planner: Planner, controller: Controller, sensor: UltrasonicArray, dt: float = 0.1
    ): ...
    def run(self, max_steps: int = 2000) -> SimulationResult:
        # SimulationResult: pose history, control history, success flag, collision flag
        ...
```

`Planner` and `Controller` as structural-typing protocols (not required base classes) is
intentional: `reeds_shepp.ReedsSheppPlanner` and `planning.hybrid_astar.HybridAStarPlanner`
both satisfy `Planner` without a shared inheritance hierarchy, and the same for the two
controllers — new algorithms can be added later without editing existing ones.

## 3. Milestones

- **M1 — Correct baseline**: extract the current prototype into the module layout above, fix
  the degrees/radians bug and any other unit issues surfaced by writing `test_vehicle.py`, keep
  the fixed-Bezier planner and Pure Pursuit controller as-is for now. Goal: a correct,
  refactored baseline before adding new algorithms on top of it.
- **M2 — Planning**: implement `reeds_shepp.py`, then `hybrid_astar.py` on top of it; replace
  the fixed Bezier path with Hybrid A* as the default planner. Validate against scenarios where
  the fixed curve would have failed (obstacle directly on the old path) but Hybrid A* succeeds.
- **M3 — Control**: implement `control/mpc.py`; make the controller selectable per scenario
  (`scenario.yaml` names which controller to use); add a scenario that runs both controllers
  on the same start/goal/obstacles for a side-by-side comparison.
- **M4 — Sensing & re-planning**: generalize the sensor to a multi-beam array; wire
  `simulation.py` so a sensor detection of an unmapped obstacle triggers a re-plan call rather
  than only braking.
- **M5 — Visualization polish**: multi-panel animation (trajectory + live sensor readings +
  speed profile), GIF export via `matplotlib.animation.PillowWriter`, used to produce the demo
  GIFs referenced in README.md.
- **M6 — Tests & CI**: fill out `tests/`, add a GitHub Actions workflow running `pytest` on
  push/PR.

## 4. Testing strategy

- `test_vehicle.py`: known-input/known-output checks for `Vehicle.update` (e.g. driving straight
  for N steps moves `x` by `v*N*dt` with `theta` unchanged; a fixed steering angle over time
  produces the expected turning radius `L / tan(delta)`).
- `test_sensors.py`: hand-computed ray/circle intersection cases (obstacle dead ahead, obstacle
  out of range, obstacle behind the beam direction).
- `test_planning.py`: for both planners, assert the returned path (a) starts at `start` and ends
  at `goal` within tolerance, (b) never comes within the vehicle's radius of any obstacle, (c)
  respects the minimum turning radius (no waypoint-to-waypoint curvature exceeds `1/turning_radius`).
- `test_control.py`: given a straight-line path and no obstacles, assert the controller's output
  converges the vehicle to the goal within a fixed number of steps and within `tol` of the
  target pose.
- `test_simulation.py` (integration): run every scenario in `scenarios/` headless (no
  animation) and assert `SimulationResult.success` and not `SimulationResult.collision` for
  each — this is the test that would have caught the "just makes bigger circles" regression
  before it reached a demo.

## 5. Dependencies & running

```
numpy
scipy        # MPC QP solve, and/or Hybrid A* heuristic support
matplotlib
pyyaml       # scenario file loading
pytest
```
(`cvxpy` is a candidate addition for `control/mpc.py` if `scipy.optimize` proves awkward for the
QP formulation — decide during M3, not up front.)

```
pip install -e .
pytest                                  # run the test suite
python -m auto_park.demo <scenario>     # run one scenario, show the animation
python -m auto_park.demo <scenario> --save out.gif   # save a GIF for the README
```

## 6. Known issues carried over from the prototype (must be resolved by M1)

- `test_parking.py` constructs `Vehicle(theta=90.0)` and `Vehicle(theta=np.pi/6)` in the same
  scenario list — some scenarios pass degrees, some pass radians, into a model that only accepts
  radians. This is the root cause of the "barely works now just makes bigger circles" commit:
  a 90.0-radian heading is over 14 full rotations away from what was intended.
- No obstacle-avoidance routing exists today — `AdaptiveParkingController` only brakes to a stop
  when the front beam is within 0.5 m, it never routes around what it detects. Resolved by M2
  (Hybrid A*) and M4 (re-planning on detection).
- Scenarios are hardcoded Python dicts in `test_parking.py`, mixing scenario *data* with
  animation/plotting *code*. Resolved by moving to `scenarios/*.yaml` + `demo.py` (M1/M5).
- No automated tests exist for any component today. Resolved by M6.
