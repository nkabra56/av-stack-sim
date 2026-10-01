"""Demo/test scenario for core/moving_obstacle.py: a pedestrian crosses the drive aisle
while the ego is mid-maneuver toward a same-heading spot, forcing a real re-plan against
the pedestrian's live position (DESIGN.md's future-extensions gap: genuinely moving obstacles).

Built directly in Python rather than as a scenarios/*.yaml file: core/scenario_loader.py's
YAML format has no notion of a moving obstacle, and ParkingHarness (not Environment) is what
turns a MovingObstacleSpec into motion, so a YAML-only scenario couldn't express this on its
own without also threading moving-obstacle specs through demo.py's harness construction.
"""

from core.environment import Environment, Spot
from core.moving_obstacle import MovingObstacleSpec
from core.vehicle import Vehicle

# Tuned by simulation, not picked by eye: forces a real stall/re-plan around x=-5 for
# seeds 1-10 (0/10 collisions); see tests/test_moving_obstacle.py for the measured margins.
PEDESTRIAN = MovingObstacleSpec(
    start_x=-5.0, start_y=-2.0, vx=0.0, vy=0.8, radius=0.5, start_time=0.8
)


def build_scenario() -> tuple[Vehicle, Environment, list[MovingObstacleSpec]]:
    """Fresh Vehicle/Environment per call: both are mutable and ParkingHarness mutates
    `environment.moving_obstacles` in place, so callers must not share instances across runs.
    The RNG seed isn't part of the scenario; pass it to ParkingHarness separately."""
    vehicle = Vehicle(x=-10.0, y=0.0, theta=0.0, wheelbase=2.7)
    environment = Environment(Spot(0.0, 0.0, 0.0), obstacles=[])
    return vehicle, environment, [PEDESTRIAN]
