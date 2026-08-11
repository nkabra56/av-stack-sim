"""Ties environment + sensor + planner + controller + vehicle together each timestep.
See DESIGN.md section 2 for the architecture this implements.
"""

from dataclasses import dataclass, field

import numpy as np

from auto_park.environment import Environment
from auto_park.interfaces import Controller, Planner
from auto_park.sensors import UltrasonicArray
from auto_park.vehicle import Vehicle

VEHICLE_RADIUS = 0.3  # collision-check buffer around the vehicle's (x, y) point


@dataclass
class SimulationResult:
    history: np.ndarray  # (N, 3) x, y, theta
    controls: np.ndarray  # (N, 2) v, delta
    success: bool
    collision: bool
    path: np.ndarray = field(repr=False, default=None)  # planned (M, 3) path, for plotting


class ParkingSimulation:
    """Runs the plan-once / control-every-step loop described in DESIGN.md section 2.

    Braking on close-range obstacle detection (rather than routing around it) is the
    M1 baseline behavior; re-planning on detection is added in M4 (see IMPLEMENTATION.md).
    """

    def __init__(
        self,
        vehicle: Vehicle,
        environment: Environment,
        planner: Planner,
        controller: Controller,
        sensor: UltrasonicArray,
        dt: float = 0.1,
        v_max: float = 1.5,
        a_max: float = 0.8,
        k_acc: float = 2.0,
        tol: float = 0.3,
        brake_distance: float = 2.0,  # >= v_max^2 / (2*a_max) stopping distance, plus margin
    ):
        self.vehicle = vehicle
        self.environment = environment
        self.planner = planner
        self.controller = controller
        self.sensor = sensor
        self.dt = dt
        self.v_max = v_max
        self.a_max = a_max
        self.k_acc = k_acc
        self.tol = tol
        self.brake_distance = brake_distance

    def _collided(self) -> bool:
        for obstacle in self.environment.obstacles:
            dist = np.hypot(self.vehicle.x - obstacle.x, self.vehicle.y - obstacle.y)
            if dist < obstacle.radius + VEHICLE_RADIUS:
                return True
        return False

    def run(self, max_steps: int = 2000) -> SimulationResult:
        start = (self.vehicle.x, self.vehicle.y, self.vehicle.theta)
        goal = (self.environment.spot.x, self.environment.spot.y, self.environment.spot.theta)
        path = self.planner.plan(start, goal, self.environment.obstacles, self.vehicle.turning_radius)

        history: list[tuple[float, float, float]] = []
        controls: list[tuple[float, float]] = []
        v = 0.0
        collision = False

        for _ in range(max_steps):
            readings = self.sensor.sense(self.vehicle, self.environment.obstacles)
            closest_range = min(readings.values()) if readings else self.sensor.max_range

            if closest_range < self.brake_distance:
                v_desired, delta = 0.0, 0.0
            else:
                v_desired, delta = self.controller.control(self.vehicle, path)

            a = np.clip(self.k_acc * (v_desired - v), -self.a_max, self.a_max)
            v = np.clip(v + a * self.dt, -self.v_max, self.v_max)
            delta = np.clip(delta, -self.vehicle.max_steer, self.vehicle.max_steer)
            self.vehicle.update(v, delta, self.dt)

            history.append((self.vehicle.x, self.vehicle.y, self.vehicle.theta))
            controls.append((v, delta))

            if self._collided():
                collision = True
                break

            dist_to_goal = np.hypot(
                self.environment.spot.x - self.vehicle.x, self.environment.spot.y - self.vehicle.y
            )
            if dist_to_goal < self.tol:
                break

        history_arr = np.array(history) if history else np.zeros((0, 3))
        controls_arr = np.array(controls) if controls else np.zeros((0, 2))
        success = (
            not collision
            and len(history_arr) > 0
            and np.hypot(
                self.environment.spot.x - history_arr[-1, 0],
                self.environment.spot.y - history_arr[-1, 1],
            )
            < self.tol
        )
        return SimulationResult(history_arr, controls_arr, success, collision, path)
