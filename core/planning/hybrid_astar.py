"""Hybrid A*: search over a discretized (x, y, theta) state space, where each
expansion step is a short arc/straight primitive consistent with the vehicle's
turning-radius limits, and the heuristic is the (obstacle-unaware) Reeds-Shepp
path length to the goal. This is what lets the planner route *around* obstacles
instead of only reacting to them at close range -- a single fixed Dubins/Reeds-Shepp
path has no way to represent "go around," this does. See DESIGN.md section 6.

Costing follows the standard practical formulation (Dolgov, Thrun, Montemerlo,
Diebel, "Practical Search Techniques in Path Planning for Autonomous Driving",
2010): a reverse-gear penalty, a cusp (direction-change) penalty, and a
steering-change penalty, which together bias the search toward smooth, mostly-
forward paths while still permitting reverse where it's the only way through (see
the parallel_between_cars scenario).

Analytic expansion (trying a direct, collision-checked Reeds-Shepp connection from
the current search node straight to the goal) is what keeps obstacle-free scenarios
fast: with zero obstacles, the very first attempt -- made on the root node,
unconditionally -- already succeeds, so the search degenerates to one
reeds_shepp_path() call, the same O(1) cost as DubinsPlanner today.
"""

import heapq
import itertools
from dataclasses import dataclass

import numpy as np

from core.environment import VEHICLE_RADIUS, Obstacle
from core.interfaces import Pose
from core.planning.dubins import _arc_points, _straight_points
from core.planning.reeds_shepp import reeds_shepp_length, reeds_shepp_path
from core.vehicle import wrap_angle

STEER_SIGNS = (-1, 0, 1)  # right, straight, left -- curvature is always 0 or
# exactly 1/turning_radius, the same two values dubins.py restricts itself to, so
# every primitive is drivable by construction.


def brake_distance_for(planner: "HybridAStarPlanner", buffer: float = 0.15) -> float:
    """A brake_distance safe to pass to ParkingHarness alongside this planner.

    ControllerNode's reactive sensor-based braking (a backstop for obstacles the
    planner didn't account for) assumes the planner never intentionally gets close to
    an obstacle -- true for DubinsPlanner/ReedsSheppPlanner, which ignore obstacles
    entirely, but no longer true for HybridAStarPlanner, whose avoidance routes
    deliberately pass within its own `vehicle_radius + safety_margin` clearance floor
    by design. Sized strictly below that floor so a genuine avoidance maneuver never
    falsely triggers the brake, while still catching truly unplanned proximity.
    """
    return planner.vehicle_radius + planner.safety_margin - buffer


class PlanningFailure(RuntimeError):
    """Raised when the search budget is exhausted with no path found. Never
    silently return a partial/best-effort path -- see DESIGN.md section 8's
    fail-loud precedent (e.g. IDM's explicit deceleration floor)."""


@dataclass
class _Node:
    x: float
    y: float
    theta: float
    g: float
    steer_idx: int  # index into STEER_SIGNS of the primitive that reached this node
    gear: int  # +1 forward, -1 reverse, 0 for the root (no direction committed yet)
    parent: "_Node | None"


def _grid_key(x: float, y: float, theta: float, xy_resolution: float, theta_bins: int) -> tuple[int, int, int]:
    bin_width = 2 * np.pi / theta_bins
    tb = int(np.floor(wrap_angle(theta) / bin_width)) % theta_bins
    return (int(np.floor(x / xy_resolution)), int(np.floor(y / xy_resolution)), tb)


def _points_collide(xy: np.ndarray, obstacles: list[Obstacle], clearance: float) -> bool:
    if not obstacles:
        return False
    ox = np.array([o.x for o in obstacles])
    oy = np.array([o.y for o in obstacles])
    orad = np.array([o.radius for o in obstacles])
    d = np.hypot(xy[:, 0:1] - ox[None, :], xy[:, 1:2] - oy[None, :])
    return bool(np.any(d < (orad[None, :] + clearance)))


def _primitive_points(pose: Pose, steer_idx: int, gear: int, turning_radius: float, length: float, n: int) -> np.ndarray:
    sign = STEER_SIGNS[steer_idx]
    if sign == 0:
        return _straight_points(pose, gear * length, n)
    return _arc_points(pose, turning_radius, gear * (length / turning_radius), sign > 0, n)


class HybridAStarPlanner:
    def __init__(
        self,
        xy_resolution: float = 0.5,
        theta_bins: int = 72,
        primitive_length: float = 2.0,
        reverse_penalty: float = 2.5,
        cusp_penalty: float = 3.0,
        steering_change_penalty: float = 0.5,
        collision_check_step: float = 0.25,
        safety_margin: float = 0.15,
        analytic_expansion_interval: int = 5,
        analytic_expansion_radius_factor: float = 2.0,
        max_expansions: int = 20_000,
        output_step: float = 0.1,
        vehicle_radius: float = VEHICLE_RADIUS,
    ):
        self.xy_resolution = xy_resolution
        self.theta_bins = theta_bins
        self.primitive_length = primitive_length
        self.reverse_penalty = reverse_penalty
        self.cusp_penalty = cusp_penalty
        self.steering_change_penalty = steering_change_penalty
        self.collision_check_step = collision_check_step
        self.safety_margin = safety_margin
        self.analytic_expansion_interval = analytic_expansion_interval
        self.analytic_expansion_radius_factor = analytic_expansion_radius_factor
        self.max_expansions = max_expansions
        self.output_step = output_step
        self.vehicle_radius = vehicle_radius
        self.last_expansions: int | None = None  # set by plan(); see IMPLEMENTATION.md's M2 entry

    def _reconstruct(self, node: _Node, start: Pose, goal: Pose, turning_radius: float) -> np.ndarray:
        steps = []
        n = node
        while n.parent is not None:
            steps.append((n.steer_idx, n.gear))
            n = n.parent
        steps.reverse()

        npts = max(2, round(self.primitive_length / self.output_step))
        pose = start
        segments = [np.array([start])]
        for steer_idx, gear in steps:
            pts = _primitive_points(pose, steer_idx, gear, turning_radius, self.primitive_length, npts)
            segments.append(pts[1:])
            pose = tuple(pts[-1])

        rs = reeds_shepp_path(pose, goal, turning_radius, step=self.output_step)
        segments.append(rs[1:])
        path = np.vstack(segments)
        path[:, 2] = wrap_angle(path[:, 2])
        return path

    def plan(self, start: Pose, goal: Pose, obstacles: list[Obstacle], turning_radius: float) -> np.ndarray:
        clearance = self.vehicle_radius + self.safety_margin
        start = (start[0], start[1], wrap_angle(start[2]))
        n_collision_pts = max(2, round(self.primitive_length / self.collision_check_step))

        counter = itertools.count()
        root = _Node(x=start[0], y=start[1], theta=start[2], g=0.0, steer_idx=1, gear=0, parent=None)
        root_h = reeds_shepp_length(start, goal, turning_radius)
        open_heap = [(root_h, next(counter), root)]
        best_g = {_grid_key(start[0], start[1], start[2], self.xy_resolution, self.theta_bins): 0.0}

        expansions = 0
        since_attempt = self.analytic_expansion_interval  # force an attempt on the very first pop

        while open_heap:
            _f, _, node = heapq.heappop(open_heap)
            key = _grid_key(node.x, node.y, node.theta, self.xy_resolution, self.theta_bins)
            if node.g > best_g.get(key, float("inf")) + 1e-9:
                continue  # stale entry, a cheaper route to this cell was already found

            expansions += 1
            if expansions > self.max_expansions:
                raise PlanningFailure(
                    f"Hybrid A* exhausted its search budget ({self.max_expansions} expansions) "
                    f"without reaching goal={goal} from start={start}."
                )

            since_attempt += 1
            dist_to_goal = float(np.hypot(node.x - goal[0], node.y - goal[1]))
            if (
                since_attempt >= self.analytic_expansion_interval
                or dist_to_goal < self.analytic_expansion_radius_factor * turning_radius
            ):
                since_attempt = 0
                candidate = reeds_shepp_path(
                    (node.x, node.y, node.theta), goal, turning_radius, step=self.collision_check_step
                )
                if candidate is not None and not _points_collide(candidate[:, :2], obstacles, clearance):
                    self.last_expansions = expansions  # exposed for tuning max_expansions, see IMPLEMENTATION.md
                    return self._reconstruct(node, start, goal, turning_radius)

            for steer_idx in range(len(STEER_SIGNS)):
                for gear in (1, -1):
                    pts = _primitive_points(
                        (node.x, node.y, node.theta), steer_idx, gear, turning_radius,
                        self.primitive_length, n_collision_pts,
                    )
                    if _points_collide(pts[:, :2], obstacles, clearance):
                        continue

                    nx, ny, ntheta = pts[-1]
                    ntheta = wrap_angle(ntheta)
                    cost = self.primitive_length * (self.reverse_penalty if gear == -1 else 1.0)
                    if node.parent is not None:
                        if gear != node.gear:
                            cost += self.cusp_penalty
                        cost += self.steering_change_penalty * abs(steer_idx - node.steer_idx)
                    new_g = node.g + cost

                    nkey = _grid_key(nx, ny, ntheta, self.xy_resolution, self.theta_bins)
                    if new_g < best_g.get(nkey, float("inf")) - 1e-9:
                        best_g[nkey] = new_g
                        h = reeds_shepp_length((nx, ny, ntheta), goal, turning_radius)
                        child = _Node(x=nx, y=ny, theta=ntheta, g=new_g, steer_idx=steer_idx, gear=gear, parent=node)
                        heapq.heappush(open_heap, (new_g + h, next(counter), child))

        raise PlanningFailure(f"Hybrid A* open set emptied without reaching goal={goal} from start={start}.")
