"""Reeds-Shepp curves: the closed-form Dubins generalization that also allows reverse gear
-- CSC (dubins.py's 4 families) plus CCC ("3-point-turn"), both forward+backward. See DESIGN.md section 6."""

import numpy as np

from core.environment import Obstacle
from core.interfaces import Pose
from core.planning.dubins import _csc_points, _mod2pi, _solve_csc, _walk_segments


def _turn_center(pose: Pose, radius: float, left: bool) -> tuple[float, float]:
    """Center of the circle a vehicle at `pose` traces turning `left`/right at `radius`.
    Same center `_arc_points` computes internally, exposed here for CCC's geometry."""
    x, y, theta = pose
    direction = 1.0 if left else -1.0
    return x - direction * radius * np.sin(theta), y + direction * radius * np.cos(theta)


def _tangent_circle_centers(c1: tuple[float, float], c3: tuple[float, float], radius: float) -> list[tuple[float, float]]:
    """Center(s) of a circle of `radius` externally tangent to both circles of the same
    `radius` centered at c1 and c3. 0, 1, or 2 solutions (2 when |c1-c3| < 2*radius)."""
    c1_arr, c3_arr = np.array(c1), np.array(c3)
    d = float(np.linalg.norm(c3_arr - c1_arr))
    if d > 2 * radius or d < 1e-9:
        return []
    a = d / 2.0
    h_sq = radius**2 - a**2
    if h_sq < 0:
        return []
    h = np.sqrt(h_sq)
    mid = (c1_arr + c3_arr) / 2.0
    perp = np.array([-(c3_arr - c1_arr)[1], (c3_arr - c1_arr)[0]]) / d
    if h < 1e-9:
        return [tuple(mid)]
    return [tuple(mid + h * perp), tuple(mid - h * perp)]


def _angle_on_circle(center: tuple[float, float], point: tuple[float, float]) -> float:
    return float(np.arctan2(point[1] - center[1], point[0] - center[0]))


def _ccc_candidates(start: Pose, goal: Pose, turning_radius: float, first_left: bool):
    """All CCC (LRL if `first_left` else RLR) candidates from start to goal: the middle
    circle is externally tangent to both turning circles; verified via tests/test_planning.py."""
    r = turning_radius
    c1 = _turn_center(start, r, left=first_left)
    c3 = _turn_center(goal, r, left=first_left)
    results = []
    for c2 in _tangent_circle_centers(c1, c3, 2 * r):
        tangent_12 = ((c1[0] + c2[0]) / 2.0, (c1[1] + c2[1]) / 2.0)
        tangent_23 = ((c2[0] + c3[0]) / 2.0, (c2[1] + c3[1]) / 2.0)

        a_start, a_t12_on_c1 = _angle_on_circle(c1, start[:2]), _angle_on_circle(c1, tangent_12)
        t = _mod2pi(a_t12_on_c1 - a_start) if first_left else _mod2pi(a_start - a_t12_on_c1)

        a_t12_on_c2, a_t23_on_c2 = _angle_on_circle(c2, tangent_12), _angle_on_circle(c2, tangent_23)
        p = _mod2pi(a_t12_on_c2 - a_t23_on_c2) if first_left else _mod2pi(a_t23_on_c2 - a_t12_on_c2)

        a_t23_on_c3, a_goal = _angle_on_circle(c3, tangent_23), _angle_on_circle(c3, goal[:2])
        q = _mod2pi(a_goal - a_t23_on_c3) if first_left else _mod2pi(a_t23_on_c3 - a_goal)

        results.append((r * (t + p + q), t, p, q))
    return results


def _solve_ccc(
    start: Pose, goal: Pose, turning_radius: float
) -> tuple[float, str, str, str, float, float, float] | None:
    """Shortest feasible CCC candidate: (length, first, mid, last, t, p, q), mirroring
    `_solve_csc`'s shape. None if neither LRL nor RLR has a solution."""
    candidates = []
    for first_left, first, mid, last in [(True, "L", "R", "L"), (False, "R", "L", "R")]:
        for length, t, p, q in _ccc_candidates(start, goal, turning_radius, first_left):
            candidates.append((length, first, mid, last, t, p, q))
    if not candidates:
        return None
    return min(candidates, key=lambda c: c[0])


def _ccc_points(
    start: Pose, first: str, mid: str, last: str, t: float, p: float, q: float, turning_radius: float, step: float = 0.1
) -> np.ndarray:
    """CCC's analog of `_csc_points`: all three legs are turns, no straight segment
    (see `_walk_segments`, shared with dubins.py's CSC composer)."""
    return _walk_segments(start, [(first, t), (mid, p), (last, q)], turning_radius, step)


def reeds_shepp_length(start: Pose, goal: Pose, turning_radius: float, include_ccc: bool = True) -> float:
    """Shortest-of-up-to-4-candidates length (CSC/CCC x forward/backward). Falls back to
    Euclidean distance only when nothing is feasible: gated behind `include_ccc` too (KNOWN_BUGS.md entry 2)."""
    families = [(_solve_csc, start, goal), (_solve_csc, goal, start)]
    euclid = float(np.hypot(goal[0] - start[0], goal[1] - start[1]))
    lengths = [euclid] if not include_ccc else []
    if include_ccc:
        families += [(_solve_ccc, start, goal), (_solve_ccc, goal, start)]
    for solve, a, b in families:
        result = solve(a, b, turning_radius)
        if result is not None:
            lengths.append(result[0])
    if lengths:
        return min(lengths)
    return euclid


def reeds_shepp_path(start: Pose, goal: Pose, turning_radius: float, step: float = 0.1, include_ccc: bool = True) -> np.ndarray | None:
    """Shortest-of-up-to-16-candidates path as an (N,3) array, or None if nothing is
    feasible. Backward candidates reverse row order, not theta: see module docstring."""
    families = [(_solve_csc, "csc", start, goal, "forward"), (_solve_csc, "csc", goal, start, "backward")]
    if include_ccc:
        families += [(_solve_ccc, "ccc", start, goal, "forward"), (_solve_ccc, "ccc", goal, start, "backward")]
    candidates = []
    for solve, family_kind, a, b, direction in families:
        result = solve(a, b, turning_radius)
        if result is not None:
            candidates.append((result[0], direction, family_kind, result[1:]))
    if not candidates:
        return None

    _length, direction, kind, segs = min(candidates, key=lambda c: c[0])
    origin = start if direction == "forward" else goal
    if kind == "csc":
        first, last, t, p, q = segs
        path = _csc_points(origin, first, last, t, p, q, turning_radius, step=step)
    else:
        first, mid, last, t, p, q = segs
        path = _ccc_points(origin, first, mid, last, t, p, q, turning_radius, step=step)

    return path if direction == "forward" else path[::-1].copy()


class ReedsSheppPlanner:
    """Satisfies the Planner protocol. Obstacles ignored: obstacle-aware planning is
    HybridAStarPlanner's job, which uses this module's functions as heuristic/connector."""

    def plan(
        self, start: Pose, goal: Pose, obstacles: list[Obstacle], turning_radius: float, step: float = 0.1
    ) -> np.ndarray:
        path = reeds_shepp_path(start, goal, turning_radius, step=step)
        if path is None:
            raise RuntimeError(
                f"No Reeds-Shepp candidate (CSC or CCC) feasible for start={start}, goal={goal}, "
                f"turning_radius={turning_radius}. Not observed in practice for any start/goal "
                f"pair (see module docstring); if this actually triggers, it's a new finding "
                f"worth its own KNOWN_BUGS.md entry, not the CCC gap this exception used to guard."
            )
        return path
