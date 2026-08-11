"""Reeds-Shepp curves: the closed-form generalization of Dubins paths that also
allows reverse gear. See DESIGN.md section 6.

Scope: **CSC family only** (the same 4 circle-straight-circle families `dubins.py`
already implements -- LSL, RSR, LSR, RSL -- each now tried in both a forward and a
backward-gear direction, 8 candidates total). The CCC (LRL/RLR, "3-point-turn")
family is deliberately not implemented here, the same kind of scoping decision
`dubins.py` already makes for its own CCC exclusion: CCC only matters when the
start/goal turning circles are closer together than ~4x turning_radius, and even in
that regime, Hybrid A*'s step-by-step primitive expansion (planning/hybrid_astar.py)
can still compose the same 3-point-turn shape out of ordinary forward+reverse
primitives -- a missing CCC family degrades search quality in a rare pocket, it
never makes a scenario unsolvable.

Reversal trick (avoids any reflect/timeflip trigonometry): a backward-gear CSC path
from A to B is exactly the point array of the ordinary FORWARD CSC solve from B to A,
with row order reversed and headings left untouched. Verified by hand against
dubins.py's `_arc_points` formula for a turning primitive -- driving a fixed steering
angle in reverse gear traces the same (x, y, theta) points as driving the mirrored
family forward, just in the opposite order, because the kinematic bicycle model's
heading is a property of the vehicle's body orientation, not its direction of travel.
"""

import numpy as np

from auto_park.environment import Obstacle
from auto_park.interfaces import Pose
from auto_park.planning.dubins import _csc_points, _solve_csc


def reeds_shepp_length(start: Pose, goal: Pose, turning_radius: float) -> float:
    """Shortest-of-8-candidates length: min(forward CSC start->goal, backward CSC
    start->goal [= forward CSC goal->start, same length]). Falls back to Euclidean
    distance for whichever direction has no feasible CSC candidate (or both), so this
    is always defined and cheap (no point generation) -- used as the Hybrid A*
    heuristic, called many times per search."""
    euclid = float(np.hypot(goal[0] - start[0], goal[1] - start[1]))
    forward = _solve_csc(start, goal, turning_radius)
    backward = _solve_csc(goal, start, turning_radius)
    lengths = [euclid]
    if forward is not None:
        lengths.append(forward[0])
    if backward is not None:
        lengths.append(backward[0])
    return min(lengths)


def reeds_shepp_path(start: Pose, goal: Pose, turning_radius: float, step: float = 0.1) -> np.ndarray | None:
    """Shortest-of-8-candidates path as a full (N,3) point array, or None if neither
    direction has a feasible CSC candidate (the CCC-only regime).

    Forward candidate: _csc_points(start, ...) directly. Backward candidate:
    _csc_points(goal, ...) (a forward path from goal to start) with row order
    reversed -- NOT theta-negated, see module docstring -- which reproduces the exact
    correct reverse-gear kinematic path from start to goal.
    """
    forward = _solve_csc(start, goal, turning_radius)
    backward = _solve_csc(goal, start, turning_radius)
    if forward is None and backward is None:
        return None

    candidates = []
    if forward is not None:
        length, first, last, t, p, q = forward
        candidates.append((length, "forward", first, last, t, p, q))
    if backward is not None:
        length, first, last, t, p, q = backward
        candidates.append((length, "backward", first, last, t, p, q))
    _length, direction, first, last, t, p, q = min(candidates, key=lambda c: c[0])

    if direction == "forward":
        return _csc_points(start, first, last, t, p, q, turning_radius, step=step)
    path = _csc_points(goal, first, last, t, p, q, turning_radius, step=step)
    return path[::-1].copy()


class ReedsSheppPlanner:
    """Satisfies the Planner protocol. Obstacles ignored (same as DubinsPlanner --
    standalone/obstacle-free use; obstacle-aware planning is HybridAStarPlanner's job,
    which uses this module's functions internally as heuristic/local-connector)."""

    def plan(
        self, start: Pose, goal: Pose, obstacles: list[Obstacle], turning_radius: float, step: float = 0.1
    ) -> np.ndarray:
        path = reeds_shepp_path(start, goal, turning_radius, step=step)
        if path is None:
            raise RuntimeError(
                f"No CSC Reeds-Shepp candidate feasible for start={start}, goal={goal}, "
                f"turning_radius={turning_radius}: start/goal are unusually close relative "
                f"to turning_radius and would need the CCC family, which is out of scope "
                f"for this planner (see module docstring)."
            )
        return path
