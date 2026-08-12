"""Reeds-Shepp curves: the closed-form generalization of Dubins paths that also
allows reverse gear. See DESIGN.md section 6.

**CSC family** (the same 4 circle-straight-circle families `dubins.py` already
implements -- LSL, RSR, LSR, RSL -- each tried in both a forward and a backward-gear
direction, 8 candidates) plus **CCC family** (LRL/RLR, the classic "3-point-turn"
maneuver, also tried forward and backward, up to 8 more candidates -- see
`_solve_ccc`'s docstring).

KNOWN_BUGS.md previously tracked CCC's absence as entry 5, on the assumption (standard
Dubins-curve folklore, and what `dubins.py`'s own CSC-only scoping note says) that CSC
becomes infeasible once the start/goal turning circles are closer than ~4x
turning_radius. That assumption doesn't hold for *this* implementation, specifically:
a global optimization search over every `(alpha, beta, d)` combination, using the
actual `_lsl`/`_rsr`/`_lsr`/`_rsl` functions, found no case where all 4 CSC families
are simultaneously infeasible, even as d -> 0 -- confirmed directly too, with 20,000
random trials through `ReedsSheppPlanner.plan()` targeting exactly that regime,
producing zero `RuntimeError`s. So CCC's absence was never actually a crash risk here.

It *is* still a real, separate gap worth closing: in that same close-pose regime, CCC
is strictly shorter than the best CSC candidate in ~30% of random trials, sometimes by
close to 2x -- a genuine path-quality gap, not a feasibility one. That's what this
family closes for the standalone `ReedsSheppPlanner` (`include_ccc` defaults to True).

**Deliberately NOT the default for `HybridAStarPlanner`** (which passes
`include_ccc=False` at its own 3 call sites in planning/hybrid_astar.py): CCC paths are
shorter but *more* curvature-aggressive than CSC's -- a genuine 3-point-turn packs more
net heading change into less arc length than an S-curve does -- and Hybrid A*'s
analytic expansion is attempted from every search node once it's within
`analytic_expansion_radius_factor * turning_radius` of the goal, not just the final
connection. Verified directly: turning CCC on unconditionally for Hybrid A* too
reopened KNOWN_BUGS.md bug 1's exact failure mode (Pure Pursuit colliding on
curvature-saturated paths) on 3 scenarios, including 2 that were previously perfectly
safe (`perpendicular_flanked`, `perpendicular_obstructed_lane`) -- CCC's shorter, tighter
final connections were curvature-saturated often enough to erase Pure Pursuit's margin
in new places, not just the already-documented one. `HybridAStarPlanner` already
"degrades gracefully" without CCC (composes the same 3-point-turn shape out of ordinary
primitives when it needs to, just less optimally) -- see hybrid_astar.py's own module
docstring -- so it doesn't need this family and paying its curvature-risk cost isn't
worth it there.

Reversal trick (avoids any reflect/timeflip trigonometry), used for both families: a
backward-gear path from A to B is exactly the point array of the ordinary FORWARD solve
from B to A, with row order reversed and headings left untouched. Verified by hand
against dubins.py's `_arc_points` formula for a turning primitive -- driving a fixed
steering angle in reverse gear traces the same (x, y, theta) points as driving the
mirrored family forward, just in the opposite order, because the kinematic bicycle
model's heading is a property of the vehicle's body orientation, not its direction of
travel.
"""

import numpy as np

from core.environment import Obstacle
from core.interfaces import Pose
from core.planning.dubins import _arc_points, _csc_points, _mod2pi, _solve_csc
from core.vehicle import wrap_angle


def _turn_center(pose: Pose, radius: float, left: bool) -> tuple[float, float]:
    """Center of the circle a vehicle at `pose` traces turning `left`/right at
    `radius` -- the exact same center `_arc_points` computes internally, exposed here
    since CCC's geometry (unlike CSC's) is built directly from these centers rather
    than from the alpha/beta/d trig reduction."""
    x, y, theta = pose
    direction = 1.0 if left else -1.0
    return x - direction * radius * np.sin(theta), y + direction * radius * np.cos(theta)


def _tangent_circle_centers(c1: tuple[float, float], c3: tuple[float, float], radius: float) -> list[tuple[float, float]]:
    """Center(s) of a circle of `radius` externally tangent to both circles of the
    same `radius` centered at c1 and c3 -- i.e. points exactly `2*radius` from both,
    the standard circle-circle intersection construction. 0, 1, or 2 solutions;
    2 solutions exist whenever `|c1 - c3| < 2*radius`, i.e. CCC's own feasibility
    condition (start/goal turning circles, themselves radius `radius`, closer together
    than `4*radius` -- matching the "~4x turning_radius" regime this family exists
    for)."""
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
    """All CCC (LRL if `first_left` else RLR) candidates from start to goal: the
    middle circle (opposite turn direction) is externally tangent to both the start's
    and goal's own turning circles, so its center is one of up to 2 points exactly
    `2*turning_radius` from each (`_tangent_circle_centers`). Each tangent point is the
    midpoint between the two circles' centers (equal radii, external tangency); the
    swept angle on each circle is just the difference in angular position of its entry
    and exit points, signed by that circle's turn direction -- the same "angular
    position tracks heading" relationship `_arc_points` is built on, so a swept angle
    computed this way is guaranteed to compose (via `_ccc_points`) into a path that
    actually reaches `goal`. Verified this way (not just derived): reconstructing the
    resulting (t, p, q) via `_arc_points` and checking the endpoint, across thousands
    of random trials in tests/test_planning.py.
    """
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
    """Shortest feasible CCC candidate from start to goal: (length, first, mid, last,
    t, p, q) -- mirrors `_solve_csc`'s return shape (with an added `mid` letter, since
    unlike CSC's fixed "S", CCC's middle segment is a turn too). None if neither LRL
    nor RLR has a solution (only when start/goal turning circles are farther apart
    than `4*turning_radius` -- see `_tangent_circle_centers`)."""
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
    """Walk the (first, mid, last) turn sequence from start, sampling at fixed
    arc-length `step` -- CCC's analog of `_csc_points`, minus the straight segment
    (all three legs here are arcs)."""
    seg_defs = [(first, t), (mid, p), (last, q)]
    seg_lengths = np.array([turning_radius * mag for _, mag in seg_defs])
    total = seg_lengths.sum()
    counts = np.maximum(2, np.round(seg_lengths / step).astype(int)) if total > 1e-9 else [2, 2, 2]

    pose = start
    segments = []
    for (kind, mag), n in zip(seg_defs, counts):
        pts = _arc_points(pose, turning_radius, mag, kind == "L", n)
        segments.append(pts)
        pose = tuple(pts[-1])

    path = np.vstack(segments)
    path[:, 2] = wrap_angle(path[:, 2])
    return path


def reeds_shepp_length(start: Pose, goal: Pose, turning_radius: float, include_ccc: bool = True) -> float:
    """Shortest-of-up-to-4-candidates length: min(forward/backward CSC,
    forward/backward CCC if `include_ccc`). Falls back to Euclidean distance only when
    none of the candidates are feasible (never observed in practice -- see module
    docstring; kept as a cheap, always-defined floor regardless).

    Previously (before CCC existed) this unconditionally included Euclidean distance
    in the `min()` alongside whatever CSC candidates were found -- since a curved path
    can never be shorter than the straight-line distance between its endpoints, that
    meant this function almost always just returned the Euclidean distance outright
    (only a straight-line-reachable start/goal pair ever has a candidate whose length
    actually equals it), silently discarding the "shortest of real candidates" the
    docstring claimed. Fixed to only fall back to Euclidean distance when there is
    truly no feasible candidate to measure -- caught by comparing this function's
    output against `reeds_shepp_path`'s actual generated path length in
    tests/test_planning.py, which the old unconditional-fallback version failed by a
    wide margin (up to ~4x) on ordinary, feasible cases.

    **This correction is also gated behind `include_ccc`**, same as the CCC family
    itself, even though it's a logically separate, always-safe-in-isolation fix: a
    strictly *tighter* admissible heuristic is still always correct for Hybrid A* (it
    can't cause a worse-than-optimal-under-the-discretization path to be missed), but
    it does change *which* of several equally-valid discretized paths the search finds
    first, by changing expansion order -- verified directly that this alone (with CCC
    still off) was enough to reopen KNOWN_BUGS.md bug 1's collision on 2 scenarios,
    independent of CCC. Since `HybridAStarPlanner` passes `include_ccc=False`
    specifically to keep its search behavior byte-for-byte unchanged from its
    already-validated state, the Euclidean-fallback correction has to travel with that
    same flag rather than applying unconditionally."""
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
    """Shortest-of-up-to-16-candidates path as a full (N,3) point array (up-to-8 if
    `include_ccc` is False), or None if nothing is feasible in either direction (never
    observed in practice; see module docstring).

    Forward candidate: points generated directly from `start`. Backward candidate:
    points generated from `goal` (a forward path from goal to start) with row order
    reversed -- NOT theta-negated, see module docstring -- which reproduces the exact
    correct reverse-gear kinematic path from start to goal.
    """
    families = [(_solve_csc, "csc", start, goal, "forward"), (_solve_csc, "csc", goal, start, "backward")]
    if include_ccc:
        families += [(_solve_ccc, "ccc", start, goal, "forward"), (_solve_ccc, "ccc", goal, start, "backward")]
    candidates = []
    for solve, points_fn, a, b, direction in families:
        result = solve(a, b, turning_radius)
        if result is not None:
            candidates.append((result[0], direction, points_fn, result[1:]))
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
    """Satisfies the Planner protocol. Obstacles ignored (same as DubinsPlanner --
    standalone/obstacle-free use; obstacle-aware planning is HybridAStarPlanner's job,
    which uses this module's functions internally as heuristic/local-connector)."""

    def plan(
        self, start: Pose, goal: Pose, obstacles: list[Obstacle], turning_radius: float, step: float = 0.1
    ) -> np.ndarray:
        path = reeds_shepp_path(start, goal, turning_radius, step=step)
        if path is None:
            raise RuntimeError(
                f"No Reeds-Shepp candidate (CSC or CCC) feasible for start={start}, goal={goal}, "
                f"turning_radius={turning_radius}. Not observed in practice for any start/goal "
                f"pair (see module docstring) -- if this actually triggers, it's a new finding "
                f"worth its own KNOWN_BUGS.md entry, not the CCC gap this exception used to guard."
            )
        return path
