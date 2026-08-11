"""Dubins path planner: the shortest curvature-constrained path for a forward-only
car between two poses, given a fixed minimum turning radius. See DESIGN.md section 6.

This is the M1 baseline planner. It replaces an earlier fixed-Bezier-curve baseline
that turned out to be kinematically infeasible near sharp heading changes -- a
generic smooth curve has no reason to respect the vehicle's actual turning radius,
and for something like a 90-degree perpendicular-parking turn compressed into a short
chord, that produced curvature far tighter than the vehicle could physically steer.
Dubins paths are built from exactly two arcs of the vehicle's own turning radius plus
a straight segment, so every generated path is trackable by construction: curvature
is always either 0 (straight) or exactly 1/turning_radius (on an arc), never more.

Uses the standard alpha/beta/d normalized formulation (LaValle, *Planning
Algorithms*, ch. 15) for all four Circle-Straight-Circle families (LSL, RSR, LSR,
RSL) and picks the shortest feasible one. All four are needed, not just the
same-direction pair (LSL/RSR): a same-heading lateral offset (e.g. pulling a couple
meters sideways into a parallel spot with no net turn) has no short same-direction
solution -- shifting sideways while ending at the same heading via same-direction
turns alone requires looping almost all the way around, whereas the
opposite-direction families (LSR/RSL) produce the short "S-curve" shift directly.
Triple-arc families (LRL, RLR) aren't implemented; they only matter when start/goal
turning circles are closer together than 4x the turning radius, which doesn't arise
in these scenarios (open lanes, well-separated poses).

Still does not avoid obstacles (ignores `obstacles`) and cannot reverse -- both
addressed by Hybrid A*/Reeds-Shepp in M2 (see IMPLEMENTATION.md).
"""

import numpy as np

from auto_park.environment import Obstacle
from auto_park.interfaces import Pose
from auto_park.vehicle import wrap_angle


def _mod2pi(theta: float) -> float:
    return theta % (2 * np.pi)


def _lsl(alpha, beta, d):
    sa, sb, ca, cb = np.sin(alpha), np.sin(beta), np.cos(alpha), np.cos(beta)
    p_sq = 2 + d * d - 2 * np.cos(alpha - beta) + 2 * d * (sa - sb)
    if p_sq < 0:
        return None
    tmp = np.arctan2(cb - ca, d + sa - sb)
    return _mod2pi(-alpha + tmp), np.sqrt(p_sq), _mod2pi(beta - tmp)


def _rsr(alpha, beta, d):
    sa, sb, ca, cb = np.sin(alpha), np.sin(beta), np.cos(alpha), np.cos(beta)
    p_sq = 2 + d * d - 2 * np.cos(alpha - beta) + 2 * d * (sb - sa)
    if p_sq < 0:
        return None
    tmp = np.arctan2(ca - cb, d - sa + sb)
    return _mod2pi(alpha - tmp), np.sqrt(p_sq), _mod2pi(-beta + tmp)


def _lsr(alpha, beta, d):
    sa, sb, ca, cb = np.sin(alpha), np.sin(beta), np.cos(alpha), np.cos(beta)
    p_sq = -2 + d * d + 2 * np.cos(alpha - beta) + 2 * d * (sa + sb)
    if p_sq < 0:
        return None
    p = np.sqrt(p_sq)
    tmp = np.arctan2(-ca - cb, d + sa + sb) - np.arctan2(-2.0, p)
    return _mod2pi(-alpha + tmp), p, _mod2pi(-_mod2pi(beta) + tmp)


def _rsl(alpha, beta, d):
    sa, sb, ca, cb = np.sin(alpha), np.sin(beta), np.cos(alpha), np.cos(beta)
    p_sq = d * d - 2 + 2 * np.cos(alpha - beta) - 2 * d * (sa + sb)
    if p_sq < 0:
        return None
    p = np.sqrt(p_sq)
    tmp = np.arctan2(ca + cb, d - sa - sb) - np.arctan2(2.0, p)
    return _mod2pi(alpha - tmp), p, _mod2pi(beta - tmp)


_FAMILIES = {"LSL": (_lsl, "L", "L"), "RSR": (_rsr, "R", "R"), "LSR": (_lsr, "L", "R"), "RSL": (_rsl, "R", "L")}


def _arc_points(pose: Pose, radius: float, angle: float, left: bool, n: int) -> np.ndarray:
    x, y, theta = pose
    direction = 1.0 if left else -1.0
    cx = x - direction * radius * np.sin(theta)
    cy = y + direction * radius * np.cos(theta)
    a0 = np.arctan2(y - cy, x - cx)
    angles = a0 + direction * np.linspace(0, angle, n)
    thetas = angles + direction * (np.pi / 2)
    return np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles), thetas])


def _straight_points(pose: Pose, distance: float, n: int) -> np.ndarray:
    x, y, theta = pose
    s = np.linspace(0, distance, n)
    return np.column_stack([x + s * np.cos(theta), y + s * np.sin(theta), np.full(n, theta)])


class DubinsPlanner:
    def plan(
        self, start: Pose, goal: Pose, obstacles: list[Obstacle], turning_radius: float, npts: int = 150
    ) -> np.ndarray:
        ex, ey = goal[0] - start[0], goal[1] - start[1]
        dist = np.hypot(ex, ey)
        chord_theta = np.arctan2(ey, ex) if dist > 1e-9 else start[2]
        d = dist / turning_radius
        alpha = _mod2pi(start[2] - chord_theta)
        beta = _mod2pi(goal[2] - chord_theta)

        candidates = []
        for name, (fn, first, last) in _FAMILIES.items():
            result = fn(alpha, beta, d)
            if result is None:
                continue
            t, p, q = result
            candidates.append((turning_radius * (t + p + q), first, last, t, p, q))
        _length, first, last, t, p, q = min(candidates, key=lambda c: c[0])

        seg_defs = [(first, t, True), ("S", turning_radius * p, False), (last, q, True)]
        seg_lengths = np.array([turning_radius * mag if is_angle else mag for _, mag, is_angle in seg_defs])
        total = seg_lengths.sum()
        counts = np.maximum(2, np.round(npts * seg_lengths / total).astype(int)) if total > 1e-9 else [2, 2, 2]

        pose = start
        segments = []
        for (kind, mag, _is_angle), n in zip(seg_defs, counts):
            pts = _straight_points(pose, mag, n) if kind == "S" else _arc_points(pose, turning_radius, mag, kind == "L", n)
            segments.append(pts)
            pose = tuple(pts[-1])

        path = np.vstack(segments)
        path[:, 2] = wrap_angle(path[:, 2])
        return path
