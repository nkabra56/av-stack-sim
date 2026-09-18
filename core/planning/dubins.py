"""Dubins path planner: the shortest curvature-constrained path for a forward-only car
between two poses, via the standard 4-family CSC (LSL/RSR/LSR/RSL) formulation. Doesn't
avoid obstacles or reverse: see Hybrid A*/Reeds-Shepp for that. DESIGN.md section 6."""

import numpy as np

from core.environment import Obstacle
from core.interfaces import Pose
from core.vehicle import wrap_angle


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


def _solve_csc(
    start: Pose, goal: Pose, turning_radius: float
) -> tuple[float, str, str, float, float, float] | None:
    """Shortest feasible CSC (Dubins) candidate: (length, first, last, t, p, q), t/q the
    arcs' swept angles and p the straight length. None if all 4 families are infeasible."""
    ex, ey = goal[0] - start[0], goal[1] - start[1]
    dist = np.hypot(ex, ey)
    chord_theta = np.arctan2(ey, ex) if dist > 1e-9 else start[2]
    d = dist / turning_radius
    alpha = _mod2pi(start[2] - chord_theta)
    beta = _mod2pi(goal[2] - chord_theta)

    candidates = []
    for fn, first, last in _FAMILIES.values():
        result = fn(alpha, beta, d)
        if result is None:
            continue
        t, p, q = result
        candidates.append((turning_radius * (t + p + q), first, last, t, p, q))
    if not candidates:
        return None
    return min(candidates, key=lambda c: c[0])


def _walk_segments(
    start: Pose, seg_defs: list[tuple[str, float]], turning_radius: float, step: float
) -> np.ndarray:
    """Walk a (kind, magnitude) segment list from start, sampling at fixed arc-length
    `step`: shared by dubins.py's CSC composer and reeds_shepp.py's CCC composer."""
    seg_lengths = np.array([mag if kind == "S" else turning_radius * mag for kind, mag in seg_defs])
    total = seg_lengths.sum()
    counts = np.maximum(2, np.round(seg_lengths / step).astype(int)) if total > 1e-9 else [2] * len(seg_defs)

    pose = start
    segments = []
    for (kind, mag), n in zip(seg_defs, counts, strict=True):
        pts = _straight_points(pose, mag, n) if kind == "S" else _arc_points(pose, turning_radius, mag, kind == "L", n)
        segments.append(pts)
        pose = tuple(pts[-1])

    path = np.vstack(segments)
    path[:, 2] = wrap_angle(path[:, 2])
    return path


def _csc_points(
    start: Pose, first: str, last: str, t: float, p: float, q: float, turning_radius: float, step: float = 0.1
) -> np.ndarray:
    return _walk_segments(start, [(first, t), ("S", turning_radius * p), (last, q)], turning_radius, step)


class DubinsPlanner:
    def plan(
        self, start: Pose, goal: Pose, obstacles: list[Obstacle], turning_radius: float, npts: int = 150
    ) -> np.ndarray:
        result = _solve_csc(start, goal, turning_radius)
        if result is None:
            raise RuntimeError(
                f"No CSC Dubins candidate feasible for start={start}, goal={goal}, "
                f"turning_radius={turning_radius} (start/goal turning circles are too "
                f"close together: the CCC-only regime this planner doesn't cover)."
            )
        length, first, last, t, p, q = result
        step = max(length / npts, 1e-6)
        return _csc_points(start, first, last, t, p, q, turning_radius, step=step)
