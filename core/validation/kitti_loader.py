"""Parses KITTI Odometry ground-truth poses into (x, y, theta, v, yaw_rate), using camera
x/z as ground-plane x/y (KITTI's x-right/y-down/z-forward convention). See DESIGN.md's
"Validation against real data" section."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.vehicle import wrap_angle

NOMINAL_DT = 0.1  # KITTI's poses-only download has no per-frame timestamps; assumed uniform at 10Hz


@dataclass
class KittiSequence:
    times: np.ndarray  # (N,) seconds, assumed uniform at NOMINAL_DT
    x: np.ndarray  # (N,) ground-plane forward position (m)
    y: np.ndarray  # (N,) ground-plane lateral position (m)
    theta: np.ndarray  # (N,) heading (rad)
    v: np.ndarray  # (N-1,) forward speed between consecutive frames (m/s)
    yaw_rate: np.ndarray  # (N-1,) heading rate between consecutive frames (rad/s)


def load_kitti_poses(path: str | Path, dt: float = NOMINAL_DT) -> KittiSequence:
    matrices = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            values = [float(v) for v in line.split()]
            matrices.append(np.array(values).reshape(3, 4))

    n = len(matrices)
    x = np.empty(n)
    y = np.empty(n)
    theta = np.empty(n)
    for i, m in enumerate(matrices):
        r, t = m[:3, :3], m[:3, 3]
        x[i] = t[2]
        y[i] = t[0]
        theta[i] = np.arctan2(r[0, 2], r[2, 2])

    times = np.arange(n) * dt

    dx = np.diff(x)
    dy = np.diff(y)
    v = np.hypot(dx, dy) / dt
    dtheta = wrap_angle(np.diff(theta))
    yaw_rate = dtheta / dt

    return KittiSequence(times=times, x=x, y=y, theta=theta, v=v, yaw_rate=yaw_rate)
