"""Parses KITTI Odometry benchmark ground-truth poses into the (x, y, theta, v,
yaw_rate) form the EKF validation needs. See DESIGN.md's "Validation against real
data" section for the axis-convention derivation.

Each line of a KITTI `poses/XX.txt` file is 12 numbers: a row-major 3x4 [R|t] matrix
mapping the left-camera frame at that timestep into the frame-0 camera frame. KITTI's
camera convention is x-right, y-down, z-forward -- so the ground-plane trajectory uses
camera x (lateral) and z (forward), not the position vector's own x/y, and heading is
the rotation about camera y, extracted as atan2(R[0,2], R[2,2]). This was verified
empirically (not just asserted) by checking that the derived heading tracks the actual
direction of travel between consecutive frames on a real sequence with turns.

KITTI's poses-only download doesn't bundle per-frame timestamps; frames are assumed
uniformly spaced at the Velodyne's nominal 10 Hz rate (dt=0.1s) -- an approximation,
not an exact per-frame timestamp, noted here rather than silently assumed.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.vehicle import wrap_angle

NOMINAL_DT = 0.1


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
