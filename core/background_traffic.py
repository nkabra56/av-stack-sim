"""Background traffic for the highway scenes: each lane's leader replays the recorded flow speed
(shifted in time) and the cars behind it follow with IDM. Cars never change lanes or react to the ego."""

import numpy as np

from core.control.acc import IDMController


def simulate_lane(
    flow_speed: np.ndarray, start_front: list[float], lengths: list[float], lag_ticks: int = 0, dt: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Front-bumper positions and speeds, each (n_cars, N). Car 0 leads and the rest follow it."""
    n, m = len(flow_speed), len(start_front)
    idm = IDMController(v0=22.0, a_max=2.0, b_comfortable=2.5, time_headway=1.3)
    pos, spd = np.zeros((m, n)), np.zeros((m, n))
    pos[:, 0] = start_front
    spd[:, 0] = flow_speed[min(max(-lag_ticks, 0), n - 1)]
    for k in range(n):
        spd[0, k] = flow_speed[min(max(k - lag_ticks, 0), n - 1)]
        for i in range(1, m):
            gap = pos[i - 1, k] - lengths[i - 1] - pos[i, k]
            if k > 0:
                spd[i, k] = max(0.0, spd[i, k - 1] + idm.control(spd[i, k - 1], gap, spd[i - 1, k - 1]) * dt)
        if k < n - 1:
            pos[:, k + 1] = pos[:, k] + spd[:, k] * dt
    return pos, spd
