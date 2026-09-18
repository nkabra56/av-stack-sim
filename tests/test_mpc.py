from unittest.mock import patch

import numpy as np

from core.control.mpc import MPCController
from core.vehicle import Vehicle


def test_mpc_falls_back_to_warm_start_when_solver_does_not_converge():
    """Code-review finding: MPCController.control used to apply result.x regardless of
    result.success. An unconverged solve can be a poor command -- fall back to the warm-started plan."""
    mpc = MPCController(wheelbase=2.7, horizon=3, v_max=1.5, delta_max=0.6)
    mpc._warm_start = np.array([0.3, 0.1, 0.3, 0.1, 0.3, 0.1])

    class _BadResult:
        success = False
        x = np.array([1.4, 0.55, 1.4, 0.55, 1.4, 0.55])  # a different, "unconverged" plan

    path = np.column_stack([np.linspace(0, 10, 20), np.zeros(20), np.zeros(20)])
    vehicle = Vehicle(x=0.0, y=0.0, theta=0.0, wheelbase=2.7)

    with patch("core.control.mpc.minimize", return_value=_BadResult()):
        v, delta = mpc.control(vehicle, path)

    assert (v, delta) == (0.3, 0.1)  # the warm start's first control, not the bad result's
