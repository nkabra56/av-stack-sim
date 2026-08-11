"""Validates estimation/ekf.py against a real driven trajectory (KITTI Odometry
ground truth) instead of only synthetic noise. See DESIGN.md's "Validation against
real data" section.

Reuses ExtendedKalmanFilter unmodified. KITTI has no recorded steering angle, only
speed and yaw rate, so each step's true (v, yaw_rate) is converted to the (v, delta)
the EKF's bicycle-model predict() expects via the inverse relation
delta = atan2(wheelbase * yaw_rate, v) -- a pure adapter, not a second process model.
Noise (odom_v_std, odom_delta_std, compass_std, position_std, position_fix_period) uses
the exact same defaults as SensorNode/VehicleNode, so this is the same filter under the
same noise assumptions used everywhere else in the project, just against a real curved
trajectory instead of a synthetic one.

Runs two passes over the same noisy odometry stream: the EKF (predict + corrections)
and dead-reckoning-only (predict only) -- the natural "what would happen without the
filter's corrections" baseline, and the basis for this module's core claim.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from auto_park.estimation.ekf import ExtendedKalmanFilter
from auto_park.validation.kitti_loader import KittiSequence, load_kitti_poses

DEFAULT_POSES_PATH = Path(__file__).parent.parent / "data" / "kitti" / "excerpt_poses.txt"


@dataclass
class ValidationResult:
    times: np.ndarray
    true: np.ndarray  # (N, 3)
    ekf: np.ndarray  # (N, 3)
    dead_reckoning: np.ndarray  # (N, 3)
    ekf_err: np.ndarray  # (N,) position error, meters
    dr_err: np.ndarray  # (N,)
    ekf_rmse: float
    dr_rmse: float


def _delta_from_yaw_rate(v: float, yaw_rate: float, wheelbase: float) -> float:
    if abs(v) < 1e-6:
        return 0.0
    return float(np.arctan2(wheelbase * yaw_rate, v))


def validate(
    sequence: KittiSequence,
    wheelbase: float = 2.7,
    seed: int = 0,
    odom_v_std: float = 0.03,
    odom_delta_std: float = 0.01,
    compass_std: float = 0.02,
    position_std: float = 0.3,
    position_fix_period: int = 10,
) -> ValidationResult:
    rng = np.random.default_rng(seed)
    n = len(sequence.x)
    dt = sequence.times[1] - sequence.times[0] if n > 1 else 0.1

    x0 = np.array([sequence.x[0], sequence.y[0], sequence.theta[0]])
    p0 = np.diag([0.25, 0.25, 0.05])
    r_landmark = np.diag([0.04, 0.001])  # unused (no landmarks in this validation), required by the constructor

    ekf = ExtendedKalmanFilter(
        x0.copy(), p0.copy(), wheelbase, odom_v_std, odom_delta_std,
        r_heading=compass_std**2, r_position=np.eye(2) * position_std**2, r_landmark=r_landmark,
    )
    dead_reckoning = ExtendedKalmanFilter(
        x0.copy(), p0.copy(), wheelbase, odom_v_std, odom_delta_std,
        r_heading=compass_std**2, r_position=np.eye(2) * position_std**2, r_landmark=r_landmark,
    )

    ekf_history = np.empty((n, 3))
    dr_history = np.empty((n, 3))
    ekf_history[0] = x0
    dr_history[0] = x0

    for i in range(1, n):
        v_true, yaw_rate_true = sequence.v[i - 1], sequence.yaw_rate[i - 1]
        delta_true = _delta_from_yaw_rate(v_true, yaw_rate_true, wheelbase)

        v_meas = v_true + rng.normal(0.0, odom_v_std)
        delta_meas = delta_true + rng.normal(0.0, odom_delta_std)

        ekf.predict(v_meas, delta_meas, dt)
        dead_reckoning.predict(v_meas, delta_meas, dt)

        ekf.update_heading(sequence.theta[i] + rng.normal(0.0, compass_std))
        if i % position_fix_period == 0:
            ekf.update_position(
                sequence.x[i] + rng.normal(0.0, position_std), sequence.y[i] + rng.normal(0.0, position_std)
            )

        ekf_history[i] = ekf.x
        dr_history[i] = dead_reckoning.x

    true = np.column_stack([sequence.x, sequence.y, sequence.theta])
    ekf_err = np.hypot(ekf_history[:, 0] - true[:, 0], ekf_history[:, 1] - true[:, 1])
    dr_err = np.hypot(dr_history[:, 0] - true[:, 0], dr_history[:, 1] - true[:, 1])

    return ValidationResult(
        times=sequence.times,
        true=true,
        ekf=ekf_history,
        dead_reckoning=dr_history,
        ekf_err=ekf_err,
        dr_err=dr_err,
        ekf_rmse=float(np.sqrt(np.mean(ekf_err**2))),
        dr_rmse=float(np.sqrt(np.mean(dr_err**2))),
    )


def plot_validation(result: ValidationResult, save_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, (ax_path, ax_err) = plt.subplots(1, 2, figsize=(13, 6))

    ax_path.plot(result.true[:, 0], result.true[:, 1], "-", color="black", lw=2, label="true (KITTI ground truth)")
    ax_path.plot(result.ekf[:, 0], result.ekf[:, 1], "--", color="tab:orange", lw=2, label="EKF estimate")
    ax_path.plot(
        result.dead_reckoning[:, 0], result.dead_reckoning[:, 1], ":", color="tab:red", lw=1.5,
        label="dead reckoning only",
    )
    ax_path.set_aspect("equal", "box")
    ax_path.set_xlabel("x (m)")
    ax_path.set_ylabel("y (m)")
    ax_path.set_title("Trajectory")
    ax_path.legend(fontsize=8)

    ax_err.plot(result.times, result.ekf_err, color="tab:orange", label=f"EKF (RMSE={result.ekf_rmse:.2f} m)")
    ax_err.plot(result.times, result.dr_err, color="tab:red", label=f"dead reckoning (RMSE={result.dr_rmse:.2f} m)")
    ax_err.set_xlabel("time (s)")
    ax_err.set_ylabel("position error (m)")
    ax_err.set_title("Error over time")
    ax_err.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate the EKF against real KITTI odometry.")
    parser.add_argument("--poses", default=str(DEFAULT_POSES_PATH), help="Path to a KITTI poses.txt file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", metavar="PATH", help="Save a trajectory/error plot to this path")
    args = parser.parse_args(argv)

    sequence = load_kitti_poses(args.poses)
    result = validate(sequence, seed=args.seed)

    improvement = (1 - result.ekf_rmse / result.dr_rmse) * 100
    print(f"Frames: {len(sequence.x)}")
    print(f"EKF RMSE:              {result.ekf_rmse:.3f} m")
    print(f"Dead-reckoning RMSE:   {result.dr_rmse:.3f} m")
    print(f"Improvement:           {improvement:.1f}%")

    if args.plot:
        plot_validation(result, args.plot)
        print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
