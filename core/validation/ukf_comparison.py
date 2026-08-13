"""Head-to-head EKF vs. UKF comparison -- the actual evidence behind estimation/ukf.py's
claim that sigma-point propagation is a genuinely different algorithm, and the answer
to DESIGN.md section 10's future-extensions question: is the EKF's linearization
"a fine approximation... but it's an approximation" actually fine here, demonstrated
with real numbers rather than just asserted?

Two comparisons, both feeding both filters the *exact same* noisy odometry/measurement
draws each step (one shared RNG stream, not two independent ones) so any RMSE
difference is attributable to the propagation method, not to different noise luck:

1. `validate_against_kitti`: real driven trajectory (KITTI Odometry ground truth,
   reusing kitti_ekf_validation.py's data/noise model exactly) -- gentle, real-world
   turning rates, the actual regime this project's EKF has always been validated
   against.
2. `validate_tight_turn_stress`: a synthetic circular arc at the vehicle's own
   physical minimum turning radius (`Vehicle.turning_radius`, ~3.9m at wheelbase=2.7/
   max_steer=0.6 -- not an arbitrary "make it extreme" number, the tightest curve this
   project's own kinematic model can actually produce), deliberately the most
   nonlinear regime the bicycle model reaches in practice, with exactly known ground
   truth (closed-form circular motion) rather than needing external data.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.estimation.ekf import ExtendedKalmanFilter
from core.estimation.ukf import UnscentedKalmanFilter
from core.validation.kitti_loader import KittiSequence, load_kitti_poses
from core.vehicle import Vehicle, wrap_angle

DEFAULT_POSES_PATH = Path(__file__).parent.parent / "data" / "kitti" / "excerpt_poses.txt"


@dataclass
class ComparisonResult:
    times: np.ndarray
    true: np.ndarray  # (N, 3)
    ekf: np.ndarray
    ukf: np.ndarray
    ekf_err: np.ndarray  # (N,) position error, meters
    ukf_err: np.ndarray
    ekf_rmse: float
    ukf_rmse: float


def _delta_from_yaw_rate(v: float, yaw_rate: float, wheelbase: float) -> float:
    if abs(v) < 1e-6:
        return 0.0
    return float(np.arctan2(wheelbase * yaw_rate, v))


def _run_both(
    true_x: np.ndarray, true_y: np.ndarray, true_theta: np.ndarray, v_true: np.ndarray, delta_true: np.ndarray,
    dt: float, wheelbase: float, seed: int, odom_v_std: float, odom_delta_std: float, compass_std: float,
    position_std: float, position_fix_period: int,
) -> ComparisonResult:
    """Shared driver for both comparisons below: both filters see identical (v, delta,
    compass, position) noise draws every step, generated from one RNG stream."""
    rng = np.random.default_rng(seed)
    n = len(true_x)
    x0 = np.array([true_x[0], true_y[0], true_theta[0]])
    p0 = np.diag([0.25, 0.25, 0.05])
    r_landmark = np.diag([0.04, 0.001])  # unused (no landmarks in either comparison), required by both constructors

    ekf = ExtendedKalmanFilter(
        x0.copy(), p0.copy(), wheelbase, odom_v_std, odom_delta_std,
        r_heading=compass_std**2, r_position=np.eye(2) * position_std**2, r_landmark=r_landmark,
    )
    ukf = UnscentedKalmanFilter(
        x0.copy(), p0.copy(), wheelbase, odom_v_std, odom_delta_std,
        r_heading=compass_std**2, r_position=np.eye(2) * position_std**2, r_landmark=r_landmark,
    )

    ekf_history = np.empty((n, 3))
    ukf_history = np.empty((n, 3))
    ekf_history[0] = x0
    ukf_history[0] = x0

    for i in range(1, n):
        v_meas = v_true[i - 1] + rng.normal(0.0, odom_v_std)
        delta_meas = delta_true[i - 1] + rng.normal(0.0, odom_delta_std)
        compass_noise = rng.normal(0.0, compass_std)
        position_noise = rng.normal(0.0, position_std, size=2)

        ekf.predict(v_meas, delta_meas, dt)
        ukf.predict(v_meas, delta_meas, dt)

        ekf.update_heading(true_theta[i] + compass_noise)
        ukf.update_heading(true_theta[i] + compass_noise)
        if i % position_fix_period == 0:
            ekf.update_position(true_x[i] + position_noise[0], true_y[i] + position_noise[1])
            ukf.update_position(true_x[i] + position_noise[0], true_y[i] + position_noise[1])

        ekf_history[i] = ekf.x
        ukf_history[i] = ukf.x

    true = np.column_stack([true_x, true_y, true_theta])
    ekf_err = np.hypot(ekf_history[:, 0] - true_x, ekf_history[:, 1] - true_y)
    ukf_err = np.hypot(ukf_history[:, 0] - true_x, ukf_history[:, 1] - true_y)

    return ComparisonResult(
        times=np.arange(n) * dt,
        true=true,
        ekf=ekf_history,
        ukf=ukf_history,
        ekf_err=ekf_err,
        ukf_err=ukf_err,
        ekf_rmse=float(np.sqrt(np.mean(ekf_err**2))),
        ukf_rmse=float(np.sqrt(np.mean(ukf_err**2))),
    )


def validate_against_kitti(
    sequence: KittiSequence,
    wheelbase: float = 2.7,
    seed: int = 0,
    odom_v_std: float = 0.03,
    odom_delta_std: float = 0.01,
    compass_std: float = 0.02,
    position_std: float = 0.3,
    position_fix_period: int = 10,
) -> ComparisonResult:
    dt = sequence.times[1] - sequence.times[0] if len(sequence.x) > 1 else 0.1
    delta_true = np.array([_delta_from_yaw_rate(v, yr, wheelbase) for v, yr in zip(sequence.v, sequence.yaw_rate)])
    return _run_both(
        sequence.x, sequence.y, sequence.theta, sequence.v, delta_true, dt, wheelbase, seed,
        odom_v_std, odom_delta_std, compass_std, position_std, position_fix_period,
    )


def validate_tight_turn_stress(
    seed: int = 0,
    odom_v_std: float = 0.03,
    odom_delta_std: float = 0.01,
    compass_std: float = 0.02,
    position_std: float = 0.3,
    position_fix_period: int = 10,
    v: float = 1.5,
    dt: float = 0.1,
    n_steps: int = 400,
) -> ComparisonResult:
    """Synthetic full-circle drive at the vehicle's own tightest physical turning
    radius -- ground truth is exact closed-form circular motion, not measured data, so
    this isolates linearization error itself rather than also including whatever
    residual noise/discretization the KITTI recording has."""
    vehicle = Vehicle(wheelbase=2.7, max_steer=0.6)
    delta = vehicle.max_steer
    wheelbase = vehicle.wheelbase
    radius = vehicle.turning_radius

    t = np.arange(n_steps) * dt
    true_theta = wrap_angle((v / wheelbase) * np.tan(delta) * t)
    # Closed-form position for constant (v, delta): the vehicle traces the circle of
    # radius `radius` centered at (0, radius), starting at the origin heading +x.
    true_x = radius * np.sin(true_theta)
    true_y = radius * (1 - np.cos(true_theta))
    v_true = np.full(n_steps, v)
    delta_true = np.full(n_steps, delta)

    return _run_both(
        true_x, true_y, true_theta, v_true, delta_true, dt, wheelbase, seed,
        odom_v_std, odom_delta_std, compass_std, position_std, position_fix_period,
    )


def plot_comparison(result: ComparisonResult, save_path: str, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, (ax_path, ax_err) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(title)

    ax_path.plot(result.true[:, 0], result.true[:, 1], "-", color="black", lw=2, label="true")
    ax_path.plot(result.ekf[:, 0], result.ekf[:, 1], "--", color="tab:orange", lw=2, label="EKF")
    ax_path.plot(result.ukf[:, 0], result.ukf[:, 1], ":", color="tab:green", lw=2, label="UKF")
    ax_path.set_aspect("equal", "box")
    ax_path.set_xlabel("x (m)")
    ax_path.set_ylabel("y (m)")
    ax_path.set_title("Trajectory")
    ax_path.legend(fontsize=8)

    ax_err.plot(result.times, result.ekf_err, color="tab:orange", label=f"EKF (RMSE={result.ekf_rmse:.3f} m)")
    ax_err.plot(result.times, result.ukf_err, color="tab:green", label=f"UKF (RMSE={result.ukf_rmse:.3f} m)")
    ax_err.set_xlabel("time (s)")
    ax_err.set_ylabel("position error (m)")
    ax_err.set_title("Error over time")
    ax_err.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare the EKF and UKF head-to-head.")
    parser.add_argument("--poses", default=str(DEFAULT_POSES_PATH), help="Path to a KITTI poses.txt file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot-dir", metavar="DIR", help="Save trajectory/error plots to this directory")
    args = parser.parse_args(argv)

    sequence = load_kitti_poses(args.poses)
    kitti_result = validate_against_kitti(sequence, seed=args.seed)
    print("=== Real KITTI trajectory (gentle, real-world turning rates) ===")
    print(f"Frames:     {len(sequence.x)}")
    print(f"EKF RMSE:   {kitti_result.ekf_rmse:.4f} m")
    print(f"UKF RMSE:   {kitti_result.ukf_rmse:.4f} m")
    print(f"Difference: {abs(kitti_result.ekf_rmse - kitti_result.ukf_rmse):.4f} m "
          f"({abs(kitti_result.ekf_rmse - kitti_result.ukf_rmse) / kitti_result.ekf_rmse * 100:.1f}% of EKF RMSE)")

    stress_result = validate_tight_turn_stress(seed=args.seed)
    print("\n=== Synthetic tight-turn stress (vehicle's own minimum turning radius) ===")
    print(f"EKF RMSE:   {stress_result.ekf_rmse:.4f} m")
    print(f"UKF RMSE:   {stress_result.ukf_rmse:.4f} m")
    print(f"Difference: {abs(stress_result.ekf_rmse - stress_result.ukf_rmse):.4f} m "
          f"({abs(stress_result.ekf_rmse - stress_result.ukf_rmse) / stress_result.ekf_rmse * 100:.1f}% of EKF RMSE)")

    if args.plot_dir:
        plot_comparison(kitti_result, f"{args.plot_dir}/ukf_vs_ekf_kitti.png", "EKF vs UKF: real KITTI trajectory")
        plot_comparison(
            stress_result, f"{args.plot_dir}/ukf_vs_ekf_stress.png", "EKF vs UKF: synthetic tight-turn stress"
        )
        print(f"\nSaved plots to {args.plot_dir}/")


if __name__ == "__main__":
    main()
