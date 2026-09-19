"""Validates control/lane_centering.py's Stanley controller against a real NGSIM-derived
lane centerline (lateral control only, no ACC yet). See DESIGN.md section 12's H3 entry."""

import argparse
from dataclasses import dataclass

import numpy as np

from core.control.lane_centering import StanleyController
from core.validation.ngsim_loader import load_lane_centerline
from core.vehicle import Vehicle

# Real per-vehicle lateral positioning std within NGSIM lane 2, before aggregation/smoothing
# (see ATTRIBUTION.md): the plausibility bar below.
REAL_LATERAL_STD_M = 0.46


@dataclass
class LaneCenteringResult:
    distance: np.ndarray  # (N,) meters along the path
    vehicle_x: np.ndarray
    vehicle_y: np.ndarray
    path: np.ndarray  # (M, 3), the reference centerline
    cross_track_error: np.ndarray  # (N,) signed, meters
    max_cte_after_settling: float
    rms_cte: float
    vehicle_theta: np.ndarray | None = None  # (N,) heading after each step
    delta: np.ndarray | None = None  # (N,) steering command applied at each step


def validate(
    initial_offset: float = 1.5,
    speed: float = 20.0,
    k: float = 0.5,
    # Convergence distance scales with initial_offset: 150m gives real margin across
    # the offsets this module is exercised with (measured directly, not assumed).
    settle_distance: float = 150.0,
    dt: float = 0.1,
) -> LaneCenteringResult:
    wheelbase = 2.7
    path = load_lane_centerline()
    vehicle = Vehicle(x=path[0, 0], y=path[0, 1] + initial_offset, theta=path[0, 2], wheelbase=wheelbase)
    controller = StanleyController(wheelbase=wheelbase, k=k)

    xs, ys, thetas, deltas, distances, ctes = [], [], [], [], [], []
    while vehicle.x < path[-1, 0]:
        delta = controller.control(vehicle, path, speed)
        vehicle.update(speed, delta, dt)

        # Measured at the front axle to match StanleyController.control()'s own
        # cross-track error; reporting rear-axle CTE would validate a different quantity.
        front_x = vehicle.x + wheelbase * np.cos(vehicle.theta)
        front_y = vehicle.y + wheelbase * np.sin(vehicle.theta)
        nearest = int(np.argmin(np.hypot(path[:, 0] - front_x, path[:, 1] - front_y)))
        cte = front_y - path[nearest, 1]

        xs.append(vehicle.x)
        ys.append(vehicle.y)
        thetas.append(vehicle.theta)
        deltas.append(delta)
        distances.append(vehicle.x - path[0, 0])
        ctes.append(cte)

    distance = np.array(distances)
    cte = np.array(ctes)
    settled = cte[distance >= settle_distance]

    return LaneCenteringResult(
        distance=distance,
        vehicle_x=np.array(xs),
        vehicle_y=np.array(ys),
        path=path,
        cross_track_error=cte,
        max_cte_after_settling=float(np.max(np.abs(settled))) if len(settled) else float("nan"),
        rms_cte=float(np.sqrt(np.mean(cte**2))),
        vehicle_theta=np.array(thetas),
        delta=np.array(deltas),
    )


def plot_validation(result: LaneCenteringResult, save_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, (ax_path, ax_cte) = plt.subplots(2, 1, figsize=(10, 7))

    ax_path.plot(result.path[:, 0], result.path[:, 1], "-", color="black", lw=2, label="real lane centerline")
    ax_path.plot(result.vehicle_x, result.vehicle_y, "--", color="tab:blue", lw=2, label="vehicle (Stanley)")
    ax_path.set_xlabel("position along road (m)")
    ax_path.set_ylabel("lateral position (m)")
    ax_path.set_title("Lane centering: real NGSIM-derived centerline vs. tracked path")
    ax_path.legend(fontsize=8)
    ax_path.set_aspect("equal", "box")

    ax_cte.plot(result.distance, result.cross_track_error, color="tab:blue")
    ax_cte.axhline(REAL_LATERAL_STD_M, color="gray", linestyle=":", label="real driver lateral std")
    ax_cte.axhline(-REAL_LATERAL_STD_M, color="gray", linestyle=":")
    ax_cte.set_xlabel("distance traveled (m)")
    ax_cte.set_ylabel("cross-track error (m)")
    ax_cte.set_title("Cross-track error vs. real driver lateral scatter")
    ax_cte.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate Stanley lane centering against a real NGSIM lane.")
    parser.add_argument("--initial-offset", type=float, default=1.5)
    parser.add_argument("--speed", type=float, default=20.0)
    parser.add_argument("--plot", metavar="PATH")
    args = parser.parse_args(argv)

    result = validate(initial_offset=args.initial_offset, speed=args.speed)
    print(f"Max CTE after settling: {result.max_cte_after_settling:.3f} m")
    print(f"RMS CTE:                {result.rms_cte:.3f} m")
    print(f"Real driver lateral std: {REAL_LATERAL_STD_M:.3f} m")

    if args.plot:
        plot_validation(result, args.plot)
        print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
