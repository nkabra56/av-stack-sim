"""Validates control/lane_centering.py's Stanley controller against a real, NGSIM-
derived lane centerline instead of a hand-authored curve. See DESIGN.md section 12's
H3 entry.

Scope note: this runs the full 2D kinematic bicycle Vehicle at a constant commanded
speed under Stanley (lateral) control alone -- it does not yet combine with ACC
(longitudinal control, H1/H2) in the same closed loop. That combination (a vehicle
whose speed comes from the ACC controller and whose steering comes from Stanley,
sharing one Vehicle) is real follow-up integration work, not built here, the same way
H1 shipped ACC standalone before H2 extended it rather than attempting everything in
one pass.

Unlike the KITTI/NGSIM-follower validations (replaying real *time-series* data through
an unmodified component), there's no real trajectory to replay here -- NGSIM records
where real drivers actually were, not a reference path independent of their own
control decisions. So this validates the controller's closed-loop tracking behavior
(does it converge, does it stay tracking) against real drivers' own lateral scatter
within the lane as the plausibility bar, the same "sanity check, not a strict target"
principle the ACC validation uses.
"""

import argparse
from dataclasses import dataclass

import numpy as np

from core.control.lane_centering import StanleyController
from core.vehicle import Vehicle
from core.validation.ngsim_loader import load_lane_centerline

# Real per-vehicle lateral positioning std within NGSIM's lane 2 before aggregation/
# smoothing (see ATTRIBUTION.md) -- what real drivers' own scatter around the lane
# center looks like, used as the plausibility bar below.
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


def validate(
    initial_offset: float = 1.5,
    speed: float = 20.0,
    k: float = 0.5,
    # Convergence distance scales with initial_offset -- a 3.0m offset takes ~94m to
    # settle under 0.3m at this speed/gain, measured directly rather than assumed;
    # 150m gives real margin across the offsets this module is actually exercised
    # with, without eating too much of the 642m path as "settled" evaluation data.
    settle_distance: float = 150.0,
    dt: float = 0.1,
) -> LaneCenteringResult:
    path = load_lane_centerline()
    vehicle = Vehicle(x=path[0, 0], y=path[0, 1] + initial_offset, theta=path[0, 2], wheelbase=2.7)
    controller = StanleyController(wheelbase=2.7, k=k)

    xs, ys, distances, ctes = [], [], [], []
    while vehicle.x < path[-1, 0]:
        delta = controller.control(vehicle, path, speed)
        vehicle.update(speed, delta, dt)

        nearest = int(np.argmin(np.hypot(path[:, 0] - vehicle.x, path[:, 1] - vehicle.y)))
        cte = vehicle.y - path[nearest, 1]

        xs.append(vehicle.x)
        ys.append(vehicle.y)
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
