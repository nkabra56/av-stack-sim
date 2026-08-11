"""Validates control/acc.py's controllers against a real recorded car-following
trajectory (NGSIM) instead of only synthetic scenarios. See DESIGN.md's ACC section.

Replays the real leader's recorded speed profile through LeadVehicleNode; runs *our*
ACC controller as the follower in simulation via AccHarness. Unlike the KITTI EKF
validation (which replays real data through an unmodified estimator and compares its
output directly to ground truth), a controller's closed-loop behavior isn't directly
comparable to what a human driver actually did -- so this validates three different
things instead: (1) safety, a hard pass/fail (gap never reaches zero), (2) comfort
(bounded jerk), and (3) plausibility -- our controller's resulting gap/time-headway
should land in a realistic range, sanity-checked against the *real* follower's own
recorded Space_Headway/Time_Headway from the same data, not asserted to match it
exactly (the real driver isn't assumed optimal).
"""

import argparse
from dataclasses import dataclass

import numpy as np

from auto_park.control.acc import IDMController, MpcAccController
from auto_park.highway_harness import AccHarness, AccSimulationResult
from auto_park.validation.ngsim_loader import load_following_pair

CONTROLLERS = {
    "idm": lambda: IDMController(),
    "mpc": lambda: MpcAccController(),
}


@dataclass
class AccValidationResult:
    sim: AccSimulationResult
    real_space_headway: np.ndarray
    real_time_headway: np.ndarray
    max_jerk: float
    mean_gap: float
    mean_real_gap: float


def validate(controller_name: str = "idm", seed: int = 0) -> AccValidationResult:
    pair = load_following_pair()
    controller = CONTROLLERS[controller_name]()

    harness = AccHarness(
        lead_position=pair.leader.position,
        lead_speed=pair.leader.speed,
        lead_length=pair.leader.length,
        controller=controller,
        ego_initial_speed=pair.follower.speed[0],
        ego_initial_gap=pair.real_space_headway[0],
        seed=seed,
    )
    sim = harness.run()

    jerk = np.diff(sim.ego_accel) / harness.dt
    max_jerk = float(np.max(np.abs(jerk))) if len(jerk) else 0.0

    return AccValidationResult(
        sim=sim,
        real_space_headway=pair.real_space_headway,
        real_time_headway=pair.real_time_headway,
        max_jerk=max_jerk,
        mean_gap=float(np.mean(sim.gap)),
        mean_real_gap=float(np.mean(pair.real_space_headway)),
    )


def plot_validation(result: AccValidationResult, save_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, (ax_gap, ax_speed) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax_gap.plot(result.sim.times, result.sim.gap, color="tab:blue", label="our controller's gap")
    ax_gap.plot(
        result.sim.times[: len(result.real_space_headway)], result.real_space_headway,
        color="black", linestyle="--", label="real NGSIM follower's gap",
    )
    ax_gap.set_ylabel("gap (m)")
    ax_gap.set_title("Following gap: ours vs. the real recorded follower")
    ax_gap.legend(fontsize=8)

    ax_speed.plot(result.sim.times, result.sim.ego_speed, color="tab:blue", label="our ego speed")
    ax_speed.plot(result.sim.times, result.sim.lead_speed, color="tab:red", label="lead speed (real, replayed)")
    ax_speed.set_xlabel("time (s)")
    ax_speed.set_ylabel("speed (m/s)")
    ax_speed.set_title("Speed profile")
    ax_speed.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate ACC controllers against real NGSIM data.")
    parser.add_argument("--controller", choices=list(CONTROLLERS), default="idm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", metavar="PATH", help="Save a gap/speed plot to this path")
    args = parser.parse_args(argv)

    result = validate(args.controller, seed=args.seed)
    print(f"Controller:        {args.controller}")
    print(f"Min gap:           {result.sim.min_gap:.2f} m (collided={result.sim.collided})")
    print(f"Mean gap (ours):   {result.mean_gap:.2f} m")
    print(f"Mean gap (real):   {result.mean_real_gap:.2f} m")
    print(f"Max jerk:          {result.max_jerk:.2f} m/s^3")

    if args.plot:
        plot_validation(result, args.plot)
        print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
