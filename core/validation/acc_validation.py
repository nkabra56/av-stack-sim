"""Validates control/acc.py's controllers against a real recorded NGSIM car-following
trajectory: safety (gap never zero), comfort (bounded jerk), plausibility vs. the real follower."""

import argparse
from dataclasses import dataclass

import numpy as np

from core.control.acc import IDMController, MpcAccController
from core.highway_harness import AccHarness, AccSimulationResult
from core.validation.ngsim_loader import load_following_pair

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


def compare_uncertainty_gain(uncertainty_gain: float, seed: int = 0) -> dict:
    """Replays this trace's real leader speed with MpcAccController at uncertainty_gain=0.0
    vs. a chosen nonzero gain, and measures the realized min gap in two windows: the trace's
    genuine deceleration/full-stop event (leader speed reaches 0 around t=21-30.5s, ramping
    down from about t=10s) and a later, smooth-flow stretch (t>=55s, leader cruising 12-18 m/s).
    """
    pair = load_following_pair()

    def run(gain: float) -> AccSimulationResult:
        harness = AccHarness(
            lead_position=pair.leader.position,
            lead_speed=pair.leader.speed,
            lead_length=pair.leader.length,
            controller=MpcAccController(uncertainty_gain=gain),
            ego_initial_speed=pair.follower.speed[0],
            ego_initial_gap=pair.real_space_headway[0],
            seed=seed,
        )
        return harness.run()

    baseline = run(0.0)
    boosted = run(uncertainty_gain)

    decel_mask = (baseline.times >= 10.0) & (baseline.times <= 35.0)
    smooth_mask = baseline.times >= 55.0

    return {
        "baseline_min_gap_decel": float(np.min(baseline.gap[decel_mask])),
        "boosted_min_gap_decel": float(np.min(boosted.gap[decel_mask])),
        "baseline_min_gap_smooth": float(np.min(baseline.gap[smooth_mask])),
        "boosted_min_gap_smooth": float(np.min(boosted.gap[smooth_mask])),
        "baseline_mean_gap_smooth": float(np.mean(baseline.gap[smooth_mask])),
        "boosted_mean_gap_smooth": float(np.mean(boosted.gap[smooth_mask])),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate ACC controllers against real NGSIM data.")
    parser.add_argument("--controller", choices=list(CONTROLLERS), default="idm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", metavar="PATH", help="Save a gap/speed plot to this path")
    parser.add_argument(
        "--compare-uncertainty-gain", type=float, metavar="GAIN",
        help="Instead of --controller, compare MpcAccController at gain=0.0 vs. this gain",
    )
    args = parser.parse_args(argv)

    if args.compare_uncertainty_gain is not None:
        stats = compare_uncertainty_gain(args.compare_uncertainty_gain, seed=args.seed)
        print(f"Uncertainty gain:            {args.compare_uncertainty_gain}")
        print(f"Min gap during decel/stop:  baseline={stats['baseline_min_gap_decel']:.3f} m, "
              f"boosted={stats['boosted_min_gap_decel']:.3f} m")
        print(f"Min gap during smooth flow: baseline={stats['baseline_min_gap_smooth']:.3f} m, "
              f"boosted={stats['boosted_min_gap_smooth']:.3f} m")
        print(f"Mean gap during smooth flow: baseline={stats['baseline_mean_gap_smooth']:.3f} m, "
              f"boosted={stats['boosted_mean_gap_smooth']:.3f} m")
        return

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
