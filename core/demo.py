"""CLI entry point: run a named scenario and show (or save) the animation.

Usage:
    python -m core.demo perpendicular_open
    python -m core.demo perpendicular_open --controller mpc
    python -m core.demo perpendicular_open --save out.gif
    python -m core.demo perpendicular_open --seed 7
"""

import argparse
import sys

from core.control.mpc import MPCController
from core.control.pure_pursuit import PurePursuitAdaptive
from core.harness import ParkingHarness
from core.planning.dubins import DubinsPlanner
from core.planning.hybrid_astar import HybridAStarPlanner
from core.planning.reeds_shepp import ReedsSheppPlanner
from core.scenario_loader import list_scenarios, load_scenario
from core.visualization.animate import render_animation

CONTROLLERS = {
    "pure_pursuit": lambda vehicle: PurePursuitAdaptive(
        wheelbase=vehicle.wheelbase, v_max=1.5, max_steer=vehicle.max_steer
    ),
    "mpc": lambda vehicle: MPCController(wheelbase=vehicle.wheelbase, delta_max=vehicle.max_steer, v_max=1.5),
}
PLANNERS = {
    "hybrid_astar": HybridAStarPlanner,
    "reeds_shepp": ReedsSheppPlanner,
    "dubins": DubinsPlanner,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run an auto-park scenario.")
    parser.add_argument("scenario", help=f"One of: {', '.join(list_scenarios())}")
    parser.add_argument("--controller", choices=list(CONTROLLERS), default="pure_pursuit")
    parser.add_argument(
        "--planner",
        choices=list(PLANNERS),
        default="hybrid_astar",
        help="hybrid_astar (default, obstacle-aware) / reeds_shepp / dubins (M1 baseline, for comparison)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the scenario's RNG seed")
    parser.add_argument("--save", metavar="PATH", help="Save the animation as a GIF instead of showing it")
    parser.add_argument(
        "--foxglove",
        metavar="PATH.mcap",
        help="Export a 3D scene for Foxglove instead of the matplotlib animation (needs `pip install -e \".[viz]\"`)",
    )
    args = parser.parse_args(argv)

    try:
        scenario = load_scenario(args.scenario)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)

    planner = PLANNERS[args.planner]()
    controller = CONTROLLERS[args.controller](scenario.vehicle)
    seed = args.seed if args.seed is not None else scenario.seed
    harness = ParkingHarness(scenario.vehicle, scenario.environment, planner, controller, seed=seed)

    result = harness.run(max_steps=1000)
    status = "reached spot" if result.success else ("collision" if result.collision else "did not converge")
    print(
        f"{scenario.name} [{args.planner}/{args.controller}, seed={seed}]: {status} "
        f"in {len(result.true_history)} steps"
    )

    title = f"{scenario.name} ({args.planner}/{args.controller})"
    if args.foxglove:
        from core.visualization.foxglove_export import render_foxglove

        render_foxglove(result, scenario.environment, title=title, save_path=args.foxglove)
    else:
        render_animation(result, scenario.environment, title=title, save_path=args.save)


if __name__ == "__main__":
    main()
