"""CLI entry point: run a named scenario and show (or save) the animation.

Usage:
    python -m auto_park.demo perpendicular_open
    python -m auto_park.demo perpendicular_open --controller mpc
    python -m auto_park.demo perpendicular_open --save out.gif
    python -m auto_park.demo perpendicular_open --seed 7
"""

import argparse
import sys

from auto_park.control.mpc import MPCController
from auto_park.control.pure_pursuit import PurePursuitAdaptive
from auto_park.harness import ParkingHarness
from auto_park.planning.dubins import DubinsPlanner
from auto_park.planning.hybrid_astar import HybridAStarPlanner, brake_distance_for
from auto_park.planning.reeds_shepp import ReedsSheppPlanner
from auto_park.scenario_loader import list_scenarios, load_scenario
from auto_park.visualization.animate import render_animation

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
    args = parser.parse_args(argv)

    try:
        scenario = load_scenario(args.scenario)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)

    planner = PLANNERS[args.planner]()
    controller = CONTROLLERS[args.controller](scenario.vehicle)
    seed = args.seed if args.seed is not None else scenario.seed
    harness_kwargs = {"brake_distance": brake_distance_for(planner)} if isinstance(planner, HybridAStarPlanner) else {}
    harness = ParkingHarness(
        scenario.vehicle, scenario.environment, planner, controller, seed=seed, **harness_kwargs
    )

    result = harness.run(max_steps=1000)
    status = "reached spot" if result.success else ("collision" if result.collision else "did not converge")
    print(
        f"{scenario.name} [{args.planner}/{args.controller}, seed={seed}]: {status} "
        f"in {len(result.true_history)} steps"
    )

    render_animation(
        result, scenario.environment, title=f"{scenario.name} ({args.planner}/{args.controller})", save_path=args.save
    )


if __name__ == "__main__":
    main()
