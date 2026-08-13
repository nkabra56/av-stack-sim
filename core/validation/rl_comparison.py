"""Evaluates a trained ParkingEnv policy against the planner+controller baseline
(HybridAStarPlanner + Pure Pursuit/MPC, via ParkingHarness) on the same scenarios --
DESIGN.md section 10's "learned parking policy... compared against the
planner+controller baseline" future extension.

**Not a strictly apples-to-apples comparison, stated plainly rather than buried**: the
RL policy is evaluated inside ParkingEnv, which (like the policy's own training loop)
uses ground-truth state directly -- no Bus/EKF/sensor-noise graph. The baseline runs
through the real, noisy ParkingHarness, exactly as it does everywhere else in this
project. Plugging the trained policy into the real noisy loop as a drop-in `Controller`
isn't possible without changing the `Controller` protocol itself: `Controller.control(
pose, path)` has no way to hand a controller live sensor readings or an explicit goal
position, both of which this policy's observation needs (`ControllerNode` sees
obstacle_ranges internally for its own speed governor, but never forwards it to the
wrapped controller). Changing that protocol to accommodate one experimental policy
would ripple across every existing controller, so this comparison instead reports what
it actually is: does an end-to-end learned policy work *at all*, evaluated on its own
best terms, against how well the existing baseline does under its real, harder
conditions. A true head-to-head under identical noisy conditions is real follow-up
work, not attempted here.
"""

import argparse
from dataclasses import dataclass

import numpy as np
from stable_baselines3 import PPO

from core.control.mpc import MPCController
from core.control.pure_pursuit import PurePursuitAdaptive
from core.harness import ParkingHarness
from core.planning.hybrid_astar import HybridAStarPlanner
from core.rl.parking_env import ParkingEnv
from core.scenario_loader import load_scenario

BASELINE_CONTROLLERS = {
    "pure_pursuit": lambda v: PurePursuitAdaptive(wheelbase=v.wheelbase, v_max=1.5, max_steer=v.max_steer),
    "mpc": lambda v: MPCController(wheelbase=v.wheelbase, delta_max=v.max_steer, v_max=1.5),
}


@dataclass
class ComparisonSummary:
    scenario_name: str
    rl_success_rate: float
    rl_collision_rate: float
    rl_mean_steps: float
    baseline_success_rate: float
    baseline_collision_rate: float
    baseline_mean_steps: float
    baseline_controller: str


def evaluate_rl_policy(model: PPO, scenario_name: str, seeds: list[int], max_steps: int = 500) -> tuple[float, float, float]:
    env = ParkingEnv(scenario_name=scenario_name, max_steps=max_steps)
    successes, collisions, steps_list = [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        successes.append(info["success"])
        collisions.append(info["collided"])
        steps_list.append(step + 1)
    return float(np.mean(successes)), float(np.mean(collisions)), float(np.mean(steps_list))


def evaluate_baseline(scenario_name: str, controller_name: str, seeds: list[int], max_steps: int = 1000) -> tuple[float, float, float]:
    successes, collisions, steps_list = [], [], []
    for seed in seeds:
        scenario = load_scenario(scenario_name)
        planner = HybridAStarPlanner()
        controller = BASELINE_CONTROLLERS[controller_name](scenario.vehicle)
        harness = ParkingHarness(scenario.vehicle, scenario.environment, planner, controller, seed=seed)
        result = harness.run(max_steps=max_steps)
        successes.append(result.success)
        collisions.append(result.collision)
        steps_list.append(len(result.true_history))
    return float(np.mean(successes)), float(np.mean(collisions)), float(np.mean(steps_list))


def compare(model_path: str, scenario_name: str, baseline_controller: str = "mpc", seeds: list[int] | None = None) -> ComparisonSummary:
    seeds = seeds if seeds is not None else [1, 2, 3, 4, 5]
    model = PPO.load(model_path)
    rl_success, rl_collision, rl_steps = evaluate_rl_policy(model, scenario_name, seeds)
    base_success, base_collision, base_steps = evaluate_baseline(scenario_name, baseline_controller, seeds)
    return ComparisonSummary(
        scenario_name=scenario_name,
        rl_success_rate=rl_success,
        rl_collision_rate=rl_collision,
        rl_mean_steps=rl_steps,
        baseline_success_rate=base_success,
        baseline_collision_rate=base_collision,
        baseline_mean_steps=base_steps,
        baseline_controller=baseline_controller,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare a trained RL parking policy against the planner+controller baseline.")
    parser.add_argument("model_path")
    parser.add_argument("scenario", nargs="?", default="perpendicular_open")
    parser.add_argument("--baseline-controller", choices=list(BASELINE_CONTROLLERS), default="mpc")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args(argv)

    result = compare(args.model_path, args.scenario, args.baseline_controller, args.seeds)
    print(f"Scenario: {result.scenario_name} (seeds={args.seeds})")
    print(f"  RL policy (ground truth):        success={result.rl_success_rate:.0%}  "
          f"collision={result.rl_collision_rate:.0%}  mean_steps={result.rl_mean_steps:.0f}")
    print(f"  Baseline ({result.baseline_controller}, real noisy loop): success={result.baseline_success_rate:.0%}  "
          f"collision={result.baseline_collision_rate:.0%}  mean_steps={result.baseline_mean_steps:.0f}")


if __name__ == "__main__":
    main()
