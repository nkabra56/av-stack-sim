"""Trains a PPO policy on ParkingEnv. See core/rl/parking_env.py for the environment
itself and core/validation/rl_comparison.py for evaluating the result against the
planner+controller baseline.

Usage:
    python -m core.rl.train perpendicular_open --timesteps 200000 --save model.zip
"""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from core.rl.parking_env import ParkingEnv


def make_env(scenario_name: str) -> Monitor:
    return Monitor(ParkingEnv(scenario_name=scenario_name))


def train(scenario_name: str, timesteps: int, seed: int = 0) -> PPO:
    env = make_env(scenario_name)
    model = PPO("MlpPolicy", env, seed=seed, verbose=1)
    model.learn(total_timesteps=timesteps)
    return model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a PPO parking policy.")
    parser.add_argument("scenario", nargs="?", default="perpendicular_open")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", metavar="PATH", default="parking_policy.zip")
    args = parser.parse_args(argv)

    model = train(args.scenario, args.timesteps, args.seed)
    model.save(args.save)
    print(f"Saved trained policy to {args.save}")


if __name__ == "__main__":
    main()
