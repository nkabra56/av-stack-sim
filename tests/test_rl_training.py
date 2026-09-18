"""Smoke test for the training pipeline: confirms PPO can train on ParkingEnv end to end
without crashing. Deliberately not a convergence test -- see test_rl_comparison.py for real numbers."""

import pytest

pytest.importorskip("stable_baselines3")

from core.rl.parking_env import ParkingEnv
from core.rl.train import train


def test_training_runs_end_to_end_without_crashing(tmp_path):
    model = train("perpendicular_open", timesteps=256, seed=0)

    save_path = tmp_path / "smoke_policy.zip"
    model.save(str(save_path))
    assert save_path.exists()

    from stable_baselines3 import PPO

    loaded = PPO.load(str(save_path))
    env = ParkingEnv(scenario_name="perpendicular_open")
    obs, _ = env.reset(seed=0)
    action, _ = loaded.predict(obs, deterministic=True)
    assert env.action_space.contains(action)
