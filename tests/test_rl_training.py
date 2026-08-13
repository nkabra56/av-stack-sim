"""Smoke test for the training pipeline itself -- confirms PPO can actually train on
ParkingEnv end to end (model construction, rollout collection, policy update, model
save/load) without crashing. Deliberately NOT a convergence test: a few thousand
timesteps is nowhere near enough for PPO to learn to park, and asserting a trained
policy's success rate here would make this test either take a very long time or be
testing noise. Real, measured policy performance lives in
core/validation/rl_comparison.py / tests/test_rl_comparison.py instead, run against a
policy trained for real outside the fast test suite.
"""

import pytest

pytest.importorskip("stable_baselines3")

from core.rl.parking_env import ParkingEnv  # noqa: E402
from core.rl.train import train  # noqa: E402


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
