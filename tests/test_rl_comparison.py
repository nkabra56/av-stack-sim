"""Pins the real, measured RL-vs-baseline comparison as a regression test against the
committed trained policy (see core/data/rl/PROVENANCE.md). No training happens here."""

from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO

from core.validation.rl_comparison import evaluate_baseline, evaluate_rl_policy

MODEL_DIR = Path(__file__).parent.parent / "core" / "data" / "rl"
MODEL_PATH = MODEL_DIR / "parking_policy_perpendicular_open.zip"
FLANKED_MODEL_PATH = MODEL_DIR / "parking_policy_perpendicular_flanked.zip"


def test_policy_artifacts_are_present():
    assert MODEL_PATH.exists()
    assert FLANKED_MODEL_PATH.exists()


def test_trained_policy_reliably_parks_without_colliding():
    """The core claim: a policy trained purely from a shaped reward (no hand-coded planning
    or control law) actually reaches the goal reliably and safely: measured across 5 seeds."""
    model = PPO.load(str(MODEL_PATH))
    success_rate, collision_rate, _mean_steps = evaluate_rl_policy(model, "perpendicular_open", seeds=[1, 2, 3, 4, 5])
    assert success_rate >= 0.8
    assert collision_rate == 0.0


def test_trained_policy_reaches_the_goal_faster_than_the_baseline():
    """The measured finding this pins down (PROVENANCE.md): on this obstacle-free scenario
    the learned policy reaches the goal in fewer steps than the baseline, not a general claim."""
    model = PPO.load(str(MODEL_PATH))
    _, _, rl_steps = evaluate_rl_policy(model, "perpendicular_open", seeds=[1, 2, 3, 4, 5])
    _, _, baseline_steps = evaluate_baseline("perpendicular_open", "mpc", seeds=[1, 2, 3, 4, 5])
    assert rl_steps < baseline_steps


def test_trained_policy_reliably_avoids_the_flanking_obstacles():
    """The harder case (PROVENANCE.md): two real obstacles to route around, the policy
    still parks reliably and never collides, though nothing in it is planner-aware."""
    model = PPO.load(str(FLANKED_MODEL_PATH))
    success_rate, collision_rate, _ = evaluate_rl_policy(model, "perpendicular_flanked", seeds=[1, 2, 3, 4, 5])
    assert success_rate >= 0.8
    assert collision_rate == 0.0
