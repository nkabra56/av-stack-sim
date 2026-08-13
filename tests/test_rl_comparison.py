"""Pins the real, measured RL-vs-baseline comparison (core/validation/rl_comparison.py)
as a regression test against the committed trained policy
(core/data/rl/parking_policy_perpendicular_open.zip, see core/data/rl/PROVENANCE.md for
how it was trained and the numbers this pins). Fast -- evaluates a handful of episodes
with an already-trained model, no training happens in this test.
"""

from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")

from core.validation.rl_comparison import evaluate_baseline, evaluate_rl_policy  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

MODEL_DIR = Path(__file__).parent.parent / "core" / "data" / "rl"
MODEL_PATH = MODEL_DIR / "parking_policy_perpendicular_open.zip"
FLANKED_MODEL_PATH = MODEL_DIR / "parking_policy_perpendicular_flanked.zip"


def test_policy_artifacts_are_present():
    assert MODEL_PATH.exists()
    assert FLANKED_MODEL_PATH.exists()


def test_trained_policy_reliably_parks_without_colliding():
    """The core claim: a policy trained purely from a shaped reward signal (no
    hand-coded planning or control law at all) actually learns to reach the goal
    reliably and safely on the scenario it was trained on -- not asserted, measured
    across 5 seeds."""
    model = PPO.load(str(MODEL_PATH))
    success_rate, collision_rate, mean_steps = evaluate_rl_policy(model, "perpendicular_open", seeds=[1, 2, 3, 4, 5])
    assert success_rate >= 0.8
    assert collision_rate == 0.0


def test_trained_policy_reaches_the_goal_faster_than_the_baseline():
    """The measured, somewhat surprising finding this pins down (see PROVENANCE.md):
    on this obstacle-free scenario, the learned policy reaches the goal in
    meaningfully fewer steps than the planner+controller baseline -- not a claim it's
    "better" in general (see rl_comparison.py's own docstring on why this isn't a
    strictly fair fight), just a real, reproducible number."""
    model = PPO.load(str(MODEL_PATH))
    _, _, rl_steps = evaluate_rl_policy(model, "perpendicular_open", seeds=[1, 2, 3, 4, 5])
    _, _, baseline_steps = evaluate_baseline("perpendicular_open", "mpc", seeds=[1, 2, 3, 4, 5])
    assert rl_steps < baseline_steps


def test_trained_policy_reliably_avoids_the_flanking_obstacles():
    """The harder case (see PROVENANCE.md): a scenario with two real obstacles to
    route around, not just an open lane -- the policy still parks reliably and never
    collides, even though nothing in its reward or observation is planner-aware."""
    model = PPO.load(str(FLANKED_MODEL_PATH))
    success_rate, collision_rate, _ = evaluate_rl_policy(model, "perpendicular_flanked", seeds=[1, 2, 3, 4, 5])
    assert success_rate >= 0.8
    assert collision_rate == 0.0
