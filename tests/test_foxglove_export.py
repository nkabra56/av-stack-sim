"""Smoke test for the Foxglove 3D scene exporter: confirms it runs end to end and
produces a non-empty .mcap file, without asserting visual correctness (needs a human)."""

import pytest

pytest.importorskip("foxglove")

from core.control.pure_pursuit import PurePursuitAdaptive
from core.harness import ParkingHarness
from core.planning.dubins import DubinsPlanner
from core.scenario_loader import load_scenario
from core.visualization.foxglove_export import render_foxglove


def test_render_foxglove_writes_nonempty_mcap(tmp_path):
    scenario = load_scenario("perpendicular_open")
    planner = DubinsPlanner()
    controller = PurePursuitAdaptive(
        wheelbase=scenario.vehicle.wheelbase, v_max=1.5, max_steer=scenario.vehicle.max_steer
    )
    harness = ParkingHarness(scenario.vehicle, scenario.environment, planner, controller, seed=1)
    result = harness.run(max_steps=200)

    out = tmp_path / "demo.mcap"
    render_foxglove(result, scenario.environment, title="test", save_path=str(out))

    assert out.exists()
    assert out.stat().st_size > 0
