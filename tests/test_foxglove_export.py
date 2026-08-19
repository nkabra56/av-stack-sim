"""Smoke test for the Foxglove 3D scene exporter -- confirms it runs end to end
against a real SimulationResult and produces a non-empty .mcap file, without
asserting anything about visual correctness (that needs a human opening the file in
the Foxglove desktop app, see FOXGLOVE_VIZ_PLAN.md)."""

import pytest

pytest.importorskip("foxglove")

from core.control.pure_pursuit import PurePursuitAdaptive  # noqa: E402
from core.harness import ParkingHarness  # noqa: E402
from core.planning.dubins import DubinsPlanner  # noqa: E402
from core.scenario_loader import load_scenario  # noqa: E402
from core.visualization.foxglove_export import render_foxglove  # noqa: E402


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
