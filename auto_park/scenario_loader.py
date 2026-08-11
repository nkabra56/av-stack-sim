"""Load scenario YAML files into Vehicle + Environment objects."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from auto_park.environment import Environment, Obstacle, Spot
from auto_park.vehicle import Vehicle

SCENARIOS_DIR = Path(__file__).parent / "scenarios"


@dataclass
class Scenario:
    name: str
    vehicle: Vehicle
    environment: Environment
    seed: int = 42


def load_scenario(name: str) -> Scenario:
    path = SCENARIOS_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in SCENARIOS_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"No scenario '{name}'. Available: {', '.join(available)}")

    data = yaml.safe_load(path.read_text())
    vehicle = Vehicle(**data["start"])
    spot = Spot(**data["spot"])
    obstacles = [Obstacle(**o) for o in data.get("obstacles", [])]
    return Scenario(
        name=data["name"],
        vehicle=vehicle,
        environment=Environment(spot, obstacles),
        seed=data.get("seed", 42),
    )


def list_scenarios() -> list[str]:
    return sorted(p.stem for p in SCENARIOS_DIR.glob("*.yaml"))
