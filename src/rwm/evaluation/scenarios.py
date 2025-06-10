"""Fixed scenario sets so that every controller is evaluated on exactly the same episodes."""

from __future__ import annotations

from pathlib import Path

from rwm.config import RESULTS_DIR, scenario_config
from rwm.simulation.scene import Scenario, ScenarioSampler, load_scenarios, save_scenarios

TEST_SEED = 2024
TEST_SET_DIR = RESULTS_DIR / "test_sets"


def generate(sim_cfg: dict, condition: str, n: int, seed: int = TEST_SEED) -> list[Scenario]:
    conditions = list(sim_cfg["test_conditions"])
    sampler = ScenarioSampler(scenario_config(sim_cfg, condition), seed=seed + 100 * conditions.index(condition))
    return [sampler.sample() for _ in range(n)]


def test_set(sim_cfg: dict, condition: str, n: int | None = None, directory: Path = TEST_SET_DIR) -> list[Scenario]:
    """Load results/test_sets/<condition>.json, creating it on first use."""
    path = Path(directory) / f"{condition}.json"
    if not path.exists():
        save_scenarios(generate(sim_cfg, condition, n or 100), path)
    scenarios = load_scenarios(path)
    return scenarios[:n] if n else scenarios
