from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"


def load_config(name: str) -> dict:
    """Load configs/<name>.yaml (or an explicit path)."""
    path = Path(name)
    if not path.suffix:
        path = CONFIG_DIR / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "shapes":
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def scenario_config(sim_cfg: dict, condition: str = "nominal") -> dict:
    """Training scenario ranges with the overrides of a test condition applied."""
    overrides = sim_cfg.get("test_conditions", {}).get(condition)
    if overrides is None:
        raise KeyError(f"unknown test condition '{condition}'")
    return deep_update(sim_cfg["scenario"], overrides)
