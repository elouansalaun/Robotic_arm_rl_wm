"""Benchmark: the three controllers on the same scenario sets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rwm.config import load_config
from rwm.evaluation.rollout import run_episode
from rwm.planning.cem import CEMPlanner, MPCController
from rwm.robotics.controllers import ScriptedPickPlace

PLANNER_KEYS = ("horizon", "population", "elites", "iterations", "init_std", "min_std", "momentum", "discount",
                "collision_weight", "action_cost", "noise_correlation", "keep_elites", "execute")


def build_planner(model, plan_cfg: dict, uncertainty_weight: float, seed: int = 0) -> CEMPlanner:
    return CEMPlanner(model, uncertainty_weight=uncertainty_weight, seed=seed,
                      **{k: plan_cfg[k] for k in PLANNER_KEYS if k in plan_cfg})


def make_controller(name: str, env, sac_run: Path | None = None, world_model: Path | None = None, seed: int = 0):
    if name == "scripted":
        return ScriptedPickPlace.from_env(env)
    if name.startswith("sac"):
        from rwm.rl.sac import SACController

        return SACController.load(sac_run, "best")
    if name.startswith("mpc"):
        from rwm.world_model.model import DynamicsEnsemble

        plan_cfg = load_config("planning")
        weight = plan_cfg["uncertainty_weight_penalized"] if name == "mpc_unc" else plan_cfg["uncertainty_weight"]
        return MPCController(build_planner(DynamicsEnsemble.load(world_model), plan_cfg, weight, seed))
    raise ValueError(name)


def evaluate_controller(env, controller, scenarios, tags: dict) -> list[dict]:
    rows = []
    for i, sc in enumerate(scenarios):
        if hasattr(controller, "plan_times"):
            controller.plan_times.clear()
            controller.uncertainties.clear()
        r = run_episode(env, controller, sc)
        r.update(tags, episode=i)
        if hasattr(controller, "plan_times"):
            r["plan_time_ms"] = 1e3 * float(np.mean(controller.plan_times))
            r["mean_uncertainty"] = float(np.mean(controller.uncertainties))
        rows.append(r)
    return rows


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x))).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(df: pd.DataFrame, by=("condition", "controller")) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(list(by), sort=False):
        k, n = int(g.success.sum()), len(g)
        lo, hi = wilson_interval(k, n)
        succ = g[g.success]
        t_lo, t_hi = bootstrap_ci(succ.time_to_success)
        row = dict(zip(by, key))
        row.update(
            episodes=n,
            success_rate=k / n,
            success_ci_low=lo,
            success_ci_high=hi,
            place_error_mm=1e3 * g.place_error.median(),
            place_error_success_mm=1e3 * succ.place_error.median() if len(succ) else np.nan,
            collision_steps=g.collision_steps.mean(),
            episodes_with_collision=(g.collision_steps > 0).mean(),
            dropped=g.dropped.mean(),
            time_to_success_steps=succ.time_to_success.mean() if len(succ) else np.nan,
            time_to_success_ci=f"[{t_lo:.1f}, {t_hi:.1f}]",
            act_time_ms=g.act_time_ms.mean(),
        )
        if "seed" in g and g.seed.nunique() > 1:
            per_seed = g.groupby("seed").success.mean()
            row["success_seed_std"] = per_seed.std()
        rows.append(row)
    return pd.DataFrame(rows)
