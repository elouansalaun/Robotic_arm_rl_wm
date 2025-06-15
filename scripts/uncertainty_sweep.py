"""MPC success as a function of the weight given to the ensemble disagreement.

    python scripts/uncertainty_sweep.py --weights 0 0.02 0.05 0.1 0.2 0.5 --episodes 30
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from rwm.config import RESULTS_DIR, load_config
from rwm.environments.pick_place_env import PickPlaceEnv
from rwm.evaluation.benchmark import build_planner, evaluate_controller, summarise
from rwm.evaluation.scenarios import test_set
from rwm.planning.cem import MPCController
from rwm.world_model.model import DynamicsEnsemble


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=float, default=[0.0, 0.02, 0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--conditions", nargs="+", default=["nominal", "heavy_slippery"])
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--world-model", type=Path, default=RESULTS_DIR / "world_model" / "ensemble.pt")
    args = parser.parse_args()

    torch.set_num_threads(1)
    sim_cfg, plan_cfg = load_config("simulation"), load_config("planning")
    model = DynamicsEnsemble.load(args.world_model)
    out = RESULTS_DIR / "benchmark" / "uncertainty_sweep"
    out.mkdir(parents=True, exist_ok=True)
    episodes_file = out / "episodes.csv"
    done = pd.read_csv(episodes_file) if episodes_file.exists() else pd.DataFrame()

    for condition in args.conditions:
        env = PickPlaceEnv(sim_cfg, condition=condition)
        scenarios = test_set(sim_cfg, condition)[: args.episodes]
        for w in args.weights:
            if len(done) and ((done.condition == condition) & (done.uncertainty_weight == w)).any():
                continue
            mpc = MPCController(build_planner(model, plan_cfg, w))
            rows = evaluate_controller(env, mpc, scenarios, dict(condition=condition, controller="mpc",
                                                                 uncertainty_weight=w, seed=0))
            df = pd.DataFrame(rows)
            df.to_csv(episodes_file, mode="a", header=not episodes_file.exists(), index=False)
            print(f"{condition:15s} weight {w:<5g} success {df.success.mean():.2f} "
                  f"uncertainty {df.mean_uncertainty.mean():.2f}", flush=True)

    summary = summarise(pd.read_csv(episodes_file), by=("condition", "uncertainty_weight"))
    summary.to_csv(out / "summary.csv", index=False)
    print(summary[["condition", "uncertainty_weight", "success_rate", "success_ci_low", "success_ci_high",
                   "collision_steps", "time_to_success_steps"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
