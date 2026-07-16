"""MPC success as a function of the weight given to the ensemble disagreement.

The weights already run by scripts/evaluate.py (mpc: 0, mpc_unc: 0.05) are
taken from results/benchmark/ground_truth/ instead of being run again.
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
    parser.add_argument("--weights", nargs="+", type=float, default=[0.02, 0.1, 0.2])
    parser.add_argument("--conditions", nargs="+", default=["nominal", "new_objects"])
    parser.add_argument("--episodes", type=int, default=50)
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
            done = pd.concat([done, df], ignore_index=True)
            done.to_csv(episodes_file, index=False)
            print(f"{condition:15s} weight {w:<5g} success {df.success.mean():.2f} "
                  f"uncertainty {df.mean_uncertainty.mean():.2f}", flush=True)

    sweep = pd.read_csv(episodes_file)
    bench_file = RESULTS_DIR / "benchmark" / "ground_truth" / "episodes.csv"
    if bench_file.exists():
        bench = pd.read_csv(bench_file)
        weights = {"mpc": plan_cfg["uncertainty_weight"], "mpc_unc": plan_cfg["uncertainty_weight_penalized"]}
        bench = bench[bench.controller.isin(weights) & bench.condition.isin(args.conditions)
                      & (bench.episode < args.episodes)].copy()
        bench["uncertainty_weight"] = bench.controller.map(weights)
        sweep = pd.concat([sweep, bench[~bench.uncertainty_weight.isin(sweep.uncertainty_weight)]], ignore_index=True)
    summary = summarise(sweep, by=("condition", "uncertainty_weight"))
    summary = summary.sort_values(["condition", "uncertainty_weight"])
    summary.to_csv(out / "summary.csv", index=False)
    print(summary[["condition", "uncertainty_weight", "success_rate", "success_ci_low", "success_ci_high",
                   "collision_steps", "time_to_success_steps"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
