"""Compare the controllers on the fixed test sets.

    # camera instead of ground-truth object pose
    MUJOCO_GL=egl python scripts/evaluate.py --observation camera --noise realistic ...

Results go to results/benchmark/<tag>/ (one row per episode + summary table).
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from rwm.config import RESULTS_DIR, load_config
from rwm.environments.pick_place_env import PickPlaceEnv
from rwm.evaluation.benchmark import evaluate_controller, make_controller, summarise
from rwm.evaluation.scenarios import test_set

DEFAULT_CONDITIONS = ["nominal", "new_positions", "new_objects", "heavy_slippery"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controllers", nargs="+", default=["scripted", "sac", "mpc", "mpc_unc"])
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--sac-runs", nargs="*", type=Path, default=[])
    parser.add_argument("--world-model", type=Path, default=RESULTS_DIR / "world_model" / "ensemble.pt")
    parser.add_argument("--observation", choices=["ground_truth", "camera"], default="ground_truth")
    parser.add_argument("--noise", default="realistic")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)
    sim_cfg = load_config("simulation")
    tag = args.tag or ("ground_truth" if args.observation == "ground_truth" else f"camera_{args.noise}")
    out = RESULTS_DIR / "benchmark" / tag
    out.mkdir(parents=True, exist_ok=True)
    episodes_file = out / "episodes.csv"
    done = pd.read_csv(episodes_file) if episodes_file.exists() else pd.DataFrame()

    for condition in args.conditions:
        scenarios = test_set(sim_cfg, condition)[: args.episodes]
        env = PickPlaceEnv(sim_cfg, condition=condition)
        if args.observation == "camera":
            from rwm.perception.estimator import CameraStateEstimator

            env.state_estimator = CameraStateEstimator(env.scene.model, load_config("vision"), noise=args.noise,
                                                       dt=env.dt, seed=0)
        for name in args.controllers:
            variants = [(run, i) for i, run in enumerate(args.sac_runs)] if name.startswith("sac") else [(None, 0)]
            for run, seed in variants:
                if len(done) and ((done.condition == condition) & (done.controller == name) & (done.seed == seed)).any():
                    continue  # already evaluated, the script can be resumed
                ctrl = make_controller(name, env, sac_run=run, world_model=args.world_model, seed=seed)
                rows = evaluate_controller(env, ctrl, scenarios, dict(condition=condition, controller=name, seed=seed))
                df = pd.DataFrame(rows)
                # rewrite the whole file: controllers do not all report the same columns
                done = pd.concat([done, df], ignore_index=True)
                done.to_csv(episodes_file, index=False)
                print(f"{condition:15s} {name:9s} seed {seed}: success {df.success.mean():.2f}  "
                      f"collisions {df.collision_steps.mean():.2f}  act {df.act_time_ms.mean():.1f} ms", flush=True)
        if env.state_estimator is not None:
            env.state_estimator.close()

    all_episodes = pd.read_csv(episodes_file)
    summary = summarise(all_episodes)
    summary.to_csv(out / "summary.csv", index=False)
    cols = ["condition", "controller", "episodes", "success_rate", "success_ci_low", "success_ci_high",
            "place_error_success_mm", "collision_steps", "time_to_success_steps", "act_time_ms"]
    print(summary[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
