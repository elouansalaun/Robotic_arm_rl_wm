"""Train the SAC baseline.

    python scripts/train_rl.py --seed 0
    python scripts/train_rl.py --seed 0 --demo-episodes 50 --name sac_demo
"""

import argparse
import shutil
import time
from pathlib import Path

import torch
import yaml

from rwm.config import CONFIG_DIR, ROOT, load_config, scenario_config
from rwm.environments.pick_place_env import PickPlaceEnv
from rwm.rl.sac import (TrainingLogger, ValidationCallback, add_demonstrations, build_sac, make_vec_env,
                        save_policy)
from rwm.simulation.scene import ScenarioSampler

VALIDATION_SEED = 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--demo-episodes", type=int)
    parser.add_argument("--name", default="sac")
    parser.add_argument("--out", default=str(ROOT / "runs" / "sac"))
    args = parser.parse_args()

    torch.set_num_threads(1)
    sim_cfg = load_config("simulation")
    cfg = load_config("rl")
    if args.timesteps:
        cfg["total_timesteps"] = args.timesteps
    if args.demo_episodes is not None:
        cfg["demo_episodes"] = args.demo_episodes

    run_dir = Path(args.out) / f"{args.name}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "rl.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    shutil.copy(CONFIG_DIR / "simulation.yaml", run_dir / "simulation.yaml")

    venv = make_vec_env(sim_cfg, args.seed, normalize=cfg["normalize_obs"])
    model = build_sac(venv, cfg, args.seed)
    if cfg["demo_episodes"] > 0:
        rate = add_demonstrations(model, venv, sim_cfg, cfg["demo_episodes"], cfg["demo_action_noise"], args.seed)
        print(f"{model.replay_buffer.demos.size()} demo transitions (demo success {rate:.2f})")

    sampler = ScenarioSampler(scenario_config(sim_cfg, "nominal"), seed=VALIDATION_SEED)
    val_scenarios = [sampler.sample() for _ in range(cfg["n_eval_scenarios"])]
    checkpoints = [int(f * cfg["total_timesteps"]) for f in cfg["checkpoint_fractions"]]
    callbacks = [
        TrainingLogger(run_dir / "train_episodes.csv"),
        ValidationCallback(PickPlaceEnv(sim_cfg), val_scenarios, cfg["eval_freq"], run_dir, checkpoints),
    ]

    t0 = time.time()
    model.learn(total_timesteps=cfg["total_timesteps"], callback=callbacks)
    save_policy(model, run_dir / "model")
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {run_dir}")


if __name__ == "__main__":
    main()
