"""Train the SAC baseline.
"""

import argparse
import shutil
import time
from pathlib import Path

import torch
import yaml

from rwm.config import CONFIG_DIR, RESULTS_DIR, ROOT, load_config, scenario_config
from rwm.environments.pick_place_env import PickPlaceEnv
from rwm.rl.sac import (ResumeCheckpoint, TrainingLogger, ValidationCallback, add_demonstrations, build_sac,
                        load_training_state, make_vec_env, save_policy, save_training_state)
from rwm.simulation.scene import ScenarioSampler

VALIDATION_SEED = 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--demo-episodes", type=int)
    parser.add_argument("--name", default="sac")
    parser.add_argument("--out", default=str(ROOT / "runs" / "sac"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    sim_cfg = load_config("simulation")
    run_dir = Path(args.out) / f"{args.name}_s{args.seed}"

    if args.resume:
        with open(run_dir / "rl.yaml") as f:
            cfg = yaml.safe_load(f)
        model = load_training_state(run_dir / "resume", sim_cfg, args.seed)
        print(f"resuming {run_dir.name} at {model.num_timesteps} steps")
    else:
        cfg = load_config("rl")
        if args.timesteps:
            cfg["total_timesteps"] = args.timesteps
        if args.demo_episodes is not None:
            cfg["demo_episodes"] = args.demo_episodes
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
        TrainingLogger(run_dir / "train_episodes.csv", resume=args.resume),
        ValidationCallback(PickPlaceEnv(sim_cfg), val_scenarios, cfg["eval_freq"], run_dir, checkpoints,
                           resume=args.resume),
        ResumeCheckpoint(run_dir, cfg["resume_every"]),
    ]

    t0 = time.time()
    remaining = cfg["total_timesteps"] - model.num_timesteps
    model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=not args.resume)
    save_policy(model, run_dir / "model")
    save_training_state(model, run_dir / "resume")
    export(run_dir)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {run_dir}")


def export(run_dir: Path) -> None:
    """Copy the logs and the selected policy to results/ (runs/ is not versioned)."""
    logs = RESULTS_DIR / "rl"
    logs.mkdir(parents=True, exist_ok=True)
    shutil.copy(run_dir / "validation.csv", logs / f"{run_dir.name}_validation.csv")
    shutil.copy(run_dir / "train_episodes.csv", logs / f"{run_dir.name}_train_episodes.csv")
    ckpt = RESULTS_DIR / "checkpoints" / run_dir.name
    ckpt.mkdir(parents=True, exist_ok=True)
    for f in ("best.zip", "best_obs_stats.npz", "rl.yaml"):
        shutil.copy(run_dir / f, ckpt / f)


if __name__ == "__main__":
    main()
