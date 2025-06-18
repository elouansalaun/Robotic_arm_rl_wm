"""Record (s, a, s') transitions for the world model.

    python scripts/collect_data.py scripted            # -> data/scripted.npz
    python scripts/collect_data.py random
    python scripts/collect_data.py sac --run runs/sac/sac_demo_s0
    python scripts/collect_data.py mpc --model results/world_model/ensemble.pt

Each source is stored in its own file; scripts/train_world_model.py merges them.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from rwm.config import ROOT, load_config
from rwm.environments.pick_place_env import PickPlaceEnv
from rwm.robotics.controllers import ScriptedPickPlace
from rwm.world_model.dataset import NoisyController, RandomController, TransitionData, record_episodes

DATA_DIR = ROOT / "data"


def collect_scripted(sim_cfg, cfg, seed):
    parts = []
    env = PickPlaceEnv(sim_cfg, seed=seed)
    ctrl = ScriptedPickPlace.from_env(env)
    parts.append(record_episodes(env, ctrl, cfg["scripted"], "scripted"))
    rng = np.random.default_rng(seed)
    for i in range(cfg["scripted_noisy"] // 25):
        noisy = NoisyController(ctrl, sigma=rng.uniform(0.1, 0.5), seed=seed + i)
        parts.append(record_episodes(env, noisy, 25, "scripted_noisy"))
    bursts = NoisyController(ctrl, sigma=0.05, burst_prob=0.04, seed=seed + 100)
    parts.append(record_episodes(env, bursts, cfg["scripted_bursts"], "scripted_bursts"))
    return TransitionData.concatenate(parts)


def collect_random(sim_cfg, cfg, seed):
    env = PickPlaceEnv(sim_cfg, seed=seed + 1)
    return record_episodes(env, RandomController(seed=seed), cfg["random"], "random")


def collect_sac(sim_cfg, cfg, seed, run_dir: Path):
    from rwm.rl.sac import SACController

    env = PickPlaceEnv(sim_cfg, seed=seed + 2)
    names = sorted(p.stem.replace("_obs_stats", "") for p in run_dir.glob("checkpoint_*_obs_stats.npz"))
    names.append("model" if (run_dir / "model.zip").exists() else "best")
    parts = []
    for i, name in enumerate(names):
        policy = SACController.load(run_dir, name)
        noisy = NoisyController(policy, sigma=0.1, seed=seed + i)
        parts.append(record_episodes(env, noisy, cfg["sac_per_checkpoint"], f"sac_{name}", log_every=20))
        print(f"  {name}: {len(parts[-1])} transitions")
    return TransitionData.concatenate(parts)


def collect_mpc(sim_cfg, cfg, seed, model_path: Path, n_episodes: int, out: Path):
    from rwm.planning.cem import CEMPlanner, MPCController
    from rwm.world_model.model import DynamicsEnsemble

    plan_cfg = load_config("planning")
    model = DynamicsEnsemble.load(model_path)
    keys = ("horizon", "population", "elites", "iterations", "init_std", "min_std", "momentum", "discount",
            "collision_weight", "uncertainty_weight", "action_cost")
    planner = CEMPlanner(model, seed=seed, **{k: plan_cfg[k] for k in keys})
    env = PickPlaceEnv(sim_cfg, seed=seed + 3)
    mpc = NoisyController(MPCController(planner), sigma=0.1, seed=seed)
    # planning costs ~0.15 s per step: stop shortly after the object is released,
    # and save every 10 episodes so a long run is never lost
    parts = []
    for start in range(0, n_episodes, 10):
        chunk = record_episodes(env, mpc, min(10, n_episodes - start), "mpc", steps_after_success=15)
        parts.append(chunk)
        data = TransitionData.concatenate(parts)
        data.save(out)
        n_ep = start + len(np.unique(chunk.episodes))
        print(f"  [mpc] {n_ep}/{n_episodes} episodes, {len(data)} transitions saved", flush=True)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["scripted", "random", "sac", "mpc"])
    parser.add_argument("--run", type=Path, help="SAC run directory")
    parser.add_argument("--model", type=Path, help="world model checkpoint (mpc source)")
    parser.add_argument("--episodes", type=int, default=100, help="mpc source only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    # one thread: data collection usually runs next to training jobs, and
    # oversubscribed torch threads slow the MPC planner down several times
    torch.set_num_threads(1)
    sim_cfg = load_config("simulation")
    cfg = load_config("world_model")["data"]
    out = args.out or DATA_DIR / f"{args.source}.npz"
    if args.source == "scripted":
        data = collect_scripted(sim_cfg, cfg, args.seed)
    elif args.source == "random":
        data = collect_random(sim_cfg, cfg, args.seed)
    elif args.source == "sac":
        data = collect_sac(sim_cfg, cfg, args.seed, args.run)
    else:
        data = collect_mpc(sim_cfg, cfg, args.seed, args.model, args.episodes, out)

    data.save(out)
    n_ep = len(np.unique(data.episodes))
    print(f"{len(data)} transitions, {n_ep} episodes, sources {data.source_names} -> {out}")
    print(f"contact transitions: {data.contacts[:, :2].any(1).mean():.2%}, collisions: {data.contacts[:, 2].mean():.2%}")


if __name__ == "__main__":
    main()
