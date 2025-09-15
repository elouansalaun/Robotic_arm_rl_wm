"""SAC baseline with Stable-Baselines3: env construction, demo seeding, logging, policy wrapper."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rwm.environments.pick_place_env import PickPlaceEnv
from rwm.evaluation.rollout import run_episode
from rwm.robotics.controllers import ScriptedPickPlace


def make_vec_env(sim_cfg: dict, seed: int, normalize: bool = True, condition: str = "nominal"):
    def _make():
        env = PickPlaceEnv(sim_cfg, condition=condition, seed=seed)
        return Monitor(env, info_keywords=("is_success", "collision_steps", "place_error"))

    venv = DummyVecEnv([_make])
    venv.seed(seed)
    if normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return venv


class MixedReplayBuffer(ReplayBuffer):
    """Replay buffer that draws a fixed fraction of every batch from demonstrations.

    Symmetric sampling as in RLPD (Ball et al., 2023): the demonstrations stay
    visible to the critic for the whole training instead of being diluted in
    the online data.
    """

    def __init__(self, *args, demo_fraction: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.demo_fraction = demo_fraction
        self.demos: ReplayBuffer | None = None

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        if self.demos is None or self.demos.size() == 0:
            return super().sample(batch_size, env)
        n_demo = int(round(batch_size * self.demo_fraction)) if self.size() > 0 else batch_size
        demo = self.demos.sample(n_demo, env)
        if n_demo == batch_size:
            return demo
        online = super().sample(batch_size - n_demo, env)
        merged = [None if a is None else torch.cat([a, b]) for a, b in zip(online, demo)]
        return ReplayBufferSamples(*merged)


def build_sac(venv, cfg: dict, seed: int) -> SAC:
    buffer_kwargs = {}
    if cfg.get("demo_episodes", 0) > 0:
        buffer_kwargs = dict(replay_buffer_class=MixedReplayBuffer,
                             replay_buffer_kwargs=dict(demo_fraction=cfg["demo_fraction"]))
    return SAC(
        "MlpPolicy",
        venv,
        learning_rate=cfg["learning_rate"],
        buffer_size=cfg["buffer_size"],
        learning_starts=cfg["learning_starts"],
        batch_size=cfg["batch_size"],
        tau=cfg["tau"],
        gamma=cfg["gamma"],
        train_freq=cfg["train_freq"],
        gradient_steps=cfg["gradient_steps"],
        ent_coef=cfg["ent_coef"],
        policy_kwargs=dict(net_arch=list(cfg["net_arch"])),
        seed=seed,
        verbose=0,
        device="cpu",
        **buffer_kwargs,
    )


def add_demonstrations(model: SAC, venv, sim_cfg: dict, n_episodes: int, noise: float, seed: int) -> float:
    """Roll out the scripted controller and store its transitions as demonstrations.

    Observations are stored unnormalised (SB3 normalises at sampling time) and
    the running observation statistics are warmed up with the same data.
    Returns the success rate of the demonstrations.
    """
    buffer = model.replay_buffer
    demos = ReplayBuffer(n_episodes * 130, venv.observation_space, venv.action_space, device="cpu")
    buffer.demos = demos
    env = PickPlaceEnv(sim_cfg, seed=seed + 10_000)
    ctrl = ScriptedPickPlace.from_env(env)
    rng = np.random.default_rng(seed)
    successes = 0
    all_obs = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ctrl.reset()
        done = False
        while not done:
            action = np.clip(ctrl.act(obs) + noise * rng.normal(size=5), -1, 1).astype(np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            demos.add(
                obs[None], next_obs[None], action[None], np.array([reward]), np.array([done]),
                [{"TimeLimit.truncated": truncated and not terminated}],
            )
            all_obs.append(obs)
            obs = next_obs
        successes += int(info["is_success"])
    if isinstance(venv, VecNormalize):
        venv.obs_rms.update(np.asarray(all_obs))
    return successes / max(n_episodes, 1)


class SACController:
    """Deterministic SAC policy with the observation normalisation baked in.

    The variance used for normalisation has a floor (1 mm, 1 mrad...): some
    features never change during training (the platform height, the tilt of an
    upright object), so their variance is ~0 and a 1 mm perception error was
    turned into a 10-sigma input. On the training data the floor changes
    nothing, those features were constant.
    """

    def __init__(self, model: SAC, obs_mean=None, obs_var=None, clip: float = 10.0, std_floor: float = 1e-3):
        self.model = model
        self.obs_mean = obs_mean
        self.obs_std = None if obs_var is None else np.sqrt(np.maximum(obs_var, std_floor**2))
        self.clip = clip

    @classmethod
    def load(cls, run_dir: str | Path, name: str = "model") -> "SACController":
        run_dir = Path(run_dir)
        model = SAC.load(run_dir / f"{name}.zip", device="cpu")
        stats = np.load(run_dir / f"{name}_obs_stats.npz")
        return cls(model, stats["mean"], stats["var"])

    @classmethod
    def from_training(cls, model: SAC) -> "SACController":
        venv = model.get_vec_normalize_env()
        if venv is None:
            return cls(model)
        return cls(model, venv.obs_rms.mean.copy(), venv.obs_rms.var.copy(), venv.clip_obs)

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_mean is not None:
            obs = np.clip((obs - self.obs_mean) / self.obs_std, -self.clip, self.clip)
        action, _ = self.model.predict(obs.astype(np.float32), deterministic=True)
        return action


def save_policy(model: SAC, path_stem: Path) -> None:
    model.save(path_stem.with_suffix(".zip"))
    venv = model.get_vec_normalize_env()
    if venv is not None:
        np.savez(f"{path_stem}_obs_stats.npz", mean=venv.obs_rms.mean, var=venv.obs_rms.var)


def _open_csv(path: Path, header: list[str], resume: bool, timestep: int = 0):
    """Open a log file, appending to it when a run is resumed.

    Rows logged after the snapshot we resume from are dropped, they will be logged again.
    Returns (file, writer, wall time offset, previous rows).
    """
    offset = 0.0
    if resume and path.exists() and path.stat().st_size > 0:
        import pandas as pd

        previous = pd.read_csv(path)
        previous = previous[previous.timestep <= timestep]
        previous.to_csv(path, index=False)
        offset = float(previous.wall_time.max()) if len(previous) else 0.0
        f = open(path, "a", newline="")
        return f, csv.writer(f), offset, previous
    f = open(path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(header)
    return f, writer, offset, None


class TrainingLogger(BaseCallback):
    """Writes one CSV row per finished training episode."""

    def __init__(self, path: Path, resume: bool = False):
        super().__init__()
        self.path = path
        self.resume = resume

    def _on_training_start(self) -> None:
        header = ["timestep", "wall_time", "return", "length", "success", "collision_steps", "place_error"]
        self.file, self.writer, offset, _ = _open_csv(self.path, header, self.resume, self.num_timesteps)
        self.t0 = time.time() - offset

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            ep = info.get("episode")
            if ep is not None:
                self.writer.writerow([
                    self.num_timesteps, round(time.time() - self.t0, 1), round(ep["r"], 3), ep["l"],
                    int(info["is_success"]), info["collision_steps"], round(info["place_error"], 4),
                ])
        return True

    def _on_training_end(self) -> None:
        self.file.close()


class ValidationCallback(BaseCallback):
    """Evaluates the deterministic policy on a fixed set of scenarios."""

    def __init__(self, env: PickPlaceEnv, scenarios: list, freq: int, run_dir: Path, checkpoints: list[int],
                 resume: bool = False):
        super().__init__()
        self.env = env
        self.scenarios = scenarios
        self.freq = freq
        self.run_dir = run_dir
        self.checkpoints = set(checkpoints)
        self.resume = resume
        self.best = -np.inf

    def _on_training_start(self) -> None:
        header = ["timestep", "wall_time", "success_rate", "mean_return", "collision_steps", "place_error"]
        self.file, self.writer, offset, previous = _open_csv(self.run_dir / "validation.csv", header, self.resume,
                                                             self.num_timesteps)
        self.t0 = time.time() - offset
        if previous is not None and len(previous):
            self.best = float((previous.success_rate + 1e-3 * previous.mean_return).max())

    def _on_step(self) -> bool:
        if self.num_timesteps in self.checkpoints:
            save_policy(self.model, self.run_dir / f"checkpoint_{self.num_timesteps}")
        if self.num_timesteps % self.freq != 0:
            return True
        ctrl = SACController.from_training(self.model)
        res = [run_episode(self.env, ctrl, sc) for sc in self.scenarios]
        success = float(np.mean([r["success"] for r in res]))
        row = [
            self.num_timesteps, round(time.time() - self.t0, 1), round(success, 3),
            round(float(np.mean([r["return"] for r in res])), 2),
            round(float(np.mean([r["collision_steps"] for r in res])), 2),
            round(float(np.median([r["place_error"] for r in res])), 4),
        ]
        self.writer.writerow(row)
        self.file.flush()
        score = success + 1e-3 * row[3]
        if score > self.best:
            self.best = score
            save_policy(self.model, self.run_dir / "best")
        print(f"[{self.num_timesteps}] success={success:.2f} return={row[3]:.1f} ({row[1]:.0f}s)", flush=True)
        return True

    def _on_training_end(self) -> None:
        self.file.close()


class ResumeCheckpoint(BaseCallback):
    """Full training state (weights, optimisers, replay buffer, normalisation) every `freq` steps.

    Background jobs are limited in time here, so a run is split in pieces:
    scripts/train_rl.py --resume picks up from the last of these snapshots.
    """

    def __init__(self, run_dir: Path, freq: int):
        super().__init__()
        self.dir = run_dir / "resume"
        self.freq = freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.freq == 0:
            save_training_state(self.model, self.dir)
        return True


def save_training_state(model: SAC, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / "model.zip")
    model.save_replay_buffer(directory / "replay_buffer.pkl")
    model.get_vec_normalize_env().save(str(directory / "vecnormalize.pkl"))
    (directory / "timesteps.txt").write_text(str(model.num_timesteps))


def load_training_state(directory: Path, sim_cfg: dict, seed: int) -> SAC:
    venv = make_vec_env(sim_cfg, seed, normalize=False)
    venv = VecNormalize.load(str(directory / "vecnormalize.pkl"), venv)
    model = SAC.load(directory / "model.zip", env=venv, device="cpu")
    model.load_replay_buffer(directory / "replay_buffer.pkl")
    return model
