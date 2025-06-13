"""Transition dataset (s, a, s') recorded from the simulator, split by episode."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from rwm.environments.state import STATE_DIM, TASK_DIM


@dataclass
class TransitionData:
    states: np.ndarray        # (N, STATE_DIM)
    actions: np.ndarray       # (N, 5)
    next_states: np.ndarray   # (N, STATE_DIM)
    tasks: np.ndarray         # (N, TASK_DIM)
    rewards: np.ndarray       # (N,)
    contacts: np.ndarray      # (N, 3) finger_left, finger_right, collision (after the step)
    episodes: np.ndarray      # (N,) episode id
    steps: np.ndarray         # (N,) step index inside the episode
    sources: np.ndarray       # (N,) index into source_names
    source_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.states)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        arrays = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "source_names"}
        np.savez_compressed(path, source_names=np.array(self.source_names), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "TransitionData":
        with np.load(path) as z:
            kwargs = {k: z[k] for k in z.files if k != "source_names"}
            names = [str(s) for s in z["source_names"]]
        return cls(source_names=names, **kwargs)

    @classmethod
    def concatenate(cls, parts: list["TransitionData"]) -> "TransitionData":
        names: list[str] = []
        for p in parts:
            names += [n for n in p.source_names if n not in names]
        out, episode_offset = [], 0
        for p in parts:
            remap = np.array([names.index(n) for n in p.source_names])
            d = {f.name: getattr(p, f.name) for f in fields(p) if f.name != "source_names"}
            d["sources"] = remap[p.sources]
            d["episodes"] = p.episodes + episode_offset
            episode_offset = d["episodes"].max() + 1
            out.append(d)
        merged = {k: np.concatenate([d[k] for d in out]) for k in out[0]}
        return cls(source_names=names, **merged)

    def subset(self, mask: np.ndarray) -> "TransitionData":
        d = {f.name: getattr(self, f.name)[mask] for f in fields(self) if f.name != "source_names"}
        return TransitionData(source_names=list(self.source_names), **d)

    def split_by_episode(self, fractions=(0.8, 0.1, 0.1), seed: int = 0):
        """Train / validation / test split with whole episodes on each side."""
        ids = np.unique(self.episodes)
        rng = np.random.default_rng(seed)
        rng.shuffle(ids)
        n_train = int(fractions[0] * len(ids))
        n_val = int(fractions[1] * len(ids))
        groups = [ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]]
        return [self.subset(np.isin(self.episodes, g)) for g in groups]

    def episode_slices(self) -> list[np.ndarray]:
        """Indices of each episode, ordered by step."""
        out = []
        for e in np.unique(self.episodes):
            idx = np.flatnonzero(self.episodes == e)
            out.append(idx[np.argsort(self.steps[idx])])
        return out


class NoisyController:
    """Exploration wrapper: Gaussian action noise and occasional bursts of random actions."""

    def __init__(self, controller, sigma: float = 0.0, burst_prob: float = 0.0, burst_len=(5, 15), seed: int = 0):
        self.controller = controller
        self.sigma = sigma
        self.burst_prob = burst_prob
        self.burst_len = burst_len
        self.rng = np.random.default_rng(seed)
        self.burst = 0
        self.random_action = np.zeros(5)

    def reset(self) -> None:
        self.controller.reset()
        self.burst = 0

    def act(self, obs):
        a = np.asarray(self.controller.act(obs), dtype=float)
        if self.burst == 0 and self.rng.random() < self.burst_prob:
            self.burst = int(self.rng.integers(*self.burst_len))
            self.random_action = self.rng.uniform(-1, 1, size=5)
        if self.burst > 0:
            self.burst -= 1
            # slowly varying random action, more realistic than white noise
            self.random_action = np.clip(self.random_action + 0.3 * self.rng.normal(size=5), -1, 1)
            return self.random_action.copy()
        return np.clip(a + self.sigma * self.rng.normal(size=a.shape), -1.0, 1.0)


class RandomController:
    """Smoothly varying random actions (free-motion and pushing data)."""

    def __init__(self, seed: int = 0, smoothing: float = 0.8):
        self.rng = np.random.default_rng(seed)
        self.smoothing = smoothing

    def reset(self) -> None:
        self.a = self.rng.uniform(-1, 1, size=5)

    def act(self, obs):
        self.a = np.clip(self.smoothing * self.a + (1 - self.smoothing) * self.rng.uniform(-1.5, 1.5, size=5), -1, 1)
        return self.a.copy()


def record_episodes(env, controller, n_episodes: int, source: str, scenarios=None,
                    steps_after_success: int | None = None, log_every: int = 0) -> TransitionData:
    """Run `controller` and record every transition with the ground-truth state.

    steps_after_success: end the episode this many steps after the first
    success (the release and retreat are still recorded, the idle hovering
    that follows is not). Useful for slow controllers such as MPC.
    """
    buf = {k: [] for k in ("states", "actions", "next_states", "tasks", "rewards", "contacts", "episodes", "steps")}
    t_start, n_success = time.time(), 0
    for ep in range(n_episodes):
        options = {"scenario": scenarios[ep]} if scenarios is not None else None
        obs, info = env.reset(options=options)
        controller.reset()
        s = env.state()
        done, t = False, 0
        while not done:
            a = np.clip(np.asarray(controller.act(obs), dtype=np.float32), -1, 1)
            obs, r, terminated, truncated, info = env.step(a)
            s_next = env.state()
            buf["states"].append(s)
            buf["actions"].append(a)
            buf["next_states"].append(s_next)
            buf["tasks"].append(env.task.copy())
            buf["rewards"].append(r)
            buf["contacts"].append([*info["finger_contacts"], info["collisions"] > 0])
            buf["episodes"].append(ep)
            buf["steps"].append(t)
            s, t = s_next, t + 1
            done = terminated or truncated
            first = info["first_success_step"]
            if steps_after_success is not None and first is not None and t >= first + steps_after_success:
                done = True
        n_success += int(info["first_success_step"] is not None)
        if log_every and (ep + 1) % log_every == 0:
            elapsed = time.time() - t_start
            print(f"  [{source}] {ep + 1}/{n_episodes} episodes, success {n_success / (ep + 1):.2f}, "
                  f"{elapsed / 60:.1f} min elapsed, ~{elapsed / (ep + 1) * (n_episodes - ep - 1) / 60:.1f} min left",
                  flush=True)
    arr = {k: np.asarray(v) for k, v in buf.items()}
    arr["states"] = arr["states"].astype(np.float32).reshape(-1, STATE_DIM)
    arr["next_states"] = arr["next_states"].astype(np.float32).reshape(-1, STATE_DIM)
    arr["tasks"] = arr["tasks"].astype(np.float32).reshape(-1, TASK_DIM)
    arr["contacts"] = arr["contacts"].astype(bool)
    return TransitionData(sources=np.zeros(len(arr["states"]), dtype=np.int64), source_names=[source], **arr)
