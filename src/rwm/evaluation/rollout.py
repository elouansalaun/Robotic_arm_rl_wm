"""Run a controller in the environment and collect per-episode metrics.

A controller is any object with `reset()` and `act(obs) -> action`.
"""

from __future__ import annotations

import time

import numpy as np

from rwm.simulation.scene import Scenario


def run_episode(env, controller, scenario: Scenario | dict | None = None, seed: int | None = None,
                record=None, frame_every: int = 1) -> dict:
    options = {"scenario": scenario} if scenario is not None else None
    obs, info = env.reset(seed=seed, options=options)
    controller.reset()
    total_reward = 0.0
    act_times = []
    frames = []
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        t0 = time.perf_counter()
        action = controller.act(obs)
        act_times.append(time.perf_counter() - t0)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if record is not None and steps % frame_every == 0:
            frames.append(record())

    out = {
        "success": bool(info["first_success_step"] is not None and info["is_success"]),
        "ever_succeeded": info["first_success_step"] is not None,
        "time_to_success": info["first_success_step"],
        "steps": steps,
        "return": total_reward,
        "place_error": info["place_error"],
        "collision_steps": info["collision_steps"],
        "dropped": bool(terminated),
        "max_lift": info["max_lift"],
        "object_kind": info["object_kind"],
        "act_time_ms": 1e3 * float(np.mean(act_times)),
    }
    if record is not None:
        out["frames"] = frames
    return out


def run_episodes(env, controller, scenarios, progress: bool = False) -> list[dict]:
    it = scenarios
    if progress:
        from tqdm import tqdm

        it = tqdm(scenarios, leave=False)
    return [run_episode(env, controller, sc) for sc in it]
