"""Pick-and-place reward, written as a function of (state, task) only.

It does not look at contacts or any other simulator internals, so the exact
same function can be evaluated by the MPC planner on states predicted by the
world model. Works on numpy arrays and torch tensors, batched or not.

Shaping, from worst to best situation:
    reach the object < hold it above the table < carry it over the target
    < put it down on the target < open the gripper < move the gripper away
Before the object is on the platform the reward pulls the gripper towards
the object and the object towards the target; once it is on the platform,
staying around it is no longer rewarded (the fingers would end up scraping
the platform), releasing and retreating are.
"""

from __future__ import annotations

import numpy as np

from rwm.environments.state import SLICES, object_up_axis_z
from rwm.simulation.scene import TABLE_Z

PLACE_XY_TOL = 0.04   # object centre within 4 cm of the platform centre
PLACE_Z_TOL = 0.01
NEAR_DIST = 0.05


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


def _backend(x):
    if _is_torch(x):
        import torch

        return torch
    return np


def _norm(x):
    return x.norm(dim=-1) if _is_torch(x) else np.linalg.norm(x, axis=-1)


def _float(cond, like):
    return cond.to(like.dtype) if _is_torch(like) else np.asarray(cond, dtype=np.float32)


def reward_terms(state, task):
    xp = _backend(state)
    tcp = state[..., SLICES["tcp_pos"]]
    obj = state[..., SLICES["obj_pos"]]
    grip = state[..., SLICES["grip"]][..., 0]
    target = task[..., 0:3]
    half_h = task[..., 3]
    half_w = task[..., 4]

    d_reach = _norm(tcp - obj)
    d_xy = _norm(obj[..., :2] - target[..., :2])
    lift = obj[..., 2] - (TABLE_Z + half_h)
    place_z = target[..., 2] + half_h

    lifted = xp.clip((lift - 0.005) / 0.015, 0.0, 1.0)
    near = _float(d_reach < NEAR_DIST, d_reach)
    held = lifted * near
    on_target = _float((d_xy < PLACE_XY_TOL) & (xp.abs(obj[..., 2] - place_z) < PLACE_Z_TOL), d_xy)
    released = _float(grip > 2 * half_w + 0.01, grip)
    tilt = 1.0 - object_up_axis_z(state[..., SLICES["obj_rot"]])
    retreat = xp.clip((tcp[..., 2] - obj[..., 2] - 0.03) / 0.05, 0.0, 1.0)

    return dict(
        reach=1.0 - xp.tanh(5.0 * d_reach),
        held=held,
        move=1.0 - xp.tanh(5.0 * d_xy),
        on_target=on_target,
        released=released,
        retreat=retreat,
        tilt=xp.clip(tilt, 0.0, 1.0),
        d_xy=d_xy,
    )


def pick_place_reward(state, task, action=None, action_cost: float = 0.005):
    t = reward_terms(state, task)
    before = 0.2 * t["reach"] + 0.3 * t["held"] + 0.5 * t["held"] * t["move"]
    after = 0.6 + 0.5 * t["move"] + t["released"] * (0.5 + 0.3 * t["retreat"])
    r = (1.0 - t["on_target"]) * before + t["on_target"] * after - 0.2 * t["tilt"]
    if action is not None:
        r = r - action_cost * (action[..., :4] ** 2).sum(-1)
    return r


def placed(state, task):
    """Geometric part of the success test (object resting on the platform)."""
    return reward_terms(state, task)["on_target"] > 0.5
