"""Pick-and-place reward, written as a function of (state, task) only.

It does not look at contacts or any other simulator internals, so the exact
same function can be evaluated by the MPC planner on states predicted by the
world model. Works on numpy arrays and torch tensors, batched or not.

Shaping, from worst to best situation:
    reach the object < close the fingers on it < hold it above the table
    < carry it over the target < lower it towards the target < put it down
    on the target < open the gripper < move the gripper away
The "lower it" term matters for the short-horizon planner: without it,
holding the object high above the platform and putting it down look the
same until the last centimetre, and MPC ends up hovering.
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
REST_SPEED = 0.05     # m/s, same threshold as the success test
NEAR_DIST = 0.05
GRASP_DIST = 0.02     # TCP within 2 cm of the object centre


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


def _backend(x):
    if _is_torch(x):
        import torch

        return torch
    return np


def _norm(x):
    return x.norm(dim=-1) if _is_torch(x) else np.linalg.norm(x, axis=-1)


def _sigmoid(x):
    return 1.0 / (1.0 + (-x).exp()) if _is_torch(x) else 1.0 / (1.0 + np.exp(-x))


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
    # "On target" means resting on the platform: an object thrown across it at the right
    # height does not count (MPC learned to let go in flight and the cylinders rolled away).
    # The thresholds are smooth: on predicted states, a hard 1 cm / 5 cm/s test was almost
    # never met and the planner hovered just above the platform.
    speed = _norm(state[..., SLICES["obj_linvel"]])
    on_target = (
        _sigmoid((PLACE_XY_TOL - d_xy) / 0.004)
        * _sigmoid((PLACE_Z_TOL - xp.abs(obj[..., 2] - place_z)) / 0.002)
        * _sigmoid((REST_SPEED - speed) / 0.01)
    )
    released = _float(grip > 2 * half_w + 0.01, grip)
    # fingers closed down to the object width around its centre: a grasp, seen from the state only
    closed_on_object = (grip < 2 * half_w + 0.006) & (grip > 2 * half_w - 0.012)
    grasped = _float(closed_on_object & (d_reach < GRASP_DIST), grip)
    tilt = 1.0 - object_up_axis_z(state[..., SLICES["obj_rot"]])
    retreat = xp.clip((tcp[..., 2] - obj[..., 2] - 0.03) / 0.05, 0.0, 1.0)
    # how close the object is to its resting height on the platform, from above
    descend = 1.0 - xp.tanh(10.0 * xp.clip(obj[..., 2] - place_z, 0.0, None))

    return dict(
        reach=1.0 - xp.tanh(5.0 * d_reach),
        grasped=grasped,
        held=held,
        move=1.0 - xp.tanh(5.0 * d_xy),
        on_target=on_target,
        released=released,
        retreat=retreat,
        descend=descend,
        tilt=xp.clip(tilt, 0.0, 1.0),
        d_xy=d_xy,
    )


def pick_place_reward(state, task, action=None, action_cost: float = 0.005):
    t = reward_terms(state, task)
    carry = t["held"] * t["move"]
    before = 0.2 * t["reach"] + 0.1 * t["grasped"] + 0.3 * t["held"] + carry * (0.5 + 0.2 * t["descend"])
    after = 0.9 + 0.5 * t["move"] + t["released"] * (0.5 + 0.3 * t["retreat"])
    r = (1.0 - t["on_target"]) * before + t["on_target"] * after - 0.2 * t["tilt"]
    if action is not None:
        r = r - action_cost * (action[..., :4] ** 2).sum(-1)
    return r


def placed(state, task):
    """Geometric part of the success test (object resting on the platform)."""
    return reward_terms(state, task)["on_target"] > 0.5
