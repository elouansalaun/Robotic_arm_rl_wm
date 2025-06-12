"""Open-loop prediction error of the dynamics model over several steps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from rwm.environments.state import SLICES, matrix_from_rot6
from rwm.world_model.dataset import TransitionData
from rwm.world_model.model import DynamicsEnsemble


def windows(data: TransitionData, horizon: int, stride: int = 1):
    """All sub-trajectories of length `horizon` that stay inside one episode."""
    s0, acts, tasks, targets, contact0 = [], [], [], [], []
    for idx in data.episode_slices():
        for t in range(0, len(idx) - horizon + 1, stride):
            w = idx[t:t + horizon]
            s0.append(data.states[w[0]])
            acts.append(data.actions[w])
            tasks.append(data.tasks[w[0]])
            targets.append(data.next_states[w])
            # contact-rich window: a finger touches the object at some point
            contact0.append(bool(data.contacts[w, :2].any()))
    return (np.asarray(s0), np.asarray(acts), np.asarray(tasks), np.asarray(targets), np.asarray(contact0))


def rotation_angle_deg(r6_a: np.ndarray, r6_b: np.ndarray) -> np.ndarray:
    Ra, Rb = matrix_from_rot6(r6_a), matrix_from_rot6(r6_b)
    cos = (np.einsum("...ij,...ij->...", Ra, Rb) - 1.0) / 2.0
    return np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0)))


def feature_errors(pred: np.ndarray, true: np.ndarray) -> dict[str, np.ndarray]:
    """Physically meaningful error per feature group (mm, deg, ...)."""
    g = lambda x, k: x[..., SLICES[k]]
    return {
        "tcp_pos_mm": 1e3 * np.linalg.norm(g(pred, "tcp_pos") - g(true, "tcp_pos"), axis=-1),
        "grip_mm": 1e3 * np.abs(g(pred, "grip") - g(true, "grip"))[..., 0],
        "obj_pos_mm": 1e3 * np.linalg.norm(g(pred, "obj_pos") - g(true, "obj_pos"), axis=-1),
        "obj_rot_deg": rotation_angle_deg(g(pred, "obj_rot"), g(true, "obj_rot")),
        "arm_q_deg": np.rad2deg(np.abs(g(pred, "arm_q") - g(true, "arm_q")).mean(-1)),
        "obj_linvel_mm_s": 1e3 * np.linalg.norm(g(pred, "obj_linvel") - g(true, "obj_linvel"), axis=-1),
    }


@torch.no_grad()
def multistep_errors(model: DynamicsEnsemble, data: TransitionData, horizons=(1, 5, 10), stride: int = 2,
                     batch: int = 4096) -> pd.DataFrame:
    H = max(horizons)
    s0, acts, tasks, targets, contact0 = windows(data, H, stride)
    preds, spreads = [], []
    for i in range(0, len(s0), batch):
        traj, _ = model.rollout(torch.as_tensor(s0[i:i + batch]), torch.as_tensor(acts[i:i + batch]),
                                torch.as_tensor(tasks[i:i + batch]))
        traj = traj.numpy()  # [K, B, H, S]
        preds.append(traj.mean(0))
        spreads.append(traj[..., SLICES["obj_pos"]].std(0).mean(-1))  # ensemble disagreement on the object
    pred = np.concatenate(preds)
    spread = np.concatenate(spreads)

    rows = []
    for h in horizons:
        err_model = feature_errors(pred[:, h - 1], targets[:, h - 1])
        err_const = feature_errors(s0, targets[:, h - 1])  # "nothing moves" baseline
        for name in err_model:
            for contact in (False, True):
                m = contact0 == contact
                rows.append(dict(horizon=h, feature=name, contact=contact, n=int(m.sum()),
                                 model_median=float(np.median(err_model[name][m])),
                                 model_mean=float(np.mean(err_model[name][m])),
                                 model_p90=float(np.percentile(err_model[name][m], 90)),
                                 constant_median=float(np.median(err_const[name][m]))))
        rows.append(dict(horizon=h, feature="ensemble_spread_obj_mm", contact=None, n=len(spread),
                         model_median=float(1e3 * np.median(spread[:, h - 1])),
                         model_mean=float(1e3 * np.mean(spread[:, h - 1]))))
    return pd.DataFrame(rows)


@torch.no_grad()
def uncertainty_vs_error(model: DynamicsEnsemble, data: TransitionData, horizon: int = 5, stride: int = 2):
    """Per window: ensemble spread and actual error on the object position at `horizon`."""
    s0, acts, tasks, targets, contact0 = windows(data, horizon, stride)
    traj, _ = model.rollout(torch.as_tensor(s0), torch.as_tensor(acts), torch.as_tensor(tasks))
    traj = traj.numpy()
    obj = traj[..., horizon - 1, SLICES["obj_pos"]]
    spread = obj.std(0).mean(-1)
    err = np.linalg.norm(obj.mean(0) - targets[:, horizon - 1, SLICES["obj_pos"]], axis=-1)
    return pd.DataFrame(dict(spread_mm=1e3 * spread, error_mm=1e3 * err, contact=contact0))
