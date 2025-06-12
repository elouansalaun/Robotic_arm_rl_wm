"""Ensemble of MLP dynamics models: (s_t, a_t) -> s_{t+1}.

Each member predicts the normalised state change Delta s = s_{t+1} - s_t and
the logit of an unwanted contact during the step. The members share nothing
but the normalisation statistics; they are evaluated in one batched matmul
(weights of shape [K, in, out]) so planning with the whole ensemble costs
about as much as with a single network.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from rwm.environments.state import SLICES, STATE_DIM, matrix_from_rot6

ACTION_DIM = 5
SIZE_DIM = 2  # object half height / half width, taken from the task vector


class EnsembleLinear(nn.Module):
    def __init__(self, n_members: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_members, d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(n_members, 1, d_out))
        bound = 1.0 / math.sqrt(d_in)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [K, B, d_in]
        return torch.baddbmm(self.bias, x, self.weight)


class Normalizer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def fit(self, x: np.ndarray) -> None:
        self.mean.copy_(torch.as_tensor(x.mean(0), dtype=torch.float32))
        self.std.copy_(torch.as_tensor(x.std(0) + 1e-6, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


class DynamicsEnsemble(nn.Module):
    def __init__(self, n_members: int = 5, hidden: int = 256, n_layers: int = 3):
        super().__init__()
        self.n_members = n_members
        d_in = STATE_DIM + ACTION_DIM + SIZE_DIM
        dims = [d_in] + [hidden] * n_layers
        self.layers = nn.ModuleList(EnsembleLinear(n_members, a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.head = EnsembleLinear(n_members, hidden, STATE_DIM + 1)
        self.act = nn.SiLU()
        self.in_norm = Normalizer(d_in)
        self.out_norm = Normalizer(STATE_DIM)
        self.config = dict(n_members=n_members, hidden=hidden, n_layers=n_layers)

    @staticmethod
    def inputs(state, action, task):
        return torch.cat([state, action, task[..., 3:5]], dim=-1)

    def fit_normalizers(self, states, actions, tasks, next_states) -> None:
        x = np.concatenate([states, actions, tasks[:, 3:5]], axis=1)
        self.in_norm.fit(x)
        self.out_norm.fit(next_states - states)

    def forward(self, x: torch.Tensor):
        """x: [K, B, d_in] raw inputs -> (normalised delta [K, B, S], collision logit [K, B])."""
        h = self.in_norm(x)
        for layer in self.layers:
            h = self.act(layer(h))
        out = self.head(h)
        return out[..., :STATE_DIM], out[..., STATE_DIM]

    def step(self, state: torch.Tensor, action: torch.Tensor, task: torch.Tensor):
        """One step for every member.

        state, action, task: [B, .] (same input for all members) or [K, B, .]
        Returns next state [K, B, S] and collision probability [K, B].
        """
        x = self.inputs(state, action, task)
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(self.n_members, -1, -1)
            state = state.unsqueeze(0).expand(self.n_members, -1, -1)
        delta_n, logit = self.forward(x)
        nxt = state + self.out_norm.inverse(delta_n)
        return project_state(nxt), torch.sigmoid(logit)

    @torch.no_grad()
    def rollout(self, state: torch.Tensor, actions: torch.Tensor, task: torch.Tensor):
        """Open-loop prediction, each member follows its own trajectory.

        state [B, S], actions [B, H, A], task [B, T] -> states [K, B, H, S], collision prob [K, B, H]
        """
        K = self.n_members
        s = state.unsqueeze(0).expand(K, -1, -1)
        a_all = actions.unsqueeze(0).expand(K, -1, -1, -1)
        t = task.unsqueeze(0).expand(K, -1, -1)
        states, probs = [], []
        for h in range(actions.shape[1]):
            s, p = self.step(s, a_all[:, :, h], t)
            states.append(s)
            probs.append(p)
        return torch.stack(states, dim=2), torch.stack(probs, dim=2)

    def save(self, path) -> None:
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path) -> "DynamicsEnsemble":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model


def project_state(s: torch.Tensor) -> torch.Tensor:
    """Bring a predicted state back onto the valid set (orthonormal rotation, bounded gripper)."""
    s = s.clone()
    R = matrix_from_rot6(s[..., SLICES["obj_rot"]])
    s[..., SLICES["obj_rot"]] = torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)
    s[..., SLICES["grip"]] = s[..., SLICES["grip"]].clamp(0.012, 0.102)
    s[..., SLICES["tcp_yaw"]] = s[..., SLICES["tcp_yaw"]].clamp(-math.pi / 2, math.pi / 2)
    return s
