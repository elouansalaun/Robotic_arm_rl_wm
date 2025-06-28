"""Model predictive control with the cross-entropy method on the learned dynamics.

At every control step:
    1. sample N action sequences of length H around the current mean
    2. roll them out in every member of the ensemble
    3. score = discounted task reward (same function as the RL reward)
               - predicted probability of unwanted contacts
               - lambda * ensemble disagreement (optional)
    4. refit the sampling distribution on the best sequences, repeat
Only the first action is executed, then the planner starts again from the
new observation (receding horizon). The previous solution, shifted by one
step, warm-starts the next optimisation.

Two changes from iCEM (Pinneri et al., 2020) matter here: the sampling noise
is correlated in time, so that coherent motions ("go down, then hold") are
actually sampled, and a fraction of the elites is carried over to the next
iteration and, shifted, to the next control step. With white noise and no
memory, the planner kept the object hovering 2 cm above the platform even
though the model predicted that putting it down was better.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from rwm.environments.reward import pick_place_reward
from rwm.environments.state import SLICES
from rwm.robotics.controllers import split_observation
from rwm.world_model.model import ACTION_DIM, DynamicsEnsemble


class CEMPlanner:
    def __init__(
        self,
        model: DynamicsEnsemble,
        horizon: int = 10,
        population: int = 256,
        elites: int = 25,
        iterations: int = 3,
        init_std: float = 0.6,
        min_std: float = 0.05,
        momentum: float = 0.1,
        discount: float = 0.95,
        collision_weight: float = 1.0,
        uncertainty_weight: float = 0.0,
        action_cost: float = 0.005,
        noise_correlation: float = 0.0,
        keep_elites: float = 0.0,
        execute: str = "mean",
        seed: int = 0,
    ):
        self.model = model
        self.H, self.N, self.n_elites, self.iterations = horizon, population, elites, iterations
        self.init_std, self.min_std, self.momentum = init_std, min_std, momentum
        self.collision_weight = collision_weight
        self.uncertainty_weight = uncertainty_weight
        self.action_cost = action_cost
        self.rho = noise_correlation
        self.n_keep = int(round(keep_elites * elites))
        self.execute = execute
        self.discounts = discount ** torch.arange(horizon, dtype=torch.float32)
        self.gen = torch.Generator().manual_seed(seed)
        self.reset()

    def reset(self) -> None:
        self.mean = torch.zeros(self.H, ACTION_DIM)
        self.kept = torch.empty(0, self.H, ACTION_DIM)
        self.last_info: dict = {}

    def _noise(self) -> torch.Tensor:
        eps = torch.randn(self.N, self.H, ACTION_DIM, generator=self.gen)
        if self.rho <= 0:
            return eps
        # AR(1) in time with unit marginal variance
        noise = torch.empty_like(eps)
        noise[:, 0] = eps[:, 0]
        scale = (1 - self.rho**2) ** 0.5
        for t in range(1, self.H):
            noise[:, t] = self.rho * noise[:, t - 1] + scale * eps[:, t]
        return noise

    @torch.no_grad()
    def evaluate(self, state: torch.Tensor, task: torch.Tensor, actions: torch.Tensor):
        """Scores of N candidate sequences. actions: [N, H, A]."""
        N = actions.shape[0]
        traj, p_coll = self.model.rollout(state.expand(N, -1), actions, task.expand(N, -1))  # [K,N,H,S], [K,N,H]
        reward = pick_place_reward(traj, task.expand_as(traj[..., :task.shape[-1]]), actions.unsqueeze(0),
                                   action_cost=self.action_cost)
        task_return = (reward * self.discounts).sum(-1).mean(0)
        collision = (p_coll * self.discounts).sum(-1).mean(0)
        # disagreement between members on where the object and the gripper end up
        spread = traj[..., SLICES["obj_pos"]].std(0).norm(dim=-1) + traj[..., SLICES["tcp_pos"]].std(0).norm(dim=-1)
        uncertainty = (spread * self.discounts).sum(-1) / 0.01  # in cm
        score = task_return - self.collision_weight * collision - self.uncertainty_weight * uncertainty
        return score, dict(task_return=task_return, collision=collision, uncertainty=uncertainty)

    @torch.no_grad()
    def plan(self, state: np.ndarray, task: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        g = torch.as_tensor(task, dtype=torch.float32).unsqueeze(0)
        mean = self.mean.clone()
        std = torch.full_like(mean, self.init_std)
        kept = self.kept
        for _ in range(self.iterations):
            samples = (mean + std * self._noise()).clamp(-1.0, 1.0)
            samples[0] = mean  # always keep the current mean as a candidate
            if len(kept):
                samples[1:1 + len(kept)] = kept
            score, terms = self.evaluate(s, g, samples)
            elite_idx = torch.topk(score, self.n_elites).indices
            elites = samples[elite_idx]
            kept = elites[: self.n_keep]
            mean = self.momentum * mean + (1 - self.momentum) * elites.mean(0)
            std = (self.momentum * std + (1 - self.momentum) * elites.std(0)).clamp_min(self.min_std)

        best = int(elite_idx[0])
        self.last_info = dict(
            plan_time=time.perf_counter() - t0,
            score=float(score[best]),
            uncertainty=float(terms["uncertainty"][best]),
            collision=float(terms["collision"][best]),
        )
        plan = samples[best] if self.execute == "best" else mean
        # warm start: shift the solution (and the kept elites) by one step
        shift = lambda x: torch.cat([x[..., 1:, :], x[..., -1:, :]], dim=-2)
        self.mean = shift(plan)
        self.kept = shift(kept)
        return plan[0].numpy()


class MPCController:
    """Controller interface (reset / act) around the planner."""

    def __init__(self, planner: CEMPlanner):
        self.planner = planner
        self.plan_times: list[float] = []
        self.uncertainties: list[float] = []

    def reset(self) -> None:
        self.planner.reset()

    def act(self, obs: np.ndarray) -> np.ndarray:
        state, task = split_observation(np.asarray(obs, dtype=np.float32))
        action = self.planner.plan(state, task)
        self.plan_times.append(self.planner.last_info["plan_time"])
        self.uncertainties.append(self.planner.last_info["uncertainty"])
        return action
