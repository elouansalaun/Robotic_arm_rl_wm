"""Training loop for the dynamics ensemble (bootstrapped members, early stopping)."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from rwm.world_model.dataset import TransitionData
from rwm.world_model.model import DynamicsEnsemble


def _tensors(data: TransitionData):
    x = np.concatenate([data.states, data.actions, data.tasks[:, 3:5]], axis=1)
    return (
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(data.next_states - data.states, dtype=torch.float32),
        torch.as_tensor(data.contacts[:, 2], dtype=torch.float32),
    )


def evaluate_loss(model: DynamicsEnsemble, data: TransitionData, batch: int = 8192) -> np.ndarray:
    """Mean squared error on the normalised deltas, one value per member."""
    x, d, _ = _tensors(data)
    target = model.out_norm(d)
    total = torch.zeros(model.n_members)
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = x[i:i + batch].unsqueeze(0).expand(model.n_members, -1, -1)
            pred, _ = model(xb)
            total += ((pred - target[i:i + batch]) ** 2).mean(-1).sum(-1)
    return (total / len(x)).numpy()


def train_ensemble(train: TransitionData, val: TransitionData, cfg: dict, log=print):
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    model = DynamicsEnsemble(cfg["n_members"], cfg["hidden"], cfg["n_layers"])
    model.fit_normalizers(train.states, train.actions, train.tasks, train.next_states)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    x, d, coll = _tensors(train)
    target = model.out_norm(d)
    n, K, B = len(x), model.n_members, cfg["batch_size"]
    # each member sees its own bootstrap resample of the training set
    boot = rng.integers(0, n, size=(K, n)) if cfg["bootstrap"] else np.tile(np.arange(n), (K, 1))
    pos_weight = torch.tensor(cfg.get("collision_pos_weight", 5.0))

    history, best, best_state, bad_epochs = [], np.inf, None, 0
    t0 = time.time()
    for epoch in range(cfg["max_epochs"]):
        model.train()
        perm = np.stack([rng.permutation(row) for row in boot])
        losses = []
        for i in range(0, n - B + 1, B):
            idx = torch.as_tensor(perm[:, i:i + B])
            pred, logit = model(x[idx])
            loss_dyn = F.mse_loss(pred, target[idx])
            loss_col = F.binary_cross_entropy_with_logits(logit, coll[idx], pos_weight=pos_weight)
            loss = loss_dyn + cfg["collision_weight"] * loss_col
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss_dyn.item())
        model.eval()
        val_loss = evaluate_loss(model, val)
        history.append(dict(epoch=epoch, train_mse=float(np.mean(losses)), val_mse=float(val_loss.mean()),
                            val_mse_members=val_loss.tolist(), time=time.time() - t0))
        if val_loss.mean() < best - 1e-5:
            best, best_state, bad_epochs = val_loss.mean(), copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
        if epoch % 5 == 0 or bad_epochs == 0:
            log(f"epoch {epoch:3d}  train {np.mean(losses):.4f}  val {val_loss.mean():.4f}  ({time.time() - t0:.0f}s)")
        if bad_epochs >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history
