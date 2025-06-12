import numpy as np
import torch

from rwm.environments.pick_place_env import PickPlaceEnv
from rwm.environments.state import SLICES, STATE_DIM
from rwm.planning.cem import CEMPlanner
from rwm.robotics.controllers import ScriptedPickPlace
from rwm.world_model.dataset import NoisyController, TransitionData, record_episodes
from rwm.world_model.evaluation import multistep_errors
from rwm.world_model.model import DynamicsEnsemble, project_state
from rwm.world_model.training import train_ensemble


def small_dataset(n_episodes=6, seed=0):
    env = PickPlaceEnv(seed=seed)
    ctrl = NoisyController(ScriptedPickPlace.from_env(env), sigma=0.2, seed=seed)
    return record_episodes(env, ctrl, n_episodes, "test")


def test_dataset_split_by_episode(tmp_path):
    data = small_dataset(6)
    assert data.states.shape[1] == STATE_DIM
    # consecutive transitions chain
    idx = data.episode_slices()[0]
    np.testing.assert_allclose(data.next_states[idx[:-1]], data.states[idx[1:]])
    train, val, test = data.split_by_episode((0.5, 0.25, 0.25), seed=0)
    assert not set(train.episodes) & set(test.episodes)
    assert len(train) + len(val) + len(test) == len(data)
    data.save(tmp_path / "d.npz")
    loaded = TransitionData.load(tmp_path / "d.npz")
    np.testing.assert_array_equal(loaded.actions, data.actions)
    merged = TransitionData.concatenate([data, loaded])
    assert len(np.unique(merged.episodes)) == 2 * len(np.unique(data.episodes))


def test_ensemble_shapes_and_projection():
    model = DynamicsEnsemble(n_members=3, hidden=32, n_layers=2)
    s = torch.randn(7, STATE_DIM)
    a = torch.rand(7, 4, 5) * 2 - 1
    task = torch.rand(7, 5)
    traj, p = model.rollout(s, a, task)
    assert traj.shape == (3, 7, 4, STATE_DIM) and p.shape == (3, 7, 4)
    r6 = project_state(torch.randn(10, STATE_DIM))[:, SLICES["obj_rot"]]
    x, y = r6[:, :3], r6[:, 3:]
    torch.testing.assert_close(x.norm(dim=-1), torch.ones(10))
    torch.testing.assert_close((x * y).sum(-1), torch.zeros(10), atol=1e-6, rtol=0)


def test_training_beats_constant_baseline(tmp_path):
    data = small_dataset(10, seed=1)
    train, val, test = data.split_by_episode((0.7, 0.15, 0.15), seed=0)
    cfg = dict(n_members=2, hidden=64, n_layers=2, lr=1e-3, weight_decay=0.0, batch_size=128, max_epochs=40,
               patience=5, bootstrap=True, collision_weight=0.1, seed=0)
    model, history = train_ensemble(train, val, cfg, log=lambda *_: None)
    assert history[-1]["val_mse"] < history[0]["val_mse"]
    err = multistep_errors(model, test, horizons=(1,))
    tcp = err[(err.feature == "tcp_pos_mm") & (~err.contact.astype(bool))].iloc[0]
    assert tcp.model_median < tcp.constant_median
    model.save(tmp_path / "m.pt")
    reloaded = DynamicsEnsemble.load(tmp_path / "m.pt")
    s = torch.as_tensor(test.states[:4])
    a = torch.as_tensor(test.actions[:4])
    t = torch.as_tensor(test.tasks[:4])
    torch.testing.assert_close(model.step(s, a, t)[0], reloaded.step(s, a, t)[0])


def test_planner_returns_valid_action():
    model = DynamicsEnsemble(n_members=2, hidden=32, n_layers=2)
    planner = CEMPlanner(model, horizon=4, population=32, elites=4, iterations=2)
    env = PickPlaceEnv(seed=0)
    env.reset()
    a = planner.plan(env.state(), env.task)
    assert a.shape == (5,) and np.all(np.abs(a) <= 1.0)
    assert planner.last_info["plan_time"] > 0
