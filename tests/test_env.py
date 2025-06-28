import warnings

import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from rwm.environments.pick_place_env import PickPlaceEnv
from rwm.environments.reward import pick_place_reward, reward_terms
from rwm.environments.state import SLICES, STATE_DIM, matrix_from_rot6, rot6_from_matrix
from rwm.evaluation.rollout import run_episode
from rwm.robotics.controllers import ScriptedPickPlace
from rwm.simulation.scene import Scenario


@pytest.fixture(scope="module")
def env():
    return PickPlaceEnv(seed=0)


def test_gymnasium_api():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(PickPlaceEnv(seed=1), skip_render_check=True)


def test_reward_numpy_and_torch_agree(env):
    env.reset(seed=3)
    states, actions = [], []
    for _ in range(40):
        a = env.action_space.sample()
        env.step(a)
        states.append(env.state())
        actions.append(a)
    S = np.array(states)
    A = np.array(actions)
    T = np.repeat(env.task[None], len(S), axis=0)
    r_np = pick_place_reward(S, T, A)
    r_t = pick_place_reward(torch.tensor(S), torch.tensor(T), torch.tensor(A)).numpy()
    np.testing.assert_allclose(r_np, r_t, atol=1e-5)


def test_reward_ordering(env):
    """Placed and released > placed and held > carried > resting on the table."""
    env.reset(options={"scenario": Scenario(object_xy=[0.5, -0.1], target_xy=[0.5, 0.15])})
    s = env.state()
    task = env.task
    h = task[3]

    def at(obj, tcp, grip):
        x = s.copy()
        x[SLICES["obj_pos"]] = obj
        x[SLICES["tcp_pos"]] = tcp
        x[SLICES["grip"]] = grip
        return float(pick_place_reward(x, task))

    place = np.array([task[0], task[1], task[2] + h])
    table = np.array([0.5, -0.1, 0.42 + h])
    r_rest = at(table, table + [0, 0, 0.2], 0.1)
    r_carry = at(place + [0, 0, 0.06], place + [0, 0, 0.06], 2 * task[4])
    r_held = at(place, place, 2 * task[4])
    r_released = at(place, place, 0.1)
    assert r_rest < r_carry < r_held < r_released


def test_rot6_round_trip():
    rng = np.random.default_rng(0)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    np.testing.assert_allclose(matrix_from_rot6(rot6_from_matrix(R)), R, atol=1e-12)


def test_state_layout(env):
    obs, _ = env.reset(seed=0)
    s = env.state()
    assert s.shape == (STATE_DIM,)
    p, _ = env.scene.tcp_pose()
    np.testing.assert_allclose(s[SLICES["tcp_pos"]], p, atol=1e-6)
    obj, _ = env.scene.object_pose()
    np.testing.assert_allclose(s[SLICES["obj_pos"]], obj, atol=1e-6)
    np.testing.assert_allclose(obs[:STATE_DIM], s)


def test_action_moves_tcp(env):
    env.reset(seed=0)
    p0 = env.state()[SLICES["tcp_pos"]].copy()
    for _ in range(5):
        env.step(np.array([1.0, 0, 0, 0, 1.0]))
    p1 = env.state()[SLICES["tcp_pos"]]
    assert p1[0] - p0[0] > 0.08  # close to 5 x 2.5 cm
    assert abs(p1[1] - p0[1]) < 0.005 and abs(p1[2] - p0[2]) < 0.005


def test_scripted_controller_succeeds(env):
    ctrl = ScriptedPickPlace.from_env(env)
    results = [run_episode(env, ctrl, seed=s) for s in range(5)]
    assert sum(r["success"] for r in results) >= 4
    assert all(r["collision_steps"] == 0 for r in results)


def test_success_needs_release(env):
    """Putting the object on the platform while still holding it is not a success."""
    ctrl = ScriptedPickPlace.from_env(env)
    obs, info = env.reset(seed=0)
    ctrl.reset()
    while ctrl.phase != "release":
        obs, _, _, _, info = env.step(ctrl.act(obs))
    for _ in range(4):  # keep squeezing until the object has come to rest on the platform
        obs, _, _, _, info = env.step(np.array([0, 0, 0, 0, -1.0]))
    assert reward_terms(env.state(), env.task)["on_target"] > 0.9
    assert info["grasped"] and not info["is_success"]
    for _ in range(10):
        obs, _, _, _, info = env.step(ctrl.act(obs))
    assert info["is_success"]
