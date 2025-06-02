import mujoco
import numpy as np
import pytest

from rwm.robotics.kinematics import ArmKinematics, rotation_log, tool_yaw, top_down_rotation
from rwm.simulation.scene import SCENE_XML


@pytest.fixture(scope="module")
def sim():
    m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    d = mujoco.MjData(m)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    return m, d, site


@pytest.fixture(scope="module")
def kin():
    return ArmKinematics()


def _set(sim, q):
    m, d, _ = sim
    d.qpos[:6] = q
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)


def test_forward_matches_mujoco(sim, kin):
    _, d, site = sim
    rng = np.random.default_rng(0)
    for _ in range(100):
        q = rng.uniform(kin.lower, kin.upper)
        _set(sim, q)
        p, R = kin.forward(q)
        np.testing.assert_allclose(p, d.site_xpos[site], atol=1e-9)
        np.testing.assert_allclose(R, d.site_xmat[site].reshape(3, 3), atol=1e-9)


def test_jacobian_matches_mujoco(sim, kin):
    m, d, site = sim
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = rng.uniform(kin.lower, kin.upper)
        _set(sim, q)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, site)
        J = kin.jacobian(q)
        np.testing.assert_allclose(J[:3], jacp[:, :6], atol=1e-9)
        np.testing.assert_allclose(J[3:], jacr[:, :6], atol=1e-9)


def test_closed_form_ik_round_trip(kin):
    rng = np.random.default_rng(2)
    solved = 0
    for _ in range(300):
        p = np.array([rng.uniform(0.32, 0.70), rng.uniform(-0.30, 0.30), rng.uniform(0.44, 0.65)])
        R = top_down_rotation(rng.uniform(-np.pi / 2, np.pi / 2))
        q = kin.inverse(p, R)
        if q is None:
            continue
        solved += 1
        p2, R2 = kin.forward(q)
        np.testing.assert_allclose(p2, p, atol=1e-9)
        np.testing.assert_allclose(R2, R, atol=1e-9)
        assert kin.within_limits(q)
    # the operating region above the table must be reachable with the tool pointing down
    assert solved == 300


def test_all_ik_branches_are_valid(kin):
    q_true = np.array([0.3, 0.5, 0.4, 0.2, 0.9, -0.4])
    p, R = kin.forward(q_true)
    sols = kin.inverse_all(p, R)
    assert len(sols) >= 2
    assert any(np.allclose(q, q_true, atol=1e-9) for q in sols)
    for q in sols:
        p2, R2 = kin.forward(q)
        np.testing.assert_allclose(p2, p, atol=1e-9)
        np.testing.assert_allclose(R2, R, atol=1e-9)


def test_dls_converges(kin):
    q0 = kin.inverse(np.array([0.45, 0.0, 0.70]), top_down_rotation(0.0))
    p_goal = np.array([0.55, 0.12, 0.50])
    R_goal = top_down_rotation(0.4)
    q, ok = kin.inverse_dls(p_goal, R_goal, q0, iters=100)
    assert ok
    p, R = kin.forward(q)
    assert np.linalg.norm(p - p_goal) < 1e-3
    assert np.linalg.norm(rotation_log(R @ R_goal.T)) < 1e-2


def test_top_down_rotation():
    for yaw in np.linspace(-1.5, 1.5, 7):
        R = top_down_rotation(yaw)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)
        np.testing.assert_allclose(R[:, 0], [0, 0, -1])
        assert np.isclose(tool_yaw(R), yaw)


def test_rotation_log():
    rng = np.random.default_rng(3)
    for _ in range(20):
        w = rng.normal(size=3)
        w *= rng.uniform(0.01, 3.0) / np.linalg.norm(w)
        # Rodrigues
        theta = np.linalg.norm(w)
        k = w / theta
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
        np.testing.assert_allclose(rotation_log(R), w, atol=1e-8)
