import numpy as np
import pytest

from rwm.config import load_config, scenario_config
from rwm.simulation.scene import TABLE_Z, PickPlaceScene, Scenario, ScenarioSampler


@pytest.fixture(scope="module")
def scene():
    return PickPlaceScene()


@pytest.mark.parametrize("kind,size", [("box", [0.02, 0.025, 0.02]), ("cylinder", [0.022, 0.028]), ("sphere", [0.025])])
def test_object_rests_on_table(scene, kind, size):
    sc = Scenario(object_kind=kind, object_size=size, object_xy=[0.5, 0.1], object_yaw=0.3, object_mass=0.2,
                  target_xy=[0.5, -0.15])
    scene.reset(sc, settle_steps=250)
    pos, _ = scene.object_pose()
    assert abs(pos[2] - (TABLE_Z + sc.half_height)) < 2e-3
    np.testing.assert_allclose(pos[:2], [0.5, 0.1], atol=2e-3)
    lin, ang = scene.object_velocity()
    assert np.linalg.norm(lin) < 1e-2 and np.linalg.norm(ang) < 1e-1


def test_object_parameters_are_applied(scene):
    sc = Scenario(object_kind="box", object_size=[0.02, 0.03, 0.025], object_mass=0.27, object_friction=0.7)
    scene.reset(sc)
    body, geom = scene.active["body"], scene.active["geom"]
    assert scene.model.body_mass[body] == pytest.approx(0.27)
    np.testing.assert_allclose(scene.model.geom_size[geom], [0.02, 0.03, 0.025])
    assert scene.model.geom_friction[geom, 0] == pytest.approx(0.7)
    hy, hz = 0.03, 0.025
    assert scene.model.body_inertia[body][0] == pytest.approx(0.27 / 3 * (hy**2 + hz**2))


def test_home_pose(scene):
    scene.reset(Scenario())
    p, R = scene.tcp_pose()
    np.testing.assert_allclose(p, [0.45, 0.0, 0.70], atol=5e-3)
    assert R[2, 0] < -0.99  # tool pointing down


def test_sampler_is_reproducible():
    cfg = load_config("simulation")
    a = [ScenarioSampler(cfg["scenario"], seed=7).sample() for _ in range(3)]
    b = [ScenarioSampler(cfg["scenario"], seed=7).sample() for _ in range(3)]
    assert [s.to_dict() for s in a] == [s.to_dict() for s in b]


def test_sampler_respects_ranges():
    cfg = load_config("simulation")
    sampler = ScenarioSampler(scenario_config(cfg, "nominal"), seed=0)
    for _ in range(200):
        s = sampler.sample()
        assert 0.40 <= s.object_xy[0] <= 0.62
        assert np.hypot(s.object_xy[0] - s.target_xy[0], s.object_xy[1] - s.target_xy[1]) > 0.15
        assert s.object_kind in ("box", "cylinder")


def test_snapshot_restores_simulation(scene, tmp_path):
    sc = ScenarioSampler(load_config("simulation")["scenario"], seed=3).sample()
    scene.reset(sc)
    ctrl = np.append(scene.home_q + 0.05, 0.02)
    scene.step(ctrl, 50)
    path = tmp_path / "snap.json"
    scene.save_snapshot(path)
    scene.step(ctrl, 100)
    ref = scene.get_state()

    other = PickPlaceScene()
    other.load_snapshot(path)
    other.step(ctrl, 100)
    np.testing.assert_allclose(other.get_state(), ref, atol=1e-10)
