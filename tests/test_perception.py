import numpy as np
import pytest

from rwm.config import load_config
from rwm.perception.camera import RGBDCamera, denoise_depth
from rwm.perception.detection import ObjectDetector, footprint
from rwm.simulation.scene import PickPlaceScene, Scenario


@pytest.fixture(scope="module")
def setup():
    cfg = load_config("vision")
    scene = PickPlaceScene()
    cam = RGBDCamera(scene.model, "front_cam", 320, 240)
    yield scene, cam, ObjectDetector(cfg)
    cam.close()


def test_projection_round_trip(setup):
    scene, cam, _ = setup
    scene.reset(Scenario())
    _, depth = cam.capture(scene.data)
    pts = cam.points_world(depth, scene.data)
    v, u = 150, 170
    uv = cam.project(pts[v, u][None], scene.data)[0]
    np.testing.assert_allclose(uv, [u + 0.5, v + 0.5], atol=1e-6)


def test_depth_hits_table(setup):
    scene, cam, _ = setup
    scene.reset(Scenario(object_xy=[0.45, -0.2], target_xy=[0.45, 0.2]))
    _, depth = cam.capture(scene.data)
    pts = cam.points_world(depth, scene.data)
    table = pts[(pts[..., 0] > 0.55) & (pts[..., 0] < 0.8) & (np.abs(pts[..., 1]) < 0.1)]
    assert len(table) > 100
    assert np.abs(np.median(table[:, 2]) - 0.42) < 1e-3


@pytest.mark.parametrize("kind,size,yaw", [("box", [0.02, 0.02, 0.02], 0.4), ("cylinder", [0.024, 0.026], 0.0),
                                           ("box", [0.016, 0.03, 0.016], -0.6)])
def test_localisation_clean(setup, kind, size, yaw):
    scene, cam, det = setup
    sc = Scenario(object_kind=kind, object_size=size, object_xy=[0.52, -0.08], object_yaw=yaw, target_xy=[0.45, 0.15])
    scene.reset(sc)
    rgb, depth = cam.capture(scene.data)
    d, target, _ = det.detect(rgb, cam.points_world(depth, scene.data))
    pos, _ = scene.object_pose()
    assert np.linalg.norm(d.position - pos) < 0.004
    assert abs(d.half_height - sc.half_height) < 0.002
    assert abs(d.half_width - sc.half_width) < 0.004
    assert np.linalg.norm(target - scene.target_position()) < 0.004
    if kind == "box":
        err = (d.yaw - yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4  # boxes: defined modulo 90 deg
        assert abs(err) < np.deg2rad(4)
        assert d.shape == "box"
    else:
        assert d.shape == "round"


def test_footprint_rectangle():
    rng = np.random.default_rng(0)
    yaw = 0.3
    local = rng.uniform([-0.02, -0.04], [0.02, 0.04], size=(500, 2))
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    pts = local @ R.T + [0.5, 0.1]
    centre, half_w, half_l, est_yaw, roundness = footprint(pts)
    np.testing.assert_allclose(centre, [0.5, 0.1], atol=2e-3)
    assert abs(half_w - 0.02) < 2e-3 and abs(half_l - 0.04) < 2e-3
    assert abs(est_yaw - yaw) < 0.05  # narrow side = local x axis
    assert roundness > 0.9


def test_denoise_keeps_edges():
    depth = np.full((40, 40), 1.0, np.float32)
    depth[:, 20:] = 0.8
    depth[5, 5] = np.nan
    out = denoise_depth(depth, 5)
    assert np.isfinite(out[5, 5]) and out[5, 5] == pytest.approx(1.0)
    assert out[10, 18] == pytest.approx(1.0) and out[10, 21] == pytest.approx(0.8)


def test_estimator_tracks_object_and_skips_near_gripper():
    from rwm.environments.state import SLICES, read_state
    from rwm.perception.estimator import CameraStateEstimator

    scene = PickPlaceScene()
    est = CameraStateEstimator(scene.model, load_config("vision"), noise="realistic", seed=0)
    sc = Scenario(object_kind="box", object_size=[0.02, 0.02, 0.02], object_xy=[0.5, -0.1], target_xy=[0.5, 0.15])
    scene.reset(sc)
    est.reset(scene)
    truth = read_state(scene)
    s = est.estimate(scene, truth)
    assert np.linalg.norm(s[SLICES["obj_pos"]] - truth[SLICES["obj_pos"]]) < 0.005
    np.testing.assert_allclose(est.task()[:3], scene.target_position(), atol=0.005)

    # gripper (open) right above the object: the camera is not trusted, the estimate is kept
    near = truth.copy()
    near[SLICES["tcp_pos"]] = truth[SLICES["obj_pos"]] + [0.0, 0.0, 0.03]
    near[SLICES["grip"]] = 0.09
    before = est.pos.copy()
    est.estimate(scene, near)
    assert est.n_skipped == 1
    np.testing.assert_array_equal(est.pos, before)
    est.close()
