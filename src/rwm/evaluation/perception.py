"""Localisation accuracy of the camera pipeline against simulator ground truth."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from rwm.perception.camera import denoise_depth
from rwm.perception.noise import LightingRandomizer, SensorNoise
from rwm.robotics.kinematics import wrap_half_pi


def yaw_error(det_yaw: float, R_obj: np.ndarray, scenario) -> float:
    """Error on the grasp axis, only meaningful for boxes (deg, NaN otherwise)."""
    if scenario.object_kind != "box":
        return np.nan
    gt = np.arctan2(R_obj[1, 0], R_obj[0, 0])
    err = abs(wrap_half_pi(det_yaw - gt))
    hx, hy = scenario.object_size[:2]
    if abs(hx - hy) < 1e-6:  # square section: both axes are valid grasp axes
        err = min(err, abs(wrap_half_pi(det_yaw - gt - np.pi / 2)))
    return float(np.rad2deg(err))


def localisation_errors(scene, camera, detector, vision_cfg: dict, noise: str, scenarios, seed: int = 0,
                        keep_frames: int = 0):
    rng = np.random.default_rng(seed)
    noise_cfg = vision_cfg["noise"][noise]
    sensor = SensorNoise(noise_cfg, rng)
    lighting = LightingRandomizer(scene.model, rng) if noise_cfg["lighting"] else None
    rows, frames = [], []
    for i, sc in enumerate(scenarios):
        scene.reset(sc)
        if lighting is not None:
            lighting.sample()
        rgb, depth = camera.capture(scene.data)
        rgb, depth = sensor.apply(rgb, depth)
        depth = denoise_depth(depth, vision_cfg["camera"].get("depth_filter", 5))
        det, target, mask = detector.detect(rgb, camera.points_world(depth, scene.data))
        pos, _ = scene.object_pose()
        R = scene.object_rotation()
        row = dict(noise=noise, object_kind=sc.object_kind, detected=det is not None, target_detected=target is not None)
        if det is not None:
            row.update(
                xy_error=float(np.linalg.norm(det.position[:2] - pos[:2])),
                z_error=float(abs(det.position[2] - pos[2])),
                yaw_error_deg=yaw_error(det.yaw, R, sc),
                half_width_error=float(abs(det.half_width - sc.half_width)),
                half_height_error=float(abs(det.half_height - sc.half_height)),
                roundness=det.roundness,
                round_object=sc.object_kind != "box",
                n_pixels=det.n_pixels,
            )
        if target is not None:
            row["target_error"] = float(np.linalg.norm(target - scene.target_position()))
        rows.append(row)
        if i < keep_frames:
            frames.append(draw_detection(rgb, mask, det, camera, scene))
    if lighting is not None:
        lighting.restore()
    return pd.DataFrame(rows), frames


def draw_detection(rgb, mask, det, camera, scene) -> np.ndarray:
    img = rgb.copy()
    if mask is not None:
        img[mask] = (0.45 * img[mask] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
    if det is not None:
        c = det.position
        axis = np.array([np.cos(det.yaw), np.sin(det.yaw), 0.0]) * 0.05
        p0, p1 = camera.project(np.stack([c - axis, c + axis]), scene.data)
        cv2.line(img, tuple(int(v) for v in p0), tuple(int(v) for v in p1), (255, 255, 0), 2)
        uv = camera.project(c[None], scene.data)[0]
        cv2.circle(img, (int(uv[0]), int(uv[1])), 3, (255, 255, 0), -1)
    gt, _ = scene.object_pose()
    uv = camera.project(gt[None], scene.data)[0]
    cv2.drawMarker(img, (int(uv[0]), int(uv[1])), (0, 255, 255), cv2.MARKER_CROSS, 10, 1)
    return img


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        d = g[g.detected]
        return pd.Series({
            "detection_rate": g.detected.mean(),
            "xy_error_median_mm": 1e3 * d.xy_error.median(),
            "xy_error_p95_mm": 1e3 * d.xy_error.quantile(0.95),
            "z_error_median_mm": 1e3 * d.z_error.median(),
            "yaw_error_median_deg": d.yaw_error_deg.median(),
            "yaw_error_p95_deg": d.yaw_error_deg.quantile(0.95),
            "width_error_median_mm": 1e3 * d.half_width_error.median() * 2,
            "target_error_median_mm": 1e3 * g.target_error.median(),
        })

    return df.groupby(["condition", "noise"], sort=False).apply(agg, include_groups=False).round(2)
