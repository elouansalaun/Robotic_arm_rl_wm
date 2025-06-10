"""Camera-based state estimation for closed-loop control.

The arm part of the state comes from proprioception (joint encoders); the
object part and the task (target position, object size) come from the RGB-D
camera. When the object is inside the closed gripper it is mostly hidden by
the fingers and the palm, so it is assumed to move rigidly with the TCP until
the gripper opens again.
"""

from __future__ import annotations

import numpy as np

from rwm.environments.state import SLICES
from rwm.perception.camera import RGBDCamera, denoise_depth
from rwm.perception.detection import ObjectDetector
from rwm.perception.noise import LightingRandomizer, SensorNoise
from rwm.robotics.kinematics import rot_z
from rwm.simulation.scene import TABLE_Z


class CameraStateEstimator:
    def __init__(self, model, vision_cfg: dict, noise: str = "clean", dt: float = 0.05, seed: int | None = None):
        cam = vision_cfg["camera"]
        self.camera = RGBDCamera(model, cam["name"], cam["width"], cam["height"])
        self.depth_filter = cam.get("depth_filter", 5)
        self.detector = ObjectDetector(vision_cfg)
        self.track_cfg = vision_cfg["tracking"]
        self.rng = np.random.default_rng(seed)
        noise_cfg = vision_cfg["noise"][noise]
        self.noise = SensorNoise(noise_cfg, self.rng)
        self.lighting = LightingRandomizer(model, self.rng) if noise_cfg["lighting"] else None
        self.dt = dt
        self.n_renders = 0
        self.n_rejected = 0

    # ------------------------------------------------------------- frames
    def observe(self, scene):
        rgb, depth = self.camera.capture(scene.data)
        rgb, depth = self.noise.apply(rgb, depth)
        depth = denoise_depth(depth, self.depth_filter)
        points = self.camera.points_world(depth, scene.data)
        self.n_renders += 1
        return self.detector.detect(rgb, points)

    # ----------------------------------------------------------- episode
    def reset(self, scene) -> None:
        if self.lighting is not None:
            self.lighting.sample()
        det, target, _ = self.observe(scene)
        self.lost = det is None
        if det is None:
            # nothing found: aim at the middle of the workspace, the controller will fail honestly
            det_pos, yaw, half_h, half_w, npix = np.array([0.5, 0.0, TABLE_Z + 0.02]), 0.0, 0.02, 0.02, 1
        else:
            det_pos, yaw, half_h, half_w, npix = det.position, det.yaw, det.half_height, det.half_width, det.n_pixels
        if target is None:
            target = np.array([0.5, 0.0, TABLE_Z + 0.02])
        self.task_estimate = np.array([*target, half_h, half_w], dtype=np.float32)
        self.pos = det_pos.copy()
        self.yaw = yaw
        self.vel = np.zeros(3)
        self.expected_pixels = npix
        self.attached = False
        self.offset = np.zeros(3)
        self.last_detection = det

    def task(self) -> np.ndarray:
        return self.task_estimate.copy()

    def _held(self, tcp: np.ndarray, grip: float) -> bool:
        half_w = self.task_estimate[4]
        closed_on_object = 1.2 * half_w < grip < 2 * half_w + 0.008
        return closed_on_object and np.linalg.norm(tcp - self.pos) < self.track_cfg["attach_distance"]

    def estimate(self, scene, true_state: np.ndarray) -> np.ndarray:
        s = true_state.copy()
        tcp = s[SLICES["tcp_pos"]].astype(float)
        grip = float(s[SLICES["grip"]][0])
        prev = self.pos.copy()

        if self.attached and grip > 2 * self.task_estimate[4] + 0.008:
            self.attached = False  # gripper opened
        if not self.attached and self._held(tcp, grip):
            self.attached = True
            self.offset = tcp - self.pos

        if self.attached:
            self.pos = tcp - self.offset
        else:
            det, _, _ = self.observe(scene)
            self.last_detection = det
            if det is not None and self._plausible(det, prev):
                # the height comes from the top face and the size measured at reset,
                # so it stays right when the object sits on the platform
                self.pos = np.array([det.position[0], det.position[1], det.z_top - self.task_estimate[3]])
                if det.n_pixels > 0.7 * self.expected_pixels:
                    self.yaw = det.yaw
            else:
                self.n_rejected += 1

        alpha = self.track_cfg["velocity_smoothing"]
        self.vel = alpha * self.vel + (1 - alpha) * (self.pos - prev) / self.dt
        s[SLICES["obj_pos"]] = self.pos
        R = rot_z(self.yaw)
        s[SLICES["obj_rot"]] = np.concatenate([R[:, 0], R[:, 1]])
        s[SLICES["obj_linvel"]] = self.vel
        s[SLICES["obj_angvel"]] = 0.0
        return s

    def _plausible(self, det, prev: np.ndarray) -> bool:
        """Reject partial views (occlusion by the gripper) and jumps."""
        enough = det.n_pixels >= self.track_cfg["min_visible_fraction"] * self.expected_pixels
        close = np.linalg.norm(det.position[:2] - prev[:2]) < 0.15
        return bool(enough and close)

    def close(self) -> None:
        if self.lighting is not None:
            self.lighting.restore()
        self.camera.close()
