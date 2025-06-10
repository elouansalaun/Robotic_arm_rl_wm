"""Simulated RGB-D camera: rendering, intrinsics, extrinsics, back-projection.

Image conventions follow OpenCV: u to the right, v down, camera frame with
x right, y down, z forward. MuJoCo cameras look along their -z axis with y
up, hence the axis flip in `extrinsics`.
"""

from __future__ import annotations

import cv2
import mujoco
import numpy as np

MJ_TO_CV = np.diag([1.0, -1.0, -1.0])


def intrinsics(fovy_deg: float, width: int, height: int) -> np.ndarray:
    f = 0.5 * height / np.tan(0.5 * np.deg2rad(fovy_deg))
    return np.array([[f, 0.0, 0.5 * width], [0.0, f, 0.5 * height], [0.0, 0.0, 1.0]])


def denoise_depth(depth: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Median filter that tolerates missing pixels (NaN)."""
    if ksize <= 1:
        return depth
    filled = np.nan_to_num(depth, nan=0.0).astype(np.float32)
    out = cv2.medianBlur(filled, ksize)
    out[out <= 0.0] = np.nan
    return out


class RGBDCamera:
    def __init__(self, model: mujoco.MjModel, name: str = "front_cam", width: int = 480, height: int = 360):
        self.model = model
        self.name = name
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if self.cam_id < 0:
            raise ValueError(f"no camera named {name}")
        self.width, self.height = width, height
        self.K = intrinsics(model.cam_fovy[self.cam_id], width, height)
        self.renderer = mujoco.Renderer(model, height, width)
        # pixel grid, centre of each pixel
        u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
        self._rays = np.stack([(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1]], axis=-1)

    def capture(self, data: mujoco.MjData, segmentation: bool = False):
        """Return (rgb uint8 HxWx3, depth float32 HxW in metres along the optical axis)."""
        r = self.renderer
        r.update_scene(data, camera=self.cam_id)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(data, camera=self.cam_id)
        depth = r.render().astype(np.float32).copy()
        r.disable_depth_rendering()
        if not segmentation:
            return rgb, depth
        r.enable_segmentation_rendering()
        r.update_scene(data, camera=self.cam_id)
        seg = r.render()[..., 0].copy()  # geom id per pixel, -1 for background
        r.disable_segmentation_rendering()
        return rgb, depth, seg

    def extrinsics(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Camera position and rotation (world <- OpenCV camera frame)."""
        pos = data.cam_xpos[self.cam_id].copy()
        R = data.cam_xmat[self.cam_id].reshape(3, 3) @ MJ_TO_CV
        return pos, R

    def points_camera(self, depth: np.ndarray) -> np.ndarray:
        """Back-project every pixel, HxWx3 points in the OpenCV camera frame."""
        z = depth[..., None]
        return np.concatenate([self._rays * z, z], axis=-1)

    def points_world(self, depth: np.ndarray, data: mujoco.MjData) -> np.ndarray:
        pos, R = self.extrinsics(data)
        return self.points_camera(depth) @ R.T + pos

    def project(self, points_world: np.ndarray, data: mujoco.MjData) -> np.ndarray:
        """World points -> pixel coordinates (u, v)."""
        pos, R = self.extrinsics(data)
        pc = (np.atleast_2d(points_world) - pos) @ R
        uv = pc[:, :2] / pc[:, 2:3]
        return uv * np.diag(self.K)[:2] + self.K[:2, 2]

    def close(self) -> None:
        self.renderer.close()
