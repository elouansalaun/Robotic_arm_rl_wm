"""Sensor imperfections: depth noise and dropout, RGB noise, lighting changes."""

from __future__ import annotations

import numpy as np


class SensorNoise:
    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def apply(self, rgb: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c, rng = self.cfg, self.rng
        if c["rgb_sigma"] > 0:
            noisy = rgb.astype(np.float32) + rng.normal(0.0, c["rgb_sigma"], rgb.shape)
            rgb = np.clip(noisy, 0, 255).astype(np.uint8)
        sigma = c["depth_sigma"] + c["depth_sigma_quadratic"] * depth**2
        if np.any(sigma > 0):
            depth = depth + rng.normal(size=depth.shape).astype(np.float32) * sigma
        if c["depth_dropout"] > 0:
            depth = depth.copy()
            depth[rng.random(depth.shape) < c["depth_dropout"]] = np.nan
        return rgb, depth


class LightingRandomizer:
    """Changes intensity and direction of the scene lights (shading and shadows move)."""

    def __init__(self, model, rng: np.random.Generator):
        self.model = model
        self.rng = rng
        self.nominal = (model.light_diffuse.copy(), model.light_dir.copy(), model.vis.headlight.ambient.copy(),
                        model.vis.headlight.diffuse.copy())

    def sample(self) -> None:
        m, rng = self.model, self.rng
        diffuse, direction, ambient, head = self.nominal
        m.light_diffuse[:] = diffuse * rng.uniform(0.3, 1.5, size=(len(diffuse), 1))
        d = direction + rng.normal(0.0, 0.4, size=direction.shape)
        d[:, 2] = -np.abs(d[:, 2]) - 0.3
        m.light_dir[:] = d / np.linalg.norm(d, axis=1, keepdims=True)
        m.vis.headlight.ambient[:] = ambient * rng.uniform(0.3, 1.6)
        m.vis.headlight.diffuse[:] = head * rng.uniform(0.3, 1.6)

    def restore(self) -> None:
        diffuse, direction, ambient, head = self.nominal
        self.model.light_diffuse[:] = diffuse
        self.model.light_dir[:] = direction
        self.model.vis.headlight.ambient[:] = ambient
        self.model.vis.headlight.diffuse[:] = head
