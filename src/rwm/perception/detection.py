"""Object and target localisation from one RGB-D frame.

Pipeline:
    1. back-project the depth map to world coordinates
    2. keep pixels inside the workspace volume above the table
    3. colour segmentation in HSV: saturated pixels are objects, green ones
       are the target platform (robot, table and distractors are grey)
    4. largest connected component -> 3D points
    5. geometry: the top face gives the height, the footprint (minimum-area
       rectangle of the points projected on the table) gives the centre, the
       width and the yaw of the narrow side
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from rwm.simulation.scene import TABLE_Z


@dataclass
class Detection:
    position: np.ndarray        # object centre, world frame
    yaw: float                  # direction of the narrow side, in [-pi/2, pi/2)
    half_height: float
    half_width: float
    half_length: float
    roundness: float            # area(points hull) / area(min rectangle): ~1 box, ~0.79 disc
    n_pixels: int
    z_top: float

    @property
    def shape(self) -> str:
        return "round" if self.roundness < 0.88 else "box"


def _largest_component(mask: np.ndarray, min_pixels: int) -> np.ndarray | None:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[best, cv2.CC_STAT_AREA] < min_pixels:
        return None
    return labels == best


def footprint(points_xy: np.ndarray):
    """Minimum-area rectangle of 2D points: centre, (half width, half length), yaw of the narrow side, roundness."""
    pts = (points_xy * 1000.0).astype(np.float32)  # mm, cv2 wants float32
    rect = cv2.minAreaRect(pts)
    corners = cv2.boxPoints(rect) / 1000.0
    e1, e2 = corners[1] - corners[0], corners[2] - corners[1]
    l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
    narrow = e1 if l1 <= l2 else e2
    yaw = float((np.arctan2(narrow[1], narrow[0]) + np.pi / 2) % np.pi - np.pi / 2)
    hull_area = cv2.contourArea(cv2.convexHull(pts)) / 1e6
    rect_area = max(l1 * l2, 1e-9)
    centre = np.asarray(rect[0]) / 1000.0
    return centre, 0.5 * min(l1, l2), 0.5 * max(l1, l2), yaw, hull_area / rect_area


class ObjectDetector:
    def __init__(self, cfg: dict):
        self.seg = cfg["segmentation"]
        self.pose_cfg = cfg["pose"]
        k = self.seg["open_kernel"]
        self.kernel = np.ones((k, k), np.uint8)

    def masks(self, rgb: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Boolean object and target masks before connected-component filtering."""
        s = self.seg
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        height = points[..., 2] - TABLE_Z
        in_roi = (
            (points[..., 0] > s["roi_x"][0]) & (points[..., 0] < s["roi_x"][1])
            & (points[..., 1] > s["roi_y"][0]) & (points[..., 1] < s["roi_y"][1])
            & (height > s["min_height"]) & (height < s["max_height"])
        )
        coloured = (sat >= s["min_saturation"]) & (val >= s["min_value"])
        green = (h >= s["target_hue"][0]) & (h <= s["target_hue"][1])
        obj = in_roi & coloured & ~green
        target = in_roi & coloured & green
        obj = cv2.morphologyEx(obj.astype(np.uint8), cv2.MORPH_OPEN, self.kernel) > 0
        target = cv2.morphologyEx(target.astype(np.uint8), cv2.MORPH_OPEN, self.kernel) > 0
        return obj, target

    def detect(self, rgb: np.ndarray, points: np.ndarray):
        """Returns (object Detection or None, target position or None, object mask)."""
        obj_mask, target_mask = self.masks(rgb, points)
        obj_mask = _largest_component(obj_mask, self.seg["min_pixels"])
        target_mask = _largest_component(target_mask, self.seg["min_pixels"])
        detection = self._object_pose(points[obj_mask]) if obj_mask is not None else None
        target = self._target_pose(points[target_mask]) if target_mask is not None else None
        return detection, target, obj_mask

    def _object_pose(self, pts: np.ndarray) -> Detection | None:
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) < self.pose_cfg["min_top_points"]:
            return None
        z_top = np.percentile(pts[:, 2], 98)
        top = pts[pts[:, 2] > z_top - self.pose_cfg["top_slice"]]
        if len(top) < self.pose_cfg["min_top_points"]:
            return None
        z_top = float(np.median(top[:, 2]))
        half_h = 0.5 * (z_top - TABLE_Z)  # valid while the object stands on the table
        # Boxes and cylinders are prisms: the top face gives the centre and the
        # orientation. The width is measured on the upper half of the object,
        # which also works for spheres and leaves out the silhouette pixels
        # that took the depth of the table behind ("flying pixels").
        centre, _, _, yaw, roundness = footprint(top[:, :2])
        upper = pts[pts[:, 2] > TABLE_Z + half_h]
        _, half_w, half_l, _, _ = footprint(upper[:, :2] if len(upper) >= 3 else top[:, :2])
        position = np.array([centre[0], centre[1], TABLE_Z + half_h])
        return Detection(position, yaw, float(half_h), float(half_w), float(half_l), float(roundness), len(pts), z_top)

    def _target_pose(self, pts: np.ndarray) -> np.ndarray | None:
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) < self.pose_cfg["min_top_points"]:
            return None
        z_top = np.percentile(pts[:, 2], 95)
        top = pts[pts[:, 2] > z_top - self.pose_cfg["top_slice"]]
        centre, *_ = footprint(top[:, :2])
        return np.array([centre[0], centre[1], np.median(top[:, 2])])
