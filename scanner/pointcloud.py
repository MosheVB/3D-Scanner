"""RGB-D frames to colored point clouds."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import open3d as o3d

from scanner.config import DEPTH_STRIDE, MAX_DEPTH_MM


def intrinsics_matrix(intrinsics: Sequence[Sequence[float]]) -> np.ndarray:
    return np.array(intrinsics, dtype=np.float64)


def rgbd_to_pointcloud(
    rgb_bgr: np.ndarray,
    depth_mm: np.ndarray,
    intrinsics: Sequence[Sequence[float]],
    *,
    min_depth_mm: int = 0,
    max_depth_mm: int = MAX_DEPTH_MM,
    stride: int = DEPTH_STRIDE,
) -> o3d.geometry.PointCloud:
    """Unproject valid depth pixels to a colored point cloud (camera frame, meters)."""
    if rgb_bgr.ndim == 2:
        rgb_bgr = np.stack([rgb_bgr] * 3, axis=-1)
    rgb = rgb_bgr[:, :, ::-1].astype(np.float64) / 255.0

    k = intrinsics_matrix(intrinsics)
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]

    h, w = depth_mm.shape
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    uu, vv = np.meshgrid(xs, ys)

    z_mm = depth_mm[vv, uu].astype(np.float64)
    valid = (z_mm > 0) & (z_mm >= min_depth_mm) & (z_mm <= max_depth_mm)
    if not np.any(valid):
        return o3d.geometry.PointCloud()

    z = z_mm[valid] / 1000.0
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    x = (u - cx) * z / fx
    # Image v grows downward; Open3D view uses Y-up so flip to match RGB layout
    y = -(v - cy) * z / fy

    points = np.stack([x, y, z], axis=1)
    colors = rgb[vv[valid], uu[valid]]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud
