"""Depth frame cleanup before TSDF / posed fusion integration."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import open3d as o3d

from scanner.pointcloud import intrinsics_matrix, rgbd_to_pointcloud

# D405 turntable working range; excludes platter below / background beyond object.
DEPTH_CLIP_MIN_MM = 100
DEPTH_CLIP_MAX_MM = 400
# Conservative RGBD depth_trunc (meters) — allows TSDF to use nearby valid depth at holes.
DEPTH_TRUNC_M = 0.40
OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO = 3.0
HOLE_FILL_MEDIAN_K = 5


def clip_depth_mm(
    depth_mm: np.ndarray,
    *,
    min_mm: int = DEPTH_CLIP_MIN_MM,
    max_mm: int = DEPTH_CLIP_MAX_MM,
) -> np.ndarray:
    """Zero depth outside the working range."""
    out = depth_mm.copy()
    out[(out < min_mm) | (out > max_mm)] = 0
    return out


def fill_small_depth_holes(
    depth_mm: np.ndarray,
    *,
    k: int = HOLE_FILL_MEDIAN_K,
) -> np.ndarray:
    """Fill isolated invalid pixels from a local median (white / textureless patches)."""
    if k < 3:
        return depth_mm
    valid = depth_mm > 0
    if not np.any(valid):
        return depth_mm
    med = cv2.medianBlur(depth_mm.astype(np.uint16), k)
    kernel = np.ones((k, k), np.uint8)
    near_valid = cv2.dilate(valid.astype(np.uint8), kernel) > 0
    holes = near_valid & ~valid
    out = depth_mm.copy()
    out[holes] = med[holes]
    return out


def _project_cloud_to_depth(
    cloud: o3d.geometry.PointCloud,
    intrinsics: Sequence[Sequence[float]],
    shape: tuple[int, int],
) -> np.ndarray:
    """Scatter filtered points back to a depth image (nearest depth per pixel)."""
    h, w = shape
    out = np.zeros((h, w), dtype=np.uint16)
    if len(cloud.points) == 0:
        return out
    k = intrinsics_matrix(intrinsics)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    pts = np.asarray(cloud.points)
    z_mm = (pts[:, 2] * 1000.0).astype(np.float64)
    u = np.rint(pts[:, 0] * fx / pts[:, 2] + cx).astype(np.int32)
    v = np.rint(-pts[:, 1] * fy / pts[:, 2] + cy).astype(np.int32)
    keep = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z_mm > 0)
    u, v, z_mm = u[keep], v[keep], z_mm[keep]
    order = np.argsort(z_mm)
    out[v[order], u[order]] = np.clip(z_mm[order], 0, 65535).astype(np.uint16)
    return out


def statistical_outlier_depth(
    depth_mm: np.ndarray,
    intrinsics: Sequence[Sequence[float]],
    *,
    nb_neighbors: int = OUTLIER_NB_NEIGHBORS,
    std_ratio: float = OUTLIER_STD_RATIO,
    stride: int = 2,
) -> np.ndarray:
    """Per-frame statistical outlier removal via unproject -> filter -> reproject."""
    cloud = rgbd_to_pointcloud(
        np.zeros((*depth_mm.shape, 3), dtype=np.uint8),
        depth_mm,
        intrinsics,
        min_depth_mm=DEPTH_CLIP_MIN_MM,
        max_depth_mm=DEPTH_CLIP_MAX_MM,
        stride=stride,
    )
    if len(cloud.points) < nb_neighbors + 5:
        return depth_mm
    filtered, _ = cloud.remove_statistical_outlier(
        nb_neighbors=int(nb_neighbors),
        std_ratio=float(std_ratio),
    )
    if len(filtered.points) < 50:
        return depth_mm
    return _project_cloud_to_depth(filtered, intrinsics, depth_mm.shape)


def preprocess_depth_frame(
    depth_mm: np.ndarray,
    intrinsics: Sequence[Sequence[float]],
    *,
    min_mm: int = DEPTH_CLIP_MIN_MM,
    max_mm: int = DEPTH_CLIP_MAX_MM,
    nb_neighbors: int = OUTLIER_NB_NEIGHBORS,
    std_ratio: float = OUTLIER_STD_RATIO,
    hole_fill_k: int = HOLE_FILL_MEDIAN_K,
    apply_outlier: bool = True,
) -> np.ndarray:
    """Clip, hole-fill, and statistical outlier filter one depth frame."""
    depth = clip_depth_mm(depth_mm, min_mm=min_mm, max_mm=max_mm)
    depth = fill_small_depth_holes(depth, k=hole_fill_k)
    if apply_outlier:
        depth = statistical_outlier_depth(
            depth,
            intrinsics,
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
    return clip_depth_mm(depth, min_mm=min_mm, max_mm=max_mm)
