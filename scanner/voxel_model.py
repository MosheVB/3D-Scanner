"""Bounded voxel grid for object-only live fusion (RAM does not grow with frames)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import open3d as o3d

from scanner.config import (
    LIVE_DEPTH_STRIDE,
    VOXEL_MODEL_EXTENT_M,
    VOXEL_MODEL_MAX_AXIS,
    VOXEL_MODEL_MIN_WEIGHT,
    VOXEL_MODEL_VOXEL_M,
)
from scanner.pointcloud import intrinsics_matrix


class ObjectVoxelModel:
    """Fixed-size voxel volume centered on the scan subject."""

    def __init__(
        self,
        center_m: np.ndarray,
        *,
        extent_m: float = VOXEL_MODEL_EXTENT_M,
        voxel_m: float = VOXEL_MODEL_VOXEL_M,
        max_axis: int = VOXEL_MODEL_MAX_AXIS,
    ) -> None:
        self.voxel_m = float(voxel_m)
        axis = int(np.ceil(float(extent_m) / self.voxel_m))
        axis = max(8, min(axis, max_axis))
        self._dims = (axis, axis, axis)
        # Physical span matches actual grid size (not requested extent if axis-capped).
        self.extent_m = axis * self.voxel_m
        self._origin = np.asarray(center_m, dtype=np.float64).reshape(3) - (
            0.5 * self.extent_m
        )

        n = int(np.prod(self._dims))
        self._weights = np.zeros(n, dtype=np.uint16)
        self._colors = np.zeros((n, 3), dtype=np.float32)
        self._observations = 0

    @property
    def observations(self) -> int:
        return self._observations

    @property
    def grid_size(self) -> tuple[int, int, int]:
        return self._dims

    @property
    def occupied_count(self) -> int:
        return int(np.count_nonzero(self._weights >= VOXEL_MODEL_MIN_WEIGHT))

    @property
    def memory_mb(self) -> float:
        return (self._weights.nbytes + self._colors.nbytes) / (1024 * 1024)

    def _indices_for_points(self, points_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if points_m.size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=bool)
        rel = (points_m - self._origin) / self.voxel_m
        ix = np.floor(rel[:, 0]).astype(np.int32)
        iy = np.floor(rel[:, 1]).astype(np.int32)
        iz = np.floor(rel[:, 2]).astype(np.int32)
        nx, ny, nz = self._dims
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
        flat = ix + nx * (iy + ny * iz)
        return flat.astype(np.int64), inside

    def integrate_masked_rgbd(
        self,
        rgb_bgr: np.ndarray,
        depth_mm: np.ndarray,
        intrinsics: Sequence[Sequence[float]],
        mask: np.ndarray,
        *,
        world_transform: np.ndarray,
        stride: int = LIVE_DEPTH_STRIDE,
    ) -> int:
        """Fuse masked depth into the voxel grid. Returns points integrated this frame."""
        if rgb_bgr.ndim == 2:
            rgb_bgr = np.stack([rgb_bgr] * 3, axis=-1)
        rgb = rgb_bgr[:, :, ::-1].astype(np.float32) / 255.0

        k = intrinsics_matrix(intrinsics)
        fx, fy = float(k[0, 0]), float(k[1, 1])
        cx, cy = float(k[0, 2]), float(k[1, 2])

        h, w = depth_mm.shape
        ys = np.arange(0, h, stride)
        xs = np.arange(0, w, stride)
        uu, vv = np.meshgrid(xs, ys)
        z_mm = depth_mm[vv, uu].astype(np.float32)
        m = mask[vv, uu]
        valid = m & (z_mm > 0)
        if not np.any(valid):
            return 0

        z = z_mm[valid] / 1000.0
        u = uu[valid].astype(np.float64)
        v = vv[valid].astype(np.float64)
        x = (u - cx) * z / fx
        y = -(v - cy) * z / fy
        cam = np.stack([x, y, z], axis=1)
        rot = world_transform[:3, :3]
        trans = world_transform[:3, 3]
        world = (cam @ rot.T) + trans

        colors = rgb[vv[valid], uu[valid]]
        flat, inside = self._indices_for_points(world)
        if not np.any(inside):
            return 0

        flat = flat[inside]
        colors = colors[inside]
        world = world[inside]

        # Average observations that land in the same voxel this frame.
        order = np.argsort(flat)
        flat = flat[order]
        colors = colors[order]
        unique, starts = np.unique(flat, return_index=True)
        counts = np.diff(np.append(starts, len(flat)))

        updated = 0
        for idx, start, count in zip(unique, starts, counts):
            w = int(self._weights[idx])
            if w == 0:
                self._colors[idx] = colors[start : start + count].mean(axis=0)
                self._weights[idx] = min(65535, int(count))
                updated += 1
            else:
                add_w = int(count)
                new_w = min(65535, w + add_w)
                if new_w > w:
                    old = self._colors[idx]
                    incoming = colors[start : start + count].mean(axis=0)
                    t = add_w / float(new_w)
                    self._colors[idx] = old * (1.0 - t) + incoming * t
                    self._weights[idx] = new_w
                    updated += 1

        self._observations += 1
        return int(world.shape[0])

    def to_point_cloud(self, *, min_weight: int = VOXEL_MODEL_MIN_WEIGHT) -> o3d.geometry.PointCloud:
        """Extract occupied voxels for display or export (bounded point count)."""
        idx = np.flatnonzero(self._weights >= min_weight)
        if idx.size == 0:
            return o3d.geometry.PointCloud()

        nx, ny, nz = self._dims
        iz = idx // (nx * ny)
        rem = idx % (nx * ny)
        iy = rem // nx
        ix = rem % nx

        centers = self._origin + (np.stack([ix, iy, iz], axis=1).astype(np.float64) + 0.5) * (
            self.voxel_m
        )
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(centers)
        cloud.colors = o3d.utility.Vector3dVector(self._colors[idx].astype(np.float64))
        return cloud
