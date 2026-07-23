"""Bounded object voxel model rendered into the capture preview (right panel)."""

from __future__ import annotations

from typing import Callable, Sequence

import cv2
import numpy as np
import open3d as o3d

from scanner.config import (
    LIVE_DEPTH_BAND_MM,
    LIVE_DEPTH_STRIDE,
    RGB_HEIGHT,
    RGB_WIDTH,
    TURNTABLE_ROTATE_STEP_DEG,
    VOXEL_MODEL_EXTENT_M,
    VOXEL_MODEL_VOXEL_M,
)
from scanner.object_mask import object_mask_from_click
from scanner.pointcloud import rgbd_to_pointcloud
from scanner.turntable_pose import (
    TurntableGeometry,
    analytical_geometry_from_frame0,
    transform_cam_to_object,
)
from scanner.voxel_model import ObjectVoxelModel


def _placeholder_bgr(width: int, height: int, message: str) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (24, 24, 28)
    cv2.putText(
        img,
        message,
        (16, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return img


class LivePointCloudView:
    """Object-only voxel fusion: background rejected, RAM capped by grid size."""

    def __init__(self) -> None:
        self._focus_mm: float | None = None
        self._seed_uv: tuple[int, int] | None = None
        self._intrinsics: Sequence[Sequence[float]] | None = None
        self._model: ObjectVoxelModel | None = None
        self._prev_cloud: o3d.geometry.PointCloud | None = None
        self._o3d_intrinsic: o3d.camera.PinholeCameraIntrinsic | None = None
        self._world_transform = np.eye(4)
        self._frame_count = 0
        self._analytical_turntable = False
        self._rotate_step_deg = TURNTABLE_ROTATE_STEP_DEG
        self._turntable_geom: TurntableGeometry | None = None
        self._vis: o3d.visualization.Visualizer | None = None
        self._bounds_set = False
        self._last_integrated = 0
        self._mask_fn: Callable[[np.ndarray], np.ndarray | None] | None = None

    def set_mask_fn(
        self, fn: Callable[[np.ndarray], np.ndarray | None] | None
    ) -> None:
        """Optional override for per-frame bool mask (web click bounds)."""
        self._mask_fn = fn

    def _frame_mask(self, depth_mm: np.ndarray) -> np.ndarray | None:
        if self._mask_fn is not None:
            custom = self._mask_fn(depth_mm)
            if custom is not None:
                return custom
        if self._seed_uv is None or self._focus_mm is None:
            return None
        return object_mask_from_click(
            depth_mm,
            self._seed_uv[0],
            self._seed_uv[1],
            center_mm=self._focus_mm,
        )

    def init_from_scan_mask(
        self,
        intrinsics: Sequence[Sequence[float]],
        *,
        seed_uv: tuple[int, int],
        focus_mm: float,
    ) -> None:
        """Initialize voxel model from web click mask seed."""
        self.set_focus(focus_mm, intrinsics, seed_uv)

    @property
    def seed_uv(self) -> tuple[int, int] | None:
        return self._seed_uv

    @property
    def intrinsics(self) -> Sequence[Sequence[float]] | None:
        return self._intrinsics

    @property
    def active(self) -> bool:
        return (
            self._focus_mm is not None
            and self._seed_uv is not None
            and self._intrinsics is not None
            and self._model is not None
        )

    @property
    def focus_mm(self) -> float | None:
        return self._focus_mm

    @property
    def point_count(self) -> int:
        if self._model is None:
            return 0
        return self._model.occupied_count

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def auto_init_from_center(
        self,
        depth_mm: np.ndarray,
        intrinsics: Sequence[Sequence[float]],
        *,
        patch: int = 21,
    ) -> bool:
        """Seed voxel model from depth at image center (turntable / web scan)."""
        h, w = depth_mm.shape[:2]
        u, v = w // 2, h // 2
        half = patch // 2
        roi = depth_mm[
            max(0, v - half) : min(h, v + half + 1),
            max(0, u - half) : min(w, u + half + 1),
        ]
        valid = roi[roi > 0]
        if valid.size < 10:
            return False
        focus_mm = float(np.median(valid))
        self.set_focus(focus_mm, intrinsics, (u, v))
        return True

    def enable_analytical_turntable(
        self, *, rotate_step_deg: float = TURNTABLE_ROTATE_STEP_DEG
    ) -> None:
        """Fixed camera: integrate each saved frame at known spin angle (no odometry)."""
        self._analytical_turntable = True
        self._rotate_step_deg = float(rotate_step_deg)
        self._turntable_geom = None

    def set_focus(
        self,
        focus_mm: float,
        intrinsics: Sequence[Sequence[float]],
        seed_uv: tuple[int, int],
    ) -> None:
        self._focus_mm = focus_mm
        self._seed_uv = seed_uv
        self._intrinsics = intrinsics
        self._world_transform = np.eye(4)
        self._frame_count = 0
        self._prev_cloud = None
        self._o3d_intrinsic = None
        self._turntable_geom = None
        self._bounds_set = False
        self._last_integrated = 0

        u, v = seed_uv
        k = np.asarray(intrinsics, dtype=np.float64)
        z = focus_mm / 1000.0
        x = (u - k[0, 2]) * z / k[0, 0]
        y = -(v - k[1, 2]) * z / k[1, 1]
        center = np.array([x, y, z], dtype=np.float64)
        self._model = ObjectVoxelModel(
            center,
            extent_m=VOXEL_MODEL_EXTENT_M,
            voxel_m=VOXEL_MODEL_VOXEL_M,
        )
        n = self._model.grid_size[0]
        print(
            f"Object voxel model {n}³ @ {VOXEL_MODEL_VOXEL_M * 1000:.1f} mm "
            f"({self._model.memory_mb:.1f} MB) — depth "
            f"{focus_mm - LIVE_DEPTH_BAND_MM:.0f}–{focus_mm + LIVE_DEPTH_BAND_MM:.0f} mm"
        )

    def _frame_cloud(self, rgb_bgr: np.ndarray, depth_mm: np.ndarray) -> o3d.geometry.PointCloud:
        assert self._focus_mm is not None and self._intrinsics is not None
        mask = self._frame_mask(depth_mm)
        if mask is None or not np.any(mask):
            return o3d.geometry.PointCloud()

        center = self._focus_mm
        band = LIVE_DEPTH_BAND_MM
        cloud = rgbd_to_pointcloud(
            rgb_bgr,
            depth_mm,
            self._intrinsics,
            min_depth_mm=max(0, int(center - band)),
            max_depth_mm=int(center + band),
            stride=LIVE_DEPTH_STRIDE,
        )
        if len(cloud.points) == 0:
            return cloud

        # Drop points outside the 2D object mask (table / background).
        k = np.asarray(self._intrinsics, dtype=np.float64)
        pts = np.asarray(cloud.points)
        fx, fy = k[0, 0], k[1, 1]
        cx, cy = k[0, 2], k[1, 2]
        z = pts[:, 2]
        u = np.rint(pts[:, 0] * fx / z + cx).astype(np.int32)
        v = np.rint(-pts[:, 1] * fy / z + cy).astype(np.int32)
        h, w = depth_mm.shape
        keep = (
            (u >= 0)
            & (u < w)
            & (v >= 0)
            & (v < h)
            & mask[v, u]
        )
        if not np.any(keep):
            return o3d.geometry.PointCloud()
        filtered = o3d.geometry.PointCloud()
        filtered.points = o3d.utility.Vector3dVector(pts[keep])
        filtered.colors = o3d.utility.Vector3dVector(np.asarray(cloud.colors)[keep])
        return filtered

    def accumulate(
        self,
        rgb_bgr: np.ndarray,
        depth_mm: np.ndarray,
        *,
        rotation_deg: float | None = None,
        tilt_deg: float = 0.0,
    ) -> None:
        """Integrate masked depth into the bounded voxel model."""
        if not self.active or self._model is None:
            return

        mask = self._frame_mask(depth_mm)
        if mask is None or not np.any(mask):
            return

        if self._analytical_turntable:
            if self._turntable_geom is None:
                self._turntable_geom = analytical_geometry_from_frame0(
                    depth_mm,
                    mask,
                    self._intrinsics or [],
                )
            if rotation_deg is None:
                rotation_deg = float(self._frame_count) * self._rotate_step_deg
            self._world_transform = transform_cam_to_object(
                self._turntable_geom,
                tilt_deg=tilt_deg,
                rotation_deg=rotation_deg,
            )
        else:
            self._world_transform = np.eye(4)

        self._last_integrated = self._model.integrate_masked_rgbd(
            rgb_bgr,
            depth_mm,
            self._intrinsics or [],
            mask,
            world_transform=self._world_transform,
            stride=LIVE_DEPTH_STRIDE,
        )
        self._frame_count += 1

    def render_panel_bgr(self, width: int = RGB_WIDTH, height: int = RGB_HEIGHT) -> np.ndarray:
        if not self.active:
            return _placeholder_bgr(width, height, "Click object to start 3D")

        assert self._model is not None
        display = self._model.to_point_cloud()
        if len(display.points) == 0:
            return _placeholder_bgr(width, height, "Masking object... move slowly")

        if self._vis is None:
            self._vis = o3d.visualization.Visualizer()
            self._vis.create_window(
                "pcl_render",
                width=width,
                height=height,
                visible=False,
            )
            opt = self._vis.get_render_option()
            if opt is not None:
                opt.point_size = 2.0
                opt.background_color = np.array([0.08, 0.08, 0.1])

        self._vis.clear_geometries()
        self._vis.add_geometry(display, reset_bounding_box=not self._bounds_set)
        self._bounds_set = True
        ctr = self._vis.get_view_control()
        if ctr is not None and self._frame_count <= 2:
            ctr.set_front([0.0, 0.0, -1.0])
            ctr.set_up([0.0, 1.0, 0.0])
            ctr.set_lookat(display.get_center())
        self._vis.poll_events()
        self._vis.update_renderer()

        buffer = np.asarray(self._vis.capture_screen_float_buffer(do_render=True))
        if buffer.size == 0:
            return _placeholder_bgr(
                width,
                height,
                f"Building... {self.point_count} voxels",
            )
        rgb = (np.clip(buffer, 0, 1) * 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if bgr.shape[0] != height or bgr.shape[1] != width:
            bgr = cv2.resize(bgr, (width, height))
        cv2.putText(
            bgr,
            f"{self.point_count} voxels | fuse {self._frame_count}",
            (8, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        return bgr

    def export_point_cloud(self) -> o3d.geometry.PointCloud:
        if self._model is None:
            return o3d.geometry.PointCloud()
        return self._model.to_point_cloud()

    def poll(self) -> None:
        if self._vis is not None:
            self._vis.poll_events()

    def close(self) -> None:
        if self._vis is not None:
            self._vis.destroy_window()
            self._vis = None
        self._focus_mm = None
        self._seed_uv = None
        self._intrinsics = None
        self._model = None
        self._prev_cloud = None
        self._turntable_geom = None
        self._o3d_intrinsic = None
        self._world_transform = np.eye(4)
        self._frame_count = 0
        self._bounds_set = False
        self._last_integrated = 0
        self._mask_fn = None
